import argparse
import os
import torch
import numpy as np

from dataset.synthetic import load_synthetic_dataset
from network.net.embedder import annealing_beta, get_embedder
from network.net.thermo_nerf import ThermoNeRF
from network.sim.simulator import DifferentiableHeatSimulator
from utils.metrics import compute_psnr, compute_ssim_windowed_3d, compute_ssim_windowed_2d, compute_k_iou
from utils.vis_utils import (
    visualize_diffusivity_combined,
    visualize_surface_temperature_time,
    visualize_error_distribution
)

def load_args_from_file(args_path):
    """Load training args from args.txt file."""
    args = argparse.Namespace()
    with open(args_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            
            # Simple type inference
            if val == 'True':
                val = True
            elif val == 'False':
                val = False
            elif val == 'None':
                val = None
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        # Keep as string (remove quotes if present)
                        if (val.startswith("'") and val.endswith("'")) or \
                           (val.startswith('"') and val.endswith('"')):
                            val = val[1:-1]
            
            setattr(args, key, val)
    
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best.tar / final.tar written by train.py')
    parser.add_argument('--args_file', type=str, default=None,
                        help='Training args.txt (default: next to the checkpoint)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write metrics.txt and figures '
                             '(default: <checkpoint dir>/output_visualization)')
    parser.add_argument('--datadir', type=str, default=None,
                        help='Override the dataset path stored in args.txt')
    cmd_args = parser.parse_args()

    # Derive defaults from checkpoint if not provided
    ckpt_dir = os.path.dirname(cmd_args.checkpoint)
    if cmd_args.args_file is None:
        cmd_args.args_file = os.path.join(ckpt_dir, 'args.txt')
    if cmd_args.output_dir is None:
        cmd_args.output_dir = os.path.join(ckpt_dir, 'output_visualization')
    
    # Create output directory
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    
    # Load training args
    print(f"Loading args from {cmd_args.args_file}...")
    args = load_args_from_file(cmd_args.args_file)
    if cmd_args.datadir is not None:
        args.datadir = cmd_args.datadir
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the ground-truth sample used for training
    print("\nLoading ground truth data...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # Same seed as training
    data = load_synthetic_dataset(args.datadir, map_location="cpu")
    
    k_gt = data['k_gt'][args.eval_idx].numpy()
    surface_gt = data['surface_gt'][args.eval_idx].numpy()
    
    print(f"Ground truth k: shape={k_gt.shape}, range=[{k_gt.min():.4f}, {k_gt.max():.4f}]")
    print(f"Ground truth surface: shape={surface_gt.shape}")
    
    # Create model and load checkpoint
    print(f"\nLoading checkpoint from {cmd_args.checkpoint}...")
    
    # Create embedder
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    
    # Create model
    model = ThermoNeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=1,
        skips=[4],
        min_diffusivity=args.min_diffusivity,
        max_diffusivity=args.max_diffusivity,
        use_grid=getattr(args, 'use_grid', False),
        grid_resolution=(args.resolution_x, args.resolution_y, args.resolution_z),
        domain_size=(args.domain_x, args.domain_y, args.domain_z),
        output_activation=getattr(args, 'output_activation', 'sigmoid'),
        mlp_activation=getattr(args, 'mlp_activation', 'softplus'),
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(cmd_args.checkpoint, map_location=device, weights_only=True)
    
    # Handle torch.compile prefix: keys may have "_orig_mod." prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_state_dict[key[10:]] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    global_step = checkpoint.get('global_step', float('inf'))
    print(f"Loaded checkpoint at step {global_step}")
    
    # Create coordinate grid
    Nx, Ny, Nz = args.resolution_x, args.resolution_y, args.resolution_z
    Lx, Ly, Lz = args.domain_x, args.domain_y, args.domain_z
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    
    x = torch.linspace(dx/2, Lx - dx/2, Nx, device=device)
    y = torch.linspace(dy/2, Ly - dy/2, Ny, device=device)
    z = torch.linspace(dz/2, Lz - dz/2, Nz, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    
    # Predict diffusivity
    print("\nPredicting diffusivity field...")
    with torch.no_grad():
        # Encoding bandwidth at the checkpoint's step (full bandwidth once annealing is done)
        beta = annealing_beta(global_step, args.multires,
                              getattr(args, 'anneal_iters', args.N_iters // 4), getattr(args, 'use_fa', True))
        print(f"Encoding bandwidth beta = {beta:.3f} (of {args.multires})")
        coords_embedded = embed_fn(coords, beta=beta)
        
        chunk = args.netchunk
        outputs = []
        for i in range(0, coords_embedded.shape[0], chunk):
            outputs.append(model(coords_embedded[i:i+chunk]).cpu())
        k_pred = torch.cat(outputs, dim=0).squeeze().numpy().reshape(Nx, Ny, Nz)
    
    print(f"Predicted k: range=[{k_pred.min():.4f}, {k_pred.max():.4f}]")
    
    # Run simulation with predicted k
    print("\nRunning simulation with predicted diffusivity...")
    simulator = DifferentiableHeatSimulator(
        domain_size=(args.domain_x, args.domain_y, args.domain_z),
        resolution=(args.resolution_x, args.resolution_y, args.resolution_z),
        dt=args.dt,
        num_steps=args.num_steps,
        jacobi_iters=args.jacobi_iters,
        mean_type=getattr(args, 'mean_type', 'harmonic'),
        device=device
    )
    
    T0 = simulator.create_gaussian_heat_source(
        center=(args.heat_source_x, args.heat_source_y, args.heat_source_z),
        intensity=args.heat_intensity,
        radius=args.heat_radius
    )
    
    k_pred_torch = torch.tensor(k_pred, dtype=torch.float32, device=device)
    with torch.no_grad():
        surface_pred = simulator(k_pred_torch, T0).cpu().numpy()
    
    print(f"Predicted surface: shape={surface_pred.shape}")
    
    # Compute metrics
    print("\n" + "="*60)
    print("Evaluation Metrics")
    print("="*60)
    
    # Diffusivity metrics
    k_mse = np.mean((k_pred - k_gt)**2)
    k_mae = np.mean(np.abs(k_pred - k_gt))
    k_max_error = np.max(np.abs(k_pred - k_gt))
    k_psnr = compute_psnr(k_pred, k_gt)
    k_ssim = compute_ssim_windowed_3d(k_pred, k_gt)
    k_iou = compute_k_iou(k_pred, k_gt, threshold=0.03)
    
    print(f"\nDiffusivity (k):")
    print(f"  GT range:     [{k_gt.min():.4f}, {k_gt.max():.4f}]")
    print(f"  Pred range:   [{k_pred.min():.4f}, {k_pred.max():.4f}]")
    print(f"  MSE:          {k_mse:.6f}")
    print(f"  MAE:          {k_mae:.6f}")
    print(f"  Max Error:    {k_max_error:.6f}")
    print(f"  PSNR:         {k_psnr:.2f} dB")
    print(f"  SSIM (windowed): {k_ssim:.4f}")
    print(f"  IOU (k<0.03): {k_iou:.4f}")
    
    # Surface temperature metrics
    surface_mse = np.mean((surface_pred - surface_gt) ** 2)
    surface_mae = np.mean(np.abs(surface_pred - surface_gt))
    final_mse = np.mean((surface_pred[-1] - surface_gt[-1]) ** 2)
    final_psnr = compute_psnr(surface_pred[-1], surface_gt[-1])
    final_ssim = compute_ssim_windowed_2d(surface_pred[-1], surface_gt[-1])
    overall_psnr = compute_psnr(surface_pred, surface_gt)

    print(f"\nSurface Temperature:")
    print(f"  MSE (all):    {surface_mse:.6f}")
    print(f"  MAE (all):    {surface_mae:.6f}")
    print(f"  MSE (final):  {final_mse:.6f}")
    print(f"  PSNR (all):   {overall_psnr:.2f} dB")
    print(f"  PSNR (final): {final_psnr:.2f} dB")
    print(f"  SSIM (final): {final_ssim:.4f}")
    # Save metrics
    with open(os.path.join(cmd_args.output_dir, 'metrics.txt'), 'w') as f:
        f.write("Diffusivity Metrics:\n")
        f.write(f"MSE: {k_mse:.6f}\n")
        f.write(f"MAE: {k_mae:.6f}\n")
        f.write(f"PSNR: {k_psnr:.2f}\n")
        f.write(f"SSIM: {k_ssim:.4f}\n\n")
        f.write(f"IOU (k<0.03): {k_iou:.4f}\n\n")
        f.write("Surface Temperature Metrics:\n")
        f.write(f"MSE (all): {surface_mse:.6f}\n")
        f.write(f"MAE (all): {surface_mae:.6f}\n")
        f.write(f"MSE (final): {final_mse:.6f}\n")
        f.write(f"PSNR (all): {overall_psnr:.2f}\n")
        f.write(f"PSNR (final): {final_psnr:.2f}\n")
        f.write(f"SSIM (final): {final_ssim:.4f}\n")
    # Visualizations
    print("\nGenerating visualizations...")
    
    visualize_diffusivity_combined(
        k_pred=torch.tensor(k_pred),
        k_gt=torch.tensor(k_gt),
        save_path=os.path.join(cmd_args.output_dir, 'diffusivity_combined.png'),
        domain_size=(args.domain_x, args.domain_y, args.domain_z)
    )
    
    # Visualize surface temperature every 10 frames
    frame_indices = list(range(0, len(surface_gt), 10))
    visualize_surface_temperature_time(
        surface_gt=torch.tensor(surface_gt),
        surface_pred=torch.tensor(surface_pred),
        save_path=os.path.join(cmd_args.output_dir, 'surface_temperature_time.png'),
        frame_indices=frame_indices
    )
    
    visualize_error_distribution(
        k_gt=torch.tensor(k_gt),
        k_pred=torch.tensor(k_pred),
        save_path=os.path.join(cmd_args.output_dir, 'error_distribution.png')
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
