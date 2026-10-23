import argparse
import os
import torch
import numpy as np

# Reuse existing utils
from dataset.synthetic import load_synthetic_dataset
from utils.metrics import compute_psnr, compute_ssim_windowed_3d, compute_ssim_windowed_2d, compute_k_iou
from utils.vis_utils import (
    visualize_diffusivity_combined,
    visualize_surface_temperature_time,
    visualize_error_distribution
)
from network.net.embedder import Embedder, annealing_beta
from network.net.thermo_nerf_pinn import ThermoNERFPINN

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
            
            if val == 'True': val = True
            elif val == 'False': val = False
            elif val == 'None': val = None
            else:
                try: val = int(val)
                except ValueError:
                    try: val = float(val)
                    except ValueError:
                        if (val.startswith("'") and val.endswith("'")) or \
                           (val.startswith('"') and val.endswith('"')):
                            val = val[1:-1]
            setattr(args, key, val)
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to final.tar')
    parser.add_argument('--args_file', type=str, default=None,
                        help='Training args.txt (default: next to the checkpoint)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write metrics.txt and figures '
                             '(default: <checkpoint dir>/output_visualization)')
    parser.add_argument('--datadir', type=str, default=None,
                        help='Override the dataset path stored in args.txt')
    cmd_args = parser.parse_args()

    ckpt_dir = os.path.dirname(cmd_args.checkpoint)
    if cmd_args.args_file is None:
        cmd_args.args_file = os.path.join(ckpt_dir, 'args.txt')
    if cmd_args.output_dir is None:
        cmd_args.output_dir = os.path.join(ckpt_dir, 'output_visualization')
    
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    
    print(f"Loading args from {cmd_args.args_file}...")
    args = load_args_from_file(cmd_args.args_file)
    if cmd_args.datadir is not None:
        args.datadir = cmd_args.datadir
    # Must match the coordinate convention used by the trainer
    normalize_coords = getattr(args, 'normalize_coords', True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load GT
    print("\nLoading ground truth data...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_synthetic_dataset(args.datadir, map_location="cpu")
    k_gt = data['k_gt'][args.eval_idx].numpy()
    surface_gt = data['surface_gt'][args.eval_idx].numpy()
    
    print(f"Ground truth k: range=[{k_gt.min():.4f}, {k_gt.max():.4f}]")
    
    # Load Model
    print(f"\nLoading PINN model from {cmd_args.checkpoint}...")
    
    # Recreate Embedders
    diff_embedder = Embedder(
        include_input=True, input_dims=3, 
        max_freq_power=args.multires-1, num_freqs=args.multires, 
        log_sampling=True, periodic_fns=[torch.sin, torch.cos]
    )
    diff_input_ch = diff_embedder.out_dim
    
    temp_embedder = Embedder(
        include_input=True, input_dims=4, 
        max_freq_power=args.multires-1, num_freqs=args.multires, 
        log_sampling=True, periodic_fns=[torch.sin, torch.cos]
    )
    temp_input_ch = temp_embedder.out_dim
    
    # Instantiate
    model = ThermoNERFPINN(
        diff_net_args={
                'D': args.netdepth, 'W': args.netwidth, 'input_ch': diff_input_ch, 'output_ch': 1,
                'skips': [4], 'min_diffusivity': args.min_diffusivity, 'max_diffusivity': args.max_diffusivity,
                'use_grid': getattr(args, 'use_grid', False), 'grid_resolution': (args.resolution_x, args.resolution_y, args.resolution_z),
                'domain_size': (args.domain_x, args.domain_y, args.domain_z),
                'output_activation': getattr(args, 'output_activation', 'sigmoid'),
                'mlp_activation': getattr(args, 'mlp_activation', 'softplus'),
        },
        temp_net_args={
            'D': args.netdepth, 'W': args.netwidth, 'input_ch': temp_input_ch, 'output_ch': 1, 'skips': [4]
        }
    ).to(device)
    
    checkpoint = torch.load(cmd_args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Predict Diffusivity
    print("\nPredicting diffusivity field...")
    Nx, Ny, Nz = args.resolution_x, args.resolution_y, args.resolution_z
    Lx, Ly, Lz = args.domain_x, args.domain_y, args.domain_z
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
    
    x = torch.linspace(dx/2, Lx - dx/2, Nx, device=device)
    y = torch.linspace(dy/2, Ly - dy/2, Ny, device=device)
    z = torch.linspace(dz/2, Lz - dz/2, Nz, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    T_max = args.dt * args.num_steps
    if normalize_coords:
        coords_in = 2.0 * (coords / torch.tensor([Lx, Ly, Lz], device=device)) - 1.0
    else:
        coords_in = coords
    
    with torch.no_grad():
        # Encoding bandwidth at the checkpoint's step (full bandwidth once annealing is done)
        beta = annealing_beta(checkpoint.get('global_step', float('inf')), args.multires,
                              getattr(args, 'anneal_iters', args.N_iters // 4), getattr(args, 'use_fa', True))
        emb = diff_embedder.embed(coords_in, beta=beta)
        outs = []
        chunk = args.netchunk
        for i in range(0, emb.shape[0], chunk):
            outs.append(model.forward_diffusivity(emb[i:i+chunk]).cpu())
        k_pred = torch.cat(outs, dim=0).squeeze().numpy().reshape(Nx, Ny, Nz)
        
    print(f"Predicted k: range=[{k_pred.min():.4f}, {k_pred.max():.4f}]")
    
    # Predict Surface Temperature (using PINN T network)
    print("\nPredicting surface temperature with PINN T-network...")
    
    time_steps = surface_gt.shape[0]
    z_surf = Lz - dz/2
    
    # Coordinates (2D surface)
    xx_s, yy_s = torch.meshgrid(x, y, indexing='ij')
    base_coords = torch.stack([xx_s.flatten(), yy_s.flatten()], dim=-1) # [N_surf, 2]
    N_surf = base_coords.shape[0]
    
    surface_preds = []
    
    # Frame i is at t = i * dt (frame 0 is the initial condition), as in the trainer.
    for i in range(time_steps):
        t_val = i * args.dt
        
        z_col = torch.full((N_surf, 1), z_surf, device=device)
        t_col = torch.full((N_surf, 1), t_val, device=device)
        
        pts = torch.cat([base_coords, z_col, t_col], dim=-1) # [N_surf, 4]
        if normalize_coords:
            pts = 2.0 * (pts / torch.tensor([Lx, Ly, Lz, T_max], device=device)) - 1.0
        
        with torch.no_grad():
            emb_t = temp_embedder.embed(pts)
            # Process in chunks if needed
            t_chunk_out = []
            for j in range(0, emb_t.shape[0], chunk):
                t_chunk_out.append(model.forward_temperature(emb_t[j:j+chunk]).cpu())
            out = torch.cat(t_chunk_out, dim=0)
            
        surface_preds.append(out.reshape(Nx, Ny))
        
    surface_pred = torch.stack(surface_preds, dim=0).numpy()
    print(f"Predicted surface: shape={surface_pred.shape}")
    
    # Metrics
    print("\n" + "="*60)
    print("Evaluation Metrics (PINN)")
    print("="*60)
    
    k_mse = np.mean((k_pred - k_gt)**2)
    k_mae = np.mean(np.abs(k_pred - k_gt))
    k_max_error = np.max(np.abs(k_pred - k_gt))
    k_psnr = compute_psnr(k_pred, k_gt)
    k_ssim = compute_ssim_windowed_3d(k_pred, k_gt)
    k_iou = compute_k_iou(k_pred, k_gt, threshold=0.03)
    
    print(f"\nDiffusivity (k):")
    print(f"  MSE:          {k_mse:.6f}")
    print(f"  PSNR:         {k_psnr:.2f} dB")
    print(f"  SSIM:         {k_ssim:.4f}")
    print(f"  IOU:          {k_iou:.4f}")

    surface_mse = np.mean((surface_pred - surface_gt) ** 2)
    surface_mae = np.mean(np.abs(surface_pred - surface_gt))
    final_mse = np.mean((surface_pred[-1] - surface_gt[-1]) ** 2)
    final_psnr = compute_psnr(surface_pred[-1], surface_gt[-1])
    final_ssim = compute_ssim_windowed_2d(surface_pred[-1], surface_gt[-1])
    overall_psnr = compute_psnr(surface_pred, surface_gt)

    print(f"\nSurface Temperature (PINN prediction):")
    print(f"  MSE (all):    {surface_mse:.6f}")
    print(f"  PSNR (all):   {overall_psnr:.2f} dB")
    print(f"  PSNR (final): {final_psnr:.2f} dB")
    
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
