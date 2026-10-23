import argparse
import os
import torch
import numpy as np
from dataset.synthetic import load_synthetic_dataset
from network.net.thermal_unet import ThermalUNet
from network.sim.simulator import DifferentiableHeatSimulator
from utils.metrics import compute_psnr, compute_ssim_windowed_3d, compute_ssim_windowed_2d, compute_k_iou
from utils.vis_utils import (
    visualize_diffusivity_combined,
    visualize_surface_temperature_time,
    visualize_error_distribution
)

def load_args_from_file(args_path):
    """Load training args from args.txt file."""
    # Simple parser for the unet args
    args = argparse.Namespace()
    with open(args_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Try to convert to int/float/bool
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                setattr(args, key, value)
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best.tar / final.tar')
    parser.add_argument('--eval_idx', type=int, required=True, help='Index of the specimen to evaluate')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write metrics.txt and figures (default: <checkpoint dir>/eval_<idx>)')
    parser.add_argument('--datadir', type=str, default='./data/synthetic_1000',
                        help='Evaluation dataset (the defect benchmark, also for the Sound-Only model)')
    parser.add_argument('--args_file', type=str, default=None,
                        help='Training args.txt (default: next to the checkpoint)')
    cmd_args = parser.parse_args()

    # Determine paths
    ckpt_dir = os.path.dirname(cmd_args.checkpoint)
    if cmd_args.args_file is None:
        cmd_args.args_file = os.path.join(ckpt_dir, 'args.txt')
    if cmd_args.output_dir is None:
        cmd_args.output_dir = os.path.join(ckpt_dir, f'eval_{cmd_args.eval_idx}')
    
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(cmd_args.checkpoint, map_location=device, weights_only=False)

    # Training args: args.txt next to the checkpoint if present, otherwise the copy
    # embedded in the checkpoint (e.g. a downloaded checkpoints/unet_sound_only.tar).
    if os.path.exists(cmd_args.args_file):
        print(f"Loading args from {cmd_args.args_file}...")
        args = load_args_from_file(cmd_args.args_file)
    else:
        print(f"No args file at {cmd_args.args_file}; using the args stored in the checkpoint")
        args = argparse.Namespace(**checkpoint.get('args', {}))
    
    print(f"Using device: {device}")
    
    # Load Data (Complete dataset, then select index)
    # UNet needs the surface input.
    print("Loading data...")
    data = load_synthetic_dataset(cmd_args.datadir, map_location="cpu")
    
    # The U-Net predicts 16 depth slices, and the surface re-simulation below uses the
    # benchmark settings (10 x 10 x 1 domain, 64 x 64 x 16 grid, 101 frames, dt = 0.05,
    # Gaussian initial condition). Other datasets are not supported here.
    k_shape = tuple(data['k_gt'].shape[1:])
    s_shape = tuple(data['surface_gt'].shape[1:])
    if k_shape != (64, 64, 16) or s_shape != (101, 64, 64):
        raise ValueError(f"evaluate_unet.py expects k_gt samples of shape (64, 64, 16) and "
                         f"surface_gt samples of shape (101, 64, 64); got {k_shape} and {s_shape}")
    # Datasets written by this release record their simulation settings; check them.
    expected = {'dt': 0.05, 'num_steps': 100, 'domain_size': {'x': 10.0, 'y': 10.0, 'z': 1.0}}
    for key, value in expected.items():
        if key in data and data[key] != value:
            raise ValueError(f"evaluate_unet.py re-simulates with the benchmark setting {key}={value}, "
                             f"but the dataset has {key}={data[key]}")
    if not 0 <= cmd_args.eval_idx < data['k_gt'].shape[0]:
        raise ValueError(f"--eval_idx {cmd_args.eval_idx} out of range for {data['k_gt'].shape[0]} samples")

    k_gt = data['k_gt'][cmd_args.eval_idx].numpy() # (H, W, D) (64, 64, 16)
    surface_gt = data['surface_gt'][cmd_args.eval_idx] # (T, H, W)
    
    # Prepare input for Model
    # Model expects (B, 1, T, H, W)
    surface_input = surface_gt.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, T, H, W)
    
    # Load Model
    print("Loading model...")
    model = ThermalUNet(
        min_diffusivity=getattr(args, 'min_diffusivity', 0.003),
        max_diffusivity=getattr(args, 'max_diffusivity', 0.25)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Predict K
    print("Predicting Diffusivity...")
    with torch.no_grad():
        k_pred_tensor = model(surface_input) # (1, 1, D, H, W) (1, 1, 16, 64, 64)
    
    # Convert to standard format (H, W, D) for metrics/vis
    # Permute from (1, 1, D, H, W) -> (D, H, W) -> (H, W, D)
    k_pred = k_pred_tensor.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    print(f"K GT Shape: {k_gt.shape}")
    print(f"K Pred Shape: {k_pred.shape}")
    
    # Run Simulation for Surface Consistency
    print("Running Simulation...")
    # Benchmark solver settings (validated against the data shapes above)
    domain_size = (10.0, 10.0, 1.0)
    resolution = (64, 64, 16)
    
    simulator = DifferentiableHeatSimulator(
        domain_size=domain_size,
        resolution=resolution,
        dt=0.05, # Standard for synthetic_1000
        num_steps=100,
        jacobi_iters=50,
        device=device
    )
    
    # Initial condition used by the dataset generator (fixed for all samples)
    heat_source_params = {
        'center': (5.0, 5.0, 0.8),
        'intensity': 100.0,
        'radius': 0.5
    }
    
    T0 = simulator.create_gaussian_heat_source(
        center=heat_source_params['center'],
        intensity=heat_source_params['intensity'],
        radius=heat_source_params['radius']
    )
    if 't0_gt' in data and not torch.allclose(T0.cpu(), data['t0_gt'][cmd_args.eval_idx], atol=1e-3):
        raise ValueError("The dataset's initial condition (t0_gt) differs from the benchmark's "
                         "Gaussian initial condition used for re-simulation.")
    
    with torch.no_grad():
        k_pred_sim = k_pred_tensor.squeeze(0).squeeze(0) # (D, H, W)
        # The simulator expects (Nx, Ny, Nz); the U-Net predicts (D, H, W).
        k_pred_sim = k_pred_sim.permute(1, 2, 0) # (H, W, D)
        
        surface_pred_seq = simulator(k_pred_sim, T0)
        surface_pred = surface_pred_seq.cpu().numpy() # (T, H, W)
        
    surface_gt_np = surface_gt.numpy()
    
    # Metrics
    print("Computing metrics...")
    
    k_mse = np.mean((k_pred - k_gt)**2)
    k_mae = np.mean(np.abs(k_pred - k_gt))
    k_max_error = np.max(np.abs(k_pred - k_gt))
    k_psnr = compute_psnr(k_pred, k_gt)
    k_ssim = compute_ssim_windowed_3d(k_pred, k_gt)
    k_iou = compute_k_iou(k_pred, k_gt, threshold=0.03)

    surface_mse = np.mean((surface_pred - surface_gt_np) ** 2)
    final_psnr = compute_psnr(surface_pred[-1], surface_gt_np[-1])
    overall_psnr = compute_psnr(surface_pred, surface_gt_np)
    
    print(f"Diffusivity Result:")
    print(f"MSE: {k_mse:.6f}, PSNR: {k_psnr:.2f}, SSIM: {k_ssim:.4f}, IOU: {k_iou:.4f}")
    
    # Save Metrics
    with open(os.path.join(cmd_args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"k_mse: {k_mse}\n")
        f.write(f"k_mae: {k_mae}\n")
        f.write(f"k_max_error: {k_max_error}\n")
        f.write(f"k_psnr: {k_psnr}\n")
        f.write(f"k_ssim: {k_ssim}\n")
        f.write(f"k_iou: {k_iou}\n")
        f.write(f"surface_mse: {surface_mse}\n")
        f.write(f"surface_psnr_final: {final_psnr}\n")
        f.write(f"surface_psnr_overall: {overall_psnr}\n")
        
    # Visualizations
    print("Saving visualizations...")
    visualize_diffusivity_combined(
        k_pred=k_pred,
        k_gt=k_gt,
        save_path=os.path.join(cmd_args.output_dir, "diffusivity_combined.png"),
        domain_size=domain_size,
        threshold=0.03
    )
    visualize_surface_temperature_time(
        surface_gt=surface_gt_np,
        surface_pred=surface_pred,
        save_path=os.path.join(cmd_args.output_dir, "surface_temperature_time.png")
    )
    visualize_error_distribution(
        k_gt=k_gt,
        k_pred=k_pred,
        save_path=os.path.join(cmd_args.output_dir, "error_distribution.png")
    )
    
    print("Done.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
    main()
