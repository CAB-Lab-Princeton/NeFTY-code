import os
import sys
if "--gpu" in sys.argv:
    gpu = sys.argv[sys.argv.index("--gpu") + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
import time
import numpy as np
import torch
import configargparse
from tqdm import tqdm

from utils.general_utils import get_best_gpu_device
from dataset.synthetic import load_synthetic_dataset
from network.trainer_pinn import ThermoNERFPINNTrainer

# Enable TensorFloat32 matmuls on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('yes', 'true', 't', 'y', '1')

def config_parser():
    """Parse command line arguments."""
    parser = configargparse.ArgumentParser()
    
    # Experiment
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, default='thermo_pinn_baseline',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs',
                        help='where to store checkpoints and logs')
    parser.add_argument("--datadir", type=str, default=None,
                        help='input data directory (None for synthetic)')
    parser.add_argument("--eval_idx", type=int, default=0,
                        help="Index of sample to train on when loading a saved dataset")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU id(s) to expose via CUDA_VISIBLE_DEVICES; command line only "
                             "(read before torch is imported). Default: keep the environment's "
                             "CUDA_VISIBLE_DEVICES")
    
    # Network
    parser.add_argument("--netdepth", type=int, default=8)
    parser.add_argument("--netwidth", type=int, default=256)
    parser.add_argument("--netchunk", type=int, default=1024*64)
    parser.add_argument("--multires", type=int, default=10)
    parser.add_argument("--i_embed", type=int, default=0)
    parser.add_argument("--anneal_iters", type=int, default=2500,
                        help='Number of iterations for frequency annealing')
    
    # Training
    parser.add_argument("--lrate", type=float, default=1e-4)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--N_iters", type=int, default=20000, 
                        help="Number of iterations (PINNs usually need more)")
    parser.add_argument("--seed", type=int, default=0, help='Random seed')
    
    # PINN Specifics
    parser.add_argument("--lambda_pde", type=float, default=0.1)
    parser.add_argument("--lambda_ic", type=float, default=1.0)
    parser.add_argument("--lambda_data", type=float, default=10.0) # Weight data heavily
    # PDE weight schedule (warmup + ramp)
    parser.add_argument("--lambda_pde_warmup", type=int, default=0,
                        help="Number of iterations to keep PDE weight at start value")
    parser.add_argument("--lambda_pde_ramp", type=int, default=2000,
                        help="Number of iterations to linearly ramp PDE weight to target")
    parser.add_argument("--lambda_pde_start", type=float, default=0.0,
                        help="Starting PDE weight for warmup/ramp schedule")
    # GradNorm-style balancing
    parser.add_argument("--use_gradnorm", type=str2bool, default=False,
                        help="Enable GradNorm-style adaptive loss weighting")
    parser.add_argument("--gradnorm_alpha", type=float, default=0.5,
                        help="GradNorm alpha (balance strength)")
    parser.add_argument("--gradnorm_warmup", type=int, default=0,
                        help="Number of iterations before GradNorm activates")
    parser.add_argument("--gradnorm_update_every", type=int, default=1,
                        help="Update GradNorm weights every N iterations")
    parser.add_argument("--n_pde", type=int, default=4096)
    parser.add_argument("--n_ic", type=int, default=1024)
    parser.add_argument("--n_data", type=int, default=1024)
    
    # Domain
    parser.add_argument("--domain_x", type=float, default=10.0)
    parser.add_argument("--domain_y", type=float, default=10.0)
    parser.add_argument("--domain_z", type=float, default=1.0)
    parser.add_argument("--resolution_x", type=int, default=32)
    parser.add_argument("--resolution_y", type=int, default=32)
    parser.add_argument("--resolution_z", type=int, default=8)
    
    # Simulation (For compatibility and IC params)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--jacobi_iters", type=int, default=30)
    
    # Heat source
    parser.add_argument("--heat_source_x", type=float, default=5.0)
    parser.add_argument("--heat_source_y", type=float, default=5.0)
    parser.add_argument("--heat_source_z", type=float, default=0.8)
    parser.add_argument("--heat_intensity", type=float, default=100.0)
    parser.add_argument("--heat_radius", type=float, default=0.5)
    
    # Ablation
    parser.add_argument("--use_grid", type=str2bool, default=False)
    parser.add_argument("--mean_type", type=str, default='harmonic')
    parser.add_argument("--output_activation", type=str, default='sigmoid')
    parser.add_argument("--use_fa", type=str2bool, default=True)
    parser.add_argument("--no_pe", type=str2bool, default=False)
    parser.add_argument("--no_sl", type=str2bool, default=False)
    parser.add_argument("--mlp_activation", type=str, default='softplus',
                        choices=['softplus', 'relu'],
                        help="Hidden activation of the diffusivity MLP")
    parser.add_argument("--normalize_coords", type=str2bool, default=True,
                        help="Map (x, y, z, t) to [-1, 1] before positional encoding "
                             "(False = raw physical coordinates)")

    # Diffusivity bounds
    parser.add_argument("--min_diffusivity", type=float, default=0.005)
    parser.add_argument("--max_diffusivity", type=float, default=0.2)
    
    # Regularization (kept for compatibility)
    parser.add_argument("--reg_type", type=str, default='tv')
    parser.add_argument("--reg_weight", type=float, default=1e-3)
    parser.add_argument("--num_loss_frames", type=int, default=10)
    
    # Logging
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_weights", type=int, default=2000)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.05)
    parser.add_argument("--early_stop_patience", type=int, default=300)
    parser.add_argument("--disable_tqdm", action="store_true")
    
    # Acceleration (ignored in PINN but kept for config compatibility)
    parser.add_argument("--use_compile", type=str2bool, default=False)
    parser.add_argument("--use_adjoint", type=str2bool, default=False)
    
    return parser

def train():
    parser = config_parser()
    args = parser.parse_args()
    
    if args.no_pe:
        args.i_embed = -1
        
    # Create log directory
    logdir = os.path.join(args.basedir, args.expname)
    os.makedirs(logdir, exist_ok=True)
    
    # Save args
    with open(os.path.join(logdir, 'args.txt'), 'w') as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            f.write('{} = {}\n'.format(arg, attr))
            
    # Device
    device_str = get_best_gpu_device(min_free_memory_gb=2.0)
    device = torch.device(device_str if "cuda" in device_str else "cpu")
    
    # Data
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.datadir is None:
        raise ValueError("Please set --datadir")
        
    data = load_synthetic_dataset(args.datadir, map_location="cpu")
    eval_idx = args.eval_idx
    surface_gt = data['surface_gt'][eval_idx].to(device)
    
    print(f"Loaded sample {eval_idx}. Surface shape: {surface_gt.shape}")
    
    # Trainer
    trainer = ThermoNERFPINNTrainer(args)
    # Ensure trainer uses same device
    trainer.device = device
    trainer.model.to(device)
    trainer.diff_embedder.freq_bands = trainer.diff_embedder.freq_bands.to(device)
    trainer.temp_embedder.freq_bands = trainer.temp_embedder.freq_bands.to(device)
    
    print("\nStarting PINN training...")
    if args.disable_tqdm:
        pbar = range(1, args.N_iters + 1)
    else:
        pbar = tqdm(range(1, args.N_iters + 1))
        
    train_start_time = time.time()
    
    for i in pbar:
        metrics = trainer.train_step(surface_gt)
        
        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {metrics['loss']:.6f} Data: {metrics['loss_data']:.6f} "
                f"PDE: {metrics['loss_pde']:.6f} PSNR (batch): {metrics['psnr']:.2f} "
                f"GradDiff: {metrics['grad_norm_diff']:.3e} GradTemp: {metrics['grad_norm_temp']:.3e} "
                f"LambdaData: {metrics['lambda_data']:.3e} LambdaIC: {metrics['lambda_ic']:.3e} "
                f"LambdaPDE: {metrics['lambda_pde']:.3e}"
            )
            if not args.disable_tqdm:
                pbar.set_description(f"Loss: {metrics['loss']:.4f} PSNR: {metrics['psnr']:.2f}")
                
        # if i % args.i_weights == 0:
        #     trainer.save_checkpoint(os.path.join(logdir, f'{i:06d}.tar'))
            
    # Final save
    trainer.save_checkpoint(os.path.join(logdir, 'final.tar'))
    print(f"Training complete. Saved to {logdir}/final.tar")

if __name__ == "__main__":
    # Force CUDA initialization
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
    train()
