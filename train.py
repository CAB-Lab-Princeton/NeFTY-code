import os
import sys
if "--gpu" in sys.argv:
    gpu = sys.argv[sys.argv.index("--gpu") + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import configargparse

from utils.general_utils import get_best_gpu_device
from dataset.synthetic import load_synthetic_dataset
from network.trainer import ThermoNeRFImplicitTrainer

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

def config_parser():
    """Parse command line arguments."""
    parser = configargparse.ArgumentParser()
    
    # Experiment
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, default='thermo_nerf_adjoint',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs',
                        help='where to store checkpoints and logs')
    parser.add_argument("--datadir", type=str, default=None,
                        help='dataset directory (containing dataset.pt) or path to a .pt file')
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
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help="lr decays by 0.1 every lrate_decay*1000 steps (ExponentialLR)")
    parser.add_argument("--N_iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0, help='Random seed for reproducibility')
    
    # Domain
    parser.add_argument("--domain_x", type=float, default=10.0)
    parser.add_argument("--domain_y", type=float, default=10.0)
    parser.add_argument("--domain_z", type=float, default=1.0)
    parser.add_argument("--resolution_x", type=int, default=32)
    parser.add_argument("--resolution_y", type=int, default=32)
    parser.add_argument("--resolution_z", type=int, default=8)
    
    # Simulation
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--jacobi_iters", type=int, default=30)
    
    # Adjoint / Acceleration
    def str2bool(v):
        if isinstance(v, bool): return v
        return v.lower() in ('yes', 'true', 't', 'y', '1')

    parser.add_argument("--use_adjoint", type=str2bool, default=False,
                        help='Use the discrete adjoint (memory independent of the number of Jacobi '
                             'iterations; stores one temperature state per time step) instead of '
                             'backpropagating through the unrolled solver (BPTT)')
    parser.add_argument("--use_compile", type=str2bool, default=False,
                        help='Use torch.compile() for acceleration')
    
    # Heat source
    parser.add_argument("--heat_source_x", type=float, default=5.0)
    parser.add_argument("--heat_source_y", type=float, default=5.0)
    parser.add_argument("--heat_source_z", type=float, default=0.8)
    parser.add_argument("--heat_intensity", type=float, default=100.0)
    parser.add_argument("--heat_radius", type=float, default=0.5)
    
    # Ablation
    parser.add_argument("--use_grid", type=str2bool, default=False,
                        help="Replace the neural field by a voxel grid (Grid Opt. baseline)")
    parser.add_argument("--mean_type", type=str, default='harmonic',
                        choices=['arithmetic', 'harmonic'])
    parser.add_argument("--output_activation", type=str, default='sigmoid',
                        choices=['sigmoid', 'softplus'])
    parser.add_argument("--use_fa", type=str2bool, default=True,
                        help="Use frequency annealing")
    parser.add_argument("--no_pe", type=str2bool, default=False, 
                        help="Disable Positional Encoding (sets i_embed=-1)")
    parser.add_argument("--no_sl", type=str2bool, default=False,
                        help="Disable the symmetry warm-up loss (enabled by default)")
    parser.add_argument("--mlp_activation", type=str, default='softplus',
                        choices=['softplus', 'relu'],
                        help="Hidden activation of the diffusivity MLP")

    # Diffusivity bounds
    parser.add_argument("--min_diffusivity", type=float, default=0.005)
    parser.add_argument("--max_diffusivity", type=float, default=0.2)
    
    # Regularization
    parser.add_argument("--reg_type", type=str, default='tv',
                        choices=['none', 'tv', 'tv_aniso'])
    parser.add_argument("--reg_weight", type=float, default=1e-3)
    parser.add_argument("--num_loss_frames", type=int, default=10)
    
    # Logging
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.05,
                        help="Minimum surface-PSNR improvement for best.tar to be updated")
    parser.add_argument("--early_stop_patience", type=int, default=300,
                        help="Interval (steps) at which best.tar is checked/updated "
                             "(training always runs N_iters steps)")
    parser.add_argument("--disable_tqdm", action="store_true",
                        help="Disable tqdm progress bar (useful for batch logging)")
    
    return parser


def train():
    """Main training function."""
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
            f.write(f'{arg} = {getattr(args, arg)}\n')
    
    # Select device
    device_str = get_best_gpu_device(min_free_memory_gb=2.0)
    device = torch.device(device_str if "cuda" in device_str else "cpu")
    
    # Create synthetic data
    np.random.seed(args.seed)  # For reproducibility
    torch.manual_seed(args.seed)
    if args.datadir is None:
        raise ValueError("Please set --datadir to the dataset directory or dataset.pt file.")
    data = load_synthetic_dataset(args.datadir, map_location="cpu")

    eval_idx = args.eval_idx
    k_gt_list = data['k_gt']
    surface_gt_list = data['surface_gt']

    print(f"Loaded 1 sample (eval_index={eval_idx})")
    print(f"Resolution: {data['resolution']}")
    print(f"Frames: {data['surface_gt'].shape[1]}")
    
    # Create trainer
    trainer = ThermoNeRFImplicitTrainer(args)
    
    # Warmup compilation (without affecting model weights)
    if trainer.use_compile:
        print("Warming up torch.compile (this may take a minute)...")
        dummy_gt = surface_gt_list[eval_idx].to(device)
        
        # Save initial state
        initial_model_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        initial_optimizer_state = trainer.optimizer.state_dict()
        initial_scheduler_state = trainer.scheduler.state_dict()
        
        start_time = time.time()
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        trainer.train_step(dummy_gt)  # Trigger compilation
        end_time = time.time()
        
        # Restore initial state
        trainer.model.load_state_dict(initial_model_state)
        trainer.optimizer.load_state_dict(initial_optimizer_state)
        trainer.scheduler.load_state_dict(initial_scheduler_state)
        trainer.global_step = 0
        trainer.best_loss = float('inf')
        print(f"Compilation warmup complete! First step took {end_time - start_time:.2f}s")

    # Training loop
    print("\nStarting training...")
    print(f"Gradients: {'discrete adjoint' if args.use_adjoint else 'BPTT through the unrolled Jacobi iterations'}")
    print(f"Device: {trainer.device}")
    
    if args.disable_tqdm:
        pbar = range(1, args.N_iters + 1)
    else:
        pbar = tqdm(range(1, args.N_iters + 1))
    
    print(f"Eval sample idx={eval_idx}, k_gt range: [{k_gt_list[eval_idx].min().item():.4f}, {k_gt_list[eval_idx].max().item():.4f}]")
    
    # Timing statistics
    train_start_time = time.time()
    step_times = []

    # Best checkpoint selection (by surface-temperature PSNR; no labels used)
    best_psnr = -float("inf")
    best_ckpt_path = os.path.join(logdir, "best.tar")
    
    for i in pbar:
        step_start_time = time.time()
        
        surface_gt = surface_gt_list[eval_idx].to(device)
        
        # train_step reports the surface PSNR of the parameters *before* its update, so the
        # candidate for best.tar is the state captured just before the step.
        check_best = (i % args.early_stop_patience == 0) or (i == args.N_iters)
        pre_step_state = trainer.snapshot() if check_best else None

        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        metrics = trainer.train_step(surface_gt)

        # K_MSE and PSNR
        with torch.no_grad():
            k_pred = metrics['k_pred']
            k_gt = k_gt_list[eval_idx].to(k_pred.device)
            k_mse = F.mse_loss(k_pred, k_gt).item()
            k_psnr = -10.0 * np.log10(k_mse + 1e-10)
        
        # Record step time
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        
        # Calculate timing stats
        elapsed_time = time.time() - train_start_time
        avg_step_time = sum(step_times[-100:]) / len(step_times[-100:])  # Rolling average of last 100 steps
        remaining_steps = args.N_iters - i
        eta_seconds = remaining_steps * avg_step_time
        
        # Format time strings
        elapsed_str = f"{elapsed_time/60:.1f}min" if elapsed_time >= 60 else f"{elapsed_time:.1f}s"
        eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds >= 60 else f"{eta_seconds:.1f}s"
        
        if not args.disable_tqdm:
            pbar.set_description(
                f"Loss: {metrics['loss']:.4f} | PSNR: {metrics['psnr']:.2f} | "
                f"K_MSE: {k_mse:.4f} | K_PSNR: {k_psnr:.2f} | | grad: {metrics['grad_norm']:.2e}"
            )
        
        if i % args.i_print == 0:
            tqdm.write(
                f"[Step {i}] Loss: {metrics['loss']:.6f}, MSE: {metrics['mse_loss']:.6f}, "
                f"Reg: {metrics['reg_loss']:.6f}, PSNR: {metrics['psnr']:.2f}, "
                f"k_MSE: {k_mse:.4f}, k_PSNR: {k_psnr:.2f}, grad: {metrics['grad_norm']:.2e}, "
                f"time: {elapsed_str}, step: {step_time*1000:.0f}ms"
            )
        
        if i % args.early_stop_patience == 0:
            if metrics['psnr'] - best_psnr > args.early_stop_min_delta:
                best_psnr = metrics['psnr']
                torch.save(pre_step_state, best_ckpt_path)
                tqdm.write(f"New best PSNR: {best_psnr:.2f}, checkpoint saved to {best_ckpt_path}")
        if i == args.N_iters:
            last_pre_step_state = pre_step_state
    
    # Final timing summary
    total_train_time = time.time() - train_start_time
    avg_step_time = sum(step_times) / len(step_times)
    print(f"\n{'='*60}")
    print(f"Training Time Summary")
    print(f"{'='*60}")
    print(f"Total training time: {total_train_time/60:.2f} minutes ({total_train_time:.1f} seconds)")
    print(f"Average step time: {avg_step_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_step_time:.2f} steps/second")
    print(f"Total steps: {args.N_iters}")
    
    # Final checkpoint
    path = os.path.join(logdir, 'final.tar')
    trainer.save_checkpoint(path)
    print(f"\nTraining complete! Final checkpoint saved to {path}")

    if metrics['psnr'] > best_psnr:
        torch.save(last_pre_step_state, best_ckpt_path)
        print(f"New best PSNR: {metrics['psnr']:.2f}, checkpoint saved to {best_ckpt_path}")
    
    # Evaluate
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    with torch.no_grad():
        k_pred = trainer.predict_diffusivity()
        k_gt_eval = k_gt_list[eval_idx].to(trainer.device)
        
        mse = F.mse_loss(k_pred, k_gt_eval).item()
        mae = torch.abs(k_pred - k_gt_eval).mean().item()
        
        print(f"Ground Truth k:  min={k_gt_eval.min().item():.4f}, max={k_gt_eval.max().item():.4f}")
        print(f"Predicted k:     min={k_pred.min().item():.4f}, max={k_pred.max().item():.4f}")
        print(f"MSE(k_pred, k_gt): {mse:.6f}")
        print(f"MAE(k_pred, k_gt): {mae:.6f}")
        
        torch.save({
            'k_gt': k_gt_list[eval_idx].clone(),
            'k_pred': k_pred.cpu(),
            'surface_gt': surface_gt_list[eval_idx].clone(),
            'seed': args.seed,
            'mse': mse,
            'mae': mae,
        }, os.path.join(logdir, 'evaluation.pt'))


if __name__ == "__main__":
    train()
