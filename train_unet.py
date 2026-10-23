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

from dataset.synthetic import load_synthetic_dataset
from network.trainer_unet import ThermalUNetTrainer

# Enable TensorFloat32
torch.set_float32_matmul_precision('high')

def config_parser():
    """Parse command line arguments."""
    parser = configargparse.ArgumentParser()
    
    # Experiment
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, default='thermo_unet_baseline',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs',
                        help='where to store checkpoints and logs')
    parser.add_argument("--datadir", type=str, default='./data/synthetic_1000',
                        help='training dataset (defect-free dataset for the Sound-Only U-Net)')
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU id(s) to expose via CUDA_VISIBLE_DEVICES; command line only "
                             "(read before torch is imported). Default: keep the environment's "
                             "CUDA_VISIBLE_DEVICES")
    
    # Training
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--lrate_decay", type=int, default=100, help="Decay step size (epochs)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help='Random seed')
    
    parser.add_argument("--use_holdout", type=lambda v: str(v).lower() in ('yes', 'true', 't', 'y', '1'),
                        default=True,
                        help="Exclude the fixed index list network.trainer_unet.HELD_OUT_INDICES "
                             "(240 samples) from training (default True, as for the released "
                             "checkpoint); False trains on every sample")

    # Diffusivity bounds
    parser.add_argument("--min_diffusivity", type=float, default=0.003)
    parser.add_argument("--max_diffusivity", type=float, default=0.25)
    
    # Logging
    parser.add_argument("--i_print", type=int, default=1)
    
    return parser

def train():
    parser = config_parser()
    args = parser.parse_args()
    
    # Create log directory
    logdir = os.path.join(args.basedir, args.expname)
    os.makedirs(logdir, exist_ok=True)
    
    # Save args
    with open(os.path.join(logdir, 'args.txt'), 'w') as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            f.write('{} = {}\n'.format(arg, attr))
            
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load Data
    print("Loading dataset...")
    data = load_synthetic_dataset(args.datadir, map_location="cpu")
    
    # Trainer
    trainer = ThermalUNetTrainer(args, data)
    
    print("\nStarting UNet training...")
    epoch_pbar = tqdm(range(1, args.epochs + 1))
    
    best_loss = float('inf')
    
    for epoch in epoch_pbar:
        loss = trainer.train_epoch()
        trainer.global_step += 1
        
        if epoch % args.i_print == 0:
            tqdm.write(f"[EPOCH {epoch}] Loss: {loss:.6f}")
            epoch_pbar.set_description(f"Loss: {loss:.6f}")
            
        if loss < best_loss:
            best_loss = loss
            trainer.save_checkpoint(os.path.join(logdir, 'best.tar'))
            
    # Final save
    trainer.save_checkpoint(os.path.join(logdir, 'final.tar'))
    print(f"Training complete. Saved to {logdir}/final.tar")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
    train()
