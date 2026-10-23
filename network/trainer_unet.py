import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import time
from .net.thermal_unet import ThermalUNet

def _held_out_indices():
    idx = []
    # Homogeneous part: 150-199, 350-399, 550-599, 750-799
    for start in (150, 350, 550, 750):
        idx.extend(range(start, start + 50))
    # Layered part: 820-824, 845-849, ..., 995-999
    for start in (820, 845, 870, 895, 920, 945, 970, 995):
        idx.extend(range(start, start + 5))
    return idx


# Fixed list of 240 dataset indices excluded from U-Net training by default
# (--use_holdout True). The datasets follow the same index schedule, so these indices
# have the same bulk type (homogeneous / 3-layer / 4-layer) as the benchmark specimens
# with the same index. The released Sound-Only checkpoint was trained this way, on the
# remaining 760 samples of the 1,000-sample defect-free set.
HELD_OUT_INDICES = _held_out_indices()


class ThermalUNetTrainer:
    def __init__(self, args, data):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Data
        surface_gt = data['surface_gt'] # (N, T, H, W)
        k_gt = data['k_gt'] # (N, H, W, D) -> Need to Permute to (N, D, H, W) for network
        
        # Dimensions
        self.T = surface_gt.shape[1]
        self.D = k_gt.shape[3]
        
        # Prepare Data
        # Permute k_gt: (N, H, W, D) -> (N, D, H, W)
        k_gt = k_gt.permute(0, 3, 1, 2)
        
        # Add Channel Dim to Surface: (N, T, H, W) -> (N, 1, T, H, W)
        surface_gt = surface_gt.unsqueeze(1)
        # Add Channel Dim to K: (N, D, H, W) -> (N, 1, D, H, W)
        k_gt = k_gt.unsqueeze(1)
        
        # Split
        train_indices, test_indices = self._get_splits(len(surface_gt))
        
        print(f"Total samples: {len(surface_gt)}")
        print(f"Train samples: {len(train_indices)}")
        print(f"Test samples: {len(test_indices)}")
        
        self.train_data = TensorDataset(surface_gt[train_indices], k_gt[train_indices])
        # We don't really use test_loader for training loop, but good to have reference if needed
        # self.test_data = TensorDataset(surface_gt[test_indices], k_gt[test_indices])
        
        self.train_loader = DataLoader(
            self.train_data, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0, # In-memory tensors, 0 is fastest
            pin_memory=True
        )
        
        # Model
        self.model = ThermalUNet(
            min_diffusivity=args.min_diffusivity,
            max_diffusivity=args.max_diffusivity
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lrate,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lrate_decay, # Decay every X epochs
            gamma=0.5
        )
        
        self.global_step = 0
        self.best_loss = float('inf')
        
    def _get_splits(self, total_samples):
        all_indices = np.arange(total_samples)
        if not getattr(self.args, 'use_holdout', True):
            return list(all_indices), []
        test_set = set(HELD_OUT_INDICES)
        train_indices = [i for i in all_indices if i not in test_set]
        return train_indices, [i for i in all_indices if i in test_set]

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (surface, k_target) in enumerate(self.train_loader):
            surface = surface.to(self.device)
            k_target = k_target.to(self.device)
            
            self.optimizer.zero_grad()
            
            k_pred = self.model(surface)
            
            loss = F.mse_loss(k_pred, k_target)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= len(self.train_loader)
        self.scheduler.step()
        
        return epoch_loss
        
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'args': vars(self.args) if hasattr(self.args, '__dict__') else self.args
        }, path)
