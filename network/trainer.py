import copy

import torch
import torch.nn.functional as F
import numpy as np

from .net.embedder import get_embedder
from .net.thermo_nerf import ThermoNeRF
from .sim.simulator import DifferentiableHeatSimulator


class ThermoNeRFImplicitTrainer:
    """NeFTY trainer: neural diffusivity field + differentiable implicit heat solver.

    With ``args.use_grid=True`` the neural field is replaced by a voxel grid and the
    same trainer runs the Grid Opt. baseline.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Domain setup
        self.domain_size = (args.domain_x, args.domain_y, args.domain_z)
        self.resolution = (args.resolution_x, args.resolution_y, args.resolution_z)

        # Positional encoding (with frequency annealing)
        self.embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        self.anneal_end_step = getattr(args, 'anneal_iters', args.N_iters // 4)
        self.num_freqs = args.multires

        # Diffusivity field
        self.model = ThermoNeRF(
            D=args.netdepth,
            W=args.netwidth,
            input_ch=input_ch,
            output_ch=1,
            skips=[4],
            min_diffusivity=args.min_diffusivity,
            max_diffusivity=args.max_diffusivity,
            use_grid=getattr(args, 'use_grid', False),
            grid_resolution=self.resolution,
            domain_size=self.domain_size,
            output_activation=getattr(args, 'output_activation', 'sigmoid'),
            mlp_activation=getattr(args, 'mlp_activation', 'softplus'),
        ).to(self.device)

        # Differentiable forward solver
        self.simulator = DifferentiableHeatSimulator(
            domain_size=self.domain_size,
            resolution=self.resolution,
            dt=args.dt,
            num_steps=args.num_steps,
            jacobi_iters=getattr(args, 'jacobi_iters', 50),
            mean_type=getattr(args, 'mean_type', 'harmonic'),
            device=self.device
        )

        # Acceleration settings
        self.use_adjoint = args.use_adjoint
        self.use_compile = args.use_compile
        if self.use_compile:
            print("Applying torch.compile() for acceleration...")
            try:
                # Compile the inner solver kernels (the outer time loop stays in Python,
                # which keeps compilation time short) and the model as a whole.
                self.simulator._jacobi_step = torch.compile(self.simulator._jacobi_step, mode='max-autotune')
                self.simulator._implicit_step = torch.compile(self.simulator._implicit_step, mode='max-autotune')
                self.simulator.apply_system_matrix = torch.compile(self.simulator.apply_system_matrix, mode='max-autotune')
                self.model = torch.compile(self.model, mode='max-autotune')
                print("  torch.compile() applied!")
            except Exception as e:
                print(f"  torch.compile() failed: {e}, using eager mode")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lrate,
            betas=(0.9, 0.999)
        )

        # Learning-rate schedule: lr is multiplied by 0.1 every (lrate_decay * 1000) steps.
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.1 ** (1 / (args.lrate_decay * 1000))
        )

        # Coordinate grid (cell centers)
        self._create_coord_grid()

        # Initial condition: Gaussian post-flash temperature
        self.heat_source_params = {
            'center': (args.heat_source_x, args.heat_source_y, args.heat_source_z),
            'intensity': args.heat_intensity,
            'radius': args.heat_radius
        }

        # Logging
        self.global_step = 0
        self.best_loss = float('inf')

        # Regularization settings
        self.reg_type = getattr(args, 'reg_type', 'none')
        self.reg_weight = getattr(args, 'reg_weight', 1e-4)
        self.num_loss_frames = getattr(args, 'num_loss_frames', 10)

    def _create_coord_grid(self):
        """Create the 3D cell-center coordinate grid used to query the field."""
        Nx, Ny, Nz = self.resolution
        Lx, Ly, Lz = self.domain_size

        dx = Lx / Nx
        dy = Ly / Ny
        dz = Lz / Nz

        x = torch.linspace(dx/2, Lx - dx/2, Nx, device=self.device)
        y = torch.linspace(dy/2, Ly - dy/2, Ny, device=self.device)
        z = torch.linspace(dz/2, Lz - dz/2, Nz, device=self.device)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        self.coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)

    def get_current_beta(self):
        """Frequency-annealing parameter beta in [0, multires]."""
        if not getattr(self.args, 'use_fa', True):
            return self.num_freqs

        if self.global_step >= self.anneal_end_step:
            return self.num_freqs  # full bandwidth

        # Linear ramp of beta from 0 to multires over anneal_iters steps
        progress = self.global_step / self.anneal_end_step
        return progress * self.num_freqs

    def predict_diffusivity(self) -> torch.Tensor:
        """Evaluate the diffusivity field on the solver grid."""
        beta = self.get_current_beta()
        coords_embedded = self.embed_fn(self.coords, beta=beta)

        chunk = self.args.netchunk
        outputs = []
        for i in range(0, coords_embedded.shape[0], chunk):
            # Clone to avoid CUDA-graph memory reuse issues with torch.compile
            outputs.append(self.model(coords_embedded[i:i+chunk]).clone())
        k_flat = torch.cat(outputs, dim=0)

        Nx, Ny, Nz = self.resolution
        k = k_flat.squeeze(-1).reshape(Nx, Ny, Nz)

        return k

    def compute_regularization(self, k: torch.Tensor) -> torch.Tensor:
        """Spatial regularizer on the diffusivity field."""
        if self.reg_type == 'none':
            return torch.tensor(0.0, device=k.device)

        dx = k[1:, :, :] - k[:-1, :, :]
        dy = k[:, 1:, :] - k[:, :-1, :]
        dz = k[:, :, 1:] - k[:, :, :-1]

        if self.reg_type == 'tv':
            # Isotropic total variation (smoothed): mean over voxels of
            # sqrt(dx^2 + dy^2 + dz^2 + eps), with forward differences.
            dx_pad = F.pad(dx, (0, 0, 0, 0, 0, 1))
            dy_pad = F.pad(dy, (0, 0, 0, 1, 0, 0))
            dz_pad = F.pad(dz, (0, 1, 0, 0, 0, 0))
            eps = 1e-8
            tv_loss = torch.sqrt(dx_pad**2 + dy_pad**2 + dz_pad**2 + eps).mean()
            return tv_loss
        elif self.reg_type == 'tv_aniso':
            return torch.abs(dx).mean() + torch.abs(dy).mean() + torch.abs(dz).mean()
        else:
            return torch.tensor(0.0, device=k.device)

    def symmetric_loss(self, k: torch.Tensor) -> torch.Tensor:
        """Mirror-symmetry penalty on k along x and y (used as a short warm-up)."""
        k_flip_x = torch.flip(k, dims=[0])
        k_flip_y = torch.flip(k, dims=[1])

        loss_x = F.mse_loss(k, k_flip_x)
        loss_y = F.mse_loss(k, k_flip_y)

        return (loss_x + loss_y) / 2.0

    def train_step(self, surface_gt: torch.Tensor) -> dict:
        """Perform one optimization step against the observed surface temperatures."""
        self.optimizer.zero_grad()

        # Predict diffusivity
        k = self.predict_diffusivity()

        # Initial temperature
        T0 = self.simulator.create_gaussian_heat_source(
            center=self.heat_source_params['center'],
            intensity=self.heat_source_params['intensity'],
            radius=self.heat_source_params['radius']
        )

        # Forward simulation (adjoint backward if use_adjoint, else BPTT)
        surface_seq_pred = self.simulator(k, T0, use_adjoint=self.use_adjoint)

        surface_gt = surface_gt.to(self.device)

        # Data term on the observed surface frames
        num_frames = min(len(surface_seq_pred), len(surface_gt))
        if self.num_loss_frames > num_frames:
            mse_loss = F.mse_loss(surface_seq_pred[:num_frames], surface_gt[:num_frames])
        else:
            num_loss_frames = min(self.num_loss_frames, num_frames)
            frame_indices = [int(i * (num_frames - 1) / max(num_loss_frames - 1, 1))
                            for i in range(num_loss_frames)]
            mse_loss = F.mse_loss(surface_seq_pred[frame_indices], surface_gt[frame_indices])

        # Regularization
        reg_loss = self.compute_regularization(k)

        # Symmetry warm-up: weight decays linearly from 100 to 0 over the first 2000 steps
        start_weight = 100.0
        decay_steps = 2000
        if self.global_step < decay_steps and not getattr(self.args, 'no_sl', False):
            decay_factor = 1.0 - (self.global_step / decay_steps)
            sym_loss = (start_weight * decay_factor) * self.symmetric_loss(k)
        else:
            sym_loss = torch.tensor(0.0, device=self.device)

        # Total loss
        loss = mse_loss + self.reg_weight * reg_loss + sym_loss

        # Backward
        loss.backward()

        # Gradient norm (before clipping), for logging
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

        # Update
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        # Metrics
        with torch.no_grad():
            final_mse = F.mse_loss(surface_seq_pred[-1], surface_gt[-1]).item()
            final_psnr = -10.0 * np.log10(final_mse/((surface_gt[-1].max().item() - surface_gt[-1].min().item())**2 + 1e-10) + 1e-10)
            if final_mse < self.best_loss:
                self.best_loss = final_mse

        return {
            'loss': loss.item(),
            'mse_loss': mse_loss.item(),
            'reg_loss': reg_loss.item(),
            'psnr': final_psnr,
            'k_pred': k.detach().cpu(),
            'k_min': k.min().item(),
            'k_max': k.max().item(),
            'k_mean': k.mean().item(),
            'grad_norm': grad_norm,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }, path)

    def snapshot(self) -> dict:
        """Copy of the current training state, in the checkpoint format."""
        return {
            'global_step': self.global_step,
            'model_state_dict': {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
            'scheduler_state_dict': copy.deepcopy(self.scheduler.state_dict()),
            'best_loss': self.best_loss,
        }

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
