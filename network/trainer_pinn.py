import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .net.embedder import Embedder
from .net.thermo_nerf_pinn import ThermoNERFPINN
from .sim.simulator import DifferentiableHeatSimulator

class ThermoNERFPINNTrainer:
    """
    PINN Baseline Trainer.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Domain setup
        self.domain_size = (args.domain_x, args.domain_y, args.domain_z)
        self.resolution = (args.resolution_x, args.resolution_y, args.resolution_z)
        
        # Embedders
        # 1. Diffusivity (Spatial 3D)
        self.diff_embed_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_power': args.multires - 1,
            'num_freqs': args.multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        self.diff_embedder = Embedder(**self.diff_embed_kwargs)
        self.diff_input_ch = self.diff_embedder.out_dim
        
        # 2. Temperature (Spacetime 4D)
        # Same encoding bandwidth as the diffusivity network
        self.temp_embed_kwargs = {
            'include_input': True,
            'input_dims': 4,
            'max_freq_power': args.multires - 1,
            'num_freqs': args.multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        self.temp_embedder = Embedder(**self.temp_embed_kwargs)
        self.temp_input_ch = self.temp_embedder.out_dim
        
        # Annealing
        self.anneal_end_step = getattr(args, 'anneal_iters', args.N_iters // 4)
        self.num_freqs = args.multires
        
        # Model
        self.model = ThermoNERFPINN(
            diff_net_args={
                'D': args.netdepth,
                'W': args.netwidth,
                'input_ch': self.diff_input_ch,
                'output_ch': 1,
                'skips': [4],
                'min_diffusivity': args.min_diffusivity,
                'max_diffusivity': args.max_diffusivity,
                'use_grid': getattr(args, 'use_grid', False),
                'grid_resolution': self.resolution,
                'domain_size': self.domain_size,
                'output_activation': getattr(args, 'output_activation', 'sigmoid'),
                'mlp_activation': getattr(args, 'mlp_activation', 'softplus'),
            },
            temp_net_args={
                'D': args.netdepth,
                'W': args.netwidth,
                'input_ch': self.temp_input_ch,
                'output_ch': 1,
                'skips': [4]
            }
        ).to(self.device)
        
        # Simulator (For IC generation and consistency)
        self.simulator = DifferentiableHeatSimulator(
            domain_size=self.domain_size,
            resolution=self.resolution,
            dt=args.dt,
            num_steps=args.num_steps,
            jacobi_iters=getattr(args, 'jacobi_iters', 50),
            mean_type=getattr(args, 'mean_type', 'harmonic'),
            device=self.device
        )
        
        # Create IC ground truth (Gaussian)
        self.heat_source_params = {
            'center': (args.heat_source_x, args.heat_source_y, args.heat_source_z),
            'intensity': args.heat_intensity,
            'radius': args.heat_radius
        }
        # We can generate this on the fly or precompute grid
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lrate,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.1 ** (1 / (args.lrate_decay * 1000))
        )
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Loss weights
        self.lambda_pde = getattr(args, 'lambda_pde', 0.1)
        self.lambda_ic = getattr(args, 'lambda_ic', 1.0)
        self.lambda_data = getattr(args, 'lambda_data', 1.0)
        # PDE schedule
        self.lambda_pde_warmup = getattr(args, 'lambda_pde_warmup', 0)
        self.lambda_pde_ramp = getattr(args, 'lambda_pde_ramp', 0)
        self.lambda_pde_start = getattr(args, 'lambda_pde_start', 0.0)

        # GradNorm-style balancing
        self.use_gradnorm = getattr(args, 'use_gradnorm', False)
        self.gradnorm_alpha = getattr(args, 'gradnorm_alpha', 0.5)
        self.gradnorm_warmup = getattr(args, 'gradnorm_warmup', 0)
        self.gradnorm_update_every = getattr(args, 'gradnorm_update_every', 1)
        self.gradnorm_eps = 1e-8
        self._gradnorm_L0 = None
        self._gradnorm_params = list(self.model.temperature_net.parameters())

        # Current adaptive weights (mutable)
        self.lambda_data_val = float(self.lambda_data)
        self.lambda_ic_val = float(self.lambda_ic)
        self.lambda_pde_val = float(self.lambda_pde)
        self._lambda_sum = self.lambda_data_val + self.lambda_ic_val + self.lambda_pde_val
        
        # Network inputs: physical coordinates mapped to [-1, 1] (True) or raw (False)
        self.normalize_coords = getattr(args, 'normalize_coords', True)

        # Batch sizes
        self.n_pde = getattr(args, 'n_pde', 4096)
        self.n_data = getattr(args, 'n_data', 1024)
        self.n_ic = getattr(args, 'n_ic', 1024)
        
        # Acceleration
        # We can compile the data fitting terms (single diff) but not PDE (double diff)
        self.use_compile = getattr(args, 'use_compile', False)
        if self.use_compile:
            print("Enabling Partial Compilation (Data+IC losses compiled, PDE eager)")
            self._compiled_compute_data_losses = torch.compile(self._compute_data_losses)
            self._is_compiled_warmed_up = False

    def get_current_beta(self):
        """Calculate current frequency annealing beta for diffusivity."""
        if not getattr(self.args, 'use_fa', True):
            return self.num_freqs

        if self.global_step >= self.anneal_end_step:
            return self.num_freqs 
        
        progress = self.global_step / self.anneal_end_step
        return progress * self.num_freqs

    def get_current_lambda_pde(self):
        """Warmup + ramp schedule for PDE weight."""
        if self.use_gradnorm:
            return self.lambda_pde_val
        if self.lambda_pde_ramp <= 0 and self.lambda_pde_warmup <= 0:
            return self.lambda_pde

        step = self.global_step
        if step < self.lambda_pde_warmup:
            return self.lambda_pde_start

        ramp_step = step - self.lambda_pde_warmup
        if self.lambda_pde_ramp <= 0:
            return self.lambda_pde

        progress = min(max(ramp_step / self.lambda_pde_ramp, 0.0), 1.0)
        return self.lambda_pde_start + (self.lambda_pde - self.lambda_pde_start) * progress

    def _gradnorm_update(self, loss_data, loss_ic, loss_pde):
        """GradNorm-style update for loss weights."""
        if not self._gradnorm_params:
            return

        if self._gradnorm_L0 is None:
            self._gradnorm_L0 = {
                'data': loss_data.detach().clamp(min=self.gradnorm_eps),
                'ic': loss_ic.detach().clamp(min=self.gradnorm_eps),
                'pde': loss_pde.detach().clamp(min=self.gradnorm_eps),
            }

        # Compute gradient norms for each task (w.r.t shared temperature net)
        g_data = torch.autograd.grad(
            self.lambda_data_val * loss_data,
            self._gradnorm_params,
            retain_graph=True,
            allow_unused=True
        )
        g_ic = torch.autograd.grad(
            self.lambda_ic_val * loss_ic,
            self._gradnorm_params,
            retain_graph=True,
            allow_unused=True
        )
        g_pde = torch.autograd.grad(
            self.lambda_pde_val * loss_pde,
            self._gradnorm_params,
            retain_graph=True,
            allow_unused=True
        )

        def _norm(grads):
            norms = [g.detach().norm() for g in grads if g is not None]
            if not norms:
                return torch.tensor(0.0, device=self.device)
            return torch.norm(torch.stack(norms), p=2)

        g_data_n = _norm(g_data)
        g_ic_n = _norm(g_ic)
        g_pde_n = _norm(g_pde)

        g_avg = (g_data_n + g_ic_n + g_pde_n) / 3.0

        # Relative inverse training rates
        L_data = loss_data.detach().clamp(min=self.gradnorm_eps)
        L_ic = loss_ic.detach().clamp(min=self.gradnorm_eps)
        L_pde = loss_pde.detach().clamp(min=self.gradnorm_eps)

        L_hat_data = L_data / self._gradnorm_L0['data']
        L_hat_ic = L_ic / self._gradnorm_L0['ic']
        L_hat_pde = L_pde / self._gradnorm_L0['pde']

        r_avg = (L_hat_data + L_hat_ic + L_hat_pde) / 3.0
        target_data = g_avg * (L_hat_data / r_avg) ** self.gradnorm_alpha
        target_ic = g_avg * (L_hat_ic / r_avg) ** self.gradnorm_alpha
        target_pde = g_avg * (L_hat_pde / r_avg) ** self.gradnorm_alpha

        # Multiplicative update and renormalize
        self.lambda_data_val *= float(target_data / (g_data_n + self.gradnorm_eps))
        self.lambda_ic_val *= float(target_ic / (g_ic_n + self.gradnorm_eps))
        self.lambda_pde_val *= float(target_pde / (g_pde_n + self.gradnorm_eps))

        # Clamp and renormalize to keep sum stable
        self.lambda_data_val = float(np.clip(self.lambda_data_val, 1e-6, 1e6))
        self.lambda_ic_val = float(np.clip(self.lambda_ic_val, 1e-6, 1e6))
        self.lambda_pde_val = float(np.clip(self.lambda_pde_val, 1e-6, 1e6))
        total = self.lambda_data_val + self.lambda_ic_val + self.lambda_pde_val
        if total > 0:
            scale = self._lambda_sum / total
            self.lambda_data_val *= scale
            self.lambda_ic_val *= scale
            self.lambda_pde_val *= scale

    def sample_domain_points(self, N):
        """Sample N random points in (x,y,z,t) domain."""
        Lx, Ly, Lz = self.domain_size
        T_max = self.args.dt * self.args.num_steps
        
        x = torch.rand(N, 1, device=self.device) * Lx
        y = torch.rand(N, 1, device=self.device) * Ly
        z = torch.rand(N, 1, device=self.device) * Lz
        t = torch.rand(N, 1, device=self.device) * T_max
        
        return x, y, z, t

    def sample_ic_points(self, N):
        """Sample N random points at t=0."""
        Lx, Ly, Lz = self.domain_size
        
        x = torch.rand(N, 1, device=self.device) * Lx
        y = torch.rand(N, 1, device=self.device) * Ly
        z = torch.rand(N, 1, device=self.device) * Lz
        
        # Compute exact GT T_initial for these points
        # Gaussian: exp(-r^2 / 2sigma^2)
        cx, cy, cz = self.heat_source_params['center']
        intensity = self.heat_source_params['intensity']
        radius = self.heat_source_params['radius']
        
        r2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
        T_gt = intensity * torch.exp(-r2 / (2 * radius**2))
        
        return torch.cat([x, y, z], dim=-1), T_gt

    def sample_data_points(self, N, surface_gt):
        """Sample N points from surface measurements."""
        # surface_gt: [Time, Nx, Ny], observed on the top cell layer (cell centers),
        # i.e. at z = Lz - dz/2.
        
        Lx, Ly, Lz = self.domain_size
        Nx, Ny, Nz = self.resolution
        dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
        
        z_surf = Lz - dz/2
        
        # Time steps
        num_frames = surface_gt.shape[0]
        # Select random time indices
        t_indices = torch.randint(0, num_frames, (N,), device=self.device)
        
        # Select random spatial indices (ix, iy)
        ix = torch.randint(0, Nx, (N,), device=self.device)
        iy = torch.randint(0, Ny, (N,), device=self.device)
        
        # Convert to physical coordinates
        # x[i] = dx/2 + i*dx
        x_vals = dx/2 + ix.float() * dx
        y_vals = dy/2 + iy.float() * dy
        z_vals = torch.full((N,), z_surf, device=self.device)
        # Frame i is observed at t = i * dt (frame 0 is the initial condition at t = 0;
        # the last frame is at t = num_steps * dt, the end of the collocation range).
        t_vals = t_indices.float() * self.args.dt
        
        # Get scalar values from GT
        # surface_gt is [Time, Nx, Ny], but we need to index [t_indices, ix, iy]
        T_gt = surface_gt[t_indices, ix, iy].unsqueeze(-1) # [N, 1]
        
        points = torch.stack([x_vals, y_vals, z_vals, t_vals], dim=-1)
        
        return points, T_gt

    def _normalize_spacetime(self, x, y, z, t):
        """Map physical coordinates to [-1, 1] for network input.

        Identity when ``normalize_coords`` is False (raw physical coordinates).
        """
        if not self.normalize_coords:
            return x, y, z, t
        Lx, Ly, Lz = self.domain_size
        T_max = self.args.dt * self.args.num_steps
        
        xn = 2.0 * (x / Lx) - 1.0
        yn = 2.0 * (y / Ly) - 1.0
        zn = 2.0 * (z / Lz) - 1.0
        tn = 2.0 * (t / T_max) - 1.0
        return xn, yn, zn, tn

    def _compute_data_losses(self, xyzt_ic, T_ic_gt, xyzt_data, T_data_gt):
        """
        Compute IC and Data fitting losses.
        Standard forward + single backward. Compile-friendly.
        """
        # Normalize inputs for stability
        # Unpack
        x_ic, y_ic, z_ic, t_ic = xyzt_ic[..., 0:1], xyzt_ic[..., 1:2], xyzt_ic[..., 2:3], xyzt_ic[..., 3:4]
        xn_ic, yn_ic, zn_ic, tn_ic = self._normalize_spacetime(x_ic, y_ic, z_ic, t_ic)
        xyzt_ic_norm = torch.cat([xn_ic, yn_ic, zn_ic, tn_ic], dim=-1)
        
        x_d, y_d, z_d, t_d = xyzt_data[..., 0:1], xyzt_data[..., 1:2], xyzt_data[..., 2:3], xyzt_data[..., 3:4]
        xn_d, yn_d, zn_d, tn_d = self._normalize_spacetime(x_d, y_d, z_d, t_d)
        xyzt_data_norm = torch.cat([xn_d, yn_d, zn_d, tn_d], dim=-1)

        # 1. IC Loss
        # Embed and predict
        xyzt_ic_emb = self.temp_embedder.embed(xyzt_ic_norm)
        T_ic_pred = self.model.forward_temperature(xyzt_ic_emb)
        loss_ic = F.mse_loss(T_ic_pred, T_ic_gt)
        
        # 2. Data Loss
        # Embed and predict
        xyzt_data_emb = self.temp_embedder.embed(xyzt_data_norm)
        T_data_pred = self.model.forward_temperature(xyzt_data_emb)
        loss_data = F.mse_loss(T_data_pred, T_data_gt)
        
        return loss_ic, loss_data

    def _compute_pde_loss(self, x_pde, y_pde, z_pde, t_pde, beta):
        """
        Compute PDE residual loss.
        Requires Double Backward. Keep Eager.
        """
        # Normalize for Network Input (but differentiate w.r.t physical x_pde)
        xn_pde, yn_pde, zn_pde, tn_pde = self._normalize_spacetime(x_pde, y_pde, z_pde, t_pde)
        
        # Predict Alpha (Diffusivity)
        # alpha depends on spatial coordinates only
        xyz_pde_norm = torch.cat([xn_pde, yn_pde, zn_pde], dim=-1)
        
        # Apply FA for diffusivity
        xyz_pde_emb = self.diff_embedder.embed(xyz_pde_norm, beta=beta)
        alpha = self.model.forward_diffusivity(xyz_pde_emb)
        
        # Predict Temperature
        xyzt_pde_norm = torch.cat([xn_pde, yn_pde, zn_pde, tn_pde], dim=-1)
        xyzt_pde_emb = self.temp_embedder.embed(xyzt_pde_norm)  # full bandwidth for T
        T = self.model.forward_temperature(xyzt_pde_emb)
        
        # Compute gradients
        # We need to manually handle gradients because torch.autograd.grad isn't fully graph-capturable in all versions
        # safely without some tricks, but standard Inductor often supports it now.
        
        grads_T = torch.autograd.grad(
            outputs=T,
            inputs=[x_pde, y_pde, z_pde, t_pde],
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        dT_dt = grads_T[3]
        dT_dx = grads_T[0]
        dT_dy = grads_T[1]
        dT_dz = grads_T[2]
        
        Fx = alpha * dT_dx
        Fy = alpha * dT_dy
        Fz = alpha * dT_dz
        
        dFx_dx = torch.autograd.grad(Fx, x_pde, torch.ones_like(Fx), create_graph=True)[0]
        dFy_dy = torch.autograd.grad(Fy, y_pde, torch.ones_like(Fy), create_graph=True)[0]
        dFz_dz = torch.autograd.grad(Fz, z_pde, torch.ones_like(Fz), create_graph=True)[0]
        
        div_flux = dFx_dx + dFy_dy + dFz_dz
        
        # Residual = dT/dt - div(alpha * grad T)
        residual = dT_dt - div_flux
        
        loss_pde = torch.mean(residual ** 2)
        return loss_pde

    def train_step(self, surface_gt):
        self.optimizer.zero_grad()
        
        # Sampling (cpu/gpu ops, not compiled for flexibility)
        # 1. IC Points
        xyz_ic, T_ic_gt = self.sample_ic_points(self.n_ic)
        t_ic = torch.zeros((self.n_ic, 1), device=self.device)
        xyzt_ic = torch.cat([xyz_ic, t_ic], dim=-1)
        
        # 2. Data Points
        xyzt_data, T_data_gt = self.sample_data_points(self.n_data, surface_gt)
        
        # 3. PDE Points
        x_pde, y_pde, z_pde, t_pde = self.sample_domain_points(self.n_pde)
        x_pde.requires_grad_(True)
        y_pde.requires_grad_(True)
        z_pde.requires_grad_(True)
        t_pde.requires_grad_(True)
        
        # Beta
        beta = self.get_current_beta()

        # 1. Compute Data Losses (Compiled if enabled)
        if self.use_compile:
            if not self._is_compiled_warmed_up:
                print("Compiling Data/IC loss computation...")
                self._compiled_compute_data_losses(xyzt_ic, T_ic_gt, xyzt_data, T_data_gt)
                self._is_compiled_warmed_up = True
            loss_ic, loss_data = self._compiled_compute_data_losses(xyzt_ic, T_ic_gt, xyzt_data, T_data_gt)
        else:
            loss_ic, loss_data = self._compute_data_losses(xyzt_ic, T_ic_gt, xyzt_data, T_data_gt)
            
        # 2. Compute PDE Loss (Always Eager)
        loss_pde = self._compute_pde_loss(x_pde, y_pde, z_pde, t_pde, beta)

        # Optionally update GradNorm weights
        if self.use_gradnorm and self.global_step >= self.gradnorm_warmup:
            if self.gradnorm_update_every > 0 and (self.global_step % self.gradnorm_update_every == 0):
                self._gradnorm_update(loss_data, loss_ic, loss_pde)

        # Total Loss
        lambda_pde = self.get_current_lambda_pde()
        lambda_data = self.lambda_data_val if self.use_gradnorm else self.lambda_data
        lambda_ic = self.lambda_ic_val if self.use_gradnorm else self.lambda_ic
        loss = (lambda_data * loss_data +
            lambda_ic * loss_ic +
            lambda_pde * loss_pde)
        
        # Backprop
        loss.backward()

        # Gradient norms of the two networks (for logging)
        diff_params = [p for p in self.model.diffusivity_net.parameters() if p.grad is not None]
        temp_params = [p for p in self.model.temperature_net.parameters() if p.grad is not None]
        diff_grad_norm = torch.tensor(0.0, device=self.device)
        temp_grad_norm = torch.tensor(0.0, device=self.device)
        if diff_params:
            diff_grad_norm = torch.sqrt(torch.sum(torch.stack([p.grad.detach().norm() ** 2 for p in diff_params])))
        if temp_params:
            temp_grad_norm = torch.sqrt(torch.sum(torch.stack([p.grad.detach().norm() ** 2 for p in temp_params])))

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        
        # Metrics for logging
        with torch.no_grad():
            # MSE on data batch
            mse = loss_data.item()
            # PSNR on data batch (approx)
            data_range = T_data_gt.max() - T_data_gt.min()
            psnr = -10.0 * np.log10(mse / (data_range.item()**2 + 1e-10))
            
            if mse < self.best_loss:
                self.best_loss = mse

        return {
            'loss': loss.item(),
            'loss_data': loss_data.item(),
            'loss_ic': loss_ic.item(),
            'loss_pde': loss_pde.item(),
            'mse_loss': mse,
            'psnr': psnr,
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm_diff': diff_grad_norm.item(),
            'grad_norm_temp': temp_grad_norm.item(),
            'lambda_pde': float(lambda_pde),
            'lambda_data': float(lambda_data),
            'lambda_ic': float(lambda_ic)
        }

    def save_checkpoint(self, path: str):
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }, path)
