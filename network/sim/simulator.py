import torch
import torch.nn as nn
import torch.nn.functional as F
from .adjoint import AdjointHeatStep

class DifferentiableHeatSimulator(nn.Module):
    """
    Differentiable heat diffusion simulator using PyTorch.
    Uses implicit Euler with Jacobi iteration.
    """
    
    def __init__(
        self,
        domain_size: tuple = (10.0, 10.0, 1.0),
        resolution: tuple = (32, 32, 8),
        dt: float = 0.1,
        num_steps: int = 50,
        jacobi_iters: int = 50,
        mean_type: str = 'harmonic',
        device: torch.device = None
    ):
        super().__init__()

        self.domain_size = domain_size
        self.resolution = resolution
        self.dt = dt
        self.num_steps = num_steps
        self.jacobi_iters = jacobi_iters
        self.mean_type = mean_type
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        Lx, Ly, Lz = domain_size
        Nx, Ny, Nz = resolution
        
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.dz = Lz / Nz
        
        self.inv_dt = 1.0 / dt
        self.inv_dx2 = 1.0 / (self.dx ** 2)
        self.inv_dy2 = 1.0 / (self.dy ** 2)
        self.inv_dz2 = 1.0 / (self.dz ** 2)
        
        # Cache for coefficients (per k tensor)
        self._coeff_cache_key = None
        self._coeff_cache = None
        print(f"DifferentiableHeatSimulator initialized:")
        print(f"  Domain: {domain_size}")
        print(f"  Resolution: {resolution}")
        print(f"  Grid spacing: dx={self.dx:.4f}, dy={self.dy:.4f}, dz={self.dz:.4f}")
        print(f"  dt={dt}, num_steps={num_steps}")

    def _get_coefficients(self, k: torch.Tensor):
        """
        Cached coefficients for a given k tensor.

        IMPORTANT: Do NOT cache when k.requires_grad is True.
        In adjoint backward we create k_detached = k.detach().requires_grad_(True),
        and caching would either (a) return stale coeffs or (b) keep autograd graphs alive.
        """
        if k.requires_grad:
            return self._compute_coefficients(k)

        key = (k.data_ptr(), k._version)
        if self._coeff_cache_key != key:
            self._coeff_cache = self._compute_coefficients(k)
            self._coeff_cache_key = key
        return self._coeff_cache
    
    def create_gaussian_heat_source(
        self,
        center: tuple = (5.0, 5.0, 0.8),
        intensity: float = 100.0,
        radius: float = 0.5
    ) -> torch.Tensor:
        """Create Gaussian initial temperature distribution."""
        Nx, Ny, Nz = self.resolution
        Lx, Ly, Lz = self.domain_size
        cx, cy, cz = center
        
        x = torch.linspace(self.dx/2, Lx - self.dx/2, Nx, device=self.device)
        y = torch.linspace(self.dy/2, Ly - self.dy/2, Ny, device=self.device)
        z = torch.linspace(self.dz/2, Lz - self.dz/2, Nz, device=self.device)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        r2 = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2
        T0 = intensity * torch.exp(-r2 / (2 * radius**2))
        
        return T0
    
    def _compute_coefficients(self, k: torch.Tensor):
        """Helper to compute stencil coefficients from diffusivity."""
        # Neighbors: periodic in x/y, replicate (zero-flux) in z
        k_xp = torch.roll(k, -1, dims=0)
        k_xm = torch.roll(k, 1, dims=0)
        k_yp = torch.roll(k, -1, dims=1)
        k_ym = torch.roll(k, 1, dims=1)

        k_padded = F.pad(k.unsqueeze(0).unsqueeze(0), (1, 1, 0, 0, 0, 0), mode='replicate').squeeze()
        k_zp = k_padded[:, :, 2:]
        k_zm = k_padded[:, :, :-2]
        
        if self.mean_type == 'harmonic':
            eps = 1e-10
            k_xp_half = 2 * k * k_xp / (k + k_xp + eps)
            k_xm_half = 2 * k * k_xm / (k + k_xm + eps)
            k_yp_half = 2 * k * k_yp / (k + k_yp + eps)
            k_ym_half = 2 * k * k_ym / (k + k_ym + eps)
            k_zp_half = 2 * k * k_zp / (k + k_zp + eps)
            k_zm_half = 2 * k * k_zm / (k + k_zm + eps)
        elif self.mean_type == 'arithmetic':
            k_xp_half = 0.5 * (k + k_xp)
            k_xm_half = 0.5 * (k + k_xm)
            k_yp_half = 0.5 * (k + k_yp)
            k_ym_half = 0.5 * (k + k_ym)
            k_zp_half = 0.5 * (k + k_zp)
            k_zm_half = 0.5 * (k + k_zm)
        else:
            raise ValueError(f"Unknown mean_type: {self.mean_type}")
        
        # Coefficients
        ax_p = k_xp_half * self.inv_dx2
        ax_m = k_xm_half * self.inv_dx2
        ay_p = k_yp_half * self.inv_dy2
        ay_m = k_ym_half * self.inv_dy2
        az_p = k_zp_half * self.inv_dz2
        az_m = k_zm_half * self.inv_dz2
        
        return ax_p, ax_m, ay_p, ay_m, az_p, az_m

    def apply_system_matrix(self, T: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Computes Matrix-Vector product: y = A * T
        where A = (1/dt * I - L).
        Used for VJP calculation in Adjoint method.
        """
        ax_p, ax_m, ay_p, ay_m, az_p, az_m = self._get_coefficients(k)
        
        # Neighbor access (must match _jacobi_step exactly)
        T_xp = torch.roll(T, -1, dims=0)
        T_xm = torch.roll(T, 1, dims=0)
        T_yp = torch.roll(T, -1, dims=1)
        T_ym = torch.roll(T, 1, dims=1)

        T_padded = F.pad(T.unsqueeze(0).unsqueeze(0), (1, 1, 0, 0, 0, 0), mode='replicate').squeeze()
        T_zp = T_padded[:, :, 2:]
        T_zm = T_padded[:, :, :-2]
        
        # Discrete Laplacian L * T
        # L T = (ax_p(T_xp - T) + ax_m(T_xm - T) + ...)
        # We can rearrange as:
        # L T = (ax_p T_xp + ...) - (ax_p + ... ) T
        
        sum_neighbors = (
            ax_p * T_xp + ax_m * T_xm +
            ay_p * T_yp + ay_m * T_ym +
            az_p * T_zp + az_m * T_zm
        )
        
        sum_coeffs = (ax_p + ax_m + ay_p + ay_m + az_p + az_m)
        
        # L * T
        laplacian_T = sum_neighbors - sum_coeffs * T
        
        # A * T = (1/dt * T - L * T)
        #       = (1/dt + sum_coeffs) * T - sum_neighbors
        # Note: In _jacobi_step, 'diag' = 1/dt + sum_coeffs
        #       'off_diag' = sum_neighbors
        # So A*T = diag * T - off_diag
        
        diag = self.inv_dt + sum_coeffs
        return diag * T - sum_neighbors

    def _jacobi_step(self, T: torch.Tensor, k: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
        """
        One Jacobi iteration for implicit Euler system.
        If b is None, assumes b = T * inv_dt (standard forward step).
        """
        # Coefficients
        ax_p, ax_m, ay_p, ay_m, az_p, az_m = self._get_coefficients(k)
        
        # Neighbors: periodic lateral boundaries
        T_xp = torch.roll(T, -1, dims=0)
        T_xm = torch.roll(T, 1, dims=0)
        T_yp = torch.roll(T, -1, dims=1)
        T_ym = torch.roll(T, 1, dims=1)

        # z-direction: always Neumann (adiabatic top/bottom)
        T_padded = F.pad(T.unsqueeze(0).unsqueeze(0), (1, 1, 0, 0, 0, 0), mode='replicate').squeeze()
        T_zp = T_padded[:, :, 2:]
        T_zm = T_padded[:, :, :-2]
        
        # Diagonal
        diag = self.inv_dt + (ax_p + ax_m) + (ay_p + ay_m) + (az_p + az_m)
        
        # Off-diagonal contribution (R * T)
        off_diag = (
            ax_p * T_xp + ax_m * T_xm +
            ay_p * T_yp + ay_m * T_ym +
            az_p * T_zp + az_m * T_zm
        )
        
        # Default RHS for forward step if not provided
        if b is None:
            b = T * self.inv_dt
        
        # Jacobi update: D * T_new = b + R * T_old
        # T_new = (b + off_diag) / diag
        T_new = (b + off_diag) / diag
        
        return T_new
    
    def _implicit_step(self, T: torch.Tensor, k: torch.Tensor, rhs: torch.Tensor = None) -> torch.Tensor:
        """
        Perform one implicit Euler time step solve: A * T_new = rhs.
        If rhs is None, uses rhs = T / dt (standard time stepping).
        """
        # Initial guess (use previous step)
        T_new = T.clone()
        
        # If rhs is passed explicitly (for Adjoint), calculate 'b' once
        b = rhs if rhs is not None else (T * self.inv_dt)
        
        # Jacobi iteration
        for _ in range(self.jacobi_iters):
            T_new = self._jacobi_step(T_new, k, b=b)
        
        return T_new
    
    def forward(self, k: torch.Tensor, T0: torch.Tensor, use_adjoint: bool = False) -> torch.Tensor:
        """
        Run heat diffusion simulation.
        """
        T = T0.clone()
        surfaces = [T[:, :, -1].clone()]  # Top surface
        
        for _ in range(self.num_steps):
            if use_adjoint:
                # Discrete adjoint: stores one temperature state per time step; memory is
                # independent of the number of Jacobi iterations
                T = AdjointHeatStep.apply(T, k, self)
            else:
                # BPTT: keeps the graph of every unrolled Jacobi iteration of every step
                T = self._implicit_step(T, k)

            surfaces.append(T[:, :, -1].clone())
        
        surface_seq = torch.stack(surfaces, dim=0)
        return surface_seq
    