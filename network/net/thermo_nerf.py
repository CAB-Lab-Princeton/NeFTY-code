import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermoNeRF(nn.Module):
    """Coordinate network for the 3D thermal diffusivity field alpha(x, y, z).

    Two parameterizations share this class:
      * neural field (default): an MLP on positionally encoded coordinates (NeFTY);
      * voxel grid (``use_grid=True``): a learnable tensor sampled with trilinear
        interpolation (the Grid Opt. baseline).

    The output is bounded to ``[min_diffusivity, max_diffusivity]`` through a
    scaled sigmoid (or ``softplus + min_diffusivity``).
    """

    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        input_ch: int = 63,
        output_ch: int = 1,
        skips: list = [4],
        min_diffusivity: float = 0.001,
        max_diffusivity: float = 0.5,
        use_grid: bool = False,
        grid_resolution: tuple = (32, 32, 8),
        domain_size: tuple = (10.0, 10.0, 1.0),
        output_activation: str = 'sigmoid',
        mlp_activation: str = 'softplus',
    ):
        super().__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.min_diffusivity = min_diffusivity
        self.max_diffusivity = max_diffusivity

        self.use_grid = use_grid
        self.grid_resolution = grid_resolution
        self.domain_size = domain_size
        self.output_activation = output_activation

        if mlp_activation == 'softplus':
            self.hidden_act = F.softplus
        elif mlp_activation == 'relu':
            self.hidden_act = F.relu
        else:
            raise ValueError(f"Unknown mlp_activation: {mlp_activation}")
        self.mlp_activation = mlp_activation

        if self.use_grid:
            # Voxel parameters live in the pre-activation space. A zero logit maps to
            # the midpoint of [min_diffusivity, max_diffusivity] under the sigmoid.
            Nx, Ny, Nz = grid_resolution
            self.grid = nn.Parameter(torch.zeros(1, 1, Nx, Ny, Nz))

        # MLP layers (the encoded input is re-injected after the layers in `skips`)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
             for i in range(D - 1)]
        )
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """Predict diffusivity for a batch of encoded coordinates x: [B, input_ch]."""
        if self.use_grid:
            # The embedder keeps the raw coordinates as the first 3 channels.
            coords = x[..., :3]
            Lx, Ly, Lz = self.domain_size

            # Physical coordinates [0, L] -> [-1, 1] for grid_sample.
            norm_x = 2 * (coords[..., 0] / Lx) - 1
            norm_y = 2 * (coords[..., 1] / Ly) - 1
            norm_z = 2 * (coords[..., 2] / Lz) - 1

            # The grid is stored as [1, 1, Nx, Ny, Nz] = [N, C, D, H, W], and
            # grid_sample expects the last dim ordered as (W, H, D) = (z, y, x).
            grid_coords = torch.stack([norm_z, norm_y, norm_x], dim=-1)
            grid_coords = grid_coords.view(1, 1, 1, -1, 3)

            raw = F.grid_sample(self.grid, grid_coords, align_corners=True, padding_mode='border')
            raw = raw.view(-1, 1)
        else:
            h = x
            for i, layer in enumerate(self.pts_linears):
                h = layer(h)
                h = self.hidden_act(h)
                if i in self.skips:
                    h = torch.cat([x, h], dim=-1)
            raw = self.output_linear(h)

        if self.output_activation == 'sigmoid':
            k = torch.sigmoid(raw) * (self.max_diffusivity - self.min_diffusivity) + self.min_diffusivity
        elif self.output_activation == 'softplus':
            k = F.softplus(raw) + self.min_diffusivity
        else:
            raise ValueError(f"Unknown activation: {self.output_activation}")
        return k
