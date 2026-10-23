import torch
import torch.nn as nn
import torch.nn.functional as F
from .thermo_nerf import ThermoNeRF

class TemperatureNetwork(nn.Module):
    """
    PINN Temperature Network.
    Predicts Temperature field T(x, y, z, t) in 4D space-time.
    """
    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        input_ch: int = 63, # Depends on embedding
        output_ch: int = 1,
        skips: list = [4]
    ):
        super().__init__()
        
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        
        # Build network layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
             for i in range(D - 1)]
        )
        
        # Output layer
        self.output_linear = nn.Linear(W, output_ch)
    
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.tanh(h) # Tanh is standard for PINNs for smooth derivatives
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)
        
        raw = self.output_linear(h)
        return raw

class ThermoNERFPINN(nn.Module):
    """
    Combined Model:
    1. Diffusivity Network (ThermoNeRF) -> alpha(x,y,z)
    2. Temperature Network (PINN) -> T(x,y,z,t)
    """
    def __init__(
        self,
        # Diffusivity Net Config
        diff_net_args: dict,
        # Temperature Net Config
        temp_net_args: dict,
    ):
        super().__init__()
        
        # 1. Diffusivity Network
        self.diffusivity_net = ThermoNeRF(**diff_net_args)
        
        # 2. Temperature Network
        self.temperature_net = TemperatureNetwork(**temp_net_args)
        
    def forward_diffusivity(self, x_spatial_embedded):
        """Predict alpha(x,y,z)."""
        return self.diffusivity_net(x_spatial_embedded)

    def forward_temperature(self, x_spacetime_embedded):
        """Predict T(x,y,z,t)."""
        return self.temperature_net(x_spacetime_embedded)
