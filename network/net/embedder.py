import torch
import torch.nn as nn
import numpy as np

class Embedder:
    """
    Positional encoding with Frequency Annealing support.
    Reference: Nerfies (Park et al., 2021)
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        self.embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        
        # Original coordinates (always kept, weight=1)
        if self.kwargs['include_input']:
            self.embed_fns.append(lambda x: x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_power']
        self.num_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            self.freq_bands = 2.**torch.linspace(0., max_freq, steps=self.num_freqs)
        else:
            self.freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=self.num_freqs)
            
        # We don't bake the functions into a list anymore, we compute dynamically
        # to apply weights easily.
        self.out_dim = out_dim + self.num_freqs * 2 * d # 2 for sin/cos
        
    def embed(self, inputs, beta=None):
        """
        Args:
            inputs: (B, 3)
            beta: float 0 to num_freqs. If None, full frequencies are used.
        """
        res = []
        
        # 1. Raw Input
        if self.kwargs['include_input']:
            res.append(inputs)
            
        # 2. Fourier Features
        # Shape: (B, 3, Num_Freqs)
        # We extend inputs to apply all freqs at once
        x = inputs.unsqueeze(-1) * self.freq_bands.to(inputs.device)
        
        # Apply sin/cos
        # Shape: (B, 3, Num_Freqs, 2)
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        features = torch.stack([x_sin, x_cos], dim=-1)
        
        # 3. Apply Frequency Annealing Window
        if beta is not None:
            # Create weights for each frequency band k
            # beta is a scalar (e.g., 3.5 means bands 0,1,2 are full, 3 is partial)
            k = torch.arange(self.num_freqs, dtype=torch.float32, device=inputs.device)
            
            # Nerfies Cosine Window
            # clamp(beta - k, 0, 1) makes the window slide
            weight = (1.0 - torch.cos(np.pi * torch.clamp(beta - k, 0, 1))) / 2.0
            
            # Reshape weight to (1, 1, Num_Freqs, 1) to broadcast
            weight = weight.reshape(1, 1, -1, 1)
            
            features = features * weight
            
        # Flatten: (B, 3 * Num_Freqs * 2)
        # Reshape to match original NeRF concatenation order: [sin(x), cos(x)...]
        # or just flatten everything, MLP doesn't care about order.
        return torch.cat([r for r in res] + [features.reshape(inputs.shape[0], -1)], dim=-1)


def annealing_beta(global_step, num_freqs, anneal_iters, use_fa=True):
    """Frequency-annealing parameter used by the trainers at ``global_step``:
    beta ramps linearly from 0 to ``num_freqs`` over ``anneal_iters`` steps."""
    if not use_fa or global_step >= anneal_iters:
        return num_freqs
    return (global_step / anneal_iters) * num_freqs


def get_embedder(multires, i=0):
    """Get positional encoding embedder."""
    if i == -1:
        return lambda x, beta=None: x, 3
    
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_power': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, beta, eo=embedder_obj: eo.embed(x, beta)
    return embed, embedder_obj.out_dim
