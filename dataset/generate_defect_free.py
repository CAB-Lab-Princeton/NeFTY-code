"""Defect-free ("sound-only") synthetic dataset used to train the Sound-Only U-Net.

Same background distribution and category schedule as ``dataset/synthetic.py``
(homogeneous for the first 80% of indices, 3-/4-layer for the rest), but no
defects are inserted.

Usage:
    python -m dataset.generate_defect_free --out_dir ./data/synthetic_1000_nodefect --num_samples 1000 --seed 42
"""
import os
import argparse

import numpy as np
import torch

from network.sim.simulator import DifferentiableHeatSimulator
from dataset.synthetic import run_custom_simulation, create_synthetic_data_phiflow, phiflow_available


def generate_nodefect_cases(
    resolution: tuple,
    domain_size: tuple,
    num_samples: int,
    heat_source_center: tuple = None,
    heat_intensity: float = 100.0,
    heat_radius: float = 0.5,
    device: torch.device = None
):
    """
    Generate defect-free diffusivity (k) and initial temperature (T0) fields.
    Returns lists of CPU tensors.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Nx, Ny, Nz = resolution
    Lx, Ly, Lz = domain_size

    if heat_source_center is None:
        heat_source_center = (Lx/2, Ly/2, 0.8*Lz)

    print(f"Generating {num_samples} defect-free geometries...")

    temp_sim = DifferentiableHeatSimulator(
        domain_size=domain_size,
        resolution=resolution,
        device=device
    )

    k_gt_list = []
    t0_gt_list = []

    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
    x = torch.linspace(dx/2, Lx - dx/2, Nx, device=device)
    y = torch.linspace(dy/2, Ly - dy/2, Ny, device=device)
    z = torch.linspace(dz/2, Lz - dz/2, Nz, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

    for i in range(num_samples):
        # Category schedule: [0, 0.8) homogeneous, [0.8, 0.9) 3 layers, [0.9, 1.0) 4 layers
        p = i / max(num_samples, 1)

        num_layers = 1
        if p < 0.8:
            num_layers = 1
        else:
            sub_p_layered = (p - 0.8) / 0.2
            if sub_p_layered < 0.5:
                num_layers = 3
            else:
                num_layers = 4

        if num_layers == 1:
            k_base = np.random.uniform(0.1, 0.2)
            k_gt = k_base * torch.ones(Nx, Ny, Nz, device=device)
        else:
            k_gt = torch.zeros(Nx, Ny, Nz, device=device)
            layer_height = Lz / num_layers
            for l_idx in range(num_layers):
                k_layer = np.random.uniform(0.1, 0.2)
                z_min = l_idx * layer_height
                z_max = (l_idx + 1) * layer_height
                if l_idx == num_layers - 1: z_max = Lz + 1e-6

                mask_layer = (zz >= z_min) & (zz < z_max)
                k_gt[mask_layer] = k_layer

        T0 = temp_sim.create_gaussian_heat_source(
            center=heat_source_center,
            intensity=heat_intensity,
            radius=heat_radius
        )

        if i % 100 == 0:
            print(f"Generated {i}/{num_samples}...")

        k_gt_list.append(k_gt.cpu())
        t0_gt_list.append(T0.cpu())

    return k_gt_list, t0_gt_list


def generate_and_save_nodefect_dataset(
    output_dir: str,
    num_samples: int = 1000,
    resolution: tuple = (64, 64, 16),
    domain_size: tuple = (10.0, 10.0, 1.0),
    num_steps: int = 100,
    dt: float = 0.05,
    use_phiflow: bool = True,
    device: torch.device = None
):
    """Generate the defect-free dataset and save it to ``<output_dir>/dataset.pt``.

    PhiFlow runs on all samples as a single batch (the sub-step count is set by the
    maximum diffusivity over the whole dataset). PhiFlow is required; generation stops
    with an error if it is unavailable or fails. ``use_phiflow=False`` (debugging only)
    stores NeFTY-solver surfaces as ``surface_gt`` and no ``surface_gt_phiflow`` key.
    """
    if use_phiflow and not phiflow_available():
        raise ImportError("PhiFlow is required to generate ground-truth observations "
                          "(pip install phiflow). Pass --no_phiflow to use the NeFTY solver "
                          "instead (debugging only).")
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting defect-free dataset generation (n={num_samples}, res={resolution})...")

    k_gt_list, t0_gt_list = generate_nodefect_cases(
        resolution=resolution,
        domain_size=domain_size,
        num_samples=num_samples,
        device=device
    )

    # A. NeFTY solver (reference)
    surface_gt_ours_list = run_custom_simulation(
        k_gt_list=k_gt_list,
        t0_gt_list=t0_gt_list,
        resolution=resolution,
        domain_size=domain_size,
        num_steps=num_steps,
        dt=dt,
        device=device
    )
    surface_ours_tensor = torch.stack(surface_gt_ours_list)

    # B. PhiFlow (ground truth), batched over all samples. Any failure aborts generation.
    surface_phiflow_tensor = None
    if use_phiflow:
        print("Running PhiFlow simulator (batched)...")
        all_k = torch.stack(k_gt_list).to(device)
        all_t0 = torch.stack(t0_gt_list).to(device)

        surface_phiflow_tensor = create_synthetic_data_phiflow(
            k_gt_tensor=all_k,
            T0_tensor=all_t0,
            domain_size=domain_size,
            dt=dt,
            num_steps=num_steps
        ).cpu()

        # Same checks as the benchmark generator: shape, finiteness, agreement with the solver
        if tuple(surface_phiflow_tensor.shape) != tuple(surface_ours_tensor.shape):
            raise RuntimeError(f"PhiFlow output has shape {tuple(surface_phiflow_tensor.shape)}, "
                               f"expected {tuple(surface_ours_tensor.shape)}; not saving the dataset.")
        if not torch.isfinite(surface_phiflow_tensor).all():
            raise RuntimeError("PhiFlow output contains non-finite values; not saving the dataset.")
        mean_mse = torch.mean((surface_ours_tensor - surface_phiflow_tensor) ** 2).item()
        if not mean_mse < 100.0:
            raise RuntimeError("PhiFlow and NeFTY-solver surfaces disagree far beyond the expected "
                               f"level (mean MSE {mean_mse:.4g}); not saving the dataset.")
    else:
        print("WARNING: --no_phiflow: observations come from the NeFTY solver (inverse crime).")

    print("Saving dataset...")

    dataset = {
        'k_gt': torch.stack(k_gt_list),
        't0_gt': torch.stack(t0_gt_list),
        'surface_gt': surface_phiflow_tensor if use_phiflow else surface_ours_tensor,
        'surface_gt_ours': surface_ours_tensor,
    }
    if use_phiflow:
        dataset['surface_gt_phiflow'] = surface_phiflow_tensor
    dataset.update({
        'resolution': {'x': resolution[0], 'y': resolution[1], 'z': resolution[2]},
        'time_steps': num_steps + 1,
        'domain_size': {'x': domain_size[0], 'y': domain_size[1], 'z': domain_size[2]},
        'dt': dt,
        'num_steps': num_steps,
    })

    save_path = os.path.join(output_dir, "dataset.pt")
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate the defect-free (sound-only) dataset")
    parser.add_argument("--out_dir", type=str, default="data/synthetic_1000_nodefect",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=int, nargs=3, default=[64, 64, 16],
                        help="Grid resolution (x y z)")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of recorded time steps (frames = num_steps + 1)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time between recorded frames")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--no_phiflow", action="store_true",
                        help="Debugging only: use the NeFTY solver instead of PhiFlow for the "
                             "observations (inverse crime); no surface_gt_phiflow key is saved")
    args = parser.parse_args()

    dev = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    generate_and_save_nodefect_dataset(
        output_dir=args.out_dir,
        num_samples=args.num_samples,
        resolution=tuple(args.resolution),
        num_steps=args.num_steps,
        dt=args.dt,
        use_phiflow=not args.no_phiflow,
        device=dev
    )


if __name__ == "__main__":
    main()
