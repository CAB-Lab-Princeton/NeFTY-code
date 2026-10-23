"""Synthetic benchmark generation for NeFTY.

Ground-truth surface temperatures are produced with PhiFlow (an independent
explicit finite-volume solver) to avoid an inverse crime. The NeFTY solver's own
forward prediction is stored alongside for reference.

Usage (paper benchmark, 1,000 samples):
    python -m dataset.synthetic --out_dir ./data/synthetic_1000 --num_samples 1000 --seed 42
"""
import os
import argparse
import importlib.util

import numpy as np
import torch

from network.sim.simulator import DifferentiableHeatSimulator


def _import_phiflow():
    """Return the ``phi.torch.flow`` module, or None if PhiFlow is not installed."""
    try:
        from phi.torch import flow
        return flow
    except ImportError as e:
        print(f"PhiFlow import warning: {e}")
        return None


def phiflow_available() -> bool:
    """True if PhiFlow can be imported (checked lazily, so training does not need it)."""
    return importlib.util.find_spec("phi") is not None


def generate_synthetic_cases(
    resolution: tuple,
    domain_size: tuple,
    num_samples: int,
    heat_source_center: tuple = None,
    heat_intensity: float = 100.0,
    heat_radius: float = 0.5,
    device: torch.device = None
):
    """
    Generate synthetic diffusivity (k) and initial temperature (T0) fields.
    Returns lists of CPU tensors.

    The sample index determines the category (fractions of ``num_samples``):
      [0.0, 0.8)  homogeneous background, split evenly into 1/2/3/4 defects
      [0.8, 0.9)  3-layer background, split evenly into 1/2/3/4 defects
      [0.9, 1.0)  4-layer background, split evenly into 1/2/3/4 defects
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Nx, Ny, Nz = resolution
    Lx, Ly, Lz = domain_size

    if heat_source_center is None:
        heat_source_center = (Lx/2, Ly/2, 0.8*Lz)

    print(f"Generating {num_samples} synthetic geometries...")

    # Helper simulator for T0 generation
    temp_sim = DifferentiableHeatSimulator(
        domain_size=domain_size,
        resolution=resolution,
        device=device
    )

    k_gt_list = []
    t0_gt_list = []

    # Cell-center coordinates
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
    x = torch.linspace(dx/2, Lx - dx/2, Nx, device=device)
    y = torch.linspace(dy/2, Ly - dy/2, Ny, device=device)
    z = torch.linspace(dz/2, Lz - dz/2, Nz, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

    for i in range(num_samples):
        p = i / max(num_samples, 1)

        n_defects = 1
        num_layers = 1

        if p < 0.8:
            # Homogeneous (80%)
            num_layers = 1
            sub_p = p / 0.8
            if sub_p < 0.25: n_defects = 1
            elif sub_p < 0.50: n_defects = 2
            elif sub_p < 0.75: n_defects = 3
            else: n_defects = 4
        else:
            # Layered (20%)
            sub_p_layered = (p - 0.8) / 0.2
            if sub_p_layered < 0.5:
                num_layers = 3
                sub_sub_p = sub_p_layered / 0.5
            else:
                num_layers = 4
                sub_sub_p = (sub_p_layered - 0.5) / 0.5

            if sub_sub_p < 0.25: n_defects = 1
            elif sub_sub_p < 0.50: n_defects = 2
            elif sub_sub_p < 0.75: n_defects = 3
            else: n_defects = 4

        # 1. Background
        if num_layers == 1:
            k_base = np.random.uniform(0.1, 0.2)
            k_gt = k_base * torch.ones(Nx, Ny, Nz, device=device)
        else:
            # Layered background (strata along z)
            k_gt = torch.zeros(Nx, Ny, Nz, device=device)
            layer_height = Lz / num_layers
            for l_idx in range(num_layers):
                k_layer = np.random.uniform(0.1, 0.2)
                z_min = l_idx * layer_height
                z_max = (l_idx + 1) * layer_height
                if l_idx == num_layers - 1: z_max = Lz + 1e-6

                mask_layer = (zz >= z_min) & (zz < z_max)
                k_gt[mask_layer] = k_layer

        # 2. Defects
        for _ in range(n_defects):
            k_defect = np.random.uniform(0.005, 0.015)

            shape_type = np.random.choice(['ellipsoid', 'box', 'cylinder'])

            # Half-extents (physical units). z: 0.15-0.35 Lz; x, y: 0.3-1.0.
            defect_rz = np.random.uniform(0.15 * Lz, 0.35 * Lz)
            defect_rx = np.random.uniform(0.3, 1.0)
            defect_ry = np.random.uniform(0.3, 1.0)

            # Lateral centers constrained to the central region [0.3, 0.7] of the domain
            x_min = max(defect_rx, Lx * 0.3)
            x_max = min(Lx - defect_rx, Lx * 0.7)

            if x_min >= x_max: cx = Lx * 0.5
            else: cx = np.random.uniform(x_min, x_max)

            y_min = max(defect_ry, Ly * 0.3)
            y_max = min(Ly - defect_ry, Ly * 0.7)

            if y_min >= y_max: cy = Ly * 0.5
            else: cy = np.random.uniform(y_min, y_max)

            # Subsurface depth
            cz = np.random.uniform(0.25 * Lz, 0.75 * Lz)

            if shape_type == 'ellipsoid':
                r2 = ((xx - cx)/defect_rx)**2 + ((yy - cy)/defect_ry)**2 + ((zz - cz)/defect_rz)**2
                defect_mask = (r2 <= 1.0).float()

            elif shape_type == 'box':
                mask_x = (torch.abs(xx - cx) <= defect_rx)
                mask_y = (torch.abs(yy - cy) <= defect_ry)
                mask_z = (torch.abs(zz - cz) <= defect_rz)
                defect_mask = (mask_x & mask_y & mask_z).float()

            elif shape_type == 'cylinder':
                # Flat-bottom-hole style: axis along z, flat top/bottom
                r2_xy = ((xx - cx)/defect_rx)**2 + ((yy - cy)/defect_ry)**2
                mask_z = (torch.abs(zz - cz) <= defect_rz)
                defect_mask = ((r2_xy <= 1.0) & mask_z).float()

            # Defect overwrites the background
            k_gt = torch.where(defect_mask > 0.5, torch.tensor(k_defect, device=device), k_gt)

        # Initial condition: Gaussian post-flash temperature
        T0 = temp_sim.create_gaussian_heat_source(
            center=heat_source_center,
            intensity=heat_intensity,
            radius=heat_radius
        )

        if i % 100 == 0:
            print(f"  Sample {i}: Type={'Layered' if num_layers > 1 else 'Homogeneous'}, "
                  f"Layers={num_layers}, Defects={n_defects}")

        k_gt_list.append(k_gt.cpu())
        t0_gt_list.append(T0.cpu())

    return k_gt_list, t0_gt_list


def run_custom_simulation(
    k_gt_list,
    t0_gt_list,
    resolution,
    domain_size,
    num_steps: int,
    dt: float,
    device: torch.device = None
):
    """
    Run the NeFTY DifferentiableHeatSimulator on a list of cases.
    Returns a list of surface temperature sequences (CPU tensors).
    """
    print(f"Running NeFTY simulator on {len(k_gt_list)} samples...")

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    simulator = DifferentiableHeatSimulator(
        domain_size=domain_size,
        resolution=resolution,
        dt=dt,
        num_steps=num_steps,
        jacobi_iters=50,
        device=device
    )

    surface_gt_list = []

    for k_gt_cpu, t0_cpu in zip(k_gt_list, t0_gt_list):
        k_gt = k_gt_cpu.to(device)
        T0 = t0_cpu.to(device)

        with torch.no_grad():
            surface_seq = simulator(k_gt, T0)

        surface_gt_list.append(surface_seq.cpu())

    return surface_gt_list


def create_synthetic_data_phiflow(
    k_gt_tensor: torch.Tensor,
    T0_tensor: torch.Tensor,
    domain_size: tuple,
    dt: float,
    num_steps: int
):
    """
    Ground-truth simulation with PhiFlow (explicit diffusion with adaptive sub-stepping).
    Supports batched inputs.

    Args:
        k_gt_tensor: (Nx, Ny, Nz) or (Batch, Nx, Ny, Nz) diffusivity
        T0_tensor: (Nx, Ny, Nz) or (Batch, Nx, Ny, Nz) initial temperature
        domain_size: (Lx, Ly, Lz)
        dt: time between recorded frames
        num_steps: number of recorded steps

    Returns:
        (num_steps+1, Nx, Ny) for 3D input, or (Batch, num_steps+1, Nx, Ny) for 4D input.
    """
    flow = _import_phiflow()
    if flow is None:
        raise ImportError("PhiFlow is not installed. Run `pip install phiflow` to use this function.")

    print(f"Running PhiFlow simulation (ground truth) on {tuple(k_gt_tensor.shape)}...")

    Lx, Ly, Lz = domain_size

    is_batched = k_gt_tensor.ndim == 4
    if not is_batched:
        k_native = k_gt_tensor.unsqueeze(0)  # (1, Nx, Ny, Nz)
        t_native = T0_tensor.unsqueeze(0)    # (1, Nx, Ny, Nz)
    else:
        k_native = k_gt_tensor
        t_native = T0_tensor

    B, Nx, Ny, Nz = k_native.shape

    # Boundary conditions matching the NeFTY solver: periodic in x/y, zero-flux in z
    bc = flow.extrapolation.combine_sides(
        x=flow.extrapolation.PERIODIC,
        y=flow.extrapolation.PERIODIC,
        z=flow.extrapolation.BOUNDARY
    )

    bounds = flow.Box(x=Lx, y=Ly, z=Lz)

    # Wrap PyTorch tensors into PhiFlow tensors with named dimensions
    k_data = flow.math.wrap(k_native, flow.math.batch('batch'), flow.math.spatial('x, y, z'))
    t_data = flow.math.wrap(t_native, flow.math.batch('batch'), flow.math.spatial('x, y, z'))

    k_grid = flow.CenteredGrid(values=k_data, extrapolation=bc, bounds=bounds)
    T_grid = flow.CenteredGrid(values=t_data, extrapolation=bc, bounds=bounds)

    # Adaptive sub-stepping for explicit stability
    dx_vals = [Lx/Nx, Ly/Ny, Lz/Nz]
    min_dx = min(dx_vals)
    dims = 3

    # Maximum diffusivity over the whole batch
    max_k = torch.max(k_native).item()

    # Max stable time step and required sub-steps
    max_stable_dt = (min_dx ** 2) / (2 * dims * max_k)
    fidelity_factor = 2.0
    required_substeps = int(np.ceil((dt / max_stable_dt) * fidelity_factor))
    num_substeps = max(10, required_substeps)

    print(f"  GT Gen: max_k={max_k:.4f}, min_dx={min_dx:.4f}")
    print(f"  GT Gen: Stability limit dt < {max_stable_dt:.6f}")
    print(f"  GT Gen: dt={dt}. Using {num_substeps} substeps (effective dt={dt/num_substeps:.6f})")

    surfaces = []

    def get_top_surface(grid):
        vals = grid.values.native(('batch', 'x', 'y', 'z'))
        return vals[:, :, :, -1]  # (Batch, Nx, Ny)

    surfaces.append(get_top_surface(T_grid))

    for _ in range(num_steps):
        # dT/dt = div(k grad T)
        T_grid = flow.diffuse.explicit(
            T_grid,
            diffusivity=k_grid,
            dt=dt,
            substeps=num_substeps
        )
        surfaces.append(get_top_surface(T_grid))

    # (Steps, Batch, Nx, Ny) -> (Batch, Steps, Nx, Ny)
    surface_seq = torch.stack(surfaces, dim=0).permute(1, 0, 2, 3)

    if not is_batched:
        return surface_seq[0]

    return surface_seq


def generate_and_save_dataset(
    output_dir: str,
    num_samples: int = 1000,
    resolution: tuple = (64, 64, 16),
    domain_size: tuple = (10.0, 10.0, 1.0),
    num_steps: int = 100,
    dt: float = 0.05,
    batch_size: int = 500,
    visualize: bool = True,
    use_phiflow: bool = True,
    device: torch.device = None
):
    """Generate the synthetic benchmark and save it to ``<output_dir>/dataset.pt``.

    Saved keys:
      k_gt               (N, Nx, Ny, Nz)          ground-truth diffusivity
      t0_gt              (N, Nx, Ny, Nz)          initial temperature
      surface_gt         (N, num_steps+1, Nx, Ny) observations used for training
                                                 (PhiFlow surfaces)
      surface_gt_phiflow (N, num_steps+1, Nx, Ny) PhiFlow surfaces
      surface_gt_ours    (N, num_steps+1, Nx, Ny) NeFTY-solver surfaces (reference)
      resolution, time_steps

    Note: the PhiFlow sub-step count is derived from the maximum diffusivity of
    each simulation batch, so ``batch_size`` is part of the generation recipe.

    PhiFlow is required: generation stops with an error if it is not installed or a
    PhiFlow batch fails. With ``use_phiflow=False`` the observations come from the NeFTY
    solver itself (``surface_gt = surface_gt_ours``, no ``surface_gt_phiflow`` key), which
    is an inverse-crime setting meant for debugging only.
    """
    if use_phiflow and not phiflow_available():
        raise ImportError("PhiFlow is required to generate ground-truth observations "
                          "(pip install phiflow). Pass --no_phiflow to use the NeFTY solver "
                          "instead (debugging only).")
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting dataset generation (n={num_samples}, res={resolution})...")

    # 1. Geometries (all at once so the category schedule spans the whole dataset)
    k_gt_list, t0_gt_list = generate_synthetic_cases(
        resolution=resolution,
        domain_size=domain_size,
        num_samples=num_samples,
        device=device
    )

    # 2. Simulations in batches
    surface_gt_ours_list = []
    surface_gt_phiflow_list = []

    print(f"Running simulations (batch size {batch_size})...")

    # Running statistics of the solver discrepancy (NeFTY solver vs PhiFlow)
    total_mse = 0.0
    total_mae = 0.0
    total_ref_energy = 0.0
    total_ref_abs = 0.0
    total_samples = 0

    iter_range = range(0, num_samples, batch_size)
    try:
        from tqdm import tqdm
        iter_range = tqdm(iter_range, desc="Simulating batches")
    except ImportError:
        pass

    for i in iter_range:
        end_idx = min(i + batch_size, num_samples)
        batch_k_cpu = k_gt_list[i:end_idx]
        batch_t0_cpu = t0_gt_list[i:end_idx]

        # A. NeFTY solver
        batch_ours = run_custom_simulation(
            k_gt_list=batch_k_cpu,
            t0_gt_list=batch_t0_cpu,
            resolution=resolution,
            domain_size=domain_size,
            num_steps=num_steps,
            dt=dt,
            device=device
        )
        surface_gt_ours_list.extend(batch_ours)

        # B. PhiFlow (ground truth). Any failure aborts generation.
        if use_phiflow:
            k_batch_dev = torch.stack(batch_k_cpu).to(device)
            t0_batch_dev = torch.stack(batch_t0_cpu).to(device)

            batch_phi_tensor = create_synthetic_data_phiflow(
                k_gt_tensor=k_batch_dev,
                T0_tensor=t0_batch_dev,
                domain_size=domain_size,
                dt=dt,
                num_steps=num_steps
            )
            batch_ours_tensor = torch.stack(batch_ours).to(device)  # (B, T, H, W)
            if tuple(batch_phi_tensor.shape) != tuple(batch_ours_tensor.shape):
                raise RuntimeError(f"PhiFlow output has shape {tuple(batch_phi_tensor.shape)}, expected "
                                   f"{tuple(batch_ours_tensor.shape)}; not saving the dataset.")
            if not torch.isfinite(batch_phi_tensor).all():
                raise RuntimeError(f"PhiFlow output of batch {i} contains non-finite values; "
                                   "not saving the dataset.")
            batch_phi = [s.cpu() for s in batch_phi_tensor]
            surface_gt_phiflow_list.extend(batch_phi)

            mse = torch.mean((batch_ours_tensor - batch_phi_tensor)**2).item()
            mae = torch.mean(torch.abs(batch_ours_tensor - batch_phi_tensor)).item()
            ref_energy = torch.mean(batch_phi_tensor ** 2).item()
            ref_abs = torch.mean(torch.abs(batch_phi_tensor)).item()

            total_mse += mse * len(batch_ours)
            total_mae += mae * len(batch_ours)
            total_ref_energy += ref_energy * len(batch_ours)
            total_ref_abs += ref_abs * len(batch_ours)
            total_samples += len(batch_ours)

    if total_samples > 0:
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        avg_ref_energy = total_ref_energy / total_samples
        avg_ref_abs = total_ref_abs / total_samples

        rel_mse = avg_mse / (avg_ref_energy + 1e-8)
        rel_mae = avg_mae / (avg_ref_abs + 1e-8)

        print(f"\nGlobal Simulator Discrepancy (MSE): {avg_mse:.8f}")
        print(f"Global Simulator Discrepancy (MAE): {avg_mae:.8f}")
        print(f"Global Simulator Discrepancy (Relative MSE): {rel_mse:.6f}")
        print(f"Global Simulator Discrepancy (Relative MAE): {rel_mae:.6f}")

        rel_mse_threshold = 0.05
        rel_mae_threshold = 0.05
        if rel_mse > rel_mse_threshold or rel_mae > rel_mae_threshold:
            print(
                "WARNING: High relative discrepancy between simulators! "
                f"(rel_mse>{rel_mse_threshold} or rel_mae>{rel_mae_threshold})"
            )
        else:
            print(
                "Discrepancy check passed "
                f"(rel_mse<{rel_mse_threshold} and rel_mae<{rel_mae_threshold})."
            )

    print("Saving full dataset...")

    dataset = {
        'k_gt': torch.stack(k_gt_list),
        't0_gt': torch.stack(t0_gt_list),
        'surface_gt_ours': torch.stack(surface_gt_ours_list),
        'resolution': {'x': resolution[0], 'y': resolution[1], 'z': resolution[2]},
        'time_steps': num_steps + 1
    }

    if use_phiflow:
        if not np.isfinite(total_mse) or total_mse / total_samples >= 100.0:
            raise RuntimeError("PhiFlow and NeFTY-solver surfaces disagree far beyond the "
                               f"expected level (mean MSE {total_mse / total_samples:.4g}); "
                               "not saving the dataset.")
        dataset['surface_gt_phiflow'] = torch.stack(surface_gt_phiflow_list)
        dataset['surface_gt'] = dataset['surface_gt_phiflow']
    else:
        print("WARNING: --no_phiflow: observations come from the NeFTY solver (inverse crime).")
        dataset['surface_gt'] = dataset['surface_gt_ours']

    # Simulation settings (checked by evaluators that re-simulate with fixed settings)
    dataset['domain_size'] = {'x': domain_size[0], 'y': domain_size[1], 'z': domain_size[2]}
    dataset['dt'] = dt
    dataset['num_steps'] = num_steps

    save_path = os.path.join(output_dir, "dataset.pt")
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")

    if visualize:
        _save_visualizations(output_dir, k_gt_list, surface_gt_ours_list,
                             surface_gt_phiflow_list, num_samples, num_steps, domain_size)

    print("Dataset generation complete.")


def _save_visualizations(output_dir, k_gt_list, surface_gt_ours_list, surface_gt_phiflow_list,
                         num_samples, num_steps, domain_size):
    """Save a few diffusivity / surface-temperature figures for visual inspection."""
    from utils.vis_utils import (
        visualize_diffusivity_slices,
        visualize_surface_temperature_time,
        visualize_3d_diffusivity_slice,
    )

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    if num_samples >= 1000:
        indices_to_vis = [
            100, 300, 500, 700,  # homogeneous
            810, 835, 955, 985   # layered
        ]
    elif num_samples >= 10:
        indices_to_vis = list(range(0, num_samples, num_samples // 5))
    else:
        indices_to_vis = list(range(num_samples))

    print("Generating visualizations...")
    for idx in indices_to_vis:
        if idx >= num_samples:
            continue

        k_val = k_gt_list[idx]
        surf_ours = surface_gt_ours_list[idx]
        surf_gt = surface_gt_phiflow_list[idx] if len(surface_gt_phiflow_list) > idx else surf_ours

        visualize_diffusivity_slices(
            k_pred=k_val,
            k_gt=k_val,
            save_path=os.path.join(vis_dir, f"sample_{idx:04d}_k_slices.png"),
            domain_size=domain_size
        )

        visualize_3d_diffusivity_slice(
            k_gt=k_val,
            k_pred=k_val,
            save_path=os.path.join(vis_dir, f"sample_{idx:04d}_k_3d.png")
        )

        # Ground truth (PhiFlow) vs NeFTY solver
        visualize_surface_temperature_time(
            surface_gt=surf_gt,
            surface_pred=surf_ours,
            save_path=os.path.join(vis_dir, f"sample_{idx:04d}_surface_compare.png"),
            frame_indices=list(range(0, num_steps + 1, 10))
        )


def load_synthetic_dataset(dataset_path: str, map_location: str | torch.device = "cpu"):
    """
    Load a dataset saved by generate_and_save_dataset().
    Accepts either a directory containing dataset.pt or a direct .pt path.
    """
    if os.path.isdir(dataset_path):
        dataset_path = os.path.join(dataset_path, "dataset.pt")

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    data = torch.load(dataset_path, map_location=map_location)
    # The frame count is taken from the tensors: the stored 'time_steps' entry of some
    # released files (e.g. 51 for the 101-frame benchmark) is stale metadata.
    if 'surface_gt' in data:
        data['time_steps'] = int(data['surface_gt'].shape[1])
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate the NeFTY synthetic benchmark")
    parser.add_argument("--out_dir", type=str, default="data/synthetic_1000",
                        help="Output directory (dataset.pt + visualizations/)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples (category schedule scales with this)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=int, nargs=3, default=[64, 64, 16],
                        help="Grid resolution (x y z)")
    parser.add_argument("--domain", type=float, nargs=3, default=[10.0, 10.0, 1.0],
                        help="Domain size (Lx Ly Lz)")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of recorded time steps (frames = num_steps + 1)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time between recorded frames")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Samples per PhiFlow batch (sets the sub-step count via the batch max diffusivity)")
    parser.add_argument("--no_vis", action="store_true", help="Skip the visualization PNGs")
    parser.add_argument("--no_phiflow", action="store_true",
                        help="Debugging only: use the NeFTY solver instead of PhiFlow for the "
                             "observations (inverse crime); no surface_gt_phiflow key is saved")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    dev = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    generate_and_save_dataset(
        output_dir=args.out_dir,
        num_samples=args.num_samples,
        resolution=tuple(args.resolution),
        domain_size=tuple(args.domain),
        num_steps=args.num_steps,
        dt=args.dt,
        batch_size=args.batch_size,
        visualize=not args.no_vis,
        use_phiflow=not args.no_phiflow,
        device=dev
    )


if __name__ == "__main__":
    main()
