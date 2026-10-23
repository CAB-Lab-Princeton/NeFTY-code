"""
Visualization utilities for ThermoNeRF training results.

This module provides helper functions for visualizing:
- Diffusivity fields (2D slices, 3D volume, isosurfaces)
- Surface temperatures
- Time evolution
- Evaluation metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import io
import os
from PIL import Image
import imageio.v2 as imageio


def visualize_diffusivity_slices(k_pred, k_gt, save_path=None, domain_size=None):
    """Visualize diffusivity field as 2D slices."""
    k_pred_np = k_pred.detach().cpu().numpy() if torch.is_tensor(k_pred) else k_pred
    k_gt_np = k_gt.detach().cpu().numpy() if torch.is_tensor(k_gt) else k_gt
    
    Nx, Ny, Nz = k_pred_np.shape
    
    # Select middle slices
    mid_x = Nx // 2
    mid_y = Ny // 2
    mid_z = Nz // 2
    
    # Convert indices to physical coordinates if domain_size is provided
    if domain_size is not None:
        Lx, Ly, Lz = domain_size
        dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
        x_phys = (mid_x + 0.5) * dx
        y_phys = (mid_y + 0.5) * dy
        z_phys = (mid_z + 0.5) * dz
    else:
        x_phys, y_phys, z_phys = mid_x, mid_y, mid_z
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Diffusivity Field Comparison', fontsize=16)
    
    # Set extent for physical coordinates if domain_size is provided
    if domain_size is not None:
        Lx, Ly, Lz = domain_size
        # XY slice extent: [xmin, xmax, ymin, ymax]
        extent_xy = [0, Lx, 0, Ly]
        # XZ slice extent: [zmin, zmax, xmin, xmax] (note: imshow uses [left, right, bottom, top])
        extent_xz = [0, Lz, 0, Lx]
    else:
        extent_xy = None
        extent_xz = None
    
    # XY slice (at middle z)
    im1 = axes[0, 0].imshow(k_gt_np[:, :, mid_z], cmap='viridis', origin='lower', extent=extent_xy)
    axes[0, 0].set_title(f'Ground Truth (z={z_phys:.2f})' if domain_size else f'Ground Truth (z={mid_z})')
    axes[0, 0].set_xlabel('Y')
    axes[0, 0].set_ylabel('X')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(k_pred_np[:, :, mid_z], cmap='viridis', origin='lower', extent=extent_xy)
    axes[0, 1].set_title(f'Predicted (z={z_phys:.2f})' if domain_size else f'Predicted (z={mid_z})')
    axes[0, 1].set_xlabel('Y')
    axes[0, 1].set_ylabel('X')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(np.abs(k_pred_np[:, :, mid_z] - k_gt_np[:, :, mid_z]), 
                            cmap='hot', origin='lower', extent=extent_xy)
    axes[0, 2].set_title(f'Error (z={z_phys:.2f})' if domain_size else f'Error (z={mid_z})')
    axes[0, 2].set_xlabel('Y')
    axes[0, 2].set_ylabel('X')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # XZ slice (at middle y)
    im4 = axes[1, 0].imshow(k_gt_np[:, mid_y, :], cmap='viridis', origin='lower', aspect='auto', extent=extent_xz)
    axes[1, 0].set_title(f'Ground Truth (y={y_phys:.2f})' if domain_size else f'Ground Truth (y={mid_y})')
    axes[1, 0].set_xlabel('Z')
    axes[1, 0].set_ylabel('X')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(k_pred_np[:, mid_y, :], cmap='viridis', origin='lower', aspect='auto', extent=extent_xz)
    axes[1, 1].set_title(f'Predicted (y={y_phys:.2f})' if domain_size else f'Predicted (y={mid_y})')
    axes[1, 1].set_xlabel('Z')
    axes[1, 1].set_ylabel('X')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(np.abs(k_pred_np[:, mid_y, :] - k_gt_np[:, mid_y, :]), 
                            cmap='hot', origin='lower', aspect='auto', extent=extent_xz)
    axes[1, 2].set_title(f'Error (y={y_phys:.2f})' if domain_size else f'Error (y={mid_y})')
    axes[1, 2].set_xlabel('Z')
    axes[1, 2].set_ylabel('X')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diffusivity visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_3d_diffusivity(k_pred, k_gt, save_path=None, threshold=None, downsample=1, domain_size=None):
    """
    Visualize 3D diffusivity field using volume rendering and isosurfaces.
    
    Args:
        k_pred: Predicted diffusivity field [Nx, Ny, Nz]
        k_gt: Ground truth diffusivity field [Nx, Ny, Nz]
        save_path: Path to save the visualization
        threshold: Threshold value for isosurface (None for auto)
        downsample: Downsample factor for faster rendering
        domain_size: Physical domain size (Lx, Ly, Lz) for coordinate conversion
    """
    k_pred_np = k_pred.detach().cpu().numpy() if torch.is_tensor(k_pred) else k_pred
    k_gt_np = k_gt.detach().cpu().numpy() if torch.is_tensor(k_gt) else k_gt
    
    # Store original resolution for coordinate conversion
    Nx_orig, Ny_orig, Nz_orig = k_pred_np.shape
    
    # Downsample for faster rendering
    if downsample > 1:
        k_pred_np = k_pred_np[::downsample, ::downsample, ::downsample]
        k_gt_np = k_gt_np[::downsample, ::downsample, ::downsample]
    
    Nx, Ny, Nz = k_pred_np.shape
    
    # Calculate physical coordinates if domain_size is provided
    if domain_size is not None:
        Lx, Ly, Lz = domain_size
        # Convert downsampled indices to physical coordinates
        # Downsampled index i corresponds to original index i * downsample
        # Physical coordinate = (i * downsample + 0.5) * (domain_size / orig_resolution)
        dx = Lx / Nx_orig
        dy = Ly / Ny_orig
        dz = Lz / Nz_orig
    else:
        dx = dy = dz = 1.0
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('3D Diffusivity Field Visualization', fontsize=16)
    
    # Method 1: Volume rendering with voxels (showing defect regions)
    if threshold is None:
        # Use mean - std as threshold to highlight defects (low diffusivity)
        threshold_pred = k_pred_np.mean() - k_pred_np.std()
        threshold_gt = k_gt_np.mean() - k_gt_np.std()
    else:
        threshold_pred = threshold_gt = threshold
    
    # Create voxel masks for defects (low diffusivity regions)
    defect_mask_pred = k_pred_np < threshold_pred
    defect_mask_gt = k_gt_np < threshold_gt
    
    # Plot 1: Ground Truth 3D voxels
    ax1 = fig.add_subplot(231, projection='3d')
    if np.any(defect_mask_gt):
        x_gt, y_gt, z_gt = np.where(defect_mask_gt)
        if domain_size is not None:
            # Convert downsampled indices to physical coordinates
            x_gt_phys = (x_gt * downsample + 0.5) * dx
            y_gt_phys = (y_gt * downsample + 0.5) * dy
            z_gt_phys = (z_gt * downsample + 0.5) * dz
        else:
            x_gt_phys, y_gt_phys, z_gt_phys = x_gt, y_gt, z_gt
        ax1.scatter(x_gt_phys, y_gt_phys, z_gt_phys, c=k_gt_np[defect_mask_gt], 
                   cmap='coolwarm', alpha=0.6, s=20)
    ax1.set_title('Ground Truth Defects (3D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    if domain_size is not None:
        ax1.set_xlim(0, Lx)
        ax1.set_ylim(0, Ly)
        ax1.set_zlim(0, Lz)
    
    # Plot 2: Predicted 3D voxels
    ax2 = fig.add_subplot(232, projection='3d')
    if np.any(defect_mask_pred):
        x_pred, y_pred, z_pred = np.where(defect_mask_pred)
        if domain_size is not None:
            # Convert downsampled indices to physical coordinates
            x_pred_phys = (x_pred * downsample + 0.5) * dx
            y_pred_phys = (y_pred * downsample + 0.5) * dy
            z_pred_phys = (z_pred * downsample + 0.5) * dz
        else:
            x_pred_phys, y_pred_phys, z_pred_phys = x_pred, y_pred, z_pred
        ax2.scatter(x_pred_phys, y_pred_phys, z_pred_phys, c=k_pred_np[defect_mask_pred], 
                   cmap='coolwarm', alpha=0.6, s=20)
    ax2.set_title('Predicted Defects (3D)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    if domain_size is not None:
        ax2.set_xlim(0, Lx)
        ax2.set_ylim(0, Ly)
        ax2.set_zlim(0, Lz)
    
    # Plot 3: Error visualization in 3D
    ax3 = fig.add_subplot(233, projection='3d')
    error_3d = np.abs(k_pred_np - k_gt_np)
    error_threshold = error_3d.mean() + error_3d.std()
    error_mask = error_3d > error_threshold
    if np.any(error_mask):
        x_err, y_err, z_err = np.where(error_mask)
        if domain_size is not None:
            # Convert downsampled indices to physical coordinates
            x_err_phys = (x_err * downsample + 0.5) * dx
            y_err_phys = (y_err * downsample + 0.5) * dy
            z_err_phys = (z_err * downsample + 0.5) * dz
        else:
            x_err_phys, y_err_phys, z_err_phys = x_err, y_err, z_err
        ax3.scatter(x_err_phys, y_err_phys, z_err_phys, c=error_3d[error_mask], 
                   cmap='hot', alpha=0.6, s=20)
    ax3.set_title('Error Regions (3D)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    if domain_size is not None:
        ax3.set_xlim(0, Lx)
        ax3.set_ylim(0, Ly)
        ax3.set_zlim(0, Lz)
    
    # Method 2: Projection views (sum along each axis)
    # Sum along Z axis (top view)
    proj_z_gt = k_gt_np.sum(axis=2)
    proj_z_pred = k_pred_np.sum(axis=2)
    
    # Set extent for physical coordinates
    if domain_size is not None:
        extent_proj = [0, Ly, 0, Lx]  # [left, right, bottom, top] for imshow
    else:
        extent_proj = None
    
    ax4 = fig.add_subplot(234)
    im4 = ax4.imshow(proj_z_gt, cmap='viridis', origin='lower', aspect='auto', extent=extent_proj)
    ax4.set_title('GT: Projection along Z')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('X')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = fig.add_subplot(235)
    im5 = ax5.imshow(proj_z_pred, cmap='viridis', origin='lower', aspect='auto', extent=extent_proj)
    ax5.set_title('Pred: Projection along Z')
    ax5.set_xlabel('Y')
    ax5.set_ylabel('X')
    plt.colorbar(im5, ax=ax5)
    
    # Error projection
    ax6 = fig.add_subplot(236)
    proj_z_error = np.abs(proj_z_pred - proj_z_gt)
    im6 = ax6.imshow(proj_z_error, cmap='hot', origin='lower', aspect='auto', extent=extent_proj)
    ax6.set_title('Error: Projection along Z')
    ax6.set_xlabel('Y')
    ax6.set_ylabel('X')
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D diffusivity visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_3d_diffusivity_isosurface(k_pred, k_gt, save_path=None, isovalue=None, domain_size=None):
    """
    Visualize 3D diffusivity field using isosurfaces.
    Creates separate visualizations for GT and predicted.
    
    If no valid isosurface can be found (data too uniform), skips visualization.
    """
    try:
        from skimage import measure
    except ImportError:
        print("Skipping isosurface visualization: scikit-image not available")
        return
    
    k_pred_np = k_pred.detach().cpu().numpy() if torch.is_tensor(k_pred) else k_pred
    k_gt_np = k_gt.detach().cpu().numpy() if torch.is_tensor(k_gt) else k_gt
    
    Nx, Ny, Nz = k_pred_np.shape
    
    # Calculate spacing for physical coordinates
    if domain_size is not None:
        Lx, Ly, Lz = domain_size
        spacing = (Lx / Nx, Ly / Ny, Lz / Nz)
        xlim = (0, Lx)
        ylim = (0, Ly)
        zlim = (0, Lz)
    else:
        spacing = (1.0, 1.0, 1.0)
        xlim = (0, Nx)
        ylim = (0, Ny)
        zlim = (0, Nz)
    
    def try_marching_cubes(data, user_isovalue, spacing):
        """Try marching cubes with multiple isovalue candidates."""
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        
        # Data too uniform - no isosurface possible
        if data_range < 1e-6:
            return None, None
        
        # Build list of candidate isovalues
        candidates = []
        if user_isovalue is not None:
            candidates.append(user_isovalue)
        
        # Add percentile-based candidates
        for pct in [50, 40, 60, 30, 70, 25, 75]:
            candidates.append(np.percentile(data, pct))
        
        # Try each candidate
        for iso in candidates:
            if data_min < iso < data_max:
                try:
                    result = measure.marching_cubes(data, iso, spacing=spacing)
                    return result, iso  # Success!
                except RuntimeError:
                    continue
        
        return None, None
    
    # Try to find isosurfaces
    result_gt, iso_gt = try_marching_cubes(k_gt_np, isovalue, spacing)
    result_pred, iso_pred = try_marching_cubes(k_pred_np, isovalue, spacing)
    
    # If neither has a valid isosurface, skip
    if result_gt is None and result_pred is None:
        print("Skipping isosurface visualization: no valid isosurface found (data too uniform)")
        return
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('3D Diffusivity Isosurface Visualization', fontsize=16)
    
    # Ground truth isosurface
    ax1 = fig.add_subplot(121, projection='3d')
    if result_gt is not None:
        verts_gt, faces_gt, normals_gt, values_gt = result_gt
        mesh_gt = Poly3DCollection(verts_gt[faces_gt], alpha=0.7, edgecolor='none')
        mesh_gt.set_facecolor([0.7, 0.7, 1.0])
        ax1.add_collection3d(mesh_gt)
        ax1.set_title(f'Ground Truth (iso={iso_gt:.4f})')
    else:
        ax1.set_title('Ground Truth (no surface)')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Predicted isosurface
    ax2 = fig.add_subplot(122, projection='3d')
    if result_pred is not None:
        verts_pred, faces_pred, normals_pred, values_pred = result_pred
        mesh_pred = Poly3DCollection(verts_pred[faces_pred], alpha=0.7, edgecolor='none')
        mesh_pred.set_facecolor([1.0, 0.7, 0.7])
        ax2.add_collection3d(mesh_pred)
        ax2.set_title(f'Predicted (iso={iso_pred:.4f})')
    else:
        ax2.set_title('Predicted (no surface)')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_zlim(zlim)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D isosurface visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_3d_diffusivity_slice(k_gt, k_pred, save_path):
    """Visualize 3D diffusivity with multiple slices."""
    Nx, Ny, Nz = k_gt.shape
    
    fig, axes = plt.subplots(2, Nz, figsize=(3*Nz, 6))
    
    vmin = min(k_gt.min(), k_pred.min())
    vmax = max(k_gt.max(), k_pred.max())
    
    for z in range(Nz):
        # Ground truth
        im1 = axes[0, z].imshow(k_gt[:, :, z].T, origin='lower', cmap='viridis',
                                 vmin=vmin, vmax=vmax)
        axes[0, z].set_title(f'z={z}')
        axes[0, z].set_xticks([])
        axes[0, z].set_yticks([])
        if z == 0:
            axes[0, z].set_ylabel('GT', fontsize=12)
        
        # Predicted
        im2 = axes[1, z].imshow(k_pred[:, :, z].T, origin='lower', cmap='viridis',
                                 vmin=vmin, vmax=vmax)
        axes[1, z].set_xticks([])
        axes[1, z].set_yticks([])
        if z == 0:
            axes[1, z].set_ylabel('Pred', fontsize=12)
    
    plt.suptitle('Diffusivity Field: All Z Slices', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

def visualize_surface_temperature_time(surface_gt, surface_pred, save_path, frame_indices=None):
    """Visualize ground truth vs predicted surface temperature with error."""
    surface_gt_np = surface_gt.detach().cpu().numpy() if torch.is_tensor(surface_gt) else surface_gt
    surface_pred_np = surface_pred.detach().cpu().numpy() if torch.is_tensor(surface_pred) else surface_pred
    
    num_frames = len(surface_gt_np)
    
    if frame_indices is None:
        # Select 5 frames evenly spaced
        frame_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
    
    n_frames = len(frame_indices)
    fig, axes = plt.subplots(3, n_frames, figsize=(4*n_frames, 12))
    
    # Handle case where n_frames=1 (axes would be 1D array)
    if n_frames == 1:
        axes = axes[:, np.newaxis]
    
    vmax = max(surface_gt_np.max(), surface_pred_np.max())
    
    for i, frame_idx in enumerate(frame_indices):
        # Ground truth
        im1 = axes[0, i].imshow(surface_gt_np[frame_idx].T, origin='lower', cmap='hot',
                                 vmin=0, vmax=vmax)
        axes[0, i].set_title(f'GT t={frame_idx}')
        if i == 0:
            axes[0, i].set_ylabel('Ground Truth')
        plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
        
        # Predicted
        im2 = axes[1, i].imshow(surface_pred_np[frame_idx].T, origin='lower', cmap='hot',
                                 vmin=0, vmax=vmax)
        axes[1, i].set_title(f'Pred t={frame_idx}')
        if i == 0:
            axes[1, i].set_ylabel('Predicted')
        plt.colorbar(im2, ax=axes[1, i], shrink=0.8)

        # Error
        error = np.abs(surface_pred_np[frame_idx] - surface_gt_np[frame_idx])
        im3 = axes[2, i].imshow(error.T, origin='lower', cmap='viridis')
        axes[2, i].set_title(f'Abs Error (Max: {error.max():.1e})')
        if i == 0:
            axes[2, i].set_ylabel('Error')
        plt.colorbar(im3, ax=axes[2, i], shrink=0.8)
    
    plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def visualize_surface_temperature(surface_pred, surface_gt, save_path=None):
    """Visualize surface temperature comparison."""
    surface_pred_np = surface_pred.detach().cpu().numpy() if torch.is_tensor(surface_pred) else surface_pred
    surface_gt_np = surface_gt.detach().cpu().numpy() if torch.is_tensor(surface_gt) else surface_gt
    
    # Use consistent colormap range for better comparison
    vmin = min(surface_gt_np.min(), surface_pred_np.min())
    vmax = max(surface_gt_np.max(), surface_pred_np.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Surface Temperature Comparison', fontsize=16)
    
    # Ground truth - using 'plasma' colormap for better contrast
    im1 = axes[0].imshow(surface_gt_np, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    plt.colorbar(im1, ax=axes[0])
    
    # Predicted
    im2 = axes[1].imshow(surface_pred_np, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title('Predicted')
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('X')
    plt.colorbar(im2, ax=axes[1])
    
    # Error - using 'viridis' for error visualization
    error = np.abs(surface_pred_np - surface_gt_np)
    im3 = axes[2].imshow(error, cmap='viridis', origin='lower')
    axes[2].set_title(f'Error (MSE: {np.mean(error**2):.6f})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('X')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved surface temperature visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_time_evolution(surface_seq_pred, surface_seq_gt, save_path=None, num_frames=None, duration=0.1, format='gif', use_final_frame_range=True):
    """
    Visualize time evolution of surface temperature as GIF animations or static images.
    
    Args:
        surface_seq_pred: Predicted surface temperature sequence [time_steps, Nx, Ny]
        surface_seq_gt: Ground truth surface temperature sequence [time_steps, Nx, Ny]
        save_path: Path for saving. Format depends on 'format' parameter:
                   - If format='gif': saves {save_path}_gt.gif and {save_path}_pred.gif
                   - If format='png': saves {save_path}.png (static comparison)
        num_frames: Number of frames to include (None for all frames in GIF, 10 for static)
        duration: Duration of each frame in seconds for GIF (default: 0.1)
        format: Output format - 'gif' for animations, 'png' for static image (default: 'gif')
        use_final_frame_range: If True, use colormap range from final frame only for consistent
                               visualization with surface_temperature.png (default: True)
    """
    surface_seq_pred_np = surface_seq_pred.detach().cpu().numpy() if torch.is_tensor(surface_seq_pred) else surface_seq_pred
    surface_seq_gt_np = surface_seq_gt.detach().cpu().numpy() if torch.is_tensor(surface_seq_gt) else surface_seq_gt
    
    if format == 'png':
        # Static visualization
        if num_frames is None:
            num_frames = 10
        
        num_steps = min(len(surface_seq_pred_np), len(surface_seq_gt_np), num_frames)
        step_indices = np.linspace(0, min(len(surface_seq_pred_np), len(surface_seq_gt_np)) - 1, 
                                   num_steps, dtype=int)
        
        fig, axes = plt.subplots(2, num_steps, figsize=(3*num_steps, 6))
        if num_steps == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Time Evolution of Surface Temperature', fontsize=16)
        
        # Determine colormap range
        if use_final_frame_range:
            # Use only final frame's range for consistency with surface_temperature.png
            vmin = min(surface_seq_gt_np[-1].min(), surface_seq_pred_np[-1].min())
            vmax = max(surface_seq_gt_np[-1].max(), surface_seq_pred_np[-1].max())
        else:
            # Use entire sequence's range
            vmin = min(surface_seq_gt_np.min(), surface_seq_pred_np.min())
            vmax = max(surface_seq_gt_np.max(), surface_seq_pred_np.max())
        
        for idx, step in enumerate(step_indices):
            # Ground truth - using 'plasma' colormap for better contrast
            im1 = axes[0, idx].imshow(surface_seq_gt_np[step], cmap='plasma', 
                                      origin='lower', vmin=vmin, vmax=vmax)
            axes[0, idx].set_title(f'GT t={step}')
            axes[0, idx].axis('off')
            if idx == 0:
                plt.colorbar(im1, ax=axes[0, idx])
            
            # Predicted
            im2 = axes[1, idx].imshow(surface_seq_pred_np[step], cmap='plasma', 
                                      origin='lower', vmin=vmin, vmax=vmax)
            axes[1, idx].set_title(f'Pred t={step}')
            axes[1, idx].axis('off')
            if idx == 0:
                plt.colorbar(im2, ax=axes[1, idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved time evolution visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return
    
    # GIF format (default)
    
    num_steps = len(surface_seq_pred_np)
    if num_frames is not None:
        num_steps = min(num_steps, num_frames)
    
    # Use all frames or sample evenly
    if num_frames is not None and num_frames < len(surface_seq_pred_np):
        step_indices = np.linspace(0, len(surface_seq_pred_np) - 1, num_frames, dtype=int)
    else:
        step_indices = np.arange(len(surface_seq_pred_np))
    
    # Determine colormap range
    if use_final_frame_range:
        # Use only final frame's range for consistency with surface_temperature.png
        vmin = min(surface_seq_gt_np[-1].min(), surface_seq_pred_np[-1].min())
        vmax = max(surface_seq_gt_np[-1].max(), surface_seq_pred_np[-1].max())
    else:
        # Use entire sequence's range
        vmin = min(surface_seq_gt_np.min(), surface_seq_pred_np.min())
        vmax = max(surface_seq_gt_np.max(), surface_seq_pred_np.max())
    
    print(f"Generating GIF with {len(step_indices)} frames...")

    # Helper function to generate frames efficiently by reusing the figure
    def generate_frames(data_seq, title_prefix):
        frames = []
        fig, ax = plt.subplots(figsize=(6, 6)) # Smaller figure for speed
        plt.tight_layout()
        
        # Initial plot
        im = ax.imshow(data_seq[step_indices[0]], cmap='plasma', origin='lower', 
                      vmin=vmin, vmax=vmax)
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        plt.colorbar(im, ax=ax)
        
        for step in step_indices:
            # print(f"Processing frame {step}/{len(data_seq)-1}", end='\r')
            # Update data instead of clearing and replotting
            im.set_data(data_seq[step])
            ax.set_title(f'{title_prefix} - Time Step {step}', fontsize=12)
            
            # Efficiently render the canvas
            fig.canvas.draw()
            
            # Extract image from buffer (much faster than savefig)
            try:
                # Modern Matplotlib (returns memoryview of RGBA)
                buf = fig.canvas.buffer_rgba()
                arr = np.frombuffer(buf, dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                image = arr.reshape((h, w, 4))[:, :, :3] # Drop alpha
            except AttributeError:
                # Older Matplotlib fallback
                try:
                    s = fig.canvas.tostring_rgb()
                    w, h = fig.canvas.get_width_height()
                    image = np.frombuffer(s, dtype=np.uint8).reshape((h, w, 3))
                except:
                     # Ultimate fallback
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    image = np.array(Image.open(buf))
            
            frames.append(image.copy())
            
        plt.close(fig)
        return frames

    # Create frames for ground truth GIF
    frames_gt = generate_frames(surface_seq_gt_np, "Ground Truth")
    
    # Create frames for predicted GIF
    frames_pred = generate_frames(surface_seq_pred_np, "Predicted")
    
    # Save GIFs
    if save_path:
        # Remove extension if present (check for common image extensions)
        base_path = save_path
        for ext in ['.gif', '.png', '.jpg', '.jpeg']:
            if save_path.lower().endswith(ext):
                base_path = save_path[:-len(ext)]
                break
        
        gif_path_gt = f"{base_path}_gt.gif"
        gif_path_pred = f"{base_path}_pred.gif"
        
        # Ensure directory exists
        dirname = os.path.dirname(gif_path_gt)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        imageio.mimsave(gif_path_gt, frames_gt, duration=duration, loop=0)
        imageio.mimsave(gif_path_pred, frames_pred, duration=duration, loop=0)
        
        print(f"Saved GIFs to:\n  {gif_path_gt}\n  {gif_path_pred}")
    else:
        # Display GIFs (requires additional setup, just show first frame)
        print("GIFs created but not saved. Provide save_path to save GIF files.")


def compute_metrics(k_pred, k_gt, surface_pred, surface_gt):
    """Compute evaluation metrics."""
    k_pred_np = k_pred.detach().cpu().numpy() if torch.is_tensor(k_pred) else k_pred
    k_gt_np = k_gt.detach().cpu().numpy() if torch.is_tensor(k_gt) else k_gt
    surface_pred_np = surface_pred.detach().cpu().numpy() if torch.is_tensor(surface_pred) else surface_pred
    surface_gt_np = surface_gt.detach().cpu().numpy() if torch.is_tensor(surface_gt) else surface_gt
    
    # Diffusivity metrics
    k_mse = np.mean((k_pred_np - k_gt_np)**2)
    k_mae = np.mean(np.abs(k_pred_np - k_gt_np))
    k_psnr = -10.0 * np.log10(k_mse + 1e-10)
    
    # Surface metrics
    surface_mse = np.mean((surface_pred_np - surface_gt_np)**2)
    surface_mae = np.mean(np.abs(surface_pred_np - surface_gt_np))
    surface_psnr = -10.0 * np.log10(surface_mse + 1e-10)
    
    return {
        'k_mse': k_mse,
        'k_mae': k_mae,
        'k_psnr': k_psnr,
        'surface_mse': surface_mse,
        'surface_mae': surface_mae,
        'surface_psnr': surface_psnr,
    }


def visualize_error_distribution(k_gt, k_pred, save_path=None):
    """Visualize error distribution."""
    k_pred_np = k_pred.detach().cpu().numpy() if torch.is_tensor(k_pred) else k_pred
    k_gt_np = k_gt.detach().cpu().numpy() if torch.is_tensor(k_gt) else k_gt
    
    diff = k_pred_np - k_gt_np
    abs_diff = np.abs(diff)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram of errors
    axes[0].hist(diff.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Error (Pred - GT)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Error Distribution\nMean={diff.mean():.4f}, Std={diff.std():.4f}')
    
    # Scatter plot
    axes[1].scatter(k_gt_np.flatten(), k_pred_np.flatten(), alpha=0.3, s=5)
    min_val = min(k_gt_np.min(), k_pred_np.min())
    max_val = max(k_gt_np.max(), k_pred_np.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    axes[1].set_xlabel('Ground Truth k')
    axes[1].set_ylabel('Predicted k')
    axes[1].set_title('Prediction vs Ground Truth')
    axes[1].legend()
    axes[1].set_aspect('equal')
    
    # Cumulative error
    sorted_errors = np.sort(abs_diff.flatten())
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[2].plot(sorted_errors, cumulative, linewidth=2)
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('Cumulative Fraction')
    axes[2].set_title('Cumulative Error Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved error distribution visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_diffusivity_combined(k_pred, k_gt, save_path=None, domain_size=None, threshold=None):
    """
    Visualize diffusivity field combining 3D volume rendering and all Z-slices.
    Rows: Prediction, Ground Truth, Error.
    Cols: 3D View, Slice 0, Slice 1, ..., Slice Nz-1.
    """
    k_pred_np = k_pred.detach().cpu().numpy() if torch.is_tensor(k_pred) else k_pred
    k_gt_np = k_gt.detach().cpu().numpy() if torch.is_tensor(k_gt) else k_gt
    
    Nx, Ny, Nz = k_pred_np.shape
    cols = Nz + 1
    rows = 3
    
    # Figure size: make it wide enough for all slices
    fig = plt.figure(figsize=(4 * cols, 12)) 
    
    vmin = min(k_gt_np.min(), k_pred_np.min())
    vmax = max(k_gt_np.max(), k_pred_np.max())
    
    # 3D Plotting Helper
    def plot_3d(ax, data, title, is_error=False):
        # Calculate physical coordinates if domain_size is provided
        if domain_size is not None:
            Lx, Ly, Lz = domain_size
            dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
            # Set aspect ratio - Z usually shorter
            ax.set_box_aspect((Lx, Ly, Lz))
            
            # Use physical limits
            ax.set_xlim(0, Lx)
            ax.set_ylim(0, Ly)
            ax.set_zlim(0, Lz)
            ax.set_zticks([0, Lz])
        else:
            dx, dy, dz = 1.0, 1.0, 1.0
            ax.set_box_aspect((Nx, Ny, Nz))
            ax.set_zticks([0, Nz])

        if is_error:
            # Error visualization
            data_mean = data.mean()
            data_std = data.std()
            thresh = data_mean + data_std
            mask = data > thresh
            cmap = 'hot'
        else:
            # Diffusivity visualization (defects)
            if threshold is None:
                thresh = data.mean() - data.std()
            else:
                thresh = threshold
            mask = data < thresh
            cmap = 'coolwarm'
            
        if np.any(mask):
            x, y, z = np.where(mask)
            if domain_size is not None:
                # Center of voxel
                x_p = (x + 0.5) * dx
                y_p = (y + 0.5) * dy
                z_p = (z + 0.5) * dz
                ax.scatter(x_p, y_p, z_p, c=data[mask], cmap=cmap, alpha=0.6, s=20)
            else:
                ax.scatter(x, y, z, c=data[mask], cmap=cmap, alpha=0.6, s=20)
                
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Row 1: Prediction
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    plot_3d(ax, k_pred_np, "Predicted 3D")
    
    for z in range(Nz):
        ax = fig.add_subplot(rows, cols, 2 + z)
        im = ax.imshow(k_pred_np[:, :, z].T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'z={z}')
        ax.axis('off')

    # Row 2: Ground Truth
    ax = fig.add_subplot(rows, cols, cols + 1, projection='3d')
    plot_3d(ax, k_gt_np, "GT 3D")
    
    for z in range(Nz):
        ax = fig.add_subplot(rows, cols, cols + 2 + z)
        im = ax.imshow(k_gt_np[:, :, z].T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')

    # Row 3: Error
    error_np = np.abs(k_pred_np - k_gt_np)
    ax = fig.add_subplot(rows, cols, 2*cols + 1, projection='3d')
    plot_3d(ax, error_np, "Error 3D", is_error=True)
    
    for z in range(Nz):
        ax = fig.add_subplot(rows, cols, 2*cols + 2 + z)
        im = ax.imshow(error_np[:, :, z].T, origin='lower', cmap='hot')
        ax.axis('off')

    plt.tight_layout()#pad=0.75, w_pad=0.1, h_pad=0.1)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined diffusivity visualization to {save_path}")
    else:
        plt.show()
    plt.close()
