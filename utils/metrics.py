import numpy as np
from skimage.metrics import structural_similarity

def compute_psnr(pred, gt, data_range=None, eps=1e-10):
    """Compute PSNR for numpy arrays."""
    mse = np.mean((pred - gt) ** 2)
    if data_range is None:
        data_range = float(gt.max() - gt.min())
    return -10.0 * np.log10(mse / (data_range ** 2 + eps) + eps)

def compute_ssim_windowed_2d(pred, gt, data_range=None):
    """Compute windowed SSIM for 2D arrays using scikit-image."""
    if data_range is None:
        data_range = float(gt.max() - gt.min())
    return float(structural_similarity(gt, pred, data_range=data_range))

def compute_ssim_windowed_3d(pred, gt, data_range=None):
    """
    Compute windowed SSIM for 3D volumes by averaging 2D SSIM over z-slices.
    Uses scikit-image structural_similarity (windowed by default).
    """
    if data_range is None:
        data_range = float(gt.max() - gt.min())
    Nz = gt.shape[2]
    ssim_vals = []
    for z in range(Nz):
        ssim = structural_similarity(
            gt[:, :, z],
            pred[:, :, z],
            data_range=data_range
        )
        ssim_vals.append(ssim)
    return float(np.mean(ssim_vals))


def compute_k_iou(k_pred, k_gt, threshold=0.03):
    """
    Compute IOU for defective regions in k based on a threshold.
    Defective = k < threshold.
    """
    pred_defect = k_pred < threshold
    gt_defect = k_gt < threshold

    intersection = np.logical_and(pred_defect, gt_defect).sum()
    union = np.logical_or(pred_defect, gt_defect).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)
