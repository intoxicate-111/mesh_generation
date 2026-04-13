import torch
import torch.nn.functional as F


def _boundary_map(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError("mask must have shape [B, H, W]")

    x = mask.unsqueeze(1)
    max_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    boundary = (max_pool - min_pool).clamp(0.0, 1.0)
    return boundary.squeeze(1)


def boundary_aware_silhouette_loss(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    boundary_weight: float = 4.0,
) -> torch.Tensor:
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("pred_mask and gt_mask must have same shape")

    base = (pred_mask - gt_mask).abs()
    boundary = _boundary_map(gt_mask)
    weights = 1.0 + boundary_weight * boundary
    return (base * weights).mean()