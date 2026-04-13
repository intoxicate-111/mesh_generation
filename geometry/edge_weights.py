from typing import List

import torch


def _anisotropic_distance(x_i: torch.Tensor, x_j: torch.Tensor, sigma_i_inv: torch.Tensor) -> torch.Tensor:
    delta = x_j - x_i
    return delta @ sigma_i_inv @ delta


def compute_edge_weights(
    points: torch.Tensor,
    sigma: torch.Tensor,
    candidate_idx: List[List[int]],
    alpha: float,
) -> List[torch.Tensor]:
    sigma_inv = torch.linalg.inv(sigma + 1e-6 * torch.eye(3, device=sigma.device).unsqueeze(0))

    weights: List[torch.Tensor] = []
    for i, neighbors in enumerate(candidate_idx):
        if not neighbors:
            weights.append(torch.empty(0, device=points.device))
            continue

        local_weights = []
        for j in neighbors:
            d_ij = _anisotropic_distance(points[i], points[j], sigma_inv[i])
            d_ji = _anisotropic_distance(points[j], points[i], sigma_inv[j])
            s_ij = torch.exp(-alpha * d_ij) * torch.exp(-alpha * d_ji)
            local_weights.append(s_ij)

        weights.append(torch.stack(local_weights))

    return weights
