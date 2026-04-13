from dataclasses import dataclass

import torch


@dataclass
class AdaptiveUpdateResult:
    points: torch.Tensor
    quat: torch.Tensor
    log_scale: torch.Tensor
    added: int
    pruned: int
    split: int
    merged: int


def _safe_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    if scores.numel() == 0 or k <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    k = min(k, scores.numel())
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.topk(scores, k=k, largest=True).indices


def densify_points(
    points: torch.Tensor,
    quat: torch.Tensor,
    log_scale: torch.Tensor,
    densify_scores: torch.Tensor,
    score_threshold: float,
    max_new_points: int,
    noise_scale: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if points.numel() == 0 or max_new_points <= 0:
        return points, quat, log_scale, 0

    candidate_mask = densify_scores > score_threshold
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return points, quat, log_scale, 0

    top_scores = densify_scores[candidate_indices]
    pick = _safe_topk_indices(top_scores, max_new_points)
    parent_indices = candidate_indices[pick]

    parent_points = points[parent_indices]
    parent_quat = quat[parent_indices]
    parent_scales = log_scale[parent_indices]

    scale = torch.exp(parent_scales).mean(dim=-1, keepdim=True)
    noise = torch.randn_like(parent_points)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    children_points = parent_points + noise * scale * noise_scale

    new_points = torch.cat([points, children_points], dim=0)
    new_quat = torch.cat([quat, parent_quat], dim=0)
    new_log_scale = torch.cat([log_scale, parent_scales], dim=0)
    return new_points, new_quat, new_log_scale, int(parent_indices.numel())


def prune_points(
    points: torch.Tensor,
    quat: torch.Tensor,
    log_scale: torch.Tensor,
    prune_scores: torch.Tensor,
    prune_threshold: float,
    min_points: int,
    max_prune_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    n = points.shape[0]
    if n <= min_points or max_prune_points <= 0:
        return points, quat, log_scale, 0

    candidate_mask = prune_scores > prune_threshold
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return points, quat, log_scale, 0

    max_allowed = max(0, n - min_points)
    prune_k = min(max_prune_points, max_allowed, int(candidate_indices.numel()))
    if prune_k <= 0:
        return points, quat, log_scale, 0

    candidate_scores = prune_scores[candidate_indices]
    pick = _safe_topk_indices(candidate_scores, prune_k)
    drop_indices = candidate_indices[pick]

    keep_mask = torch.ones(n, dtype=torch.bool, device=points.device)
    keep_mask[drop_indices] = False

    return points[keep_mask], quat[keep_mask], log_scale[keep_mask], int(drop_indices.numel())


def adaptive_point_update(
    points: torch.Tensor,
    quat: torch.Tensor,
    log_scale: torch.Tensor,
    densify_scores: torch.Tensor,
    prune_scores: torch.Tensor,
    split_scores: torch.Tensor,
    merge_edge_matrix: torch.Tensor,
    densify_threshold: float,
    prune_threshold: float,
    split_threshold: float,
    merge_distance_threshold: float,
    merge_scale_rel_threshold: float,
    merge_neighbor_jaccard_threshold: float,
    max_new_points: int,
    max_prune_points: int,
    max_split_points: int,
    max_merge_pairs: int,
    min_points: int,
) -> AdaptiveUpdateResult:
    split_points_out, split_quat_out, split_scale_out, split_count = split_points(
        points,
        quat,
        log_scale,
        split_scores=split_scores,
        score_threshold=split_threshold,
        max_split_points=max_split_points,
    )

    if split_count > 0:
        split_appended = split_points_out.shape[0] - points.shape[0]
        densify_scores = torch.cat(
            [
                densify_scores,
                torch.zeros(split_appended, device=densify_scores.device, dtype=densify_scores.dtype),
            ],
            dim=0,
        )
        prune_scores = torch.cat(
            [
                prune_scores,
                torch.zeros(split_appended, device=prune_scores.device, dtype=prune_scores.dtype),
            ],
            dim=0,
        )

    new_points, new_quat, new_log_scale, added = densify_points(
        split_points_out,
        split_quat_out,
        split_scale_out,
        densify_scores,
        score_threshold=densify_threshold,
        max_new_points=max_new_points,
    )

    if added > 0:
        appended_count = new_points.shape[0] - split_points_out.shape[0]
        pad = torch.zeros(appended_count, device=prune_scores.device, dtype=prune_scores.dtype)
        prune_scores = torch.cat([prune_scores, pad], dim=0)

    if merge_edge_matrix.numel() > 0:
        old_n = merge_edge_matrix.shape[0]
        new_n = new_points.shape[0]
        if new_n > old_n:
            padded = torch.zeros((new_n, new_n), device=merge_edge_matrix.device, dtype=merge_edge_matrix.dtype)
            padded[:old_n, :old_n] = merge_edge_matrix
            merge_edge_matrix = padded
        elif new_n < old_n:
            merge_edge_matrix = merge_edge_matrix[:new_n, :new_n]

    merged_points, merged_quat, merged_log_scale, merged, keep_indices = merge_points(
        new_points,
        new_quat,
        new_log_scale,
        edge_mat=merge_edge_matrix,
        merge_distance_threshold=merge_distance_threshold,
        merge_scale_rel_threshold=merge_scale_rel_threshold,
        merge_neighbor_jaccard_threshold=merge_neighbor_jaccard_threshold,
        max_merge_pairs=max_merge_pairs,
        min_points=min_points,
    )

    if merged > 0:
        prune_scores = prune_scores[keep_indices]

    final_points, final_quat, final_log_scale, pruned = prune_points(
        merged_points,
        merged_quat,
        merged_log_scale,
        prune_scores,
        prune_threshold=prune_threshold,
        min_points=min_points,
        max_prune_points=max_prune_points,
    )

    return AdaptiveUpdateResult(
        points=final_points,
        quat=final_quat,
        log_scale=final_log_scale,
        added=added,
        pruned=pruned,
        split=split_count,
        merged=merged,
    )


def split_points(
    points: torch.Tensor,
    quat: torch.Tensor,
    log_scale: torch.Tensor,
    split_scores: torch.Tensor,
    score_threshold: float,
    max_split_points: int,
    axis_offset_scale: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    n = points.shape[0]
    if n == 0 or max_split_points <= 0:
        return points, quat, log_scale, 0

    candidate_mask = split_scores > score_threshold
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return points, quat, log_scale, 0

    picked = _safe_topk_indices(split_scores[candidate_indices], max_split_points)
    split_indices = candidate_indices[picked]
    if split_indices.numel() == 0:
        return points, quat, log_scale, 0

    updated_points = points.clone()
    updated_quat = quat.clone()
    updated_log_scale = log_scale.clone()

    children_points = []
    children_quat = []
    children_log_scale = []

    for idx in split_indices.tolist():
        scales = torch.exp(updated_log_scale[idx])
        principal_axis = int(torch.argmax(scales).item())
        offset = torch.zeros(3, device=points.device, dtype=points.dtype)
        offset[principal_axis] = scales[principal_axis] * axis_offset_scale

        updated_points[idx] = updated_points[idx] + offset

        child_point = points[idx] - offset
        child_quat = updated_quat[idx]
        child_scale = updated_log_scale[idx].clone()
        child_scale[principal_axis] = child_scale[principal_axis] - 0.35
        updated_log_scale[idx, principal_axis] = updated_log_scale[idx, principal_axis] - 0.35

        children_points.append(child_point)
        children_quat.append(child_quat)
        children_log_scale.append(child_scale)

    if not children_points:
        return points, quat, log_scale, 0

    out_points = torch.cat([updated_points, torch.stack(children_points, dim=0)], dim=0)
    out_quat = torch.cat([updated_quat, torch.stack(children_quat, dim=0)], dim=0)
    out_log_scale = torch.cat([updated_log_scale, torch.stack(children_log_scale, dim=0)], dim=0)
    return out_points, out_quat, out_log_scale, int(len(children_points))


def merge_points(
    points: torch.Tensor,
    quat: torch.Tensor,
    log_scale: torch.Tensor,
    edge_mat: torch.Tensor,
    merge_distance_threshold: float,
    merge_scale_rel_threshold: float,
    merge_neighbor_jaccard_threshold: float,
    max_merge_pairs: int,
    min_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    n = points.shape[0]
    if n <= min_points or max_merge_pairs <= 0:
        return points, quat, log_scale, 0, torch.arange(n, device=points.device)

    if edge_mat.ndim != 2 or edge_mat.shape[0] != edge_mat.shape[1] or edge_mat.shape[0] != n:
        edge_mat = torch.zeros((n, n), device=points.device, dtype=points.dtype)

    dmat = torch.cdist(points, points)
    avg_scale = torch.exp(log_scale).mean(dim=1)
    neighbor_mask = edge_mat > 0

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(dmat[i, j].item())
            if dist > merge_distance_threshold:
                continue

            scale_gap = abs(float(avg_scale[i].item() - avg_scale[j].item()))
            rel_gap = scale_gap / max(float(avg_scale[i].item()), float(avg_scale[j].item()), 1e-6)
            if rel_gap > merge_scale_rel_threshold:
                continue

            neigh_i = neighbor_mask[i]
            neigh_j = neighbor_mask[j]
            inter = torch.logical_and(neigh_i, neigh_j).sum().item()
            union = torch.logical_or(neigh_i, neigh_j).sum().item()
            jaccard = float(inter / max(union, 1))
            if jaccard < merge_neighbor_jaccard_threshold:
                continue

            score = (merge_distance_threshold - dist) + jaccard
            candidates.append((score, i, j))

    if not candidates:
        return points, quat, log_scale, 0, torch.arange(n, device=points.device)

    candidates.sort(key=lambda x: x[0], reverse=True)
    used = torch.zeros(n, dtype=torch.bool, device=points.device)
    keep_mask = torch.ones(n, dtype=torch.bool, device=points.device)
    merged_count = 0

    for _, i, j in candidates:
        if merged_count >= max_merge_pairs:
            break
        if used[i] or used[j]:
            continue
        if int(keep_mask.sum().item()) - 1 < min_points:
            break

        merged_pos = 0.5 * (points[i] + points[j])
        merged_quat = quat[i] + quat[j]
        merged_scale = torch.log(0.5 * (torch.exp(log_scale[i]) + torch.exp(log_scale[j])).clamp_min(1e-8))

        points[i] = merged_pos
        quat[i] = merged_quat
        log_scale[i] = merged_scale

        used[i] = True
        used[j] = True
        keep_mask[j] = False
        merged_count += 1

    if merged_count == 0:
        return points, quat, log_scale, 0, torch.arange(n, device=points.device)

    keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    return points[keep_mask], quat[keep_mask], log_scale[keep_mask], int(merged_count), keep_indices