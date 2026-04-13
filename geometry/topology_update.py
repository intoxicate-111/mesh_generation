from itertools import combinations
from typing import List, Sequence, Tuple

import torch


def build_sparse_edge_matrix(
    candidate_idx: List[List[int]],
    edge_weights: List[torch.Tensor],
    num_points: int,
    min_weight: float,
    topk_per_point: int,
) -> torch.Tensor:
    device = edge_weights[0].device if edge_weights else "cpu"
    edge_mat = torch.zeros((num_points, num_points), dtype=torch.float32, device=device)
    for i, neigh in enumerate(candidate_idx):
        if i >= len(edge_weights) or not neigh:
            continue
        local_w = edge_weights[i]
        if local_w.numel() == 0:
            continue

        score_pairs: List[Tuple[float, int]] = []
        for k, j in enumerate(neigh):
            if j < 0 or j >= num_points or k >= local_w.numel() or j == i:
                continue
            w = float(local_w[k].item())
            if w >= min_weight:
                score_pairs.append((w, int(j)))

        if not score_pairs:
            continue

        score_pairs.sort(key=lambda t: t[0], reverse=True)
        for w, j in score_pairs[:topk_per_point]:
            edge_mat[i, j] = max(edge_mat[i, j], w)
            edge_mat[j, i] = max(edge_mat[j, i], w)

    return edge_mat


def _triangle_aspect_ratio(verts: torch.Tensor, tri: Sequence[int]) -> float:
    i, j, k = tri
    a = (verts[i] - verts[j]).norm().item()
    b = (verts[j] - verts[k]).norm().item()
    c = (verts[k] - verts[i]).norm().item()
    longest = max(a, b, c)
    shortest = max(min(a, b, c), 1e-6)
    return longest / shortest


def _triangle_area(verts: torch.Tensor, tri: Sequence[int]) -> float:
    i, j, k = tri
    return 0.5 * torch.cross(verts[j] - verts[i], verts[k] - verts[i], dim=0).norm().item()


def propose_faces_from_edges(
    edge_mat: torch.Tensor,
    max_faces_per_anchor: int,
) -> List[Tuple[int, int, int]]:
    n = edge_mat.shape[0]
    faces: List[Tuple[int, int, int]] = []

    for i in range(n):
        neigh = torch.nonzero(edge_mat[i] > 0, as_tuple=False).flatten().tolist()
        if len(neigh) < 2:
            continue

        local_candidates: List[Tuple[float, Tuple[int, int, int]]] = []
        for j, k in combinations(neigh, 2):
            if edge_mat[j, k] <= 0:
                continue
            score = float(edge_mat[i, j].item() * edge_mat[j, k].item() * edge_mat[k, i].item())
            tri = tuple(sorted((i, j, k)))
            local_candidates.append((score, tri))

        if not local_candidates:
            continue

        local_candidates.sort(key=lambda t: t[0], reverse=True)
        for _, tri in local_candidates[:max_faces_per_anchor]:
            faces.append(tri)

    return faces


def filter_valid_faces(
    verts: torch.Tensor,
    faces: List[Tuple[int, int, int]],
    edge_mat: torch.Tensor,
    min_area: float,
    max_aspect_ratio: float,
) -> torch.Tensor:
    unique = set()
    valid: List[Tuple[int, int, int]] = []

    for tri in faces:
        tri_sorted = tuple(sorted(tri))
        if tri_sorted in unique:
            continue
        i, j, k = tri_sorted

        if i == j or j == k or i == k:
            continue
        if edge_mat[i, j] <= 0 or edge_mat[j, k] <= 0 or edge_mat[k, i] <= 0:
            continue

        area = _triangle_area(verts, tri_sorted)
        if area < min_area:
            continue

        aspect = _triangle_aspect_ratio(verts, tri_sorted)
        if aspect > max_aspect_ratio:
            continue

        unique.add(tri_sorted)
        valid.append(tri_sorted)

    if not valid:
        return torch.empty((0, 3), dtype=torch.long, device=verts.device)
    return torch.tensor(valid, dtype=torch.long, device=verts.device)


def build_faces_from_edge_graph(
    verts: torch.Tensor,
    edge_mat: torch.Tensor,
    max_faces_per_anchor: int = 6,
    min_area: float = 1e-6,
    max_aspect_ratio: float = 15.0,
) -> torch.Tensor:
    raw_faces = propose_faces_from_edges(edge_mat=edge_mat, max_faces_per_anchor=max_faces_per_anchor)
    return filter_valid_faces(
        verts=verts,
        faces=raw_faces,
        edge_mat=edge_mat,
        min_area=min_area,
        max_aspect_ratio=max_aspect_ratio,
    )