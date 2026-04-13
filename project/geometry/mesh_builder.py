import torch
from typing import List, Tuple


def build_fixed_topology_mesh(num_verts: int, device: str = "cpu"):
    if num_verts < 3:
        raise ValueError("num_verts must be >= 3")

    verts = torch.randn(num_verts, 3, device=device) * 0.1
    faces = []
    for i in range(1, num_verts - 1):
        faces.append([0, i, i + 1])
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    return verts, faces


def build_edges_from_candidate_graph(
    candidate_idx: List[List[int]],
    edge_weights: List[torch.Tensor],
    min_weight: float = 0.2,
) -> List[Tuple[int, int, float]]:
    edges: List[Tuple[int, int, float]] = []
    for i, neigh in enumerate(candidate_idx):
        if not neigh or i >= len(edge_weights):
            continue

        local_w = edge_weights[i]
        for k, j in enumerate(neigh):
            if j <= i or k >= local_w.numel():
                continue
            w = float(local_w[k].item())
            if w >= min_weight:
                edges.append((i, j, w))
    return edges


def triangle_scores_from_edge_matrix(edge_score_matrix: torch.Tensor) -> torch.Tensor:
    if edge_score_matrix.ndim != 2 or edge_score_matrix.size(0) != edge_score_matrix.size(1):
        raise ValueError("edge_score_matrix must be square with shape [N, N]")

    n = edge_score_matrix.size(0)
    tri_scores = torch.zeros((n, n, n), device=edge_score_matrix.device)

    for i in range(n):
        for j in range(i + 1, n):
            sij = edge_score_matrix[i, j]
            if sij <= 0:
                continue
            for k in range(j + 1, n):
                sjk = edge_score_matrix[j, k]
                ski = edge_score_matrix[k, i]
                if sjk > 0 and ski > 0:
                    score = sij * sjk * ski
                    tri_scores[i, j, k] = score
                    tri_scores[i, k, j] = score
                    tri_scores[j, i, k] = score
                    tri_scores[j, k, i] = score
                    tri_scores[k, i, j] = score
                    tri_scores[k, j, i] = score

    return tri_scores
