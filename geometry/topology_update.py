from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch

from geometry.mesh_builder import triangle_scores_from_edge_matrix


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


def propose_faces_from_triangle_scores(
    tri_scores: torch.Tensor,
    score_threshold: float,
    max_faces: int,
) -> List[Tuple[int, int, int]]:
    n = tri_scores.shape[0]
    candidates: List[Tuple[float, Tuple[int, int, int]]] = []

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                score = float(tri_scores[i, j, k].item())
                if score >= score_threshold:
                    candidates.append((score, (i, j, k)))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [tri for _, tri in candidates[:max_faces]]


def _face_normal(verts: torch.Tensor, tri: Sequence[int]) -> torch.Tensor:
    i, j, k = tri
    n = torch.cross(verts[j] - verts[i], verts[k] - verts[i], dim=0)
    return n / n.norm().clamp_min(1e-8)


def _orient_face_outward(verts: torch.Tensor, tri: Tuple[int, int, int]) -> Tuple[int, int, int]:
    center = verts.mean(dim=0)
    i, j, k = tri
    normal = _face_normal(verts, tri)
    face_center = (verts[i] + verts[j] + verts[k]) / 3.0
    if torch.dot(normal, face_center - center) < 0:
        return (i, k, j)
    return tri


def filter_valid_faces(
    verts: torch.Tensor,
    faces: List[Tuple[int, int, int]],
    edge_mat: torch.Tensor,
    min_area: float,
    max_aspect_ratio: float,
    min_normal_dot: float,
) -> torch.Tensor:
    unique = set()
    valid: List[Tuple[int, int, int]] = []
    edge_normals: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)

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

        oriented = _orient_face_outward(verts, tri_sorted)
        normal = _face_normal(verts, oriented)
        reject = False
        for a, b in ((oriented[0], oriented[1]), (oriented[1], oriented[2]), (oriented[2], oriented[0])):
            e = (min(a, b), max(a, b))
            if e in edge_normals and edge_normals[e]:
                ref_n = torch.stack(edge_normals[e], dim=0).mean(dim=0)
                ref_n = ref_n / ref_n.norm().clamp_min(1e-8)
                if torch.dot(normal, ref_n) < min_normal_dot:
                    reject = True
                    break

        if reject:
            continue

        unique.add(tri_sorted)
        valid.append(oriented)
        for a, b in ((oriented[0], oriented[1]), (oriented[1], oriented[2]), (oriented[2], oriented[0])):
            e = (min(a, b), max(a, b))
            edge_normals[e].append(normal)

    if not valid:
        return torch.empty((0, 3), dtype=torch.long, device=verts.device)
    return torch.tensor(valid, dtype=torch.long, device=verts.device)


def build_faces_from_edge_graph(
    verts: torch.Tensor,
    edge_mat: torch.Tensor,
    triangle_score_threshold: float = 1e-4,
    max_faces: int = 1024,
    min_area: float = 1e-6,
    max_aspect_ratio: float = 15.0,
    min_normal_dot: float = -0.2,
) -> torch.Tensor:
    tri_scores = triangle_scores_from_edge_matrix(edge_mat)
    raw_faces = propose_faces_from_triangle_scores(
        tri_scores=tri_scores,
        score_threshold=triangle_score_threshold,
        max_faces=max_faces,
    )
    return filter_valid_faces(
        verts=verts,
        faces=raw_faces,
        edge_mat=edge_mat,
        min_area=min_area,
        max_aspect_ratio=max_aspect_ratio,
        min_normal_dot=min_normal_dot,
    )