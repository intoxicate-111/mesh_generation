from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch


def _face_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.numel() == 0:
        return torch.empty((0, 3), device=verts.device, dtype=verts.dtype)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=1)
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _edge_face_map(faces: torch.Tensor) -> Dict[Tuple[int, int], List[int]]:
    mapping: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, tri in enumerate(faces.tolist()):
        i, j, k = tri
        for a, b in ((i, j), (j, k), (k, i)):
            e = (min(a, b), max(a, b))
            mapping[e].append(idx)
    return mapping


def _unique_edges(faces: torch.Tensor) -> torch.Tensor:
    if faces.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=faces.device)
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)
    edges = torch.sort(edges, dim=1).values
    edges = torch.unique(edges, dim=0)
    return edges


def _triangle_aspect_ratios(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.numel() == 0:
        return torch.empty(0, device=verts.device, dtype=verts.dtype)
    a = (verts[faces[:, 0]] - verts[faces[:, 1]]).norm(dim=1)
    b = (verts[faces[:, 1]] - verts[faces[:, 2]]).norm(dim=1)
    c = (verts[faces[:, 2]] - verts[faces[:, 0]]).norm(dim=1)
    stacked = torch.stack([a, b, c], dim=1)
    return stacked.max(dim=1).values / stacked.min(dim=1).values.clamp_min(1e-8)


def _face_adjacency_pairs(faces: torch.Tensor) -> List[Tuple[int, int]]:
    edge_map = _edge_face_map(faces)
    pairs: List[Tuple[int, int]] = []
    for face_ids in edge_map.values():
        if len(face_ids) < 2:
            continue
        for i in range(len(face_ids)):
            for j in range(i + 1, len(face_ids)):
                pairs.append((face_ids[i], face_ids[j]))
    return pairs


def _connected_components_count(faces: torch.Tensor) -> int:
    if faces.numel() == 0:
        return 0

    pairs = _face_adjacency_pairs(faces)
    f = faces.shape[0]
    if f == 0:
        return 0

    parent = list(range(f))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)

    roots = {find(i) for i in range(f)}
    return len(roots)


def chamfer_distance_points(pred_points: torch.Tensor, target_points: torch.Tensor) -> float:
    if pred_points.numel() == 0 or target_points.numel() == 0:
        return float("nan")
    d = torch.cdist(pred_points, target_points)
    return float(d.min(dim=1).values.mean().item() + d.min(dim=0).values.mean().item())


def compute_mesh_quality_metrics(
    verts: torch.Tensor,
    faces: torch.Tensor,
    edge_mat: Optional[torch.Tensor] = None,
    target_points: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    normals = _face_normals(verts, faces)
    pairs = _face_adjacency_pairs(faces)
    if pairs and normals.numel() > 0:
        dots = []
        flips = 0
        for i, j in pairs:
            dot = float(torch.dot(normals[i], normals[j]).item())
            dots.append(dot)
            if dot < 0.0:
                flips += 1
        metrics["normal_consistency"] = float(sum(dots) / max(len(dots), 1))
        metrics["flip_ratio"] = float(flips / max(len(dots), 1))
    else:
        metrics["normal_consistency"] = float("nan")
        metrics["flip_ratio"] = float("nan")

    aspect = _triangle_aspect_ratios(verts, faces)
    if aspect.numel() > 0:
        metrics["triangle_aspect_ratio_mean"] = float(aspect.mean().item())
        metrics["triangle_aspect_ratio_max"] = float(aspect.max().item())
    else:
        metrics["triangle_aspect_ratio_mean"] = float("nan")
        metrics["triangle_aspect_ratio_max"] = float("nan")

    edges = _unique_edges(faces)
    if edges.numel() > 0:
        edge_lengths = (verts[edges[:, 0]] - verts[edges[:, 1]]).norm(dim=1)
        metrics["edge_length_mean"] = float(edge_lengths.mean().item())
        metrics["edge_length_std"] = float(edge_lengths.std(unbiased=False).item())
    else:
        metrics["edge_length_mean"] = float("nan")
        metrics["edge_length_std"] = float("nan")

    edge_face_map = _edge_face_map(faces)
    non_manifold = sum(1 for faces_on_edge in edge_face_map.values() if len(faces_on_edge) > 2)
    metrics["non_manifold_edge_count"] = float(non_manifold)
    metrics["component_count"] = float(_connected_components_count(faces))

    if edge_mat is not None and edge_mat.numel() > 0:
        degree = (edge_mat > 0).sum(dim=1).float()
        metrics["graph_degree_mean"] = float(degree.mean().item())
        metrics["graph_degree_std"] = float(degree.std(unbiased=False).item())
    else:
        metrics["graph_degree_mean"] = float("nan")
        metrics["graph_degree_std"] = float("nan")

    if target_points is not None:
        metrics["chamfer_distance"] = chamfer_distance_points(verts, target_points)
    else:
        metrics["chamfer_distance"] = float("nan")

    return metrics