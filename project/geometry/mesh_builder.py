import torch


def build_fixed_topology_mesh(num_verts: int, device: str = "cpu"):
    if num_verts < 3:
        raise ValueError("num_verts must be >= 3")

    verts = torch.randn(num_verts, 3, device=device) * 0.1
    faces = []
    for i in range(1, num_verts - 1):
        faces.append([0, i, i + 1])
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    return verts, faces
