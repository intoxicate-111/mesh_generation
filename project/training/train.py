import torch
import torch.nn.functional as F

from geometry.edge_weights import compute_edge_weights
from geometry.mesh_builder import build_fixed_topology_mesh
from geometry.spatial_blocks import build_spatial_blocks, query_candidate_neighbors
from models.point_representation import PointCloudParams
from rendering.renderer import PyTorch3DSilhouetteRenderer


def laplacian_like_loss(points: torch.Tensor, neighbors: list[list[int]]) -> torch.Tensor:
    losses = []
    for i, neigh in enumerate(neighbors):
        if not neigh:
            continue
        mean_neigh = points[torch.tensor(neigh, device=points.device)].mean(dim=0)
        losses.append(((points[i] - mean_neigh) ** 2).mean())
    if not losses:
        return torch.tensor(0.0, device=points.device)
    return torch.stack(losses).mean()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    num_points = 64
    cell_size = 0.35
    alpha = 1.5

    params = PointCloudParams(num_points=num_points, device=device)
    renderer = PyTorch3DSilhouetteRenderer(image_size=128, device=device).to(device)
    _, faces = build_fixed_topology_mesh(num_verts=num_points, device=device)

    optimizer = torch.optim.Adam(params.parameters(), lr=1e-2)

    with torch.no_grad():
        gt_points = torch.randn(num_points, 3, device=device) * 0.35
        gt_points[:, 2] = gt_points[:, 2] * 0.2
        gt_mask = renderer(gt_points, faces)

    for step in range(101):
        optimizer.zero_grad()

        points = params.points
        sigma = params.covariance()

        blocks = build_spatial_blocks(points.detach(), cell_size=cell_size)
        neighbors = query_candidate_neighbors(points.detach(), blocks, cell_size=cell_size)

        weights = compute_edge_weights(points, sigma, neighbors, alpha=alpha)
        if weights and any(w.numel() > 0 for w in weights):
            edge_reg = torch.stack([w.mean() for w in weights if w.numel() > 0]).mean()
        else:
            edge_reg = torch.tensor(0.0, device=device)

        pred_verts = torch.tanh(points)
        pred_mask = renderer(pred_verts, faces)

        render_loss = F.mse_loss(pred_mask, gt_mask)
        smooth_loss = laplacian_like_loss(points, neighbors)
        loss = render_loss + 0.05 * smooth_loss - 0.01 * edge_reg

        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(
                f"step={step:03d} total={loss.item():.6f} "
                f"render={render_loss.item():.6f} smooth={smooth_loss.item():.6f}"
            )


if __name__ == "__main__":
    main()
