import torch
import torch.nn.functional as F
from pathlib import Path

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


def edge_length_loss(points: torch.Tensor, neighbors: list[list[int]]) -> torch.Tensor:
    lengths = []
    for i, neigh in enumerate(neighbors):
        for j in neigh:
            if j > i:
                lengths.append((points[i] - points[j]).norm())

    if not lengths:
        return torch.tensor(0.0, device=points.device)

    edge_lengths = torch.stack(lengths)
    return ((edge_lengths - edge_lengths.mean()) ** 2).mean()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    num_points = 64
    cell_size = 0.35
    alpha = 1.5
    steps = 101

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = PointCloudParams(num_points=num_points, device=device)
    renderer = PyTorch3DSilhouetteRenderer(image_size=128, device=device).to(device)
    _, faces = build_fixed_topology_mesh(num_verts=num_points, device=device)

    optimizer = torch.optim.Adam(params.parameters(), lr=1e-2)

    with torch.no_grad():
        gt_points = torch.randn(num_points, 3, device=device) * 0.35
        gt_points[:, 2] = gt_points[:, 2] * 0.2
        gt_mask = renderer(gt_points, faces)
        torch.save(
            {"gt_points": gt_points.cpu(), "faces": faces.cpu(), "gt_mask": gt_mask.cpu()},
            output_dir / "target.pt",
        )

    history = []

    for step in range(steps):
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
        e_len_loss = edge_length_loss(pred_verts, neighbors)
        loss = render_loss + 0.05 * smooth_loss + 0.02 * e_len_loss - 0.01 * edge_reg

        loss.backward()
        optimizer.step()

        history.append(
            {
                "step": step,
                "total": float(loss.item()),
                "render": float(render_loss.item()),
                "smooth": float(smooth_loss.item()),
                "edge_len": float(e_len_loss.item()),
                "edge_reg": float(edge_reg.item()),
            }
        )

        if step % 20 == 0:
            print(
                f"step={step:03d} total={loss.item():.6f} "
                f"render={render_loss.item():.6f} smooth={smooth_loss.item():.6f} "
                f"edge_len={e_len_loss.item():.6f}"
            )

    with torch.no_grad():
        final_verts = torch.tanh(params.points)
        final_mask = renderer(final_verts, faces)

    torch.save(
        {
            "points": params.points.detach().cpu(),
            "quat": params.quat.detach().cpu(),
            "log_scale": params.log_scale.detach().cpu(),
            "faces": faces.detach().cpu(),
        },
        output_dir / "checkpoint_final.pt",
    )
    torch.save(
        {"pred_mask": final_mask.detach().cpu(), "gt_mask": gt_mask.detach().cpu()},
        output_dir / "masks_final.pt",
    )

    loss_csv = output_dir / "loss_history.csv"
    with loss_csv.open("w", encoding="utf-8") as f:
        f.write("step,total,render,smooth,edge_len,edge_reg\n")
        for row in history:
            f.write(
                f"{row['step']},{row['total']:.8f},{row['render']:.8f},"
                f"{row['smooth']:.8f},{row['edge_len']:.8f},{row['edge_reg']:.8f}\n"
            )

    print(f"saved: {output_dir / 'checkpoint_final.pt'}")
    print(f"saved: {output_dir / 'masks_final.pt'}")
    print(f"saved: {loss_csv}")


if __name__ == "__main__":
    main()
