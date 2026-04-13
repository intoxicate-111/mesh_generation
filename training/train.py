import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data.multiview_dataset import load_multiview_supervision
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


def save_obj(path: Path, verts: torch.Tensor, faces: torch.Tensor) -> None:
    verts_cpu = verts.detach().cpu()
    faces_cpu = faces.detach().cpu()

    with path.open("w", encoding="utf-8") as f:
        for v in verts_cpu:
            f.write(f"v {v[0].item():.6f} {v[1].item():.6f} {v[2].item():.6f}\n")
        for tri in faces_cpu:
            i, j, k = int(tri[0].item()) + 1, int(tri[1].item()) + 1, int(tri[2].item()) + 1
            f.write(f"f {i} {j} {k}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mesh with multi-view supervision")
    parser.add_argument("--views_json", type=str, required=True, help="Path to views.json")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--num_points", type=int, default=64)
    parser.add_argument("--cell_size", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    num_points = args.num_points
    cell_size = args.cell_size
    alpha = args.alpha
    steps = args.steps

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = PointCloudParams(num_points=num_points, device=device)
    renderer = PyTorch3DSilhouetteRenderer(
        image_size=(args.image_height, args.image_width),
        device=device,
    ).to(device)
    _, faces = build_fixed_topology_mesh(num_verts=num_points, device=device)

    optimizer = torch.optim.Adam(params.parameters(), lr=args.lr)

    gt_mask, cameras = load_multiview_supervision(
        views_json_path=args.views_json,
        image_height=args.image_height,
        image_width=args.image_width,
        device=device,
    )
    torch.save(
        {
            "faces": faces.cpu(),
            "gt_mask": gt_mask.cpu(),
            "view_count": gt_mask.shape[0],
            "image_height": args.image_height,
            "image_width": args.image_width,
            "views_json": str(Path(args.views_json).resolve()),
        },
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
        pred_mask = renderer(pred_verts, faces, cameras)

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
        final_mask = renderer(final_verts, faces, cameras)

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
    mesh_obj_path = output_dir / "mesh_final.obj"
    save_obj(mesh_obj_path, final_verts, faces)

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
    print(f"saved: {mesh_obj_path}")
    print(f"saved: {loss_csv}")
    print(f"views used: {gt_mask.shape[0]}")


if __name__ == "__main__":
    main()
