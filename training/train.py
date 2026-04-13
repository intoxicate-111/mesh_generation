import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data.multiview_dataset import load_multiview_supervision
from geometry.dynamic_points import adaptive_point_update
from geometry.edge_weights import compute_edge_weights
from geometry.mesh_builder import build_fixed_topology_mesh
from geometry.topology_update import build_faces_from_edge_graph, build_sparse_edge_matrix
from losses.boundary_losses import boundary_aware_silhouette_loss
from geometry.spatial_blocks import build_spatial_blocks, query_candidate_neighbors
from models.point_representation import PointCloudParams
from rendering.renderer import PyTorch3DSilhouetteRenderer
from utils.optimizer_remap import rebuild_adam_optimizer


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


def _normalize_positive(values: torch.Tensor) -> torch.Tensor:
    return values / values.mean().clamp_min(1e-6)


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
    parser.add_argument("--faces_per_pixel", type=int, default=8)
    parser.add_argument("--max_views_per_batch", type=int, default=4)
    parser.add_argument("--boundary_loss_weight", type=float, default=0.4)
    parser.add_argument("--adaptive_interval", type=int, default=20)
    parser.add_argument("--max_new_points", type=int, default=4)
    parser.add_argument("--max_prune_points", type=int, default=3)
    parser.add_argument("--max_split_points", type=int, default=2)
    parser.add_argument("--max_merge_pairs", type=int, default=2)
    parser.add_argument("--min_points", type=int, default=24)
    parser.add_argument("--densify_threshold", type=float, default=1.25)
    parser.add_argument("--prune_threshold", type=float, default=0.8)
    parser.add_argument("--split_threshold", type=float, default=1.35)
    parser.add_argument("--merge_distance_threshold", type=float, default=0.06)
    parser.add_argument("--merge_scale_rel_threshold", type=float, default=0.35)
    parser.add_argument("--merge_neighbor_jaccard_threshold", type=float, default=0.3)
    parser.add_argument("--prune_min_degree", type=int, default=1)
    parser.add_argument("--edge_min_weight", type=float, default=0.15)
    parser.add_argument("--edge_topk", type=int, default=8)
    parser.add_argument("--max_faces_per_anchor", type=int, default=6)
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
        faces_per_pixel=args.faces_per_pixel,
        device=device,
    ).to(device)
    _, fallback_faces = build_fixed_topology_mesh(num_verts=num_points, device=device)
    faces = fallback_faces

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
        added_this_step = 0
        pruned_this_step = 0
        split_this_step = 0
        merged_this_step = 0

        points = params.points
        sigma = params.covariance()

        blocks = build_spatial_blocks(points.detach(), cell_size=cell_size)
        neighbors = query_candidate_neighbors(points.detach(), blocks, cell_size=cell_size)

        weights = compute_edge_weights(points, sigma, neighbors, alpha=alpha)
        edge_mat = build_sparse_edge_matrix(
            candidate_idx=neighbors,
            edge_weights=weights,
            num_points=points.shape[0],
            min_weight=args.edge_min_weight,
            topk_per_point=args.edge_topk,
        ).to(device)

        pred_verts = torch.tanh(points)
        dynamic_faces = build_faces_from_edge_graph(
            verts=pred_verts,
            edge_mat=edge_mat,
            max_faces_per_anchor=args.max_faces_per_anchor,
        )
        if dynamic_faces.numel() > 0:
            faces = dynamic_faces
        else:
            _, faces = build_fixed_topology_mesh(num_verts=points.shape[0], device=device)

        if edge_mat.numel() > 0 and torch.any(edge_mat > 0):
            edge_reg = edge_mat[edge_mat > 0].mean()
        else:
            edge_reg = torch.tensor(0.0, device=device)

        pred_mask = renderer(
            pred_verts,
            faces,
            cameras,
            max_views_per_batch=args.max_views_per_batch,
        )

        render_loss = F.mse_loss(pred_mask, gt_mask)
        boundary_loss = boundary_aware_silhouette_loss(pred_mask, gt_mask)
        smooth_loss = laplacian_like_loss(points, neighbors)
        e_len_loss = edge_length_loss(pred_verts, neighbors)
        loss = (
            render_loss
            + args.boundary_loss_weight * boundary_loss
            + 0.05 * smooth_loss
            + 0.02 * e_len_loss
            - 0.01 * edge_reg
        )

        loss.backward()

        if params.points.grad is not None:
            point_grad_norm = params.points.grad.detach().norm(dim=1)
        else:
            point_grad_norm = torch.zeros(points.shape[0], device=device)

        optimizer.step()

        if (
            args.adaptive_interval > 0
            and step > 0
            and step % args.adaptive_interval == 0
            and points.shape[0] >= args.min_points
        ):
            with torch.no_grad():
                degree = (edge_mat > 0).sum(dim=1).float()
                scales = torch.exp(params.log_scale.detach())
                scale_score = _normalize_positive(scales.mean(dim=1))
                grad_score = _normalize_positive(point_grad_norm)
                densify_scores = 0.6 * scale_score + 0.4 * grad_score
                anisotropy = scales.amax(dim=1) / scales.amin(dim=1).clamp_min(1e-6)
                split_scores = _normalize_positive(0.7 * anisotropy + 0.3 * grad_score)

                low_grad = 1.0 - point_grad_norm / point_grad_norm.max().clamp_min(1e-6)
                low_connectivity = (degree <= float(args.prune_min_degree)).float()
                prune_scores = 0.7 * low_connectivity + 0.3 * low_grad

                update = adaptive_point_update(
                    points=params.points.detach(),
                    quat=params.quat.detach(),
                    log_scale=params.log_scale.detach(),
                    densify_scores=densify_scores,
                    prune_scores=prune_scores,
                    split_scores=split_scores,
                    merge_edge_matrix=edge_mat.detach(),
                    densify_threshold=args.densify_threshold,
                    prune_threshold=args.prune_threshold,
                    split_threshold=args.split_threshold,
                    merge_distance_threshold=args.merge_distance_threshold,
                    merge_scale_rel_threshold=args.merge_scale_rel_threshold,
                    merge_neighbor_jaccard_threshold=args.merge_neighbor_jaccard_threshold,
                    max_new_points=args.max_new_points,
                    max_prune_points=args.max_prune_points,
                    max_split_points=args.max_split_points,
                    max_merge_pairs=args.max_merge_pairs,
                    min_points=args.min_points,
                )

                added_this_step = update.added
                pruned_this_step = update.pruned
                split_this_step = update.split
                merged_this_step = update.merged
                if added_this_step > 0 or pruned_this_step > 0 or split_this_step > 0 or merged_this_step > 0:
                    params.reset_parameters(update.points, update.quat, update.log_scale)
                    optimizer = rebuild_adam_optimizer(optimizer, params.parameters(), lr_override=args.lr)

        history.append(
            {
                "step": step,
                "total": float(loss.item()),
                "render": float(render_loss.item()),
                "boundary": float(boundary_loss.item()),
                "smooth": float(smooth_loss.item()),
                "edge_len": float(e_len_loss.item()),
                "edge_reg": float(edge_reg.item()),
                "points": int(params.num_points),
                "faces": int(faces.shape[0]),
                "added": int(added_this_step),
                "pruned": int(pruned_this_step),
                "split": int(split_this_step),
                "merged": int(merged_this_step),
            }
        )

        if step % 20 == 0:
            print(
                f"step={step:03d} total={loss.item():.6f} "
                f"render={render_loss.item():.6f} boundary={boundary_loss.item():.6f} "
                f"smooth={smooth_loss.item():.6f} edge_len={e_len_loss.item():.6f} "
                f"points={params.num_points} faces={faces.shape[0]} "
                f"+{added_this_step}/-{pruned_this_step} split={split_this_step} merge={merged_this_step}"
            )

    with torch.no_grad():
        final_verts = torch.tanh(params.points)
        final_sigma = params.covariance()
        final_blocks = build_spatial_blocks(params.points.detach(), cell_size=cell_size)
        final_neighbors = query_candidate_neighbors(params.points.detach(), final_blocks, cell_size=cell_size)
        final_weights = compute_edge_weights(params.points, final_sigma, final_neighbors, alpha=alpha)
        final_edge_mat = build_sparse_edge_matrix(
            candidate_idx=final_neighbors,
            edge_weights=final_weights,
            num_points=params.num_points,
            min_weight=args.edge_min_weight,
            topk_per_point=args.edge_topk,
        ).to(device)
        final_faces = build_faces_from_edge_graph(
            verts=final_verts,
            edge_mat=final_edge_mat,
            max_faces_per_anchor=args.max_faces_per_anchor,
        )
        if final_faces.numel() == 0:
            _, final_faces = build_fixed_topology_mesh(num_verts=params.num_points, device=device)

        final_mask = renderer(
            final_verts,
            final_faces,
            cameras,
            max_views_per_batch=args.max_views_per_batch,
        )

    torch.save(
        {
            "points": params.points.detach().cpu(),
            "quat": params.quat.detach().cpu(),
            "log_scale": params.log_scale.detach().cpu(),
            "faces": final_faces.detach().cpu(),
        },
        output_dir / "checkpoint_final.pt",
    )
    torch.save(
        {"pred_mask": final_mask.detach().cpu(), "gt_mask": gt_mask.detach().cpu()},
        output_dir / "masks_final.pt",
    )
    mesh_obj_path = output_dir / "mesh_final.obj"
    save_obj(mesh_obj_path, final_verts, final_faces)

    loss_csv = output_dir / "loss_history.csv"
    with loss_csv.open("w", encoding="utf-8") as f:
        f.write("step,total,render,boundary,smooth,edge_len,edge_reg,points,faces,added,pruned,split,merged\n")
        for row in history:
            f.write(
                f"{row['step']},{row['total']:.8f},{row['render']:.8f},"
                f"{row['boundary']:.8f},{row['smooth']:.8f},{row['edge_len']:.8f},"
                f"{row['edge_reg']:.8f},{row['points']},{row['faces']},"
                f"{row['added']},{row['pruned']},{row['split']},{row['merged']}\n"
            )

    print(f"saved: {output_dir / 'checkpoint_final.pt'}")
    print(f"saved: {output_dir / 'masks_final.pt'}")
    print(f"saved: {mesh_obj_path}")
    print(f"saved: {loss_csv}")
    print(f"views used: {gt_mask.shape[0]}")


if __name__ == "__main__":
    main()
