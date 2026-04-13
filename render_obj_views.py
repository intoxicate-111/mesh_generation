import argparse
import json
from pathlib import Path

import torch
from PIL import Image
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes


def load_mesh(mesh_path: Path, device: torch.device) -> Meshes:
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        return load_objs_as_meshes([str(mesh_path)], device=device)
    if suffix == ".stl":
        tm = trimesh.load(str(mesh_path), force="mesh")
        if tm.faces is None or len(tm.faces) == 0:
            raise ValueError(f"No valid faces in STL: {mesh_path}")
        verts = torch.tensor(tm.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(tm.faces, dtype=torch.int64, device=device)
        return Meshes(verts=[verts], faces=[faces])
    raise ValueError(f"Unsupported mesh format: {mesh_path.suffix}. Use .obj or .stl")


def normalize_mesh(mesh: Meshes) -> Meshes:
    verts = mesh.verts_packed()
    v_min = verts.min(dim=0).values
    v_max = verts.max(dim=0).values
    center = (v_min + v_max) * 0.5

    centered = verts - center
    scale = centered.norm(dim=1).max().clamp_min(1e-6)

    norm_verts = [(v - center) / scale for v in mesh.verts_list()]
    return Meshes(verts=norm_verts, faces=mesh.faces_list())


def main() -> None:
    parser = argparse.ArgumentParser(description="Render multi-view silhouettes from an OBJ/STL mesh")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--num_views", type=int, default=24)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--dist", type=float, default=2.7)
    parser.add_argument("--elev", type=float, default=10.0)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable_normalize", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    mesh_path = Path(args.mesh_path).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(mesh_path, device=device)
    if not args.disable_normalize:
        mesh = normalize_mesh(mesh)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=args.image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    azims = torch.linspace(0.0, 360.0, steps=args.num_views + 1)[:-1]
    views = []

    for idx, azim in enumerate(azims.tolist()):
        r, t = look_at_view_transform(dist=args.dist, elev=args.elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(R=r, T=t, fov=args.fov, device=device)

        rendered = renderer(meshes_world=mesh, cameras=cameras)
        mask = rendered[0, ..., 3].clamp(0.0, 1.0)
        mask_u8 = (mask.detach().cpu().numpy() * 255.0).astype("uint8")

        image_name = f"view_{idx:03d}.png"
        image_path = images_dir / image_name
        Image.fromarray(mask_u8, mode="L").save(image_path)

        views.append(
            {
                "image": f"images/{image_name}",
                "R": r[0].detach().cpu().tolist(),
                "T": t[0].detach().cpu().tolist(),
            }
        )

    views_json = {
        "data_root": str(output_dir).replace("\\", "/"),
        "fov": args.fov,
        "views": views,
    }

    views_path = output_dir / "views.json"
    with views_path.open("w", encoding="utf-8") as f:
        json.dump(views_json, f, indent=2)

    print(f"saved: {views_path}")
    print(f"saved images: {images_dir}")
    print(f"normalize mesh: {not args.disable_normalize}")


if __name__ == "__main__":
    main()
