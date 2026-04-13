import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
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


def _direction_to_elev_azim(direction: torch.Tensor) -> tuple[float, float]:
    unit = direction / direction.norm().clamp_min(1e-8)
    x, y, z = float(unit[0].item()), float(unit[1].item()), float(unit[2].item())
    elev = math.degrees(math.asin(max(-1.0, min(1.0, y))))
    azim = math.degrees(math.atan2(x, z))
    return elev, azim


def build_cube_like_angles(num_views: int) -> list[tuple[float, float]]:
    if num_views < 1:
        raise ValueError("num_views must be >= 1")

    side = 1
    while 6 * side * side < num_views:
        side += 1

    coords = torch.linspace(-1.0 + 1.0 / side, 1.0 - 1.0 / side, steps=side)
    face_dirs: list[list[torch.Tensor]] = [[] for _ in range(6)]

    for u in coords:
        for v in coords:
            face_dirs[0].append(torch.tensor([1.0, u.item(), v.item()]))
            face_dirs[1].append(torch.tensor([-1.0, u.item(), v.item()]))
            face_dirs[2].append(torch.tensor([u.item(), 1.0, v.item()]))
            face_dirs[3].append(torch.tensor([u.item(), -1.0, v.item()]))
            face_dirs[4].append(torch.tensor([u.item(), v.item(), 1.0]))
            face_dirs[5].append(torch.tensor([u.item(), v.item(), -1.0]))

    samples: list[torch.Tensor] = []
    idx = 0
    while len(samples) < num_views:
        progress = False
        for f in range(6):
            if idx < len(face_dirs[f]):
                samples.append(face_dirs[f][idx])
                progress = True
                if len(samples) == num_views:
                    break
        if not progress:
            break
        idx += 1

    return [_direction_to_elev_azim(d) for d in samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render multi-view silhouettes from an OBJ/STL mesh")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument("--dist", type=float, default=2.7)
    parser.add_argument("--elev", type=float, default=10.0)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable_normalize", action="store_true")
    parser.add_argument("--bin_size", type=int, default=0)
    parser.add_argument("--max_faces_per_bin", type=int, default=200000)
    args = parser.parse_args()

    device = torch.device(args.device)
    mesh_path = Path(args.mesh_path).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "images"
    rgb_dir = output_dir / "images_rgb"
    images_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(mesh_path, device=device)
    if not args.disable_normalize:
        mesh = normalize_mesh(mesh)
    if mesh.textures is None:
        verts_features = [torch.ones_like(v, device=device) for v in mesh.verts_list()]
        mesh = Meshes(verts=mesh.verts_list(), faces=mesh.faces_list(), textures=TexturesVertex(verts_features))

    raster_settings = RasterizationSettings(
        image_size=(args.image_height, args.image_width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=args.bin_size,
        max_faces_per_bin=args.max_faces_per_bin,
    )
    rgb_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device),
    )
    mask_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(),
    )
    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])

    angle_pairs = build_cube_like_angles(args.num_views)

    views = []

    for idx, (elev, azim) in enumerate(angle_pairs):
        r, t = look_at_view_transform(dist=args.dist, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(R=r, T=t, fov=args.fov, device=device)

        rgb = rgb_renderer(meshes_world=mesh, cameras=cameras, lights=lights)[0, ..., :3].clamp(0.0, 1.0)
        rendered = mask_renderer(meshes_world=mesh, cameras=cameras)
        mask = rendered[0, ..., 3].clamp(0.0, 1.0)
        rgb_u8 = (rgb.detach().cpu().numpy() * 255.0).astype("uint8")
        mask_u8 = (mask.detach().cpu().numpy() * 255.0).astype("uint8")

        image_name = f"view_{idx:03d}.png"
        image_path = images_dir / image_name
        rgb_path = rgb_dir / image_name
        Image.fromarray(mask_u8, mode="L").save(image_path)
        Image.fromarray(rgb_u8, mode="RGB").save(rgb_path)

        views.append(
            {
                "image": f"images/{image_name}",
                "rgb_image": f"images_rgb/{image_name}",
                "R": r[0].detach().cpu().tolist(),
                "T": t[0].detach().cpu().tolist(),
                "elev": float(elev),
                "azim": float(azim),
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
    print(f"saved rgb images: {rgb_dir}")
    print(f"normalize mesh: {not args.disable_normalize}")


if __name__ == "__main__":
    main()
