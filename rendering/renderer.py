import math
from typing import Tuple

import torch
import torch.nn as nn
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes


class PyTorch3DSilhouetteRenderer(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int] = (1080, 1920),
        sigma: float = 1e-4,
        faces_per_pixel: int = 8,
        bin_size: int = 0,
        max_faces_per_bin: int = 200000,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.image_size = image_size

        blend_params = BlendParams(sigma=sigma, gamma=1e-4)
        blur_radius = math.log(1.0 / 1e-4 - 1.0) * sigma
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin,
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )
        self.device = device

    def forward(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        cameras: FoVPerspectiveCameras,
        max_views_per_batch: int = 4,
    ) -> torch.Tensor:
        if verts.ndim != 2 or verts.size(-1) != 3:
            raise ValueError("verts must have shape [V, 3]")
        if faces.ndim != 2 or faces.size(-1) != 3:
            raise ValueError("faces must have shape [F, 3]")
        if len(cameras) < 1:
            raise ValueError("cameras must contain at least one view")
        if max_views_per_batch < 1:
            raise ValueError("max_views_per_batch must be >= 1")

        view_count = len(cameras)
        alpha_outputs = []
        for start in range(0, view_count, max_views_per_batch):
            end = min(start + max_views_per_batch, view_count)
            index_list = list(range(start, end))
            cameras_batch = cameras[index_list]
            batch_size = end - start
            verts_batch = [verts for _ in range(batch_size)]
            faces_batch = [faces for _ in range(batch_size)]
            meshes = Meshes(verts=verts_batch, faces=faces_batch)
            rendered = self.renderer(meshes_world=meshes, cameras=cameras_batch)
            alpha_outputs.append(rendered[..., 3])

        return torch.cat(alpha_outputs, dim=0)
