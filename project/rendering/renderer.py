import math

import torch
import torch.nn as nn
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


class PyTorch3DSilhouetteRenderer(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        sigma: float = 1e-4,
        faces_per_pixel: int = 50,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.image_size = image_size

        r, t = look_at_view_transform(dist=2.7, elev=10.0, azim=20.0)
        cameras = FoVPerspectiveCameras(R=r, T=t, device=device)

        blend_params = BlendParams(sigma=sigma, gamma=1e-4)
        blur_radius = math.log(1.0 / 1e-4 - 1.0) * sigma
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )

    def forward(self, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        if verts.ndim != 2 or verts.size(-1) != 3:
            raise ValueError("verts must have shape [V, 3]")
        if faces.ndim != 2 or faces.size(-1) != 3:
            raise ValueError("faces must have shape [F, 3]")

        meshes = Meshes(verts=[verts], faces=[faces])
        rendered = self.renderer(meshes)
        return rendered[..., 3].squeeze(0)
