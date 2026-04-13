import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras


def _load_mask(image_path: Path, image_height: int, image_width: int, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("L").resize((image_width, image_height), Image.BILINEAR)
    mask = torch.from_numpy(np.array(image)).float() / 255.0
    return mask.to(device)


def load_multiview_supervision(
    views_json_path: str,
    image_height: int,
    image_width: int,
    device: str,
) -> tuple[torch.Tensor, FoVPerspectiveCameras]:
    views_path = Path(views_json_path)
    with views_path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    data_root = Path(config.get("data_root", views_path.parent))
    fov = float(config.get("fov", 60.0))

    views = config.get("views", [])
    if not views:
        raise ValueError("views.json must contain a non-empty 'views' list")

    masks = []
    rotations = []
    translations = []

    for item in views:
        image_rel = item.get("image")
        r = item.get("R")
        t = item.get("T")
        if image_rel is None or r is None or t is None:
            raise ValueError("Each view must include image, R, and T")

        image_path = (data_root / image_rel).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        masks.append(
            _load_mask(
                image_path,
                image_height=image_height,
                image_width=image_width,
                device=device,
            )
        )
        rotations.append(torch.tensor(r, dtype=torch.float32, device=device))
        translations.append(torch.tensor(t, dtype=torch.float32, device=device))

    target_masks = torch.stack(masks, dim=0)
    R = torch.stack(rotations, dim=0)
    T = torch.stack(translations, dim=0)

    cameras = FoVPerspectiveCameras(R=R, T=T, fov=fov, device=device)
    return target_masks, cameras
