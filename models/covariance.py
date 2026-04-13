import torch


def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)

    two = 2.0
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    r00 = ww + xx - yy - zz
    r01 = two * (xy - wz)
    r02 = two * (xz + wy)

    r10 = two * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = two * (yz - wx)

    r20 = two * (xz - wy)
    r21 = two * (yz + wx)
    r22 = ww - xx - yy + zz

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def build_covariance(quat: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    r = quaternion_to_rotation_matrix(quat)
    s = torch.exp(log_scale)
    diag = torch.diag_embed(s * s)
    return r @ diag @ r.transpose(-1, -2)
