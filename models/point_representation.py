import torch
import torch.nn as nn

from .covariance import build_covariance


class PointCloudParams(nn.Module):
    def __init__(self, num_points: int, device: str = "cpu") -> None:
        super().__init__()
        self.points = nn.Parameter(torch.randn(num_points, 3, device=device) * 0.1)
        quat = torch.zeros(num_points, 4, device=device)
        quat[:, 0] = 1.0
        self.quat = nn.Parameter(quat)
        self.log_scale = nn.Parameter(torch.zeros(num_points, 3, device=device))

    def covariance(self) -> torch.Tensor:
        return build_covariance(self.quat, self.log_scale)

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    def reset_parameters(
        self,
        points: torch.Tensor,
        quat: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> None:
        self.points = nn.Parameter(points)
        self.quat = nn.Parameter(quat)
        self.log_scale = nn.Parameter(log_scale)
