from __future__ import annotations

import torch
import torch.nn as nn


class CoupledXYDbhLoss(nn.Module):
    """
    total = xy_weight * mean(dx^2 + dy^2) + dbh_weight * SmoothL1(dbh)
    """

    def __init__(
        self,
        xy_weight: float = 1.0,
        dbh_weight: float = 1.0,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.xy_weight = float(xy_weight)
        self.dbh_weight = float(dbh_weight)
        self.dbh_loss_fn = nn.SmoothL1Loss(reduction="mean", beta=smooth_l1_beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dx = pred[:, 0] - target[:, 0]
        dy = pred[:, 1] - target[:, 1]

        xy_sq_dist = dx.square() + dy.square()
        xy_loss = xy_sq_dist.mean()

        dbh_loss = self.dbh_loss_fn(pred[:, 2], target[:, 2])

        return self.xy_weight * xy_loss + self.dbh_weight * dbh_loss

    @torch.no_grad()
    def components(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        dx = pred[:, 0] - target[:, 0]
        dy = pred[:, 1] - target[:, 1]

        xy_sq_dist = dx.square() + dy.square()
        xy_loss = xy_sq_dist.mean()
        xy_mean_euclid = torch.sqrt(xy_sq_dist + 1e-12).mean()

        dbh_loss = self.dbh_loss_fn(pred[:, 2], target[:, 2])
        total = self.xy_weight * xy_loss + self.dbh_weight * dbh_loss

        return {
            "xy_loss_unweighted": float(xy_loss.item()),
            "xy_mean_euclidean_error": float(xy_mean_euclid.item()),
            "dbh_loss_unweighted": float(dbh_loss.item()),
            "xy_loss_weighted": float((self.xy_weight * xy_loss).item()),
            "dbh_loss_weighted": float((self.dbh_weight * dbh_loss).item()),
            "total_loss": float(total.item()),
        }