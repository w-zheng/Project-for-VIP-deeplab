from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor




def compute_sile(x: Tensor, y: Tensor, num: int) -> Tensor:
    """
    Scale invariant logarithmic error.
    """
    log_err = torch.log1p(x) - torch.log1p(y)

    sile_1 = log_err.square().sum() / num
    sile_2 = log_err.sum() / (num**2)

    return sile_1 - sile_2


def compute_rel(x: Tensor, y: Tensor, num: int) -> Tuple[Tensor, Tensor]:
    """
    Square relative error and absolute relative error
    """
    err = x - y
    err_rel = err / y.clamp(1e-6)
    are = err_rel.abs().sum() / num

    sre = err_rel.square().sum() / num
    sre = sre.clamp(1e-8).sqrt()

    return are, sre


class DepthLoss(nn.Module):
    def __init__(self, weight_sile=1.0, weight_are=1.0, weight_sre=1.0, **kwargs):
        super().__init__(**kwargs)

        self.weight_sile = weight_sile
        self.weight_are = weight_are
        self.weight_sre = weight_sre

    def forward(self,
        x: torch.Tensor,# truth
        y: torch.Tensor,
        ) -> Tensor:

        mask = (y > 0)
        if not mask.any():
            return x.sum() * 0.0

        y = y[mask]
        x = x[mask]
        n = y.numel()

        sile = compute_sile(x, y, n)
        are, sre = compute_rel(x, y, n)

        loss = sile * self.weight_sile + are * self.weight_are + sre * self.weight_sre

        return loss


# Aliases for legacy code
DepthInstanceLoss = DepthLoss
DepthFlatLoss = DepthLoss
