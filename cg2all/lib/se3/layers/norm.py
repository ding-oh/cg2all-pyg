"""Norm-based SE(3) nonlinearity — verbatim from SE3Transformer fork."""

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from ..fiber import Fiber


class NormSE3(nn.Module):
    NORM_CLAMP = 2 ** -24

    def __init__(self, fiber: Fiber, nonlinearity: nn.Module = nn.ReLU()):
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nonlinearity

        if len(set(fiber.channels)) == 1:
            self.group_norm = nn.GroupNorm(
                num_groups=len(fiber.degrees), num_channels=sum(fiber.channels)
            )
        else:
            self.layer_norms = nn.ModuleDict(
                {str(degree): nn.LayerNorm(channels) for degree, channels in fiber}
            )

    def forward(
        self, features: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        output = {}
        if hasattr(self, "group_norm"):
            norms = [
                features[str(d)].norm(dim=-1, keepdim=True).clamp(min=self.NORM_CLAMP)
                for d in self.fiber.degrees
            ]
            fused_norms = torch.cat(norms, dim=-2)
            new_norms = self.nonlinearity(
                self.group_norm(fused_norms.squeeze(-1))
            ).unsqueeze(-1)
            new_norms = torch.chunk(new_norms, chunks=len(self.fiber.degrees), dim=-2)
            for norm, new_norm, d in zip(norms, new_norms, self.fiber.degrees):
                output[str(d)] = features[str(d)] / norm * new_norm
        else:
            for degree, feat in features.items():
                norm = feat.norm(dim=-1, keepdim=True).clamp(min=self.NORM_CLAMP)
                new_norm = self.nonlinearity(
                    self.layer_norms[degree](norm.squeeze(-1)).unsqueeze(-1)
                )
                output[degree] = new_norm * feat / norm
        return output
