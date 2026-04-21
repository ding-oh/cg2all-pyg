"""1x1 SE(3)-equivariant linear — verbatim from SE3Transformer fork."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..fiber import Fiber


class LinearSE3(nn.Module):
    def __init__(self, fiber_in: Fiber, fiber_out: Fiber):
        super().__init__()
        self.weights = nn.ParameterDict(
            {
                str(degree_out): nn.Parameter(
                    torch.randn(channels_out, fiber_in[degree_out])
                    / np.sqrt(fiber_in[degree_out])
                )
                for degree_out, channels_out in fiber_out
            }
        )

    def forward(
        self, features: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        return {
            degree: self.weights[degree] @ features[degree]
            for degree, weight in self.weights.items()
        }
