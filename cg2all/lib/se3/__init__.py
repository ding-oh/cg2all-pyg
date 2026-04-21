"""DGL-free port of the SE(3)-Transformer layers used by cg2all.

State_dict compatible with the legacy `se3_transformer` fork used for
training the published checkpoints on Zenodo 8393343. Graph topology ops
live in `cg2all.lib.graph`; the rest of the math is a near-verbatim copy of
the fork with DGL calls swapped for the shim.
"""

from .fiber import Fiber
from .layers import (
    AttentionBlockSE3,
    AttentionSE3,
    ConvSE3,
    ConvSE3FuseLevel,
    LinearSE3,
    NormSE3,
    RadialProfile,
    VersatileConvSE3,
)
from .snippets import InteractionModule, LinearModule
from .transformer import SE3Transformer, SE3TransformerPooled

__all__ = [
    "AttentionBlockSE3",
    "AttentionSE3",
    "ConvSE3",
    "ConvSE3FuseLevel",
    "Fiber",
    "InteractionModule",
    "LinearModule",
    "LinearSE3",
    "NormSE3",
    "RadialProfile",
    "SE3Transformer",
    "SE3TransformerPooled",
    "VersatileConvSE3",
]
