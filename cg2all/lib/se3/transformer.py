"""SE3Transformer top-level module — ported from fork. Accepts a `Graph`
(instead of a DGLGraph). Edge relative positions are read from `graph.edge["rel_pos"]`.
Pooling (unused by cg2all) is intentionally not ported.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .basis import get_basis, update_basis_with_fused
from .fiber import Fiber
from .layers.attention import AttentionBlockSE3
from .layers.convolution import ConvSE3, ConvSE3FuseLevel
from .layers.norm import NormSE3
from ..graph import Graph


class Sequential(nn.Sequential):
    """Sequential module with arbitrary forward args and kwargs."""

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


def get_populated_edge_features(
    relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None
):
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if "0" in edge_features:
        edge_features["0"] = torch.cat([edge_features["0"], r[..., None]], dim=1)
    else:
        edge_features["0"] = r[..., None]
    return edge_features


class SE3Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        num_heads: int,
        channels_div: int,
        fiber_edge: Fiber = Fiber({}),
        return_type: Optional[int] = None,
        pooling: Optional[str] = None,
        norm: bool = True,
        use_layer_norm: bool = True,
        nonlinearity: nn.Module = nn.ReLU(),
        mid_dim: int = 32,
        tensor_cores: bool = False,
        low_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert pooling is None, "pooling not ported (cg2all does not use it)"
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.pooling = None
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory

        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            self.fuse_level = (
                ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL
            )

        graph_modules = []
        for _ in range(num_layers):
            graph_modules.append(
                AttentionBlockSE3(
                    fiber_in=fiber_in,
                    fiber_out=fiber_hidden,
                    fiber_edge=fiber_edge,
                    num_heads=num_heads,
                    channels_div=channels_div,
                    use_layer_norm=use_layer_norm,
                    nonlinearity=nonlinearity,
                    max_degree=self.max_degree,
                    mid_dim=mid_dim,
                    fuse_level=self.fuse_level,
                    low_memory=low_memory,
                )
            )
            if norm:
                graph_modules.append(NormSE3(fiber_hidden, nonlinearity=nonlinearity))
            fiber_in = fiber_hidden

        graph_modules.append(
            ConvSE3(
                fiber_in=fiber_in,
                fiber_out=fiber_out,
                fiber_edge=fiber_edge,
                self_interaction=True,
                use_layer_norm=use_layer_norm,
                nonlinearity=nonlinearity,
                mid_dim=mid_dim,
                max_degree=self.max_degree,
            )
        )
        self.graph_modules = Sequential(*graph_modules)

    def forward(
        self,
        graph: Graph,
        node_feats: Dict[str, Tensor],
        edge_feats: Optional[Dict[str, Tensor]] = None,
        basis: Optional[Dict[str, Tensor]] = None,
    ):
        rel_pos = graph.edge["rel_pos"]
        basis = basis or get_basis(
            rel_pos,
            max_degree=self.max_degree,
            compute_gradients=False,
            use_pad_trick=self.tensor_cores and not self.low_memory,
            amp=torch.is_autocast_enabled(),
        )
        basis = update_basis_with_fused(
            basis,
            self.max_degree,
            use_pad_trick=self.tensor_cores and not self.low_memory,
            fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL,
        )
        edge_feats = get_populated_edge_features(rel_pos, edge_feats)
        node_feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis)
        if self.return_type is not None:
            return node_feats[str(self.return_type)]
        return node_feats


class SE3TransformerPooled(nn.Module):
    """Kept for API parity; pooling path not actually used by cg2all."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SE3TransformerPooled is not ported — cg2all does not use the pooled variant."
        )
