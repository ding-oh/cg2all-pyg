"""LinearModule / InteractionModule — convenience wrappers used by cg2all.lib.libmodel.

Ported DGL-free: InteractionModule.forward now takes a `Graph`.
"""

from typing import Optional

import torch
import torch.nn as nn

from .fiber import Fiber
from .layers import LinearSE3, NormSE3
from .transformer import SE3Transformer
from ..graph import Graph


class LinearModule(nn.Module):
    def __init__(
        self,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        n_layer: Optional[int] = 2,
        use_norm: Optional[bool] = True,
        nonlinearity: Optional[nn.Module] = nn.ReLU(),
        **kwargs,
    ):
        super().__init__()
        linear_module = []
        if n_layer >= 2:
            if use_norm:
                linear_module.append(NormSE3(Fiber(fiber_in), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(fiber_in), Fiber(fiber_hidden)))
            for _ in range(n_layer - 2):
                if use_norm:
                    linear_module.append(NormSE3(Fiber(fiber_hidden), nonlinearity=nonlinearity))
                linear_module.append(LinearSE3(Fiber(fiber_hidden), Fiber(fiber_hidden)))
            if use_norm:
                linear_module.append(NormSE3(Fiber(fiber_hidden), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(fiber_hidden), Fiber(fiber_out)))
        else:
            if use_norm:
                linear_module.append(NormSE3(Fiber(fiber_in), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(fiber_in), Fiber(fiber_out)))
        self.linear_module = nn.Sequential(*linear_module)

    def forward(self, x):
        return self.linear_module(x)


class InteractionModule(nn.Module):
    def __init__(
        self,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        fiber_edge: Optional[Fiber] = Fiber({}),
        n_layer: Optional[int] = 2,
        n_head: Optional[int] = 2,
        use_norm: Optional[bool] = True,
        use_layer_norm: Optional[bool] = True,
        nonlinearity: Optional[nn.Module] = nn.ReLU(),
        low_memory: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__()
        self.graph_module = SE3Transformer(
            num_layers=n_layer,
            fiber_in=fiber_in,
            fiber_hidden=fiber_hidden,
            fiber_out=fiber_out,
            num_heads=n_head,
            channels_div=2,
            fiber_edge=fiber_edge,
            norm=use_norm,
            use_layer_norm=use_layer_norm,
            nonlinearity=nonlinearity,
            low_memory=low_memory,
        )

    def forward(self, batch: Graph, node_feats, edge_feats):
        return self.graph_module(batch, node_feats=node_feats, edge_feats=edge_feats)
