"""DGL-free graph container and shim operations for cg2all.

Replaces the five DGL primitives used by the vendored SE3Transformer
(`edges()`, `edata["rel_pos"]`, `dgl.ops.edge_softmax`, `dgl.ops.e_dot_v`,
`dgl.ops.copy_e_sum`) and the two graph-construction calls
(`dgl.radius_graph`, per-batch unbatch/slice). Output values must match DGL
elementwise so existing checkpoints load with `strict=True`.

Unit tests live in `tests/test_graph_shim.py` (added separately).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class Graph:
    """Mirrors the subset of `torch_geometric.data.Data` that cg2all needs.

    Using a plain dataclass keeps the PyG import optional at the graph layer
    (PyG is only imported by the radius_graph helper and the DataLoader).
    """

    pos: Tensor  # (N, 3) float
    edge_index: Tensor  # (2, E) long; edge_index[0]=src, edge_index[1]=dst
    num_nodes: int
    node: Dict[str, Tensor] = field(default_factory=dict)
    edge: Dict[str, Tensor] = field(default_factory=dict)
    batch: Optional[Tensor] = None  # (N,) long, per-node graph index when batched

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def edges(self) -> Tuple[Tensor, Tensor]:
        return self.edge_index[0], self.edge_index[1]

    def to(self, device) -> "Graph":
        return Graph(
            pos=self.pos.to(device),
            edge_index=self.edge_index.to(device),
            num_nodes=self.num_nodes,
            node={k: v.to(device) for k, v in self.node.items()},
            edge={k: v.to(device) for k, v in self.edge.items()},
            batch=self.batch.to(device) if self.batch is not None else None,
        )


def radius_graph(
    pos: Tensor,
    r: float,
    self_loop: bool = False,
    batch: Optional[Tensor] = None,
) -> Tensor:
    """Radius graph over node positions. Returns edge_index of shape (2, E).

    Matches DGL `dgl.radius_graph(pos, r, self_loop=...)` semantics: an edge
    (u, v) is produced for every ordered pair with `||pos[u] - pos[v]|| < r`
    (or <= r in DGL; torch_cluster uses <=). Self-loops are excluded by
    default. `torch_cluster.radius_graph` follows the PyG convention, which is
    the same shape we use here.
    """
    from torch_cluster import radius_graph as _rg

    return _rg(pos, r=r, batch=batch, loop=self_loop)


def _expand_index_like(index: Tensor, ref: Tensor) -> Tensor:
    while index.ndim < ref.ndim:
        index = index.unsqueeze(-1)
    return index.expand_as(ref)


def edge_softmax(edge_values: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
    """Per-destination softmax — replacement for `dgl.ops.edge_softmax`.

    For each dst node, softmax over incoming edges. Shape of `edge_values` is
    preserved. `edge_index[1]` selects the dst node per edge.
    """
    dst = edge_index[1]
    idx = _expand_index_like(dst, edge_values)

    neg_inf = torch.full(
        (num_nodes,) + edge_values.shape[1:],
        float("-inf"),
        dtype=edge_values.dtype,
        device=edge_values.device,
    )
    e_max = neg_inf.scatter_reduce(0, idx, edge_values, reduce="amax", include_self=True)
    e_max = torch.where(torch.isfinite(e_max), e_max, torch.zeros_like(e_max))
    e_exp = (edge_values - e_max[dst]).exp()
    e_sum = torch.zeros_like(e_max).scatter_add_(0, idx, e_exp)
    return e_exp / e_sum[dst].clamp_min(1e-20)


def e_dot_v(edge_key: Tensor, node_query: Tensor, edge_index: Tensor) -> Tensor:
    """Replacement for `dgl.ops.e_dot_v(graph, key, query)`.

    Called by SE3Transformer attention.py:76. Semantics: for each edge e=(u,v),
    compute <key[e], query[v]> along the last dimension. The SE3 fork reshapes
    key/query to (*, num_heads, features_per_head) so the sum is over the last
    axis; we preserve an extra singleton dim to match the DGL output.
    """
    dst = edge_index[1]
    return (edge_key * node_query[dst]).sum(dim=-1, keepdim=True)


def copy_e_sum(edge_value: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
    """Replacement for `dgl.ops.copy_e_sum(graph, e)`.

    Aggregates edge features into dst-node buckets by summation. Shape:
    (E, *F) -> (num_nodes, *F).
    """
    dst = edge_index[1]
    out_shape = (num_nodes,) + tuple(edge_value.shape[1:])
    out = torch.zeros(out_shape, dtype=edge_value.dtype, device=edge_value.device)
    idx = _expand_index_like(dst, edge_value)
    out.scatter_add_(0, idx, edge_value)
    return out


def has_edges_between(
    edge_index: Tensor, src: Tensor, dst: Tensor, num_nodes: int
) -> Tensor:
    """For each (src[i], dst[i]) pair, return bool whether that edge exists in edge_index.

    Matches DGL's `graph.has_edges_between(src, dst)`. Returns shape (len(src),).
    """
    key = edge_index[0].to(torch.long) * num_nodes + edge_index[1].to(torch.long)
    query = src.to(torch.long) * num_nodes + dst.to(torch.long)
    return torch.isin(query, key)


def edge_ids(
    edge_index: Tensor, src: Tensor, dst: Tensor, num_nodes: int
) -> Tensor:
    """Return the edge index for each given (src, dst) pair. Each pair must exist exactly
    once in edge_index (matching DGL's `graph.edge_ids(src, dst)` default semantics).
    """
    key = edge_index[0].to(torch.long) * num_nodes + edge_index[1].to(torch.long)
    query = src.to(torch.long) * num_nodes + dst.to(torch.long)
    sorter = torch.argsort(key)
    key_sorted = key[sorter]
    positions = torch.searchsorted(key_sorted, query)
    return sorter[positions]


def subgraph(g: Graph, node_idx: Tensor) -> Graph:
    """Node-induced subgraph. Matches `dgl.subgraph(nodes)` semantics (relabelled nodes,
    edge features gathered by the surviving edges).
    """
    from torch_geometric.utils import subgraph as _pyg_subgraph

    node_idx = node_idx.to(g.edge_index.device)
    edge_index_new, _, edge_mask = _pyg_subgraph(
        node_idx,
        g.edge_index,
        num_nodes=g.num_nodes,
        relabel_nodes=True,
        return_edge_mask=True,
    )
    num_nodes_new = int(node_idx.numel())
    node_new = {k: v[node_idx] for k, v in g.node.items()}
    edge_new = {k: v[edge_mask] for k, v in g.edge.items()}
    return Graph(
        pos=g.pos[node_idx],
        edge_index=edge_index_new,
        num_nodes=num_nodes_new,
        node=node_new,
        edge=edge_new,
        batch=None if g.batch is None else g.batch[node_idx],
    )


def batch_graphs(graphs: List[Graph]) -> Graph:
    """Concatenate Graphs into a single Graph with offset edge_index."""
    if len(graphs) == 1:
        g = graphs[0]
        if g.batch is None:
            g = Graph(
                pos=g.pos,
                edge_index=g.edge_index,
                num_nodes=g.num_nodes,
                node=g.node,
                edge=g.edge,
                batch=torch.zeros(g.num_nodes, dtype=torch.long, device=g.pos.device),
            )
        return g

    pos = torch.cat([g.pos for g in graphs], dim=0)
    offsets = torch.tensor([0] + [g.num_nodes for g in graphs[:-1]]).cumsum(0)
    edge_index = torch.cat(
        [g.edge_index + off for g, off in zip(graphs, offsets)], dim=1
    )
    num_nodes = int(sum(g.num_nodes for g in graphs))
    batch = torch.cat(
        [torch.full((g.num_nodes,), i, dtype=torch.long) for i, g in enumerate(graphs)]
    )

    node: Dict[str, Tensor] = {}
    for key in graphs[0].node:
        node[key] = torch.cat([g.node[key] for g in graphs], dim=0)
    edge: Dict[str, Tensor] = {}
    for key in graphs[0].edge:
        edge[key] = torch.cat([g.edge[key] for g in graphs], dim=0)

    return Graph(
        pos=pos,
        edge_index=edge_index,
        num_nodes=num_nodes,
        node=node,
        edge=edge,
        batch=batch.to(pos.device),
    )


def batch_size(g: Graph) -> int:
    """Number of graphs bundled into `g`. Matches `batch.batch_size` in DGL."""
    if g.batch is None:
        return 1
    if g.batch.numel() == 0:
        return 0
    return int(g.batch.max().item()) + 1


def slice_batch(g: Graph, batch_index: int, store_ids: bool = False) -> Graph:
    """Extract one graph from a batched Graph. Matches `dgl.slice_batch(g, i, store_ids=...)`.

    When `store_ids=True`, the per-node index into the original batched graph is saved under
    `node["_ID"]` (mirrors DGL's store_ids behaviour).
    """
    if g.batch is None:
        if batch_index != 0:
            raise IndexError(f"slice_batch: requested index {batch_index} on unbatched graph")
        out = Graph(
            pos=g.pos,
            edge_index=g.edge_index,
            num_nodes=g.num_nodes,
            node=dict(g.node),
            edge=dict(g.edge),
            batch=None,
        )
        if store_ids:
            out.node["_ID"] = torch.arange(g.num_nodes, device=g.pos.device)
        return out

    node_ids = (g.batch == batch_index).nonzero(as_tuple=False).squeeze(-1)
    out = subgraph(g, node_ids)
    out.batch = None
    if store_ids:
        out.node["_ID"] = node_ids
    return out


def unbatch_graphs(g: Graph) -> List[Graph]:
    """Inverse of batch_graphs. Replacement for `dgl.unbatch(batch)`."""
    if g.batch is None:
        return [g]

    num_graphs = int(g.batch.max().item()) + 1
    out: List[Graph] = []
    for i in range(num_graphs):
        node_mask = g.batch == i
        node_ids = node_mask.nonzero(as_tuple=False).squeeze(-1)
        remap = torch.full((g.num_nodes,), -1, dtype=torch.long, device=g.pos.device)
        remap[node_ids] = torch.arange(node_ids.numel(), device=g.pos.device)
        src, dst = g.edge_index[0], g.edge_index[1]
        edge_mask = node_mask[src] & node_mask[dst]
        sub_edge_index = torch.stack([remap[src[edge_mask]], remap[dst[edge_mask]]], dim=0)

        out.append(
            Graph(
                pos=g.pos[node_ids],
                edge_index=sub_edge_index,
                num_nodes=int(node_ids.numel()),
                node={k: v[node_ids] for k, v in g.node.items()},
                edge={k: v[edge_mask] for k, v in g.edge.items()},
                batch=None,
            )
        )
    return out
