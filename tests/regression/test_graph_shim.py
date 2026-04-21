"""Self-consistency tests for `cg2all.lib.graph` — no DGL required.

These lock in the invariants the shim promises: per-dst softmax sums to 1,
copy_e_sum equals an explicit scatter, e_dot_v equals an explicit gather+dot,
and batch/unbatch round-trip. A separate cross-validation test against DGL's
own implementations lives in `test_graph_shim_vs_dgl.py` (runs only in the
legacy env).
"""
from __future__ import annotations

import pytest
import torch

from cg2all.lib.graph import (
    Graph,
    batch_graphs,
    copy_e_sum,
    e_dot_v,
    edge_softmax,
    unbatch_graphs,
)


def _toy_edge_index(num_nodes: int = 5) -> torch.Tensor:
    # Hand-picked edges covering self-loops, multi-in-degree, isolated dst.
    return torch.tensor(
        [
            [0, 1, 2, 2, 3, 4, 4],  # src
            [1, 2, 1, 3, 3, 3, 2],  # dst  (node 0 has no incoming; node 3 has 3 incoming)
        ],
        dtype=torch.long,
    )


def test_edge_softmax_sums_to_one_per_dst():
    torch.manual_seed(0)
    num_nodes = 5
    edge_index = _toy_edge_index(num_nodes)
    e = torch.randn(edge_index.shape[1])
    out = edge_softmax(e, edge_index, num_nodes)

    assert out.shape == e.shape
    # group by dst, sum must equal 1 for each dst that has incoming edges
    for dst in edge_index[1].unique().tolist():
        mask = edge_index[1] == dst
        assert torch.isclose(out[mask].sum(), torch.tensor(1.0), atol=1e-6), dst


def test_copy_e_sum_matches_explicit_scatter():
    num_nodes = 5
    edge_index = _toy_edge_index(num_nodes)
    e = torch.arange(edge_index.shape[1] * 4, dtype=torch.float32).view(-1, 4)

    got = copy_e_sum(e, edge_index, num_nodes)

    expected = torch.zeros(num_nodes, 4)
    for i, dst in enumerate(edge_index[1].tolist()):
        expected[dst] += e[i]
    assert torch.allclose(got, expected)


def test_e_dot_v_matches_gather_and_dot():
    num_nodes = 5
    edge_index = _toy_edge_index(num_nodes)
    num_heads, feat = 3, 7
    key = torch.randn(edge_index.shape[1], num_heads, feat)
    query = torch.randn(num_nodes, num_heads, feat)

    got = e_dot_v(key, query, edge_index)

    expected = (key * query[edge_index[1]]).sum(dim=-1, keepdim=True)
    assert torch.allclose(got, expected)


def test_batch_unbatch_roundtrip():
    g1 = Graph(
        pos=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        num_nodes=4,
        node={"feat": torch.randn(4, 2)},
        edge={"rel_pos": torch.randn(3, 3)},
    )
    g2 = Graph(
        pos=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
        num_nodes=3,
        node={"feat": torch.randn(3, 2)},
        edge={"rel_pos": torch.randn(2, 3)},
    )
    batched = batch_graphs([g1, g2])
    assert batched.num_nodes == 7
    assert batched.num_edges == 5
    assert batched.batch is not None and batched.batch.tolist() == [0, 0, 0, 0, 1, 1, 1]

    parts = unbatch_graphs(batched)
    assert len(parts) == 2
    assert torch.equal(parts[0].pos, g1.pos)
    assert torch.equal(parts[1].pos, g2.pos)
    assert torch.equal(parts[0].edge_index, g1.edge_index)
    assert torch.equal(parts[1].edge_index, g2.edge_index)
    assert torch.equal(parts[0].node["feat"], g1.node["feat"])
    assert torch.equal(parts[1].edge["rel_pos"], g2.edge["rel_pos"])


@pytest.mark.skipif(
    not pytest.importorskip("torch_cluster", reason="torch_cluster not installed"),
    reason="torch_cluster required for radius_graph",
)
def test_radius_graph_matches_brute_force():
    from cg2all.lib.graph import radius_graph

    torch.manual_seed(42)
    pos = torch.randn(20, 3)
    r = 1.0
    ei = radius_graph(pos, r=r, self_loop=False)

    # Brute-force reference
    dist = torch.cdist(pos, pos)
    mask = (dist <= r) & ~torch.eye(pos.shape[0], dtype=torch.bool)
    expected_pairs = set(zip(*torch.nonzero(mask, as_tuple=True)))
    expected_pairs = {(int(a), int(b)) for a, b in expected_pairs}

    got_pairs = {(int(s), int(d)) for s, d in zip(ei[0].tolist(), ei[1].tolist())}
    assert got_pairs == expected_pairs
