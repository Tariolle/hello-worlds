"""Tests for the TCP montage graph construction."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from eb_jepa.graph_jepa.tcp_graph import (CONTRALATERAL_PAIRS, EDGE_CONTRA,
                                          EDGE_SELF, EDGE_SHARED, TCP_CHANNELS,
                                          build_dense_adjacency, build_tcp_graph,
                                          graph_metadata, split_electrodes)


def test_channel_count_and_split():
    assert len(TCP_CHANNELS) == 22
    assert split_electrodes("Fp1-F7") == ("Fp1", "F7")
    assert split_electrodes("C3-Cz") == ("C3", "Cz")


def test_edge_index_shapes_and_symmetry():
    edge_index, edge_type = build_tcp_graph(add_self_loops=True)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_type.shape[0]
    assert edge_index.dtype == torch.long
    # every directed edge has its reverse (undirected graph)
    pairs = {(int(i), int(j)) for i, j in zip(edge_index[0], edge_index[1])}
    for i, j in pairs:
        assert (j, i) in pairs, f"missing reverse edge for {(i, j)}"
    # edge types are valid
    assert set(int(t) for t in edge_type) <= {EDGE_SHARED, EDGE_CONTRA, EDGE_SELF}


def test_self_loops():
    edge_index, edge_type = build_tcp_graph(add_self_loops=True)
    self_edges = [(int(i), int(j)) for i, j, t in
                  zip(edge_index[0], edge_index[1], edge_type) if int(t) == EDGE_SELF]
    assert len(self_edges) == 22
    assert all(i == j for i, j in self_edges)
    # and absent when disabled
    _, et2 = build_tcp_graph(add_self_loops=False)
    assert (et2 == EDGE_SELF).sum() == 0


def test_shared_electrode_edges():
    idx = {c: i for i, c in enumerate(TCP_CHANNELS)}
    edge_index, edge_type = build_tcp_graph()
    edges = {(int(i), int(j)): int(t)
             for i, j, t in zip(edge_index[0], edge_index[1], edge_type)}
    # Fp1-F7 and F7-T3 share F7 -> shared edge
    assert edges.get((idx["Fp1-F7"], idx["F7-T3"])) == EDGE_SHARED
    assert edges.get((idx["T3-T5"], idx["T5-O1"])) == EDGE_SHARED
    # Fp1-F7 and T5-O1 share nothing -> no edge
    assert (idx["Fp1-F7"], idx["T5-O1"]) not in edges


def test_contralateral_edges():
    idx = {c: i for i, c in enumerate(TCP_CHANNELS)}
    edge_index, edge_type = build_tcp_graph()
    edges = {(int(i), int(j)): int(t)
             for i, j, t in zip(edge_index[0], edge_index[1], edge_type)}
    for a, b in CONTRALATERAL_PAIRS:
        assert edges.get((idx[a], idx[b])) == EDGE_CONTRA, f"{a}<->{b} missing"
        assert edges.get((idx[b], idx[a])) == EDGE_CONTRA


def test_dense_adjacency():
    adj = build_dense_adjacency(normalize=True)
    for key in ("shared", "contra", "self", "combined"):
        assert adj[key].shape == (22, 22)
        assert torch.allclose(adj[key], adj[key].T, atol=1e-5), f"{key} not symmetric"
    # self adjacency is exactly the identity support
    assert torch.count_nonzero(adj["self"]) == 22
    # un-normalised combined has finite, non-negative entries
    raw = build_dense_adjacency(normalize=False)["combined"]
    assert (raw >= 0).all()


def test_metadata():
    meta = graph_metadata()
    assert meta["n_nodes"] == 22
    assert meta["n_self"] == 22
    assert meta["n_contra"] == 2 * len(CONTRALATERAL_PAIRS)
    assert meta["n_shared"] > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("TCP_GRAPH_TESTS_OK")
