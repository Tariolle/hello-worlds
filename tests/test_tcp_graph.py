import torch

from src.graphs.tcp_graph import (
    EDGE_CONTRALATERAL,
    EDGE_SELF,
    EDGE_SHARED,
    TCP_CHANNELS,
    build_tcp_graph,
    dense_adjacency,
)


def _has_edge(edge_index, edge_type, channels, a, b, typ):
    idx = {name: i for i, name in enumerate(channels)}
    src, dst = idx[a], idx[b]
    for k in range(edge_index.shape[1]):
        if int(edge_index[0, k]) == src and int(edge_index[1, k]) == dst:
            if int(edge_type[k]) == typ:
                return True
    return False


def test_tcp_graph_contains_expected_edges():
    edge_index, edge_type = build_tcp_graph(TCP_CHANNELS)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_type.numel()
    assert _has_edge(edge_index, edge_type, TCP_CHANNELS, "Fp1-F7", "F7-T3", EDGE_SHARED)
    assert _has_edge(edge_index, edge_type, TCP_CHANNELS, "Fp1-F7", "Fp2-F8", EDGE_CONTRALATERAL)
    assert _has_edge(edge_index, edge_type, TCP_CHANNELS, "C3-Cz", "Cz-C4", EDGE_CONTRALATERAL)
    for i in range(len(TCP_CHANNELS)):
        assert ((edge_index[0] == i) & (edge_index[1] == i) & (edge_type == EDGE_SELF)).any()


def test_dense_adjacency_is_normalized():
    adj = dense_adjacency(TCP_CHANNELS)
    assert adj.shape == (22, 22)
    assert torch.allclose(adj.sum(dim=1), torch.ones(22))

