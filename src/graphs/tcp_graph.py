"""TCP bipolar montage graph construction.

The 22 TCP bipolar derivations are graph nodes. Edges encode local anatomical
continuity through shared electrodes, explicit left/right homologous pairs, and
optional self-loops for dense message passing layers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


TCP_CHANNELS = [
    "Fp1-F7", "F7-T3", "T3-T5", "T5-O1",
    "Fp2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3", "T3-C3", "C3-Cz", "Cz-C4", "C4-T4", "T4-A2",
    "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
    "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
]

CONTRALATERAL_PAIRS = [
    ("Fp1-F7", "Fp2-F8"),
    ("F7-T3", "F8-T4"),
    ("T3-T5", "T4-T6"),
    ("T5-O1", "T6-O2"),
    ("Fp1-F3", "Fp2-F4"),
    ("F3-C3", "F4-C4"),
    ("C3-P3", "C4-P4"),
    ("P3-O1", "P4-O2"),
    ("T3-C3", "C4-T4"),
    ("C3-Cz", "Cz-C4"),
]

EDGE_SHARED = 0
EDGE_CONTRALATERAL = 1
EDGE_SELF = 2


@dataclass(frozen=True)
class TCPGraph:
    channels: list[str]
    edge_index: torch.LongTensor
    edge_type: torch.LongTensor


def _electrodes(channel: str) -> tuple[str, str]:
    parts = [p.strip() for p in channel.split("-")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"TCP channel must look like 'A-B', got {channel!r}")
    return parts[0], parts[1]


def _add_directed(edges: list[tuple[int, int, int]], src: int, dst: int, typ: int) -> None:
    edges.append((src, dst, typ))


def _add_undirected(edges: list[tuple[int, int, int]], a: int, b: int, typ: int) -> None:
    _add_directed(edges, a, b, typ)
    _add_directed(edges, b, a, typ)


def build_tcp_graph(
    channels: list[str] | None = None,
    include_self_loops: bool = True,
    undirected: bool = True,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Return ``(edge_index, edge_type)`` for the TCP channel graph.

    ``edge_index`` follows the PyG convention ``LongTensor[2, num_edges]`` even
    when PyG is not installed. ``edge_type`` uses 0=shared-electrode,
    1=contralateral, and 2=self-loop.
    """
    channels = list(TCP_CHANNELS if channels is None else channels)
    index = {name: i for i, name in enumerate(channels)}
    electrodes = [set(_electrodes(ch)) for ch in channels]
    edges: list[tuple[int, int, int]] = []

    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            if electrodes[i] & electrodes[j]:
                if undirected:
                    _add_undirected(edges, i, j, EDGE_SHARED)
                else:
                    _add_directed(edges, i, j, EDGE_SHARED)

    for left, right in CONTRALATERAL_PAIRS:
        if left in index and right in index:
            if undirected:
                _add_undirected(edges, index[left], index[right], EDGE_CONTRALATERAL)
            else:
                _add_directed(edges, index[left], index[right], EDGE_CONTRALATERAL)

    if include_self_loops:
        for i in range(len(channels)):
            _add_directed(edges, i, i, EDGE_SELF)

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        return edge_index, edge_type

    edge_index = torch.tensor([[src, dst] for src, dst, _ in edges], dtype=torch.long).T
    edge_type = torch.tensor([typ for _, _, typ in edges], dtype=torch.long)
    return edge_index, edge_type


def dense_adjacency(
    channels: list[str] | None = None,
    include_self_loops: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Build a dense channel adjacency matrix for plain PyTorch graph layers."""
    channels = list(TCP_CHANNELS if channels is None else channels)
    edge_index, _edge_type = build_tcp_graph(
        channels=channels, include_self_loops=include_self_loops, undirected=True
    )
    adj = torch.zeros((len(channels), len(channels)), dtype=torch.float32)
    if edge_index.numel() > 0:
        adj[edge_index[1], edge_index[0]] = 1.0
    if normalize:
        degree = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        adj = adj / degree
    return adj


def graph_metadata(channels: list[str] | None = None) -> dict:
    channels = list(TCP_CHANNELS if channels is None else channels)
    edge_index, edge_type = build_tcp_graph(channels)
    return {
        "channels": channels,
        "edge_index": edge_index.tolist(),
        "edge_type": edge_type.tolist(),
        "edge_type_names": {
            str(EDGE_SHARED): "shared_electrode",
            str(EDGE_CONTRALATERAL): "contralateral",
            str(EDGE_SELF): "self_loop",
        },
    }
