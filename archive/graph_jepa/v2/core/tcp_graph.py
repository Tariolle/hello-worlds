"""TCP bipolar montage as a graph — TCP-Graph-JEPA.

The 22 TCP (Temporal-Central-Parasagittal) bipolar derivations of the TUH
``01_tcp_ar`` montage are treated as the nodes of a graph. Edges encode the two
priors a clinician uses when reading a scalp EEG:

  * **shared-electrode** edges — two derivations that share a physical electrode
    are spatially adjacent on the scalp (e.g. ``Fp1-F7`` and ``F7-T3`` share F7);
  * **contralateral** ("brain-twin") edges — left/right homologous derivations
    (e.g. ``Fp1-F7`` and ``Fp2-F8``); abnormality is often a *break* of this
    left-right symmetry, so an explicit edge lets the model predict one side
    from the other;
  * **self-loops** — optional, so a graph layer can keep a node's own state.

Each TCP channel name ``"A-B"`` literally encodes its electrode pair, so the
referential electrodes are recovered by splitting on ``"-"`` (``"Fp1-F7"`` ->
anode ``Fp1``, cathode ``F7``). The EDF label for an electrode follows the TUH
convention ``"EEG {NAME}-REF"`` (see ``edf_label``).

No PyTorch-Geometric dependency: ``build_tcp_graph`` returns a plain
``edge_index`` LongTensor (PyG-compatible) and ``build_dense_adjacency`` returns
dense per-relation adjacency matrices for the in-repo dense message passing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

# Canonical 22-channel TCP bipolar montage (TUH 01_tcp_ar). Order is fixed and
# is the node order everywhere downstream (features, masking, heatmaps).
TCP_CHANNELS: List[str] = [
    "Fp1-F7", "F7-T3", "T3-T5", "T5-O1",
    "Fp2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3", "T3-C3", "C3-Cz", "Cz-C4", "C4-T4", "T4-A2",
    "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
    "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
]

# Explicit left<->right homologous pairs (the "brain-twin" edges).
CONTRALATERAL_PAIRS: List[Tuple[str, str]] = [
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

# Hemisphere grouping (used by contralateral masking). The central pair
# C3-Cz / Cz-C4 straddles the midline; we assign each to its lateral electrode.
LEFT_CHANNELS: List[str] = [
    "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "A1-T3", "T3-C3",
    "C3-Cz", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
]
RIGHT_CHANNELS: List[str] = [
    "Fp2-F8", "F8-T4", "T4-T6", "T6-O2", "T4-A2", "C4-T4",
    "Cz-C4", "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
]

# Edge-type codes.
EDGE_SHARED = 0
EDGE_CONTRA = 1
EDGE_SELF = 2


def split_electrodes(channel: str) -> Tuple[str, str]:
    """``"Fp1-F7"`` -> ``("Fp1", "F7")`` (anode, cathode)."""
    a, b = channel.split("-")
    return a, b


def electrode_set(channel: str) -> frozenset:
    return frozenset(split_electrodes(channel))


def edf_label(electrode: str, template: str = "EEG {E}-REF") -> str:
    """Electrode name -> EDF channel label (TUH AR convention, upper-cased)."""
    return template.format(E=electrode.upper())


def shared_electrode_pairs(channels: List[str]) -> List[Tuple[int, int]]:
    """Index pairs (i<j) of channels that share at least one electrode."""
    pairs = []
    sets = [electrode_set(c) for c in channels]
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            if sets[i] & sets[j]:
                pairs.append((i, j))
    return pairs


def build_tcp_graph(
    channels: Optional[List[str]] = None,
    add_self_loops: bool = True,
    contralateral: bool = True,
    shared: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the TCP montage graph.

    Returns ``(edge_index, edge_type)`` where ``edge_index`` is a
    ``LongTensor[2, num_edges]`` (both directions for every undirected edge) and
    ``edge_type`` is a ``LongTensor[num_edges]`` with values
    ``0=shared-electrode, 1=contralateral, 2=self-loop``.

    A pair that would qualify as both shared and contralateral keeps its first
    seen type (shared), and self-loops are added last; duplicates are dropped.
    """
    channels = channels or TCP_CHANNELS
    index = {c: i for i, c in enumerate(channels)}
    seen: Dict[Tuple[int, int], int] = {}

    def _add(i: int, j: int, etype: int, force: bool = False):
        key = (i, j)
        if force or key not in seen:
            seen[key] = etype

    if shared:
        for i, j in shared_electrode_pairs(channels):
            _add(i, j, EDGE_SHARED)
            _add(j, i, EDGE_SHARED)
    if contralateral:
        # contralateral pairs are explicitly requested as brain-twin edges; when a
        # pair also shares an electrode (only C3-Cz <-> Cz-C4), the contralateral
        # label takes precedence.
        for a, b in CONTRALATERAL_PAIRS:
            if a in index and b in index:
                _add(index[a], index[b], EDGE_CONTRA, force=True)
                _add(index[b], index[a], EDGE_CONTRA, force=True)
    if add_self_loops:
        for i in range(len(channels)):
            _add(i, i, EDGE_SELF)

    if not seen:  # degenerate (e.g. shared=contra=self=False)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        return edge_index, edge_type

    edges = sorted(seen.items())  # deterministic ordering
    edge_index = torch.tensor([[i for (i, _), _ in edges],
                               [j for (_, j), _ in edges]], dtype=torch.long)
    edge_type = torch.tensor([t for _, t in edges], dtype=torch.long)
    return edge_index, edge_type


def _sym_norm(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric degree normalisation D^-1/2 (A) D^-1/2 of a dense adjacency."""
    deg = adj.sum(dim=1)
    dinv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    return dinv.unsqueeze(1) * adj * dinv.unsqueeze(0)


def build_dense_adjacency(
    channels: Optional[List[str]] = None,
    add_self_loops: bool = True,
    contralateral: bool = True,
    shared: bool = True,
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """Dense per-relation adjacency matrices for the in-repo message passing.

    Returns a dict with ``shared``, ``contra``, ``self`` keys, each a
    ``[C, C]`` float matrix (symmetrically normalised when ``normalize``), plus
    ``combined`` = normalised sum of the enabled relations. Self-loops are added
    to every relation's normalisation domain only via the ``self`` matrix; the
    ``shared``/``contra`` matrices are normalised on their own support.
    """
    channels = channels or TCP_CHANNELS
    C = len(channels)
    edge_index, edge_type = build_tcp_graph(
        channels, add_self_loops=add_self_loops, contralateral=contralateral,
        shared=shared)
    mats = {
        "shared": torch.zeros(C, C),
        "contra": torch.zeros(C, C),
        "self": torch.zeros(C, C),
    }
    name = {EDGE_SHARED: "shared", EDGE_CONTRA: "contra", EDGE_SELF: "self"}
    for k in range(edge_index.shape[1]):
        i, j, t = int(edge_index[0, k]), int(edge_index[1, k]), int(edge_type[k])
        mats[name[t]][i, j] = 1.0

    combined = mats["shared"] + mats["contra"] + mats["self"]
    if normalize:
        for key in ("shared", "contra", "self"):
            mats[key] = _sym_norm(mats[key])
        combined = _sym_norm(combined)
    mats["combined"] = combined
    return mats


def graph_metadata(channels: Optional[List[str]] = None) -> dict:
    """Lightweight, serialisable description saved into checkpoints."""
    channels = channels or TCP_CHANNELS
    edge_index, edge_type = build_tcp_graph(channels)
    return {
        "channels": list(channels),
        "n_nodes": len(channels),
        "n_edges": int(edge_index.shape[1]),
        "n_shared": int((edge_type == EDGE_SHARED).sum()),
        "n_contra": int((edge_type == EDGE_CONTRA).sum()),
        "n_self": int((edge_type == EDGE_SELF).sum()),
        "contralateral_pairs": [list(p) for p in CONTRALATERAL_PAIRS],
    }
