"""TCP-Graph-JEPA — graph-based JEPA anomaly detection for EEG (TUAB).

Treat the 22 TCP bipolar derivations as graph nodes; pretrain a masked-latent
JEPA on normal EEG; score anomalies by latent unpredictability. Additive to the
existing two-view SIGReg/PEIRA EEG-JEPA track (nothing here modifies it).

Sub-modules:
  * ``tcp_graph`` — montage graph (shared-electrode + contralateral + self-loops)
  * ``features``  — log-bandpower [C,T,F] features + train-only normalisation
  * ``windows``   — EDF/tensor/synthetic [C,T,F] windowed dataset
  * ``masking``   — channel-time masking (random / channel / contralateral)
  * ``model``     — TCPGraphJEPA (temporal + dense relational graph + EMA target)
  * ``scoring``   — per-window/file anomaly scores + channel-time heatmaps
  * ``metrics``   — AUROC/AUPRC/balanced-acc/F1 + threshold selection
"""
from .tcp_graph import (TCP_CHANNELS, build_dense_adjacency, build_tcp_graph,
                        graph_metadata)
from .features import DEFAULT_BANDS, FeatureConfig, FeatureStats, log_bandpower
from .masking import MaskConfig, make_mask
from .model import ModelConfig, TCPGraphJEPA, build_model
from .scoring import ScoringConfig, score_recording, top_anomalies
from .windows import GraphEEGConfig, GraphEEGDataset, make_graph_loader

__all__ = [
    "TCP_CHANNELS", "build_tcp_graph", "build_dense_adjacency", "graph_metadata",
    "FeatureConfig", "FeatureStats", "log_bandpower", "DEFAULT_BANDS",
    "MaskConfig", "make_mask",
    "ModelConfig", "TCPGraphJEPA", "build_model",
    "ScoringConfig", "score_recording", "top_anomalies",
    "GraphEEGConfig", "GraphEEGDataset", "make_graph_loader",
]
