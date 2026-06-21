"""Shared checkpoint loading for the TCP-Graph-JEPA eval / visualisation scripts."""
import torch

from archive.graph_jepa.v2.core.config import bind
from archive.graph_jepa.v2.core.features import FeatureStats
from archive.graph_jepa.v2.core.model import ModelConfig, build_model
from archive.graph_jepa.v2.core.scoring import ScoringConfig
from archive.graph_jepa.v2.core.windows import GraphEEGConfig


def load_checkpoint(path, device=None):
    """Rebuild ``(model, data_cfg, scoring_cfg, stats)`` from a saved checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device, weights_only=False)
    model_cfg = bind(ModelConfig, state["model_cfg"])
    model = build_model(model_cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    data_cfg = bind(GraphEEGConfig, state["data_cfg"])
    scoring_cfg = bind(ScoringConfig, state.get("scoring_cfg", {}))
    stats = FeatureStats.from_dict(state["feature_stats"])
    return model, data_cfg, scoring_cfg, stats
