"""Shared checkpoint loading for the TCP-Graph-JEPA eval / visualisation scripts."""
import torch

from eb_jepa.graph_jepa.config import bind
from eb_jepa.graph_jepa.features import FeatureStats
from eb_jepa.graph_jepa.model import ModelConfig, build_model
from eb_jepa.graph_jepa.scoring import ScoringConfig
from eb_jepa.graph_jepa.windows import GraphEEGConfig


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
