import torch

from src.eval.anomaly_scoring import aggregate_scores, score_windows
from src.models.tcp_graph_jepa import TCPGraphJEPA, TCPGraphJEPAConfig


def test_anomaly_scoring_shapes_and_injected_region():
    torch.manual_seed(0)
    model = TCPGraphJEPA(
        TCPGraphJEPAConfig(
            channels=22,
            time_steps=20,
            feature_dim=5,
            hidden_dim=16,
            temporal_layers=1,
            temporal_heads=4,
            graph_layers=1,
            dropout=0.0,
            mask_ratio=0.25,
        )
    )
    x = torch.randn(1, 22, 20, 5) * 0.1
    x[:, 4, 8:12, :] += 10.0
    mask = torch.zeros(1, 22, 20, dtype=torch.bool)
    mask[:, 4, 8:12] = True
    out = score_windows(model, x, explicit_mask=mask)
    heat = out["heatmap"]
    assert heat.shape == (1, 22, 20)
    assert out["window_score"].shape == (1,)
    injected = heat[0, 4, 8:12].mean()
    background = aggregate_scores(heat[0], method="mean")
    assert injected > background
