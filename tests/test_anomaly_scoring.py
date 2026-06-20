"""Tests for anomaly scoring, aggregation, and metrics."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from eb_jepa.graph_jepa.masking import MaskConfig
from eb_jepa.graph_jepa.metrics import (evaluate, normal_quantile_threshold,
                                        youden_threshold)
from eb_jepa.graph_jepa.model import ModelConfig, build_model
from eb_jepa.graph_jepa.scoring import (ScoringConfig, score_recording,
                                        top_anomalies, window_error_maps,
                                        window_scores_from_heat)


def test_heatmap_and_score_shapes():
    B, C, T, F = 3, 22, 30, 5
    model = build_model(ModelConfig(feature_dim=F, hidden_dim=32, max_time=T,
                                    n_temporal_layers=1, dropout=0.0))
    x = torch.randn(B, C, T, F)
    cm = torch.ones(B, C, dtype=torch.bool)
    scfg = ScoringConfig(n_masks=4, mask_ratio=0.5)
    heat = window_error_maps(model, x, cm, scfg)
    assert heat.shape == (B, C, T)
    scores = window_scores_from_heat(heat, cm, scfg)
    assert scores.shape == (B,)
    fs, win, mh = score_recording(model, x, cm[0], scfg)
    assert win.shape == (B,) and mh.shape == (C, T) and np.isfinite(fs)


def test_aggregations_differ():
    heat = torch.zeros(1, 4, 10)
    heat[0, 0, 0] = 100.0                        # one large outlier
    cm = torch.ones(1, 4, dtype=torch.bool)
    mean_s = window_scores_from_heat(heat, cm, ScoringConfig(window_agg="mean"))[0]
    max_s = window_scores_from_heat(heat, cm, ScoringConfig(window_agg="max"))[0]
    topk = window_scores_from_heat(heat, cm,
                                   ScoringConfig(window_agg="top_k_mean", top_k_frac=0.1))[0]
    assert max_s == 100.0
    assert mean_s < topk <= max_s


def test_injected_anomaly_scores_higher():
    # train a tiny model on normal windows, then check the injected block lights up
    from examples.eeg.graph_jepa.smoke_graph_jepa import run
    ratio = run(steps=80, seed=0, verbose=False)
    assert ratio > 1.0, f"anomaly region not above average (ratio={ratio:.2f})"


def test_evaluate_drops_nonfinite_scores():
    scores = np.array([0.1, 0.2, np.nan, 0.9, 1.0, np.inf])
    labels = np.array([0, 0, 0, 1, 1, 1])
    m = evaluate(scores, labels)            # must not raise on NaN/Inf
    assert m["n_dropped_nonfinite"] == 2
    assert np.isfinite(m["auroc"])


def test_features_nan_inf_robust():
    from eb_jepa.graph_jepa.features import FeatureConfig, fit_feature_stats, log_bandpower
    raw = np.random.randn(22, 1400).astype("float32")
    raw[5, 500] = np.nan; raw[7, :] = np.inf
    feat = log_bandpower(raw, FeatureConfig(sfreq=200))
    assert np.isfinite(feat).all()
    batch = np.random.randn(4, 22, 70, 5).astype("float32"); batch[0, 3] = np.nan
    st = fit_feature_stats(batch)
    assert np.isfinite(st.mean).all() and np.isfinite(st.std).all()


def test_metrics_separable():
    scores = np.array([0.1, 0.2, 0.15, 0.9, 1.0, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    m = evaluate(scores, labels)
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert len(m["confusion_matrix"]) == 2


def test_threshold_helpers():
    normal = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    thr = normal_quantile_threshold(normal, q=0.95)
    assert 0.4 <= thr <= 0.5
    yj = youden_threshold(np.array([0.1, 0.2, 0.9, 1.0]), np.array([0, 0, 1, 1]))
    assert np.isfinite(yj)


def test_top_anomalies_skips_unavailable():
    C, T = 22, 10
    mh = np.zeros((C, T)); mh[3, 5] = 9.0; mh[8, 2] = 99.0   # ch 8 unavailable
    from eb_jepa.graph_jepa.tcp_graph import TCP_CHANNELS
    cm = np.ones(C, bool); cm[8] = False
    tops = top_anomalies(mh, TCP_CHANNELS, 0.1, channel_mask=cm, top_n=3)
    assert tops[0]["channel"] == TCP_CHANNELS[3]               # 8 skipped despite higher
    assert all(t["channel"] != TCP_CHANNELS[8] for t in tops)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("ANOMALY_SCORING_TESTS_OK")
