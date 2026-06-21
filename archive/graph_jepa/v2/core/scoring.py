"""Anomaly scoring for TCP-Graph-JEPA.

The anomaly signal is the model's own JEPA error: a region is anomalous when its
masked latent is *not predictable* from the surrounding spatial-temporal graph
context. For a window we mask many times so every channel-time position is hidden
in at least one pass, and average the per-position error into a dense
``[C, T]`` heatmap. Window scores aggregate the heatmap; file scores aggregate
window scores. All aggregations (``mean`` / ``max`` / ``top_k_mean``) are
configurable; the default is the mean of the top-10% errors, which is sensitive
to focal abnormality without being dominated by a single outlier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .masking import MaskConfig, make_mask
from .model import TCPGraphJEPA


@dataclass
class ScoringConfig:
    n_masks: int = 8                 # masking passes per window (heatmap coverage)
    mask_ratio: float = 0.5          # higher than training for fast full coverage
    mask_mode: str = "random"
    window_agg: str = "top_k_mean"   # mean | max | top_k_mean
    file_agg: str = "top_k_mean"     # mean | max | top_k_mean
    top_k_frac: float = 0.1
    seed: int = 0


def _aggregate(values: np.ndarray, agg: str, top_k_frac: float) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return float("nan")
    if agg == "mean":
        return float(values.mean())
    if agg == "max":
        return float(values.max())
    if agg == "top_k_mean":
        k = max(1, int(round(top_k_frac * values.size)))
        return float(np.sort(values)[-k:].mean())
    raise ValueError(f"unknown aggregation: {agg!r}")


@torch.no_grad()
def window_error_maps(model: TCPGraphJEPA, x: torch.Tensor,
                      channel_mask: torch.Tensor, cfg: ScoringConfig,
                      device=None) -> torch.Tensor:
    """``x: [B, C, T, F]`` -> dense error heatmaps ``[B, C, T]`` (avg over masks).

    Positions that happen never to be masked across the passes are filled with the
    per-sample mean of the observed (masked, available) errors.
    """
    model.eval()
    device = device or next(model.parameters()).device
    x = x.to(device)
    cm = channel_mask.to(device)
    B, C, T, _ = x.shape
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    err_sum = torch.zeros(B, C, T, device=device)
    cnt = torch.zeros(B, C, T, device=device)
    mcfg = MaskConfig(mode=cfg.mask_mode, mask_ratio=cfg.mask_ratio)
    for _ in range(cfg.n_masks):
        mask = make_mask(B, C, T, mcfg, channel_mask=cm, generator=gen,
                         device="cpu", channels=model.cfg.channels).to(device)
        out = model(x, mask, channel_mask=cm)
        v = out["valid"].float()
        err_sum += out["err"] * v
        cnt += v
    heat = err_sum / cnt.clamp_min(1.0)
    # fill never-masked positions with each sample's observed mean
    observed = cnt > 0
    for b in range(B):
        if observed[b].any():
            fill = heat[b][observed[b]].mean()
            heat[b][~observed[b]] = fill
    return heat.cpu()


def window_scores_from_heat(heat: torch.Tensor, channel_mask: torch.Tensor,
                            cfg: ScoringConfig) -> np.ndarray:
    """``heat: [B, C, T]`` -> ``[B]`` window scores (available positions only)."""
    heat = heat.cpu().numpy()
    cm = channel_mask.cpu().numpy().astype(bool)
    if cm.ndim == 1:
        cm = np.broadcast_to(cm, (heat.shape[0], heat.shape[1]))
    scores = np.empty(heat.shape[0], dtype=np.float64)
    for b in range(heat.shape[0]):
        vals = heat[b][cm[b]]              # [available_channels, T] -> flat
        scores[b] = _aggregate(vals, cfg.window_agg, cfg.top_k_frac)
    return scores


@torch.no_grad()
def score_recording(model: TCPGraphJEPA, x: torch.Tensor,
                    channel_mask: torch.Tensor, cfg: ScoringConfig,
                    device=None):
    """Score one recording's windows ``x: [N, C, T, F]``.

    Returns ``(file_score, window_scores[N], mean_heatmap[C, T])``.
    """
    cm = channel_mask
    if cm.dim() == 1:
        cm = cm.unsqueeze(0).expand(x.shape[0], -1)
    heat = window_error_maps(model, x, cm, cfg, device=device)        # [N,C,T]
    win = window_scores_from_heat(heat, cm, cfg)                      # [N]
    file_score = _aggregate(win, cfg.file_agg, cfg.top_k_frac)
    mean_heat = heat.mean(dim=0).numpy()                             # [C,T]
    return file_score, win, mean_heat


@torch.no_grad()
def score_file_loader(model: TCPGraphJEPA, loader, cfg: ScoringConfig,
                      device=None, max_files: Optional[int] = None):
    """Iterate a ``mode='file'`` loader -> ``(scores, labels, paths)`` arrays."""
    device = device or next(model.parameters()).device
    scores, labels, paths = [], [], []
    seen = 0
    for batch in loader:
        x, label, cm, ok, path = batch
        # x: [B, N, C, T, F]
        for b in range(x.shape[0]):
            # skip unreadable files and files with no constructible channels
            # (an all-False channel_mask would otherwise yield a NaN file score)
            if not bool(ok[b]) or not bool(cm[b].any()):
                continue
            fs, _, _ = score_recording(model, x[b], cm[b], cfg, device=device)
            if not np.isfinite(fs):
                continue
            scores.append(float(fs)); labels.append(int(label[b]))
            paths.append(path[b] if isinstance(path, (list, tuple)) else str(path[b]))
            seen += 1
            if max_files is not None and seen >= max_files:
                return np.asarray(scores), np.asarray(labels), paths
    return np.asarray(scores), np.asarray(labels), paths


def top_anomalies(mean_heat: np.ndarray, channels, frame_sec: float,
                  channel_mask: Optional[np.ndarray] = None, top_n: int = 10):
    """Rank channel-time cells of a ``[C, T]`` heatmap; return a list of dicts
    ``{channel, time_sec, score}`` for the top-N (skipping unavailable channels)."""
    C, T = mean_heat.shape
    cm = np.ones(C, bool) if channel_mask is None else np.asarray(channel_mask, bool)
    cand = []
    for c in range(C):
        if not cm[c]:
            continue
        for t in range(T):
            cand.append((float(mean_heat[c, t]), c, t))
    cand.sort(reverse=True)
    out = []
    for score, c, t in cand[:top_n]:
        out.append({"channel": channels[c], "channel_index": c,
                    "time_sec": round(t * frame_sec, 3), "score": score})
    return out
