"""Latent prediction-error anomaly scoring for TCP-Graph-JEPA."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def latent_error(pred: torch.Tensor, target: torch.Tensor, norm: str = "l1") -> torch.Tensor:
    diff = pred - target
    if norm == "l2":
        return diff.pow(2).sum(dim=-1).sqrt()
    if norm == "l1":
        return diff.abs().mean(dim=-1)
    raise ValueError(f"unknown error norm: {norm!r}")


@torch.no_grad()
def score_windows(
    model,
    x: torch.Tensor,
    n_masks: int = 8,
    mask_ratio: float | None = None,
    mask_mode: str = "random",
    error_norm: str = "l1",
    explicit_mask: torch.BoolTensor | None = None,
) -> dict:
    """Return channel-time heatmaps and window scores for ``x [B,C,T,F]``."""
    model.eval()
    x = x.to(next(model.parameters()).device)
    bsz, channels, time_steps, _feat = x.shape
    heat = torch.zeros(bsz, channels, time_steps, device=x.device)
    counts = torch.zeros_like(heat)

    masks = [explicit_mask.to(x.device)] if explicit_mask is not None else []
    while len(masks) < max(1, int(n_masks)):
        masks.append(
            model.make_mask(
                bsz,
                time_steps,
                x.device,
                mode=mask_mode,
                mask_ratio=mask_ratio,
            )
        )

    target = model.encode_target(x)
    for mask in masks:
        pred = model.predict_context(x, mask)
        err = latent_error(pred, target, norm=error_norm)
        heat += err * mask.float()
        counts += mask.float()
    heat = heat / counts.clamp_min(1.0)
    window_scores = aggregate_scores(heat, method="top_k_mean", top_k=0.10, dims=(1, 2))
    return {
        "heatmap": heat.detach().cpu(),
        "window_score": window_scores.detach().cpu(),
        "mask_count": counts.detach().cpu(),
    }


def aggregate_scores(
    scores: torch.Tensor,
    method: str = "top_k_mean",
    top_k: float = 0.10,
    dims: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Aggregate score tensors by mean, max, or top-k mean."""
    if dims is None:
        dims = tuple(range(scores.ndim))
    method = method.lower()
    if method == "mean":
        return scores.mean(dim=dims)
    if method == "max":
        return scores.amax(dim=dims)
    if method == "top_k_mean":
        moved = scores
        keep_dims = [d for d in range(scores.ndim) if d not in dims]
        perm = keep_dims + list(dims)
        moved = moved.permute(perm)
        base_shape = moved.shape[: len(keep_dims)]
        flat = moved.reshape(*base_shape, -1)
        k = max(1, int(round(flat.shape[-1] * float(top_k))))
        return flat.topk(k, dim=-1).values.mean(dim=-1)
    raise ValueError(f"unknown aggregation method: {method!r}")


def aggregate_file_scores(
    file_ids: list[str],
    window_scores: torch.Tensor | np.ndarray,
    method: str = "top_k_mean",
    top_k: float = 0.10,
) -> dict[str, float]:
    grouped = defaultdict(list)
    values = window_scores.detach().cpu().numpy() if torch.is_tensor(window_scores) else np.asarray(window_scores)
    for file_id, score in zip(file_ids, values):
        grouped[str(file_id)].append(float(score))
    out = {}
    for file_id, vals in grouped.items():
        arr = torch.tensor(vals, dtype=torch.float32)
        out[file_id] = float(aggregate_scores(arr, method=method, top_k=top_k).item())
    return out


def threshold_from_normal(scores, quantile: float = 0.95) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("cannot estimate threshold from empty scores")
    return float(np.quantile(scores, quantile))


def youden_threshold(y_true, scores) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def binary_metrics(y_true, scores, threshold: float | None = None) -> dict:
    """File-level normal/abnormal metrics. Label 1 is abnormal."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if threshold is None:
        threshold = youden_threshold(y_true, scores)
    pred = (scores >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
        metrics["auprc"] = float(average_precision_score(y_true, scores))
        _precision, _recall, _thr = precision_recall_curve(y_true, scores)
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None
    return metrics
