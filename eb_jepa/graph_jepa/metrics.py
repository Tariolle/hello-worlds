"""Anomaly-detection metrics + threshold selection for TCP-Graph-JEPA.

Scores are anomaly scores (higher = more abnormal); labels are 0=normal,
1=abnormal. When abnormal labels are available we report AUROC / AUPRC / balanced
accuracy / F1 / confusion matrix and pick a threshold by Youden's J. When only
normal data is available, ``normal_quantile_threshold`` gives a conformal-style
cutoff at a chosen quantile of the normal scores.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def normal_quantile_threshold(normal_scores, q: float = 0.95) -> float:
    """Conformal-style cutoff = ``q``-quantile of validation *normal* scores."""
    return float(np.quantile(np.asarray(normal_scores, dtype=np.float64), q))


def orient_scores(scores, labels):
    """Calibrate the anomaly-score polarity on labelled data.

    The naive JEPA prior is "anomaly = high latent error", but for TUAB the
    abnormal class turns out to be *more* latent-predictable (lower error), so the
    discriminative direction is inverted. We pick the single sign bit that makes
    "higher score = abnormal" from the labels and return
    ``(oriented_scores, direction, auroc_raw)`` where ``direction`` is +1 (native)
    or -1 (inverted). ``oriented`` has AUROC = max(auroc_raw, 1-auroc_raw).
    """
    from sklearn.metrics import roc_auc_score
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    finite = np.isfinite(scores)
    if finite.sum() == 0 or labels[finite].min() == labels[finite].max():
        return scores, 1, float("nan")
    auroc_raw = float(roc_auc_score(labels[finite], scores[finite]))
    direction = 1 if auroc_raw >= 0.5 else -1
    return scores * direction, direction, auroc_raw


def separation(auroc: float) -> float:
    """Direction-agnostic separability: ``max(auroc, 1 - auroc)``."""
    return float(max(auroc, 1.0 - auroc))


def youden_threshold(scores, labels) -> float:
    """Threshold maximising TPR - FPR (Youden's J) on a labelled set."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def evaluate(scores, labels, threshold: Optional[float] = None) -> Dict:
    """Full metric bundle. ``threshold`` defaults to Youden's J on this set."""
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, roc_auc_score)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    # drop any non-finite scores so one bad recording can't NaN-poison the run
    finite = np.isfinite(scores)
    n_dropped = int((~finite).sum())
    scores, labels = scores[finite], labels[finite]
    out: Dict = {"n": int(len(scores)), "n_pos": int(labels.sum()),
                 "n_neg": int((labels == 0).sum()), "n_dropped_nonfinite": n_dropped}
    two_class = out["n_pos"] > 0 and out["n_neg"] > 0
    if two_class:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
    else:
        out["auroc"] = float("nan"); out["auprc"] = float("nan")

    if threshold is None and two_class:
        threshold = youden_threshold(scores, labels)
    out["threshold"] = float(threshold) if threshold is not None else float("nan")

    if threshold is not None:
        pred = (scores >= threshold).astype(int)
        out["balanced_accuracy"] = float(balanced_accuracy_score(labels, pred)) \
            if two_class else float("nan")
        out["f1"] = float(f1_score(labels, pred, zero_division=0))
        cm = confusion_matrix(labels, pred, labels=[0, 1])
        out["confusion_matrix"] = cm.tolist()
        tn, fp, fn, tp = cm.ravel()
        out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return out
