"""Riemannian geometry EEG classifiers for covariance-based decoding.

Pipeline:
  1. Read recording windows from ``EEGDataset``.
  2. Estimate one channel covariance matrix per window.
  3. Aggregate each recording's windows into one SPD matrix.
  4. Classify on the SPD manifold with either:
       * MDM: minimum distance to class Riemannian means.
       * tangent-logreg: AIRM/log-Euclidean tangent features + logistic probe.

The implementation is intentionally usable without pyRiemann at runtime for the
main path: covariance estimation uses scikit-learn OAS/Ledoit-Wolf/SCM, and the
SPD geometry below implements the AIRM Karcher mean, distance, and tangent map
directly. pyRiemann is still supported as a fallback for extra covariance
estimators.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


METRIC_ALIASES = {
    "airm": "riemann",
    "riemannian": "riemann",
    "le": "logeuclid",
    "log-euclid": "logeuclid",
    "log_euclid": "logeuclid",
    "euclidean": "euclid",
}


def parse_classes(raw: str | Iterable[str] | None) -> list[str] | None:
    if not raw:
        return None
    if isinstance(raw, str):
        return [name.strip() for name in raw.split(",") if name.strip()]
    return [str(name) for name in raw]


def normalize_metric(metric: str) -> str:
    metric = str(metric).lower().strip()
    return METRIC_ALIASES.get(metric, metric)


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def eig_floor(A: np.ndarray, rel_floor: float = 1e-10) -> float:
    scale = float(np.trace(sym(A)) / A.shape[-1])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return max(float(rel_floor), float(rel_floor) * scale)


def sym_func(A: np.ndarray, fn, eig_min: float | None = None) -> np.ndarray:
    vals, vecs = np.linalg.eigh(sym(A))
    if eig_min is not None:
        vals = np.maximum(vals, eig_min)
    out = (vecs * fn(vals)) @ vecs.T
    return sym(out)


def nearest_spd(A: np.ndarray, rel_floor: float = 1e-10) -> np.ndarray:
    """Symmetrize and eigenvalue-floor a matrix so downstream log/sqrt is safe."""
    return sym_func(A, lambda x: x, eig_min=eig_floor(A, rel_floor))


def logm_spd(A: np.ndarray, rel_floor: float = 1e-10) -> np.ndarray:
    return sym_func(A, np.log, eig_min=eig_floor(A, rel_floor))


def expm_sym(A: np.ndarray) -> np.ndarray:
    return sym_func(A, lambda x: np.exp(np.clip(x, -50.0, 50.0)))


def powm_spd(A: np.ndarray, power: float, rel_floor: float = 1e-10) -> np.ndarray:
    return sym_func(A, lambda x: np.power(x, power), eig_min=eig_floor(A, rel_floor))


def _normalise_weights(n: int, sample_weight: np.ndarray | None) -> np.ndarray:
    if sample_weight is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"sample_weight shape must be ({n},), got {w.shape}")
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("sample_weight must have a positive finite sum")
    return w / total


def euclidean_mean(covs: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    covs = np.asarray(covs, dtype=np.float64)
    w = _normalise_weights(len(covs), sample_weight)
    return nearest_spd(np.tensordot(w, covs, axes=(0, 0)))


def logeuclid_mean(covs: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    covs = np.asarray(covs, dtype=np.float64)
    w = _normalise_weights(len(covs), sample_weight)
    mean_log = np.zeros_like(covs[0], dtype=np.float64)
    for weight, cov in zip(w, covs):
        mean_log += weight * logm_spd(cov)
    return nearest_spd(expm_sym(mean_log))


def riemann_mean(
    covs: np.ndarray,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> np.ndarray:
    """Affine-invariant Riemannian (Karcher/Frechet) mean of SPD matrices."""
    covs = np.asarray(covs, dtype=np.float64)
    if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
        raise ValueError(f"expected [n, c, c] SPD matrices, got {covs.shape}")
    if len(covs) == 1:
        return nearest_spd(covs[0])

    w = _normalise_weights(len(covs), sample_weight)
    mean = logeuclid_mean(covs, w)
    for _ in range(max_iter):
        sqrt_mean = powm_spd(mean, 0.5)
        invsqrt_mean = powm_spd(mean, -0.5)
        update = np.zeros_like(mean)
        for weight, cov in zip(w, covs):
            whitened = nearest_spd(invsqrt_mean @ cov @ invsqrt_mean)
            update += weight * logm_spd(whitened)
        if float(np.linalg.norm(update, ord="fro")) < tol:
            break
        mean = nearest_spd(sqrt_mean @ expm_sym(update) @ sqrt_mean)
    return mean


def mean_spd(
    covs: np.ndarray,
    metric: str = "riemann",
    sample_weight: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> np.ndarray:
    metric = normalize_metric(metric)
    if metric == "riemann":
        return riemann_mean(covs, sample_weight=sample_weight, max_iter=max_iter, tol=tol)
    if metric == "logeuclid":
        return logeuclid_mean(covs, sample_weight=sample_weight)
    if metric == "euclid":
        return euclidean_mean(covs, sample_weight=sample_weight)
    raise ValueError(f"unknown SPD mean metric: {metric!r}")


def distance_spd(A: np.ndarray, B: np.ndarray, metric: str = "riemann") -> float:
    metric = normalize_metric(metric)
    A, B = nearest_spd(A), nearest_spd(B)
    if metric == "riemann":
        invsqrt_A = powm_spd(A, -0.5)
        evals = np.linalg.eigvalsh(nearest_spd(invsqrt_A @ B @ invsqrt_A))
        return float(np.linalg.norm(np.log(np.maximum(evals, eig_floor(B)))))
    if metric == "logeuclid":
        return float(np.linalg.norm(logm_spd(A) - logm_spd(B), ord="fro"))
    if metric == "euclid":
        return float(np.linalg.norm(A - B, ord="fro"))
    raise ValueError(f"unknown SPD distance metric: {metric!r}")


def upper_tri_vec(S: np.ndarray) -> np.ndarray:
    """Vectorize symmetric matrices with sqrt(2) off-diagonal isometry weight."""
    S = np.asarray(S)
    d = S.shape[-1]
    idx = np.triu_indices(d)
    out = S[..., idx[0], idx[1]].copy()
    weights = np.ones(len(idx[0]), dtype=out.dtype)
    weights[idx[0] != idx[1]] = np.sqrt(2.0)
    return out * weights


def tangent_space(
    covs: np.ndarray,
    reference: np.ndarray | None = None,
    metric: str = "riemann",
    reference_metric: str = "riemann",
    max_iter: int = 50,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Project SPD matrices to tangent vectors at ``reference``.

    For the AIRM path, the vectorized object is log(R^-1/2 C R^-1/2), which is
    the identity-transported tangent representation used by pyRiemann-style
    tangent-space classifiers.
    """
    covs = np.asarray(covs, dtype=np.float64)
    if reference is None:
        reference = mean_spd(covs, reference_metric, max_iter=max_iter, tol=tol)
    reference = nearest_spd(reference)
    metric = normalize_metric(metric)

    if metric == "riemann":
        invsqrt_ref = powm_spd(reference, -0.5)
        mats = [logm_spd(nearest_spd(invsqrt_ref @ cov @ invsqrt_ref)) for cov in covs]
    elif metric == "logeuclid":
        log_ref = logm_spd(reference)
        mats = [logm_spd(cov) - log_ref for cov in covs]
    elif metric == "euclid":
        mats = [nearest_spd(cov) - reference for cov in covs]
    else:
        raise ValueError(f"unknown tangent metric: {metric!r}")
    return upper_tri_vec(np.stack(mats)), reference


def estimate_covariances(
    windows: np.ndarray,
    estimator: str = "oas",
    regularization: float = 1e-6,
) -> np.ndarray:
    """Estimate one channel covariance per window from [n_windows, channels, time]."""
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3:
        raise ValueError(f"expected [n_windows, channels, time], got {windows.shape}")
    name = estimator.lower().strip()

    if name in {"scm", "sample", "cov"}:
        x = windows - windows.mean(axis=-1, keepdims=True)
        denom = max(1, x.shape[-1] - 1)
        covs = x @ np.swapaxes(x, -1, -2) / denom
    elif name in {"oas", "oracle_approximating_shrinkage"}:
        from sklearn.covariance import OAS

        covs = [OAS(assume_centered=False).fit(win.T).covariance_ for win in windows]
        covs = np.stack(covs)
    elif name in {"lwf", "ledoit_wolf", "ledoit-wolf"}:
        from sklearn.covariance import LedoitWolf

        covs = [LedoitWolf(assume_centered=False).fit(win.T).covariance_ for win in windows]
        covs = np.stack(covs)
    else:
        try:
            from pyriemann.estimation import Covariances
        except ImportError as exc:
            raise ImportError(
                f"covariance estimator {estimator!r} requires pyRiemann; "
                "use scm/oas/lwf or install pyriemann"
            ) from exc
        covs = Covariances(estimator=estimator).transform(windows)

    eye = np.eye(covs.shape[-1], dtype=np.float64)
    covs = np.stack([nearest_spd(cov + regularization * eye) for cov in covs])
    return covs


def aggregate_covariances(
    covs: np.ndarray,
    method: str = "riemann",
    max_iter: int = 50,
    tol: float = 1e-7,
) -> np.ndarray:
    method = normalize_metric(method)
    return mean_spd(covs, metric=method, max_iter=max_iter, tol=tol)


def eeg_config(
    data_root: str | None = None,
    label_scheme: str = "tuab",
    class_names: str | Iterable[str] | None = None,
    n_windows: int | None = None,
    n_channels: int | None = None,
    sfreq: int | None = None,
    window_sec: float | None = None,
) -> EEGConfig:
    cfg = EEGConfig(mode="probe")
    if data_root:
        cfg.data_root = data_root
    cfg.label_scheme = label_scheme
    parsed_classes = parse_classes(class_names)
    if parsed_classes:
        cfg.class_names = parsed_classes
    if n_windows is not None:
        cfg.n_windows = int(n_windows)
    if n_channels is not None:
        cfg.n_channels = int(n_channels)
    if sfreq is not None:
        cfg.sfreq = int(sfreq)
    if window_sec is not None:
        cfg.window_sec = float(window_sec)
    return cfg


def recording_covariances(
    split: str,
    data_root: str | None = None,
    label_scheme: str = "tuab",
    class_names: str | Iterable[str] | None = None,
    n_windows: int | None = None,
    estimator: str = "oas",
    aggregation: str = "riemann",
    n_channels: int | None = None,
    sfreq: int | None = None,
    window_sec: float | None = None,
    cov_regularization: float = 1e-6,
    riemann_max_iter: int = 50,
    riemann_tol: float = 1e-7,
    return_label_names: bool = False,
    progress_every: int = 200,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]:
    """Build recording-level SPD covariances and labels for a split."""
    cfg = eeg_config(
        data_root=data_root,
        label_scheme=label_scheme,
        class_names=class_names,
        n_windows=n_windows,
        n_channels=n_channels,
        sfreq=sfreq,
        window_sec=window_sec,
    )
    cfg.split = split
    ds = EEGDataset(cfg)
    covs, ys = [], []
    for i in range(len(ds)):
        wins, label, ok = ds[i]
        if not ok:
            continue
        window_covs = estimate_covariances(
            wins.numpy(), estimator=estimator, regularization=cov_regularization
        )
        rec_cov = aggregate_covariances(
            window_covs, method=aggregation, max_iter=riemann_max_iter, tol=riemann_tol
        )
        covs.append(rec_cov)
        ys.append(int(label))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{split}] {i + 1}/{len(ds)}", flush=True)
    if not covs:
        raise RuntimeError(f"No readable recordings for split={split!r}")
    out = (np.stack(covs), np.asarray(ys, dtype=int))
    if return_label_names:
        return (*out, list(ds.label_names or []))
    return out


class RiemannianMDMClassifier:
    """Minimum distance to class mean classifier on SPD matrices."""

    def __init__(
        self,
        mean_metric: str = "riemann",
        distance_metric: str = "riemann",
        max_iter: int = 50,
        tol: float = 1e-7,
    ):
        self.mean_metric = mean_metric
        self.distance_metric = distance_metric
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.covmeans_ = np.stack(
            [
                mean_spd(
                    X[y == cls],
                    metric=self.mean_metric,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
                for cls in self.classes_
            ]
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(
            [
                [
                    distance_spd(cov, center, metric=self.distance_metric)
                    for center in self.covmeans_
                ]
                for cov in X
            ],
            dtype=np.float64,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self.transform(X)
        return self.classes_[np.argmin(distances, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances = self.transform(X)
        logits = -(distances ** 2)
        logits -= logits.max(axis=1, keepdims=True)
        proba = np.exp(logits)
        return proba / proba.sum(axis=1, keepdims=True)


@dataclass
class TangentLogRegModel:
    reference_: np.ndarray
    scaler_: object
    classifier_: object
    classes_: np.ndarray
    tangent_metric: str

    def _features(self, X: np.ndarray) -> np.ndarray:
        Z, _ = tangent_space(X, reference=self.reference_, metric=self.tangent_metric)
        return self.scaler_.transform(Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier_.predict(self._features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier_.predict_proba(self._features(X))


def fit_tangent_logreg(
    Ctr: np.ndarray,
    ytr: np.ndarray,
    mean_metric: str = "riemann",
    tangent_metric: str = "riemann",
    class_weight: str | dict | None = "balanced",
    max_iter: int = 3000,
    riemann_max_iter: int = 50,
    riemann_tol: float = 1e-7,
) -> TangentLogRegModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Ztr, reference = tangent_space(
        Ctr,
        metric=tangent_metric,
        reference_metric=mean_metric,
        max_iter=riemann_max_iter,
        tol=riemann_tol,
    )
    scaler = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
    clf.fit(scaler.transform(Ztr), ytr)
    return TangentLogRegModel(
        reference_=reference,
        scaler_=scaler,
        classifier_=clf,
        classes_=clf.classes_,
        tangent_metric=tangent_metric,
    )


def aligned_proba(classes: np.ndarray, raw_proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
    proba = np.zeros((raw_proba.shape[0], len(labels)), dtype=raw_proba.dtype)
    label_to_col = {int(label): i for i, label in enumerate(labels)}
    for src_col, cls in enumerate(classes):
        dst_col = label_to_col.get(int(cls))
        if dst_col is not None:
            proba[:, dst_col] = raw_proba[:, src_col]
    return proba


def safe_auroc(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: np.ndarray,
    scored_labels: np.ndarray | None = None,
) -> float | None:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    scored = {int(s) for s in (labels if scored_labels is None else scored_labels)}
    keep = [int(lab) for lab in labels if int(lab) in scored and (y_true == lab).sum() > 0]
    if len(keep) < 2:
        return None
    col_of = {int(lab): i for i, lab in enumerate(labels)}
    cols = [col_of[lab] for lab in keep]
    mask = np.isin(y_true, keep)
    y_k, p_k = y_true[mask], proba[np.ix_(mask, cols)]
    try:
        if len(keep) == 2:
            score = roc_auc_score((y_k == keep[1]).astype(int), p_k[:, 1])
        else:
            score = roc_auc_score(
                y_k, p_k, labels=keep, multi_class="ovr", average="macro"
            )
    except ValueError:
        return None
    return round(float(score), 4) if np.isfinite(score) else None


def score_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
    )

    labels = np.arange(int(max(y_true.max(), pred.max(), classes.max())) + 1)
    if label_names is None:
        label_names = [str(i) for i in labels]
    if len(label_names) < len(labels):
        label_names = [*label_names, *[str(i) for i in labels[len(label_names) :]]]

    aligned = aligned_proba(classes, proba, labels)
    per_class_recall = recall_score(
        y_true, pred, labels=labels, average=None, zero_division=0
    )
    return {
        "acc": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "f1": round(
            float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
            4,
        ),
        "auroc": safe_auroc(y_true, aligned, labels, scored_labels=classes),
        "classes": list(label_names[: len(labels)]),
        "n_eval": int(len(y_true)),
        "per_class_recall": {
            str(label_names[i]): round(float(per_class_recall[i]), 4)
            for i in range(len(labels))
        },
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }


def fit_score_riemannian(
    Ctr: np.ndarray,
    ytr: np.ndarray,
    Cev: np.ndarray,
    yev: np.ndarray,
    label_names: list[str] | None = None,
    classifier: str = "tangent-logreg",
    mean_metric: str = "riemann",
    distance_metric: str = "riemann",
    tangent_metric: str = "riemann",
    riemann_max_iter: int = 50,
    riemann_tol: float = 1e-7,
) -> dict:
    """Fit a Riemannian EEG classifier and return held-out metrics."""
    classifier = classifier.lower().strip().replace("_", "-")
    if classifier == "mdm":
        model = RiemannianMDMClassifier(
            mean_metric=mean_metric,
            distance_metric=distance_metric,
            max_iter=riemann_max_iter,
            tol=riemann_tol,
        ).fit(Ctr, ytr)
    elif classifier in {"tangent-logreg", "ts-logreg", "logreg"}:
        model = fit_tangent_logreg(
            Ctr,
            ytr,
            mean_metric=mean_metric,
            tangent_metric=tangent_metric,
            riemann_max_iter=riemann_max_iter,
            riemann_tol=riemann_tol,
        )
        classifier = "tangent-logreg"
    else:
        raise ValueError("classifier must be 'mdm' or 'tangent-logreg'")

    pred = model.predict(Cev)
    proba = model.predict_proba(Cev)
    metrics = score_predictions(yev, pred, proba, model.classes_, label_names)
    metrics.update(
        {
            "n_train": int(len(ytr)),
            "riemann_classifier": classifier,
            "mean_metric": normalize_metric(mean_metric),
            "distance_metric": normalize_metric(distance_metric),
            "tangent_metric": normalize_metric(tangent_metric),
        }
    )
    return metrics


def run_recording_riemannian(
    data_root: str | None = None,
    label_scheme: str = "tuab",
    class_names: str | Iterable[str] | None = None,
    train_split: str = "train",
    eval_split: str = "eval",
    n_windows: int | None = None,
    estimator: str = "oas",
    aggregation: str = "riemann",
    classifier: str = "tangent-logreg",
    mean_metric: str = "riemann",
    distance_metric: str = "riemann",
    tangent_metric: str = "riemann",
    n_channels: int | None = None,
    sfreq: int | None = None,
    window_sec: float | None = None,
    cov_regularization: float = 1e-6,
    riemann_max_iter: int = 50,
    riemann_tol: float = 1e-7,
) -> dict:
    """Read splits, compute recording SPD covariances, fit, and score."""
    print(
        f"[riemann] train covariances "
        f"(estimator={estimator}, aggregation={aggregation})...",
        flush=True,
    )
    Ctr, ytr, label_names = recording_covariances(
        train_split,
        data_root=data_root,
        label_scheme=label_scheme,
        class_names=class_names,
        n_windows=n_windows,
        estimator=estimator,
        aggregation=aggregation,
        n_channels=n_channels,
        sfreq=sfreq,
        window_sec=window_sec,
        cov_regularization=cov_regularization,
        riemann_max_iter=riemann_max_iter,
        riemann_tol=riemann_tol,
        return_label_names=True,
    )
    print("[riemann] eval covariances...", flush=True)
    Cev, yev = recording_covariances(
        eval_split,
        data_root=data_root,
        label_scheme=label_scheme,
        class_names=label_names,
        n_windows=n_windows,
        estimator=estimator,
        aggregation=aggregation,
        n_channels=n_channels,
        sfreq=sfreq,
        window_sec=window_sec,
        cov_regularization=cov_regularization,
        riemann_max_iter=riemann_max_iter,
        riemann_tol=riemann_tol,
    )
    metrics = fit_score_riemannian(
        Ctr,
        ytr,
        Cev,
        yev,
        label_names=label_names,
        classifier=classifier,
        mean_metric=mean_metric,
        distance_metric=distance_metric,
        tangent_metric=tangent_metric,
        riemann_max_iter=riemann_max_iter,
        riemann_tol=riemann_tol,
    )
    metrics["cov_estimator"] = estimator
    metrics["cov_aggregation"] = normalize_metric(aggregation)
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--data_root", dest="data_root")
    parser.add_argument("--label-scheme", choices=["tuab", "folders"], default="tuab")
    parser.add_argument(
        "--classes",
        help="comma-separated class folder order, e.g. normal,seizure,dementia",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument(
        "--cov-estimator",
        "--estimator",
        dest="estimator",
        default="oas",
        help="covariance estimator: oas, lwf, scm, or a pyRiemann estimator",
    )
    parser.add_argument(
        "--aggregation",
        default="riemann",
        choices=["riemann", "logeuclid", "euclid"],
        help="recording-level mean of window covariance matrices",
    )
    parser.add_argument(
        "--classifier",
        default="tangent-logreg",
        choices=["tangent-logreg", "mdm"],
        help="SPD classifier: tangent-space logistic regression or MDM",
    )
    parser.add_argument(
        "--mean-metric",
        default="riemann",
        choices=["riemann", "logeuclid", "euclid"],
        help="metric for class means and tangent reference",
    )
    parser.add_argument(
        "--distance-metric",
        default="riemann",
        choices=["riemann", "logeuclid", "euclid"],
        help="distance metric for MDM",
    )
    parser.add_argument(
        "--tangent-metric",
        default="riemann",
        choices=["riemann", "logeuclid", "euclid"],
        help="tangent map metric for tangent-logreg",
    )
    parser.add_argument("--cov-regularization", type=float, default=1e-6)
    parser.add_argument("--riemann-max-iter", type=int, default=50)
    parser.add_argument("--riemann-tol", type=float, default=1e-7)
    parser.add_argument("--n-channels", type=int, default=None)
    parser.add_argument("--sfreq", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(sys.argv[1:] if argv is None else argv)
    metrics = run_recording_riemannian(
        data_root=args.data_root,
        label_scheme=args.label_scheme,
        class_names=args.classes,
        train_split=args.train_split,
        eval_split=args.eval_split,
        n_windows=args.n_windows,
        estimator=args.estimator,
        aggregation=args.aggregation,
        classifier=args.classifier,
        mean_metric=args.mean_metric,
        distance_metric=args.distance_metric,
        tangent_metric=args.tangent_metric,
        n_channels=args.n_channels,
        sfreq=args.sfreq,
        window_sec=args.window_sec,
        cov_regularization=args.cov_regularization,
        riemann_max_iter=args.riemann_max_iter,
        riemann_tol=args.riemann_tol,
    )
    print("[riemann] held-out:", metrics)


if __name__ == "__main__":
    main()
