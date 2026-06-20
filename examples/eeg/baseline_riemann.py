"""Classical Riemannian baseline: covariance features on the SPD manifold.

Per recording: spatial channel covariance of each 10 s window -> Euclidean mean
-> one SPD matrix; project all recordings to the tangent space at their
Riemannian mean (pyRiemann) -> standardize -> logistic regression. On TUH
Abnormal this reaches ~0.86 accuracy with NO deep network (Gemein et al. 2020),
so it (a) sanity-checks the EDF data pipeline before any GPU run and (b) is the
complexity reference the JEPA frozen probe is measured against.

The baseline follows the same labelled dataset conventions as ``eval.py``:
TUAB binary folders by default, or arbitrary diagnosis folders via
``--label-scheme folders`` and optional ``--classes``.

Run:
  python -m examples.eeg.baseline_riemann --data-root <TUAB_PREPROCESSED>
  python -m examples.eeg.baseline_riemann --data-root <ROOT> \
      --label-scheme folders --classes normal,seizure,dementia
"""
import argparse
import sys

import numpy as np

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


def _parse_classes(raw):
    if not raw:
        return None
    if isinstance(raw, str):
        return [name.strip() for name in raw.split(",") if name.strip()]
    return [str(name) for name in raw]


def _cov_data_cfg(
    data_root=None,
    label_scheme="tuab",
    class_names=None,
    n_windows=None,
    n_channels=None,
    sfreq=None,
    window_sec=None,
):
    cfg = EEGConfig(mode="probe")
    if data_root:
        cfg.data_root = data_root
    cfg.label_scheme = label_scheme
    parsed_classes = _parse_classes(class_names)
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


def _recording_covariances(
    split,
    data_root=None,
    label_scheme="tuab",
    class_names=None,
    n_windows=None,
    estimator="oas",
    n_channels=None,
    sfreq=None,
    window_sec=None,
    return_label_names=False,
):
    """[N_rec, C, C] recording-level mean covariances + labels."""
    from pyriemann.estimation import Covariances

    cfg = _cov_data_cfg(
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
    est = Covariances(estimator=estimator)
    covs, ys = [], []
    for i in range(len(ds)):
        wins, label, ok = ds[i]               # [N, C, T], int, bool
        if not ok:
            continue
        c = est.transform(wins.numpy().astype(np.float64))  # [N, C, C]
        c = c.mean(axis=0)                    # recording-level mean covariance
        covs.append(0.5 * (c + c.T))          # guard against numerical asymmetry
        ys.append(int(label))
        if (i + 1) % 200 == 0:
            print(f"  [{split}] {i + 1}/{len(ds)}", flush=True)
    if not covs:
        raise RuntimeError(f"No readable recordings for split={split!r}")
    out = (np.stack(covs), np.array(ys))
    if return_label_names:
        return (*out, list(ds.label_names or []))
    return out


def _safe_auroc(y_true, proba, labels, scored_labels=None):
    """Macro one-vs-rest AUROC over classes that are both present in ``y_true``
    and learned by the classifier. A class the model never saw (all-zero score
    column) or with no eval support is dropped from both axes instead of
    silently dragging the macro mean toward chance; the score is finite-guarded
    so an undefined metric becomes ``None`` rather than a bare ``nan``. Returns
    ``None`` when fewer than two scorable classes remain."""
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
                y_k, p_k, labels=keep, multi_class="ovr", average="macro")
    except ValueError:
        return None
    return round(float(score), 4) if np.isfinite(score) else None


def _aligned_proba(clf, raw_proba, labels):
    """Expand ``clf.predict_proba`` columns to the explicit label axis."""
    proba = np.zeros((raw_proba.shape[0], len(labels)), dtype=raw_proba.dtype)
    label_to_col = {int(label): i for i, label in enumerate(labels)}
    for src_col, cls in enumerate(clf.classes_):
        dst_col = label_to_col.get(int(cls))
        if dst_col is not None:
            proba[:, dst_col] = raw_proba[:, src_col]
    return proba


def fit_score_riemann(Ctr, ytr, Cev, yev, label_names=None):
    """Fit tangent-space logistic regression and return eval metrics."""
    from pyriemann.tangentspace import TangentSpace
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
    )

    labels = np.arange(int(max(ytr.max(), yev.max())) + 1)
    if label_names is None:
        label_names = [str(i) for i in labels]
    if len(label_names) < len(labels):
        label_names = [*label_names, *[str(i) for i in labels[len(label_names):]]]

    clf = make_pipeline(
        TangentSpace(metric="riemann"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    clf.fit(Ctr, ytr)
    pred = clf.predict(Cev)
    lr = clf.named_steps["logisticregression"]
    raw_proba = clf.predict_proba(Cev)
    proba = _aligned_proba(lr, raw_proba, labels)
    per_class_recall = recall_score(
        yev, pred, labels=labels, average=None, zero_division=0)
    return {
        "acc": round(float(accuracy_score(yev, pred)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(yev, pred)), 4),
        "f1": round(float(f1_score(yev, pred, labels=labels, average="macro",
                                   zero_division=0)), 4),
        "auroc": _safe_auroc(yev, proba, labels, scored_labels=lr.classes_),
        "classes": list(label_names[:len(labels)]),
        "n_train": int(len(ytr)),
        "n_eval": int(len(yev)),
        "per_class_recall": {
            str(label_names[i]): round(float(per_class_recall[i]), 4)
            for i in range(len(labels))
        },
        "confusion_matrix": confusion_matrix(yev, pred, labels=labels).tolist(),
    }


def run_recording_riemann(
    data_root=None,
    label_scheme="tuab",
    class_names=None,
    train_split="train",
    eval_split="eval",
    n_windows=None,
    estimator="oas",
    n_channels=None,
    sfreq=None,
    window_sec=None,
):
    """Read both splits, compute recording covariances, fit and score."""
    print("[riemann] train covariances...", flush=True)
    Ctr, ytr, label_names = _recording_covariances(
        train_split, data_root, label_scheme, class_names, n_windows, estimator,
        n_channels, sfreq, window_sec, return_label_names=True)
    # Freeze the train label order for eval. This matches eval.py and prevents a
    # diagnosis that appears only in eval from shifting labels.
    print("[riemann] eval covariances...", flush=True)
    Cev, yev = _recording_covariances(
        eval_split, data_root, label_scheme, label_names, n_windows, estimator,
        n_channels, sfreq, window_sec)
    return fit_score_riemann(Ctr, ytr, Cev, yev, label_names)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--data_root", dest="data_root")
    parser.add_argument("--label-scheme", choices=["tuab", "folders"], default="tuab")
    parser.add_argument("--classes",
                        help="comma-separated class folder order, e.g. normal,seizure,dementia")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument("--estimator", default="oas",
                        help="pyRiemann covariance estimator, e.g. oas, lwf, scm")
    parser.add_argument("--n-channels", type=int, default=None)
    parser.add_argument("--sfreq", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    return parser.parse_args(argv)


def main():
    args = _parse_args(sys.argv[1:])
    metrics = run_recording_riemann(
        data_root=args.data_root,
        label_scheme=args.label_scheme,
        class_names=args.classes,
        train_split=args.train_split,
        eval_split=args.eval_split,
        n_windows=args.n_windows,
        estimator=args.estimator,
        n_channels=args.n_channels,
        sfreq=args.sfreq,
        window_sec=args.window_sec,
    )
    print("[riemann] held-out:", metrics)


if __name__ == "__main__":
    main()
