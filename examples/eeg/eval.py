"""EEG — downstream evaluation (patient-disjoint frozen probes).

The feature-extraction harness is provided: per recording, encode N evenly-spaced
10 s windows with the FROZEN encoder and mean-pool them into ONE embedding. We
implement the probe + metric.

GOLDEN RULE — patient-disjoint split: fit the probe on `train` patients, score on
`eval` patients (no subject overlap). The held-out-patient number is the only one
that answers transferability.

REPORTING HONESTY for TUAB: report balanced accuracy on the full 2717/276 split
at recording level. Do NOT report plain accuracy on a reduced subset.

Run:  python -m examples.eeg.eval --ckpt <.../latest.pth.tar> [--floor]

For diagnosis datasets arranged as root/{train,eval}/<class_name>/**/*.edf:
      python -m examples.eeg.eval --ckpt <...> --data-root <ROOT> \
        --label-scheme folders
"""
import argparse
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, device, data_cfg=None, pool="mean",
                     return_label_names=False):
    """Frozen encoder -> [N_rec, D] recording-level features + labels.

    Per recording, encode its N windows and mean-pool over windows. `pool` sets the
    per-window temporal pooling of the feature map [B*N, D, T']:
      * "mean"    -> time-mean only                 -> [N_rec, D]   (default)
      * "meanstd" -> concat(time-mean, time-std)    -> [N_rec, 2D]  (ablation: keeps
                     second-order temporal structure, abnormality is power/variance-driven)
    """
    cfg = EEGConfig(**(data_cfg or {}))
    cfg.split, cfg.mode = split, "probe"
    ds = EEGDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False,
                                         num_workers=cfg.num_workers, pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:          # wins: [B, N, C, T]
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        if pool == "meanstd":
            fm = encoder.feature_map(flat)                              # [B*N, D, T']
            zz = torch.cat([fm.mean(dim=-1), fm.std(dim=-1)], dim=1)    # [B*N, 2D]
        else:
            zz = encoder.represent(flat)                               # [B*N, D]
        z = zz.reshape(B, N, -1).mean(dim=1).cpu().numpy()             # [B, D or 2D]
        for k in range(B):
            if bool(ok[k]):                  # drop unreadable recordings
                X.append(z[k]); y.append(int(labels[k]))
    X, y = np.stack(X), np.array(y)
    names = list(ds.label_names or [])
    if names:  # surface the resolved label order + per-class counts for this split
        counts = np.bincount(y, minlength=len(names))
        summary = ", ".join(f"{n}={int(counts[i])}" for i, n in enumerate(names))
        print(f"[eeg-eval] {split} classes: {summary}  (n_recordings={len(y)})",
              flush=True)
    if return_label_names:
        return X, y, names
    return X, y


def _aligned_proba(clf, raw_proba, labels):
    """Expand ``clf.predict_proba`` columns onto the explicit 0..K-1 label axis.

    predict_proba columns follow ``clf.classes_`` (only classes present in train).
    This maps them back to the full ``labels`` axis so a class learned out of order
    or absent from train lands in the right column (or a zero column), rather than
    the binary ``proba[:, 1]`` / multiclass ``labels`` paths silently assuming the
    columns are contiguous 0..K-1. Mirrors baseline_riemann._aligned_proba.
    """
    proba = np.zeros((raw_proba.shape[0], len(labels)), dtype=raw_proba.dtype)
    label_to_col = {int(label): i for i, label in enumerate(labels)}
    for src_col, cls in enumerate(clf.classes_):
        dst_col = label_to_col.get(int(cls))
        if dst_col is not None:
            proba[:, dst_col] = raw_proba[:, src_col]
    return proba


def _safe_auroc(y_true, proba, labels, scored_labels=None):
    """Macro one-vs-rest AUROC over classes both present in ``y_true`` and learned
    by the classifier; a class with no eval support or never learned (all-zero
    score column) is dropped from both axes instead of dragging the macro toward
    chance. Returns None when fewer than two scorable classes remain or the metric
    is undefined — read None as 'not reported', not zero.
    """
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
            score = roc_auc_score(y_k, p_k, labels=keep, multi_class="ovr", average="macro")
    except ValueError:
        return None
    return round(float(score), 4) if np.isfinite(score) else None


def probe(Xtr, ytr, Xev, yev, label_names=None):
    """Patient-disjoint linear probe on FROZEN features. Standardize on TRAIN
    stats only, fit LogisticRegression(class_weight='balanced'), score on
    held-out patients. Supports binary TUAB and multiclass diagnosis folders."""
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

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pe = clf.predict(sc.transform(Xev))
    proba = _aligned_proba(clf, clf.predict_proba(sc.transform(Xev)), labels)
    per_class_recall = recall_score(yev, pe, labels=labels, average=None, zero_division=0)
    macro_f1 = round(float(f1_score(yev, pe, labels=labels, average="macro",
                                    zero_division=0)), 4)
    return {"acc": round(float(accuracy_score(yev, pe)), 4),
            "balanced_acc": round(float(balanced_accuracy_score(yev, pe)), 4),
            "f1": macro_f1,  # macro-averaged; the benchmark harness reads "f1"
            "auroc": _safe_auroc(yev, proba, labels, scored_labels=clf.classes_),
            "classes": list(label_names[:len(labels)]),
            "n_train": int(len(ytr)),
            "n_eval": int(len(yev)),
            "per_class_recall": {
                str(label_names[i]): round(float(per_class_recall[i]), 4)
                for i in range(len(labels))
            },
            "confusion_matrix": confusion_matrix(yev, pe, labels=labels).tolist()}


def _parse_classes(raw):
    if not raw:
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--floor", action="store_true",
                        help="also evaluate a random encoder with the same architecture")
    parser.add_argument("--data-root",
                        help="override checkpoint data_root; use for imported diagnosis datasets")
    parser.add_argument("--label-scheme", choices=["tuab", "folders"],
                        help="tuab keeps normal/abnormal; folders maps class subfolders to labels")
    parser.add_argument("--classes",
                        help="comma-separated class folder names/order, e.g. seizure,dementia,normal")
    parser.add_argument("--pool", default="mean", choices=["mean", "meanstd"],
                        help="per-window temporal pooling: mean | meanstd (2nd-order ablation)")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    return parser.parse_args(argv)


def _data_cfg_with_overrides(state_cfg, args):
    data_cfg = OmegaConf.to_container(state_cfg.data, resolve=True)
    if args.data_root:
        data_cfg["data_root"] = args.data_root
    if args.label_scheme:
        data_cfg["label_scheme"] = args.label_scheme
    classes = _parse_classes(args.classes)
    if classes:
        data_cfg["class_names"] = classes
    return data_cfg


def main():
    args = _parse_args(sys.argv[1:])
    pool = args.pool
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    data_cfg = _data_cfg_with_overrides(cfg, args)

    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    print(f"[eeg-eval] pool={pool} | extracting TRAIN embeddings (fit set)...", flush=True)
    Xtr, ytr, label_names = extract_features(
        encoder, args.train_split, device, data_cfg, pool=pool, return_label_names=True)
    data_cfg["class_names"] = label_names
    print("[eeg-eval] extracting EVAL embeddings (held-out patients)...", flush=True)
    Xev, yev = extract_features(encoder, args.eval_split, device, data_cfg, pool=pool)
    print(f"[eeg-eval] TRAINED (pool={pool}):", probe(Xtr, ytr, Xev, yev, label_names))

    if args.floor:  # same architecture, untrained -> random-encoder floor
        # Seed the floor init so the ~0.79 reference is reproducible and matches
        # benchmark.py's seeded floor (it was previously unseeded -> drifted run to
        # run). NB: BatchNorm uses init running stats (0/1) here, so this is an
        # untrained-architecture floor with un-adapted BN, not a data-calibrated one.
        seed = int(cfg.meta.get("seed", 0))
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        rnd = build_encoder(cfg.model).to(device).eval()
        Rtr, ry = extract_features(rnd, args.train_split, device, data_cfg, pool=pool)
        Rev, rey = extract_features(rnd, args.eval_split, device, data_cfg, pool=pool)
        print(f"[eeg-eval] RANDOM floor (pool={pool}):", probe(Rtr, ry, Rev, rey, label_names))


if __name__ == "__main__":
    main()
