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
def extract_features(encoder, split, device, data_cfg=None, return_label_names=False):
    """Frozen encoder -> [N_rec, D] recording-level features + labels.

    One embedding per recording: encode its N windows and mean-pool them.
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
        z = encoder.represent(flat).reshape(B, N, -1).mean(dim=1).cpu().numpy()  # [B, D]
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


def _safe_auroc(y_true, proba, labels):
    """One-vs-rest macro AUROC, or None when it is undefined.

    Returns None (instead of raising) when a class in ``labels`` has no samples in
    ``y_true`` or the probability columns don't line up — e.g. a diagnosis with no
    eval recordings. The whole AUROC is then dropped rather than computed over the
    present classes, so read None as 'not reported', not zero.
    """
    from sklearn.metrics import roc_auc_score

    try:
        if len(labels) == 2:
            return round(float(roc_auc_score(y_true, proba[:, 1])), 4)
        return round(float(roc_auc_score(
            y_true, proba, labels=labels, multi_class="ovr", average="macro")), 4)
    except ValueError:
        return None


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
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pe = clf.predict(sc.transform(Xev))
    proba = clf.predict_proba(sc.transform(Xev))
    per_class_recall = recall_score(yev, pe, labels=labels, average=None, zero_division=0)
    macro_f1 = round(float(f1_score(yev, pe, labels=labels, average="macro",
                                    zero_division=0)), 4)
    return {"acc": round(float(accuracy_score(yev, pe)), 4),
            "balanced_acc": round(float(balanced_accuracy_score(yev, pe)), 4),
            "f1": macro_f1,  # macro-averaged; the benchmark harness reads "f1"
            "auroc": _safe_auroc(yev, proba, labels),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    data_cfg = _data_cfg_with_overrides(cfg, args)

    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    print("[eeg-eval] extracting TRAIN embeddings (fit set)...", flush=True)
    Xtr, ytr, label_names = extract_features(
        encoder, args.train_split, device, data_cfg, return_label_names=True)
    data_cfg["class_names"] = label_names
    print("[eeg-eval] extracting EVAL embeddings (held-out patients)...", flush=True)
    Xev, yev = extract_features(encoder, args.eval_split, device, data_cfg)
    print("[eeg-eval] TRAINED:", probe(Xtr, ytr, Xev, yev, label_names))

    if args.floor:  # same architecture, untrained -> random-encoder floor
        rnd = build_encoder(cfg.model).to(device).eval()
        Rtr, ry = extract_features(rnd, args.train_split, device, data_cfg)
        Rev, rey = extract_features(rnd, args.eval_split, device, data_cfg)
        print("[eeg-eval] RANDOM floor:", probe(Rtr, ry, Rev, rey, label_names))


if __name__ == "__main__":
    main()
