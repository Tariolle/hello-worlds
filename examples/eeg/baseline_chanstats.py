"""Trivial per-channel statistics baseline: [mean, std] per channel -> linear probe.

For each recording we take N evenly-spaced windows, compute every channel's mean
and standard deviation over time (per window), average those over the windows, and
stack them into a ``2 * n_channels`` feature vector. The SAME linear probe as
``eval.py`` (StandardScaler + class-balanced LogisticRegression) is fit on TRAIN and
scored on the held-out EVAL split. This is a 0-parameter "are trivial per-channel
statistics enough?" reference for the frozen head-to-head.

IMPORTANT -- run on RAW windows (the default here, ``normalize=False``). The main
pipeline z-scores each channel per window, which forces per-channel mean=0 / std=1
and makes this baseline EXACTLY chance. On the raw uV signal the per-channel std is
band power, which IS informative for TUAB abnormality (power-driven). Pass
``--zscore`` to reproduce the degenerate (chance) variant.

Run on the cluster (CPU only, no GPU):
  python -m examples.eeg.baseline_chanstats --data-root <TUAB_PREPROCESSED> --n-windows 8
"""
import argparse
import sys

import numpy as np

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


def _chan_stat_features(split, data_root=None, n_windows=None, normalize=False,
                        class_names=None, return_label_names=False):
    """[N_rec, 2*C] per-channel (mean, std) features + labels, averaged over windows."""
    cfg = EEGConfig(mode="probe")
    if data_root:
        cfg.data_root = data_root
    cfg.split = split
    cfg.normalize = normalize
    if class_names:
        cfg.class_names = class_names
    if n_windows is not None:
        cfg.n_windows = int(n_windows)
    ds = EEGDataset(cfg)
    X, y = [], []
    for i in range(len(ds)):
        wins, label, ok = ds[i]                       # [N, C, T], int, bool
        if not ok:
            continue
        w = wins.numpy().astype(np.float64)           # raw uV when normalize=False
        mu = w.mean(axis=2).mean(axis=0)              # [C] per-channel mean
        sd = w.std(axis=2).mean(axis=0)               # [C] per-channel std (= power)
        X.append(np.concatenate([mu, sd]))            # [2C]
        y.append(int(label))
        if (i + 1) % 200 == 0:
            print(f"  [{split}] {i + 1}/{len(ds)}", flush=True)
    if not X:
        raise RuntimeError(f"No readable recordings for split={split!r}")
    out = (np.stack(X), np.array(y))
    if return_label_names:
        return (*out, list(ds.label_names or []))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", "--data_root", dest="data_root")
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--zscore", action="store_true",
                    help="use the pipeline's per-channel z-scored windows (=> chance); default RAW uV")
    args = ap.parse_args(sys.argv[1:])

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    normalize = bool(args.zscore)
    print(f"[chanstats] normalize(z-score)={normalize}  -> {'CHANCE (z-scored)' if normalize else 'RAW power'}",
          flush=True)
    Xtr, ytr, names = _chan_stat_features("train", args.data_root, args.n_windows,
                                          normalize, return_label_names=True)
    Xev, yev = _chan_stat_features("eval", args.data_root, args.n_windows, normalize)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xev)
    ba = balanced_accuracy_score(yev, pred)
    acc = accuracy_score(yev, pred)
    auroc = None
    if len(np.unique(ytr)) == 2:
        auroc = roc_auc_score(yev, clf.predict_proba(Xev)[:, 1])
    print(f"[chanstats] RESULT  balanced_acc={ba:.4f}  acc={acc:.4f}"
          + (f"  auroc={auroc:.4f}" if auroc is not None else "")
          + f"  | classes={names}  n_train={len(ytr)} n_eval={len(yev)} dim={Xtr.shape[1]}",
          flush=True)


if __name__ == "__main__":
    main()
