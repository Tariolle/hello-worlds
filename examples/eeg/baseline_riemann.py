"""Classical Riemannian baseline — the 0-parameter complexity yardstick.

Per recording: spatial channel covariance of each 10 s window -> Euclidean mean
-> one SPD matrix; project all recordings to the tangent space at their
Riemannian mean (pyRiemann) -> standardize -> logistic regression. On TUH
Abnormal this reaches ~0.86 accuracy with NO deep network (Gemein et al. 2020),
so it (a) sanity-checks the EDF data pipeline before any GPU run and (b) is the
complexity reference the JEPA frozen probe is measured against.

Run:  python -m examples.eeg.baseline_riemann [--data_root <TUAB_PREPROCESSED>]
"""
import sys

import numpy as np

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


def _recording_covariances(split, data_root):
    """[N_rec, C, C] recording-level mean covariances + labels."""
    from pyriemann.estimation import Covariances
    cfg = EEGConfig(split=split, mode="probe")
    if data_root:
        cfg.data_root = data_root
    ds = EEGDataset(cfg)
    est = Covariances(estimator="oas")
    covs, ys = [], []
    for i in range(len(ds)):
        wins, label, ok = ds[i]               # [N, C, T], int, bool
        if not ok:
            continue
        c = est.transform(wins.numpy().astype(np.float64))  # [N, C, C]
        covs.append(c.mean(axis=0))           # recording-level mean covariance
        ys.append(int(label))
        if (i + 1) % 200 == 0:
            print(f"  [{split}] {i + 1}/{len(ds)}", flush=True)
    return np.stack(covs), np.array(ys)


def main():
    argv = sys.argv
    data_root = argv[argv.index("--data_root") + 1] if "--data_root" in argv else None
    from pyriemann.tangentspace import TangentSpace
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    print("[riemann] train covariances...", flush=True)
    Ctr, ytr = _recording_covariances("train", data_root)
    print("[riemann] eval covariances...", flush=True)
    Cev, yev = _recording_covariances("eval", data_root)

    clf = make_pipeline(TangentSpace(metric="riemann"), StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Ctr, ytr)
    pe = clf.predict(Cev)
    se = clf.predict_proba(Cev)[:, 1]
    print("[riemann] held-out-patient:",
          {"acc": round(float(accuracy_score(yev, pe)), 4),
           "balanced_acc": round(float(balanced_accuracy_score(yev, pe)), 4),
           "auroc": round(float(roc_auc_score(yev, se)), 4)})


if __name__ == "__main__":
    main()
