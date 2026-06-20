"""Classical anomaly-detection baselines on flattened TCP bandpower features.

For comparison with TCP-Graph-JEPA: fit an unsupervised detector on NORMAL
training recordings and score the eval split (normal vs abnormal) by AUROC/AUPRC.
Per recording we average the log-bandpower over its windows and flatten to a
``C*T*F`` (or ``C*F`` with ``--pool-time``) vector.

  * Isolation Forest
  * One-Class SVM

(The Riemannian covariance-distance baseline already lives in
``examples/eeg/baseline_riemann.py``; this module is the bandpower-space
counterpart and reuses the graph-JEPA feature pipeline.)

Run:
    python -m examples.eeg.graph_jepa.baselines --data-root <TUAB_PREPROCESSED>
"""
import argparse

import numpy as np
import torch

from eb_jepa.graph_jepa.metrics import evaluate
from eb_jepa.graph_jepa.windows import GraphEEGConfig, make_graph_loader


def _recording_vectors(loader, pool_time):
    X, y = [], []
    for x, label, cm, ok, _ in loader:
        for b in range(x.shape[0]):
            if not bool(ok[b]):
                continue
            feat = x[b].numpy()              # [N, C, T, F]
            feat = feat.mean(axis=0)        # [C, T, F]
            if pool_time:
                feat = feat.mean(axis=1)    # [C, F]
            X.append(feat.reshape(-1)); y.append(int(label[b]))
    return np.asarray(X), np.asarray(y)


def main():
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--pool-time", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    def cfg(split):
        c = GraphEEGConfig(data_root=a.data_root, split=split, mode="file",
                           n_windows=a.n_windows, batch_size=4, num_workers=a.workers)
        return c

    tr_loader, _ = make_graph_loader(cfg("train"), shuffle=False)
    ev_loader, _ = make_graph_loader(cfg("eval"), shuffle=False)
    print("[baselines] extracting features ...", flush=True)
    Xtr, ytr = _recording_vectors(tr_loader, a.pool_time)
    Xev, yev = _recording_vectors(ev_loader, a.pool_time)
    Xtr_n = Xtr[ytr == 0]                    # fit on NORMAL only
    print(f"[baselines] train-normal={len(Xtr_n)} eval={len(Xev)} dim={Xtr.shape[1]}",
          flush=True)

    sc = StandardScaler().fit(Xtr_n)
    Xtr_s, Xev_s = sc.transform(Xtr_n), sc.transform(Xev)

    for name, model in [
        ("IsolationForest", IsolationForest(n_estimators=200, random_state=0)),
        ("OneClassSVM", OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")),
    ]:
        model.fit(Xtr_s)
        scores = -model.score_samples(Xev_s) if hasattr(model, "score_samples") \
            else -model.decision_function(Xev_s)
        m = evaluate(scores, yev)
        print(f"[baselines] {name:16s} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f}",
              flush=True)
    print("GRAPH_JEPA_BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()
