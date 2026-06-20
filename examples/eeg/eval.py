"""EEG — downstream evaluation (the patient-disjoint abnormality probe).

The feature-extraction harness is provided: per recording, encode N evenly-spaced
10 s windows with the FROZEN encoder and mean-pool them into ONE embedding. We
implement the probe + metric.

GOLDEN RULE — patient-disjoint split: fit the probe on `train` patients, score on
`eval` patients (no subject overlap). The held-out-patient number is the only one
that answers transferability.

REPORTING HONESTY: this reports balanced accuracy on the full 2717/276 split at
recording level. Do NOT report plain accuracy on a reduced subset.

Run:  python -m examples.eeg.eval --ckpt <.../latest.pth.tar> [--floor]
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.geometry import tangent_features
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, device, data_cfg=None, pool="mean"):
    """Frozen encoder -> [N_rec, D] recording-level features + labels.

    Per recording, encode its N windows and mean-pool over windows. `pool` sets the
    per-window temporal pooling of the feature map [B*N, D, T']:
      * "mean"    -> time-mean only                 -> [N_rec, D]   (default)
      * "meanstd" -> concat(time-mean, time-std)    -> [N_rec, 2D]  (ablation: keeps
                     second-order temporal structure, abnormality is power/variance-driven)
      * "tangent" -> Log-Euclidean SPD tangent covariance features from encoder.cov_features
                     -> [N_rec, d_cov(d_cov+1)/2]
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
        elif pool == "tangent":
            zz = tangent_features(encoder.cov_features(flat))           # [B*N, d_cov(d_cov+1)/2]
        else:
            zz = encoder.represent(flat)                               # [B*N, D]
        z = zz.reshape(B, N, -1).mean(dim=1).cpu().numpy()             # [B, D or 2D]
        for k in range(B):
            if bool(ok[k]):                  # drop unreadable recordings
                X.append(z[k]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


def probe(Xtr, ytr, Xev, yev):
    """Patient-disjoint linear probe on FROZEN features. Standardize on TRAIN
    stats only, fit LogisticRegression(class_weight='balanced'), score on held-out
    patients. Returns accuracy / balanced-accuracy / AUROC (normal=0, abnormal=1)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pe = clf.predict(sc.transform(Xev))
    se = clf.predict_proba(sc.transform(Xev))[:, 1]
    return {"acc": round(float(accuracy_score(yev, pe)), 4),
            "balanced_acc": round(float(balanced_accuracy_score(yev, pe)), 4),
            "auroc": round(float(roc_auc_score(yev, se)), 4)}


def main():
    argv = sys.argv
    ckpt = argv[argv.index("--ckpt") + 1]
    pool = argv[argv.index("--pool") + 1] if "--pool" in argv else "mean"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    import dataclasses
    _tuab_keys = {f.name for f in dataclasses.fields(EEGConfig)}
    data_cfg = {k: v for k, v in OmegaConf.to_container(cfg.data, resolve=True).items()
                if k in _tuab_keys}

    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    print(f"[eeg-eval] pool={pool} | extracting TRAIN embeddings (fit set)...", flush=True)
    Xtr, ytr = extract_features(encoder, "train", device, data_cfg, pool)
    print("[eeg-eval] extracting EVAL embeddings (held-out patients)...", flush=True)
    Xev, yev = extract_features(encoder, "eval", device, data_cfg, pool)
    print(f"[eeg-eval] TRAINED (pool={pool}):", probe(Xtr, ytr, Xev, yev))

    if "--floor" in argv:  # same architecture, untrained -> random-encoder floor
        rnd = build_encoder(cfg.model).to(device).eval()
        Rtr, ry = extract_features(rnd, "train", device, data_cfg, pool)
        Rev, rey = extract_features(rnd, "eval", device, data_cfg, pool)
        print(f"[eeg-eval] RANDOM floor (pool={pool}):", probe(Rtr, ry, Rev, rey))


if __name__ == "__main__":
    main()
