"""Corruption-robustness of a FROZEN EEG representation.

Fit a linear probe on CLEAN train features, then evaluate balanced accuracy on EVAL
under increasing corruption applied to the input windows before the frozen encoder:
  * channel dropout (zeroing) at p in {0.1, 0.25, 0.5},
  * additive Gaussian noise at sigma in {0.1, 0.25, 0.5} (z-scored units).

Question: do geometry/covariance (tangent) reps degrade more gracefully than
ambient/random? Covariance pooling over time should be less sensitive to transient
corruption — the one axis where the geometry arm has a mechanistic reason to win.

NOTE: channel dropout here is the standard ZERO-PAD variant (keeps the 19-ch input
shape the encoder expects). The "remove-vs-zero-pad" confound (Beyond-Accuracy 2026)
would require montage surgery and is out of scope; this is the zero-pad curve.

Run:  python -u -m examples.eeg.robustness_probe --ckpt <...> --data-root <TUAB> [--random]
"""
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder


def _corrupt(flat, kind, level):
    if level == 0 or kind == "none":
        return flat
    if kind == "drop":                                   # zero a fraction of channels per sample
        mask = (torch.rand(flat.shape[0], flat.shape[1], 1, device=flat.device) > level).float()
        return flat * mask
    if kind == "noise":                                  # additive Gaussian (z-units)
        return flat + level * torch.randn_like(flat)
    return flat


@torch.no_grad()
def extract(encoder, split, device, data_cfg, kind="none", level=0.0):
    cfg = EEGConfig(**(data_cfg or {})); cfg.split, cfg.mode = split, "probe"
    ds = EEGDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=cfg.num_workers)
    X, y = [], []
    for wins, labels, ok in loader:
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device)
        flat = _corrupt(flat, kind, level)
        z = encoder.represent(flat).reshape(B, N, -1).mean(1).cpu().numpy()
        for k in range(B):
            if bool(ok[k]):
                X.append(z[k]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--random", action="store_true", help="untrained encoder (random floor)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(dev)
    tag = "RANDOM" if a.random else f"{cfg.model.ssl.get('reg_type','?')}/{cfg.model.ssl.get('reg_space','?')}"
    if not a.random:
        enc.load_state_dict(state["encoder"])
    enc.eval()

    dcfg = {"data_root": a.data_root, "label_scheme": "tuab", "n_channels": int(cfg.model.n_channels),
            "sfreq": 200, "window_sec": 10.0, "n_windows": 16, "num_workers": 8}

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    Xtr, ytr = extract(enc, "train", dev, dcfg, "none", 0.0)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)

    def ba(kind, lvl):
        Xev, yev = extract(enc, "eval", dev, dcfg, kind, lvl)
        return balanced_accuracy_score(yev, clf.predict(sc.transform(Xev)))

    clean = ba("none", 0.0)
    print(f"[robust] enc={tag} | clean BA={clean:.4f}", flush=True)
    for lvl in (0.1, 0.25, 0.5):
        v = ba("drop", lvl); print(f"[robust] enc={tag} drop p={lvl}: BA={v:.4f} (drop {clean - v:+.4f})", flush=True)
    for lvl in (0.1, 0.25, 0.5):
        v = ba("noise", lvl); print(f"[robust] enc={tag} noise s={lvl}: BA={v:.4f} (drop {clean - v:+.4f})", flush=True)
    print("ROBUST_DONE", flush=True)


if __name__ == "__main__":
    main()
