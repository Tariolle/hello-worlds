"""JEPA-SCORE (post-hoc): input->embedding Jacobian typicality as an OOD/artifact signal.

For a FROZEN encoder we estimate, per eval recording, the squared Jacobian Frobenius
norm ||d represent(x)/dx||_F^2 (input sensitivity) via a Hutchinson estimator: with
v ~ N(0,I), E_v[||J v||^2] = ||J||_F^2, and J v is one forward-mode JVP. We then test
whether this score:
  (1) SEPARATES clean vs corrupted (noise-injected) windows  -> artifact detection,
  (2) PREDICTS probe MIS-classification on TUAB eval.
Read-only on existing checkpoints; no training. (Encoder in eval mode -> BatchNorm uses
running stats, so the batched JVP is per-sample = block-diagonal Jacobian.)

Run (GPU): python -u -m examples.eeg.jepa_score --ckpt <...> --data-root <TUAB> [--random]
"""
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.eval import extract_features, probe
from examples.eeg.main import build_encoder


def jac_fro2(encoder, x, K=8):
    """Hutchinson estimate of ||J||_F^2 per sample, J = d represent(x)/dx. x: [B,C,T]."""
    acc = torch.zeros(x.shape[0], device=x.device)
    for _ in range(K):
        v = torch.randn_like(x)
        _, jv = torch.autograd.functional.jvp(
            lambda z: encoder.represent(z), (x,), (v,), create_graph=False)
        acc = acc + (jv ** 2).sum(dim=1)
    return (acc / K).detach().cpu().numpy()


def per_recording_scores(encoder, split, device, data_cfg, noise=0.0, K=8):
    cfg = EEGConfig(**data_cfg); cfg.split, cfg.mode = split, "probe"
    ds = EEGDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=cfg.num_workers)
    scores, labels = [], []
    for wins, labs, ok in loader:
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device)
        if noise > 0:
            flat = flat + noise * torch.randn_like(flat)
        s = jac_fro2(encoder, flat, K).reshape(B, N).mean(1)
        for k in range(B):
            if bool(ok[k]):
                scores.append(float(s[k])); labels.append(int(labs[k]))
    return np.asarray(scores), np.asarray(labels)


def main():
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--random", action="store_true"); ap.add_argument("-K", type=int, default=8)
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

    # (2) misclassification: fit probe on train represent features, predict eval
    Xtr, ytr = extract_features(enc, "train", dev, dcfg)
    Xev, yev = extract_features(enc, "eval", dev, dcfg)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xev))
    mis = (pred != yev).astype(int)

    # JEPA-SCORE on clean eval (aligned with the probe's recording order via same loader)
    s_clean, _ = per_recording_scores(enc, "eval", dev, dcfg, noise=0.0, K=a.K)
    # (1) artifact: scores on noise-corrupted eval, pooled vs clean
    s_corr, _ = per_recording_scores(enc, "eval", dev, dcfg, noise=0.5, K=a.K)

    auc_mis = roc_auc_score(mis, s_clean) if mis.sum() and mis.sum() < len(mis) else float("nan")
    y_corr = np.r_[np.zeros(len(s_clean)), np.ones(len(s_corr))]
    auc_art = roc_auc_score(y_corr, np.r_[s_clean, s_corr])
    print(f"[jepa-score] enc={tag} K={a.K} n_eval={len(s_clean)} misclassified={int(mis.sum())}", flush=True)
    print(f"[jepa-score] AUROC(score -> misclassification) = {auc_mis:.4f}  "
          f"(|.5| = {abs(auc_mis-0.5):.4f}; >.5 means high score => error)", flush=True)
    print(f"[jepa-score] AUROC(score -> corrupted vs clean) = {auc_art:.4f}  "
          f"(artifact detectability; .5 = no signal)", flush=True)
    print(f"[jepa-score] mean score clean={np.mean(s_clean):.3e} corrupted={np.mean(s_corr):.3e}", flush=True)
    print("JEPA_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
