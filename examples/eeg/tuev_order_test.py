"""TUEV confirmatory test: 1st-order (represent) vs 2nd-order (covariance) latent.

Same TUEV events, same frozen multiclass probe (StandardScaler + balanced LogisticRegression,
patient-disjoint), reported as silhouette AND balanced accuracy for BOTH latents, for ours
and a random-init floor. Decides whether TUEV's near-zero covariance-silhouette is
"second-order-blindness" (1st-order separates/decodes, 2nd-order doesn't) vs dead features.

Run (GPU): python -u -m examples.eeg.tuev_order_test --tuev-root <TUEV_PREPROCESSED> --work $WORK
"""
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder
from examples.eeg.tuev_probe import build_split, extract, run_probe
from examples.eeg.riemann_latent_tuev import event_spd, euclid_tangent


def _sil(X, y):
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    return float(silhouette_score(StandardScaler().fit_transform(X), y))


def _sil_airm(mats, y):
    from sklearn.metrics import silhouette_score
    from pyriemann.utils.distance import pairwise_distance
    D = pairwise_distance(mats, metric="riemann"); np.fill_diagonal(D, 0.0)
    return float(silhouette_score(D, y, metric="precomputed"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tuev-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--train-cap", type=int, default=200)
    ap.add_argument("--eval-cap", type=int, default=120, help="balanced eval subset/class")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    tr = build_split(a.tuev_root, "train", a.train_cap, rng)
    ev = build_split(a.tuev_root, "eval", a.eval_cap, rng)
    print(f"[order] train={len(tr)} eval={len(ev)} events", flush=True)

    ckpt = f"{a.work}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar"
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])

    rows = []
    for label, rand in [("SIGReg-ambient (ours)", False), ("Random-init (floor)", True)]:
        enc = build_encoder(cfg.model).to(dev)
        if not rand:
            enc.load_state_dict(state["encoder"])
        enc.eval()

        # 1st-order: represent() (time-mean-pooled features) — what the probe normally uses
        X1tr, y1tr = extract(enc, tr, dev)
        X1ev, y1ev = extract(enc, ev, dev)
        r1 = run_probe(X1tr, y1tr, X1ev, y1ev, f"{label} | 1st-order represent")
        s1 = _sil(X1ev, y1ev)

        # 2nd-order: covariance -> Log-Euclidean tangent — what the manifold viz uses
        Mtr, y2tr = event_spd(enc, tr, dev)
        Mev, y2ev = event_spd(enc, ev, dev)
        Ttr, Tev = euclid_tangent(Mtr), euclid_tangent(Mev)
        r2 = run_probe(Ttr, y2tr, Tev, y2ev, f"{label} | 2nd-order cov-tangent")
        s2 = _sil(Tev, y2ev)
        s2a = _sil_airm(Mev, y2ev)

        rows.append((label, r1["balanced_acc"], s1, r2["balanced_acc"], s2, s2a))

    print("\n================ TUEV: 1st-order vs 2nd-order (6-class, balanced) ================", flush=True)
    print(f"{'encoder':<24} | {'1st BA':>7} {'1st sil':>8} | {'2nd BA':>7} {'2nd silE':>9} {'2nd silAIRM':>11}", flush=True)
    for lab, ba1, s1, ba2, s2, s2a in rows:
        print(f"{lab:<24} | {ba1:>7.4f} {s1:>8.4f} | {ba2:>7.4f} {s2:>9.4f} {s2a:>11.4f}", flush=True)
    print("(chance balanced-acc for 6 classes = 0.1667)", flush=True)
    print("TUEV_ORDER_DONE", flush=True)


if __name__ == "__main__":
    main()
