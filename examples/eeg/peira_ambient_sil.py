"""Latent silhouette for PEIRA-ambient (c3) on TUAB (recording-level) and TUSZ (event-level),
to complete the PEIRA vanilla-vs-tangent comparison on the geometry axis. Prints Euc + AIRM,
matching the recipes used by riemann_latent.py (TUAB) and riemann_latent_tusz.py (TUSZ).

Run (GPU): python -u -m examples.eeg.peira_ambient_sil --tuab-root <..> --tusz-root <..> --work $WORK
"""
import argparse
from collections import Counter

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder
from examples.eeg.riemann_latent import recording_spd
from examples.eeg.riemann_latent_tuev import event_spd, euclid_tangent
from examples.eeg.riemann_latent_tusz import build_tusz_split, TUSZ_CLASSES


def sils(mats, y):
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from pyriemann.utils.distance import pairwise_distance
    euc = silhouette_score(StandardScaler().fit_transform(euclid_tangent(mats)), y)
    D = pairwise_distance(mats, metric="riemann"); np.fill_diagonal(D, 0.0)
    airm = silhouette_score(D, y, metric="precomputed")
    return euc, airm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tuab-root", required=True)
    ap.add_argument("--tusz-root", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    ck = f"{a.work}/checkpoints/c3_peira_ambient_s1/latest.pth.tar"
    state = torch.load(ck, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(dev)
    enc.load_state_dict(state["encoder"]); enc.eval()

    # TUAB — recording-level SPD (same recipe as riemann_latent.py)
    dcfg = {"data_root": a.tuab_root, "label_scheme": "tuab", "n_channels": int(cfg.model.n_channels),
            "sfreq": 200, "window_sec": 10.0, "n_windows": 16, "num_workers": 8}
    Mt, yt = recording_spd(enc, "eval", dev, dcfg)
    e, ai = sils(Mt, yt)
    print(f"[peira-amb] TUAB cov-SPD: Euc={e:.4f} AIRM={ai:.4f}  (n={len(yt)})", flush=True)

    # TUSZ — event-level SPD (same recipe as riemann_latent_tusz.py: cap 80/class, min 15)
    c2i = {c: i for i, c in enumerate(TUSZ_CLASSES)}
    raw = build_tusz_split(a.tusz_root, "eval", 80, rng, c2i)
    cnt = Counter(l for _, _, l in raw)
    keep = sorted([k for k, v in cnt.items() if v >= 15])
    remap = {o: i for i, o in enumerate(keep)}
    items = [(p, s, remap[l]) for p, s, l in raw if l in remap]
    Ms, ys = event_spd(enc, items, dev)
    e2, ai2 = sils(Ms, ys)
    print(f"[peira-amb] TUSZ cov-SPD: Euc={e2:.4f} AIRM={ai2:.4f}  "
          f"(n={len(ys)}, classes={[TUSZ_CLASSES[k] for k in keep]})", flush=True)
    print("PEIRA_AMB_DONE", flush=True)


if __name__ == "__main__":
    main()
