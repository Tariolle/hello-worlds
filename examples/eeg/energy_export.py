"""Export the TUAB ours-encoder energy landscape (UMAP embedding + KDE energy grid + points)
to results/energy/energy_data.npz, for the Manim 3-D animation. Coarse 70x70 grid for render
perf. Energy = -log p_hat (density proxy, not a learned EBM energy).

Run (GPU): python -u -m examples.eeg.energy_export --data-root <TUAB> --work $WORK
"""
import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.eval import extract_features
from examples.eeg.main import build_encoder


def main():
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import gaussian_kde
    import umap
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = f"{a.work}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar"
    state = torch.load(ck, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(dev)
    enc.load_state_dict(state["encoder"]); enc.eval()
    dcfg = {"data_root": a.data_root, "label_scheme": "tuab", "n_channels": int(cfg.model.n_channels),
            "sfreq": 200, "window_sec": 10.0, "n_windows": 16, "num_workers": 8}
    X, y = extract_features(enc, "eval", dev, dcfg)
    Xs = StandardScaler().fit_transform(X)
    emb = umap.UMAP(n_components=2, random_state=0).fit_transform(Xs)

    kde = gaussian_kde(emb.T)
    pad = 1.0
    gx, gy = np.mgrid[emb[:, 0].min() - pad: emb[:, 0].max() + pad: 70j,
                      emb[:, 1].min() - pad: emb[:, 1].max() + pad: 70j]
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    energy = -np.log(dens + 1e-12)
    energy = np.clip(energy, energy.min(), np.percentile(energy, 97))
    # energy of each data point (for placing dots on the surface)
    pe = -np.log(kde(emb.T) + 1e-12)
    pe = np.clip(pe, energy.min(), np.percentile(energy, 97))

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "energy", "energy_data.npz"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, gx=gx, gy=gy, energy=energy, emb=emb, y=y, point_energy=pe)
    print(f"saved {out}: grid={energy.shape} points={emb.shape}", flush=True)
    print("ENERGY_EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
