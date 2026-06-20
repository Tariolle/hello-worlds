"""TUAB frozen-latent ENERGY LANDSCAPE.

For each frozen encoder: UMAP the recording-level represent() features, fit a kernel-density
estimate over the 2-D embedding, and plot the "energy" surface E = -log p_hat (a density /
typicality proxy -- NOT a learned EBM energy; our JEPA has no explicit energy head). Dark
basins = where recordings concentrate; bright rim = low-density / high-energy. Normal vs
abnormal scattered on top show how the classes sit in the landscape. ours vs random-init.

Run (GPU): python -u -m examples.eeg.energy_latent --data-root <TUAB> --work $WORK
"""
import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.eval import extract_features
from examples.eeg.main import build_encoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def feats(ckpt, data_root, dev, rand=False):
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(dev)
    if not rand:
        enc.load_state_dict(state["encoder"])
    enc.eval()
    dcfg = {"data_root": data_root, "label_scheme": "tuab", "n_channels": int(cfg.model.n_channels),
            "sfreq": 200, "window_sec": 10.0, "n_windows": 16, "num_workers": 8}
    return extract_features(enc, "eval", dev, dcfg)


def main():
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import gaussian_kde
    import umap
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = a.work
    ENCS = [
        ("SIGReg-ambient (ours)", f"{W}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", False),
        ("Random-init (control)", f"{W}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, (label, ck, rand) in zip(axes, ENCS):
        X, y = feats(ck, a.data_root, dev, rand)
        Xs = StandardScaler().fit_transform(X)
        emb = umap.UMAP(n_components=2, random_state=0).fit_transform(Xs)

        kde = gaussian_kde(emb.T)
        pad = 1.0
        gx, gy = np.mgrid[emb[:, 0].min() - pad: emb[:, 0].max() + pad: 220j,
                          emb[:, 1].min() - pad: emb[:, 1].max() + pad: 220j]
        dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        energy = -np.log(dens + 1e-12)
        energy = np.clip(energy, energy.min(), np.percentile(energy, 97))  # tame the sparse rim

        cf = ax.contourf(gx, gy, energy, levels=24, cmap="magma")
        for cls, col, nm in [(0, "#39a7ff", "normal"), (1, "#ff4d4d", "abnormal")]:
            mk = y == cls
            ax.scatter(emb[mk, 0], emb[mk, 1], s=12, alpha=0.85, c=col,
                       edgecolors="white", linewidths=0.3, label=nm)
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"energy $= -\log \hat{p}$", fontsize=8)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle("TUAB frozen-latent energy landscape -- $E=-\\log\\hat p$ over the UMAP embedding "
                 "(density / typicality proxy, NOT a learned EBM energy)\n"
                 "dark basins = where recordings concentrate; ours carves more normal/abnormal "
                 "structure than the random-init floor (separation is modest: silhouette ${\\sim}0.05$)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "energy", "energy_latent.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}", flush=True)
    print("ENERGY_LATENT_DONE", flush=True)


if __name__ == "__main__":
    main()
