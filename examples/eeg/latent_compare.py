"""Compare FROZEN latent spaces across encoders (ours vs random vs baselines).

For each encoder: recording-level frozen features on TUAB eval -> 2-D embedding
(t-SNE, and UMAP if `umap-learn` is installed) coloured by normal/abnormal, with the
silhouette score (computed on the full-dim features, label-separation, higher=better).
Read-only on existing checkpoints.

Run (GPU):
  python -u -m examples.eeg.latent_compare --data-root <TUAB> \
      --enc "SIGReg (ours)|<.../c1_sigreg_ambient_s1/latest.pth.tar>" \
      --enc "Random|<...same ckpt...>|random" \
      --enc "SIGReg-tangent|<.../c2.../latest.pth.tar>" \
      --enc "PEIRA-tangent|<.../c4.../latest.pth.tar>"
"""
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.eval import extract_features
from examples.eeg.main import build_encoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def feats(ckpt, data_root, device, random=False):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(device)
    if not random:
        enc.load_state_dict(state["encoder"])
    enc.eval()
    dcfg = {"data_root": data_root, "label_scheme": "tuab", "n_channels": int(cfg.model.n_channels),
            "sfreq": 200, "window_sec": 10.0, "n_windows": 16, "num_workers": 8}
    return extract_features(enc, "eval", device, dcfg)


def embed_2d(X):
    """t-SNE 2-D + (optional) UMAP 2-D. Returns {method: coords}."""
    from sklearn.manifold import TSNE
    out = {"t-SNE": TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)}
    try:
        import umap
        out["UMAP"] = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
    except Exception:
        pass
    return out


def main():
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True, help="$WORK (checkpoints under <work>/checkpoints/)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = a.work
    # All four read via the SAME frozen recipe (represent() ambient 256-d -> mean-pool
    # -> StandardScaler -> silhouette); labels name the SSL *training* variant only.
    # "Random-init" = same architecture, weights never trained (a control isolating the
    # contribution of SSL training, NOT a baseline we claim to beat).
    ENCS = [
        ("SIGReg-ambient (ours)", f"{W}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", False),
        ("Random-init (control)", f"{W}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", True),
        ("SIGReg-tangent", f"{W}/checkpoints/c2_sigreg_tangent_s1/latest.pth.tar", False),
        ("PEIRA-tangent", f"{W}/checkpoints/c4_peira_tangent_s1/latest.pth.tar", False),
    ]
    encs = []
    for label, ckpt, rand in ENCS:
        X, y = feats(ckpt, a.data_root, dev, random=rand)
        Xs = StandardScaler().fit_transform(X)
        sil = silhouette_score(Xs, y)
        encs.append((label, Xs, y, sil, embed_2d(Xs)))
        print(f"[latent] {label}: n={len(y)} silhouette={sil:.4f}", flush=True)

    methods = list(encs[0][4].keys())  # t-SNE (+UMAP)
    nrow, ncol = len(methods), len(encs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow), squeeze=False)
    for j, (label, Xs, y, sil, emb) in enumerate(encs):
        for i, m in enumerate(methods):
            ax = axes[i][j]; C = emb[m]
            for cls, col, nm in [(0, "#2f80ed", "normal"), (1, "#e74c3c", "abnormal")]:
                mk = y == cls
                ax.scatter(C[mk, 0], C[mk, 1], s=6, alpha=0.6, c=col, label=nm)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{label}\nsilhouette={sil:.3f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(m, fontsize=10)
    axes[0][-1].legend(fontsize=7, markerscale=1.5, loc="upper right")
    fig.suptitle("Frozen latent space on TUAB eval — normal vs abnormal (recording-level)\n"
                 "same frozen ambient represent() readout for all; silhouette is on full-dim "
                 "features (the 2-D map is illustrative)")
    fig.tight_layout()
    import os
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "latent", "latent_compare.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}", flush=True)
    print("LATENT_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
