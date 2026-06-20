"""Visualise the frozen latent space: best JEPA encoder vs random encoder.

Per-recording mean-pooled features (the exact probe representation, eval.py:extract_features)
on the patient-disjoint EVAL split, projected to 2D with PCA and t-SNE, coloured by label
(normal/abnormal). Separation is quantified per panel by the 2D silhouette score; the real
full-D linear-probe BalAcc is annotated for reference (2D silhouette UNDER-states separability
that a linear probe sees in the full space, so read it as a lower bound on structure).

  python -u -m examples.eeg.latent_viz --ckpt <best/latest.pth.tar> --out results/latent
"""
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from examples.eeg.main import build_encoder  # noqa: E402
from examples.eeg.eval import extract_features  # noqa: E402

a = sys.argv
ckpt = a[a.index("--ckpt") + 1]
out = a[a.index("--out") + 1] if "--out" in a else "results/latent"
# reference full-D linear-probe BalAcc (held-out patients), for the suptitle
ba_jepa = float(a[a.index("--ba_jepa") + 1]) if "--ba_jepa" in a else 0.833
ba_rand = float(a[a.index("--ba_rand") + 1]) if "--ba_rand" in a else 0.787
os.makedirs(out, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load(ckpt, map_location=dev, weights_only=False)
cfg = OmegaConf.create(state["cfg"])
data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

enc = build_encoder(cfg.model).to(dev)
enc.load_state_dict(state["encoder"]); enc.eval()
rnd = build_encoder(cfg.model).to(dev).eval()   # same arch, untrained = random floor

print("[viz] extracting EVAL features (best JEPA + random)...", flush=True)
Xj, y = extract_features(enc, "eval", dev, data_cfg, "mean")
Xr, yr = extract_features(rnd, "eval", dev, data_cfg, "mean")
assert np.array_equal(y, yr), "label order mismatch between encoders"
print(f"[viz] {len(y)} eval recordings | {int((y == 1).sum())} abnormal / {int((y == 0).sum())} normal", flush=True)


def reduce(X):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0).fit_transform(Xs)
    perp = max(5, min(30, (len(X) - 1) // 3))
    try:
        tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                    learning_rate="auto", random_state=0).fit_transform(Xs)
    except Exception as e:  # t-SNE is the cherry on top; never let it kill the PCA panel
        print(f"[viz] t-SNE failed ({e}); using PCA in its place", flush=True)
        tsne = pca
    return pca, tsne


def sil(emb):
    try:
        return float(silhouette_score(emb, y))
    except Exception:
        return float("nan")


(pj, tj) = reduce(Xj)
(pr, tr) = reduce(Xr)

panels = [("Random encoder · PCA", pr), ("Best JEPA · PCA", pj),
          ("Random encoder · t-SNE", tr), ("Best JEPA · t-SNE", tj)]

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, (title, emb) in zip(axes.ravel(), panels):
    for lab, c, name in [(0, "#1f77b4", "normal"), (1, "#d62728", "abnormal")]:
        m = y == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=16, c=c, alpha=0.6, label=name, edgecolors="none")
    ax.set_title(f"{title}   (silhouette {sil(emb):+.3f})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].legend(fontsize=8, loc="best", framealpha=0.9)
fig.suptitle(
    "Frozen latent space on TUAB eval (held-out patients) — best JEPA vs random encoder\n"
    f"full-D linear-probe BalAcc: JEPA {ba_jepa:.3f}  vs  random {ba_rand:.3f}  (+{ba_jepa - ba_rand:.3f})",
    fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{out}/latent_space.png", dpi=140)
print(f"[viz] saved {out}/latent_space.png", flush=True)

json.dump({"y": y.tolist(),
           "jepa": {"pca": pj.tolist(), "tsne": tj.tolist(), "sil_pca": sil(pj), "sil_tsne": sil(tj)},
           "random": {"pca": pr.tolist(), "tsne": tr.tolist(), "sil_pca": sil(pr), "sil_tsne": sil(tr)}},
          open(f"{out}/latent_coords.json", "w"))
print(f"[viz] wrote {out}/latent_coords.json", flush=True)
