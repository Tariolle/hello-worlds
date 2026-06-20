"""Multiclass manifold-aware latent viz on TUEV (6 event classes).

Generalizes the TUAB AIRM-vs-Euclidean SPD-latent viz to >2 labels, so we can see how
visualization quality scales with the number of classes. Per TUEV event window we build the
frozen encoder's SPD covariance (cov_features -> temporal covariance) and embed it two ways:
Euclidean (tangent) t-SNE vs AIRM Riemannian t-SNE (de Surrel's rtsne -> Poincare disk).
Multiclass silhouette (Euclidean tangent vs AIRM geodesic) is the quantitative number.

Cross-TASK, same-site (TUEV shares the TUH 19-ch montage; encoder is TUAB-pretrained, frozen).

Run (GPU): python -u -m examples.eeg.riemann_latent_tuev --tuev-root <TUEV_PREP200> --work $WORK
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.geometry import temporal_covariance, spd_logm, upper_tri_vec
from examples.eeg.main import build_encoder
from examples.eeg.tuev_probe import build_split, CLASSES, N_CH, WIN, _zscore

import pyedflib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def event_spd(encoder, items, device):
    """TUEV event windows -> [N, d, d] SPD covariance of cov_features + labels (0..5)."""
    by_path = defaultdict(list)
    for path, start, lab in items:
        by_path[path].append((start, lab))
    mats, labs, buf, blab = [], [], [], []

    def flush():
        if not buf:
            return
        t = torch.from_numpy(np.stack(buf)).to(device)
        C = temporal_covariance(encoder.cov_features(t))   # [B, d_cov, d_cov]
        mats.extend(list(C.float().cpu().numpy())); labs.extend(blab)
        buf.clear(); blab.clear()

    for path, lst in by_path.items():
        try:
            f = pyedflib.EdfReader(path)
        except Exception:
            continue
        try:
            if f.signals_in_file < N_CH:
                continue
            nsamp = int(min(f.getNSamples()[:N_CH]))
            for start, lab in lst:
                if start < 0 or start + WIN > nsamp:
                    continue
                x = np.empty((N_CH, WIN), dtype=np.float32)
                try:
                    for c in range(N_CH):
                        x[c] = f.readSignal(c, start, WIN)
                except Exception:
                    continue
                buf.append(_zscore(x, axis=1)); blab.append(lab)
                if len(buf) >= 128:
                    flush()
        finally:
            f._close()
    flush()
    return np.stack(mats), np.asarray(labs)


def euclid_tangent(mats):
    C = torch.from_numpy(mats).double()
    return upper_tri_vec(spd_logm(C)).numpy()


def spd2_to_disk(M):
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 1, 1]
    det = np.clip(a * c - b * b, 1e-9, None)
    tau = (b + 1j * np.sqrt(det)) / np.clip(a, 1e-9, None)
    w = (tau - 1j) / (tau + 1j)
    return np.real(w), np.imag(w)


def main():
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tuev-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--per-class-cap", type=int, default=80, help="events/class for the viz (balance + rtsne cost)")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    items = build_split(a.tuev_root, "eval", a.per_class_cap, rng)
    print(f"[tuev-riem] {len(items)} event windows (cap {a.per_class_cap}/class)", flush=True)

    ENCS = [
        ("SIGReg-ambient (ours)", f"{a.work}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", False),
        ("Random-init (control)", f"{a.work}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", True),
    ]
    results = []
    for label, ckpt, rand in ENCS:
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        cfg = OmegaConf.create(state["cfg"])
        enc = build_encoder(cfg.model).to(dev)
        if not rand:
            enc.load_state_dict(state["encoder"])
        enc.eval()
        mats, y = event_spd(enc, items, dev)
        dist = ", ".join(f"{CLASSES[i]}={int(c)}" for i, c in enumerate(np.bincount(y, minlength=6)))
        print(f"[tuev-riem] {label}: {mats.shape[0]} events SPD({mats.shape[-1]}) | {dist}", flush=True)

        tan = StandardScaler().fit_transform(euclid_tangent(mats))
        sil_euc = silhouette_score(tan, y)
        emb_euc = TSNE(n_components=2, perplexity=a.perplexity, init="pca", random_state=0).fit_transform(tan)

        sil_airm, disk = float("nan"), None
        try:
            from pyriemann.utils.distance import pairwise_distance
            D = pairwise_distance(mats, metric="riemann"); np.fill_diagonal(D, 0.0)
            sil_airm = silhouette_score(D, y, metric="precomputed")
        except Exception as e:  # noqa: BLE001
            print(f"[tuev-riem] {label}: AIRM silhouette failed: {e}", flush=True)
        try:
            import rtsne
            Y = np.asarray(rtsne.RiemannianTSNE(perplexity=a.perplexity, max_iter=a.max_iter).fit_transform(mats))
            disk = np.stack(spd2_to_disk(Y), axis=1)
        except Exception as e:  # noqa: BLE001
            print(f"[tuev-riem] {label}: rtsne failed: {e}", flush=True)
        print(f"[tuev-riem] {label}: silhouette euclidean(tan)={sil_euc:.4f} AIRM(geodesic)={sil_airm:.4f}", flush=True)
        results.append((label, y, emb_euc, sil_euc, disk, sil_airm))

    cmap = plt.get_cmap("tab10")
    n = len(results)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 8.0), squeeze=False)
    for j, (label, y, emb_euc, sil_euc, disk, sil_airm) in enumerate(results):
        for k in range(6):
            mk = y == k
            ax[0][j].scatter(emb_euc[mk, 0], emb_euc[mk, 1], s=8, alpha=0.6, color=cmap(k), label=CLASSES[k])
        ax[0][j].set_title(f"{label}\nEuclidean t-SNE  sil={sil_euc:.3f}", fontsize=9)
        ax[0][j].set_xticks([]); ax[0][j].set_yticks([])
        if disk is not None:
            th = np.linspace(0, 2 * np.pi, 200)
            ax[1][j].plot(np.cos(th), np.sin(th), color="#bbb", lw=0.8)
            for k in range(6):
                mk = y == k
                ax[1][j].scatter(disk[mk, 0], disk[mk, 1], s=8, alpha=0.6, color=cmap(k))
            ax[1][j].set_aspect("equal"); ax[1][j].set_xlim(-1.05, 1.05); ax[1][j].set_ylim(-1.05, 1.05)
        ax[1][j].set_title(f"AIRM Riemannian t-SNE (Poincare)\nsil={sil_airm:.3f}", fontsize=9)
        ax[1][j].set_xticks([]); ax[1][j].set_yticks([])
    ax[0][0].legend(fontsize=7, markerscale=1.4, ncol=2, loc="upper right")
    fig.suptitle("Frozen SPD latents on TUEV eval — 6 event classes (cross-task, same-site)\n"
                 "Euclidean (tangent) t-SNE vs AIRM Riemannian t-SNE; silhouette is full-dim",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "riemann", "riemann_latent_tuev.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}", flush=True)
    print("RIEMANN_TUEV_DONE", flush=True)


if __name__ == "__main__":
    main()
