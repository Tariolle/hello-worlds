"""Manifold-aware latent visualization: AIRM Riemannian t-SNE vs Euclidean t-SNE.

For each frozen encoder we build ONE recording-level SPD covariance from the learned
cov_features (temporal covariance over all windows' features), then embed the cloud of
SPD matrices two ways:

  * Euclidean: vec(logm(C)) at the identity (the LE tangent we already use) -> StandardScaler
    -> sklearn t-SNE.  Silhouette on the full-dim tangent vector (Euclidean).
  * Riemannian: the raw SPD matrices -> rtsne.RiemannianTSNE under the Affine-Invariant
    Riemannian Metric (de Surrel, r_tSNE_in_C) -> embeds onto SPD(2) ~= the hyperbolic
    plane, rendered in the Poincare disk.  Silhouette on full-dim AIRM geodesic distances
    (pyriemann), the metric-faithful number.

The point is not a leaderboard: it shows whether respecting the SPD geometry (AIRM) reveals
structure that flattening-to-tangent + Euclidean t-SNE misses. Read-only on checkpoints.

Run (GPU for extraction):
  python -u -m examples.eeg.riemann_latent --data-root <TUAB> --work $WORK
"""
import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.geometry import temporal_covariance, spd_logm, upper_tri_vec
from examples.eeg.main import build_encoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def recording_spd(encoder, split, device, dcfg):
    """[N_rec, d, d] one SPD temporal covariance per recording (over all windows)."""
    cfg = EEGConfig(**dcfg); cfg.split, cfg.mode = split, "probe"
    ds = EEGDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False,
                                         num_workers=cfg.num_workers)
    mats, labels = [], []
    for wins, labs, ok in loader:                 # wins: [B, N, C, T]
        for k in range(wins.shape[0]):
            if not bool(ok[k]):
                continue
            w = wins[k].to(device)                # [N, C, T]
            fm = encoder.cov_features(w)          # [N, d_cov, T']
            d = fm.shape[1]
            feat = fm.permute(1, 0, 2).reshape(1, d, -1)   # [1, d_cov, N*T']
            C = temporal_covariance(feat)[0]      # [d_cov, d_cov] SPD
            mats.append(C.float().cpu().numpy())
            labels.append(int(labs[k]))
    return np.stack(mats), np.asarray(labels)


def euclid_tangent(mats):
    """SPD matrices -> Log-Euclidean tangent vectors vec_sqrt2(logm(C)). [N, d(d+1)/2]."""
    C = torch.from_numpy(mats).double()
    return upper_tri_vec(spd_logm(C)).numpy()


def spd2_to_disk(M):
    """SPD(2) -> Poincare disk point. M=[[a,b],[b,c]] -> tau=(b+i*sqrt(det))/a in H,
    scale-invariant; then w=(tau-i)/(tau+i), |w|<1."""
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
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--perplexity", type=float, default=30.0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = a.work
    ENCS = [
        ("SIGReg-ambient (ours)", f"{W}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", False),
        ("Random-init (control)", f"{W}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", True),
        ("SIGReg-tangent",        f"{W}/checkpoints/c2_sigreg_tangent_s1/latest.pth.tar", False),
        ("PEIRA-tangent",         f"{W}/checkpoints/c4_peira_tangent_s1/latest.pth.tar", False),
    ]

    results = []
    for label, ckpt, rand in ENCS:
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        cfg = OmegaConf.create(state["cfg"])
        enc = build_encoder(cfg.model).to(dev)
        if not rand:
            enc.load_state_dict(state["encoder"])
        enc.eval()
        dcfg = {"data_root": a.data_root, "label_scheme": "tuab",
                "n_channels": int(cfg.model.n_channels), "sfreq": 200,
                "window_sec": 10.0, "n_windows": 16, "num_workers": 8}
        mats, y = recording_spd(enc, "eval", dev, dcfg)
        print(f"[riem] {label}: {mats.shape[0]} recordings, SPD({mats.shape[-1]})", flush=True)

        # --- Euclidean branch ---
        tan = euclid_tangent(mats)
        tan_s = StandardScaler().fit_transform(tan)
        sil_euc = silhouette_score(tan_s, y)
        emb_euc = TSNE(n_components=2, perplexity=a.perplexity, init="pca",
                       random_state=0).fit_transform(tan_s)

        # --- Riemannian branch (AIRM) --- decoupled: silhouette and embedding fail independently
        sil_airm, disk = float("nan"), None
        try:
            from pyriemann.utils.distance import pairwise_distance
            D = pairwise_distance(mats, metric="riemann")
            sil_airm = silhouette_score(D, y, metric="precomputed")
        except Exception as e:  # noqa: BLE001
            print(f"[riem] {label}: AIRM silhouette failed: {e}", flush=True)
        try:
            import rtsne
            m = rtsne.RiemannianTSNE(perplexity=a.perplexity, max_iter=a.max_iter)
            Y = np.asarray(m.fit_transform(mats))         # [N, 2, 2] SPD
            disk = np.stack(spd2_to_disk(Y), axis=1)      # [N, 2]
        except Exception as e:  # noqa: BLE001
            print(f"[riem] {label}: rtsne embedding failed: {e}", flush=True)
        print(f"[riem] {label}: silhouette  euclidean(tan)={sil_euc:.4f}  "
              f"AIRM(geodesic)={sil_airm:.4f}", flush=True)
        results.append((label, y, emb_euc, sil_euc, disk, sil_airm))

    # --- figure: rows = {Euclidean t-SNE, AIRM Riemannian t-SNE}, cols = encoders ---
    n = len(results)
    fig, ax = plt.subplots(2, n, figsize=(3.5 * n, 7.2), squeeze=False)
    for j, (label, y, emb_euc, sil_euc, disk, sil_airm) in enumerate(results):
        for cls, col, nm in [(0, "#2f80ed", "normal"), (1, "#e74c3c", "abnormal")]:
            mk = y == cls
            ax[0][j].scatter(emb_euc[mk, 0], emb_euc[mk, 1], s=7, alpha=0.6, c=col, label=nm)
        ax[0][j].set_title(f"{label}\nEuclidean t-SNE  sil={sil_euc:.3f}", fontsize=9)
        ax[0][j].set_xticks([]); ax[0][j].set_yticks([])
        if disk is not None:
            th = np.linspace(0, 2 * np.pi, 200)
            ax[1][j].plot(np.cos(th), np.sin(th), color="#bbb", lw=0.8)
            for cls, col in [(0, "#2f80ed"), (1, "#e74c3c")]:
                mk = y == cls
                ax[1][j].scatter(disk[mk, 0], disk[mk, 1], s=7, alpha=0.6, c=col)
            ax[1][j].set_aspect("equal"); ax[1][j].set_xlim(-1.05, 1.05); ax[1][j].set_ylim(-1.05, 1.05)
        ax[1][j].set_title(f"AIRM Riemannian t-SNE (Poincare)\nsil={sil_airm:.3f}", fontsize=9)
        ax[1][j].set_xticks([]); ax[1][j].set_yticks([])
    ax[0][-1].legend(fontsize=7, markerscale=1.6, loc="upper right")
    fig.suptitle("Frozen SPD latents on TUAB eval — Euclidean (tangent) t-SNE vs AIRM Riemannian "
                 "t-SNE\nsilhouette is full-dim (Euclidean tangent vs AIRM geodesic); 2-D maps "
                 "are illustrative", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "riemann", "riemann_latent.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}", flush=True)
    print("RIEMANN_LATENT_DONE", flush=True)


if __name__ == "__main__":
    main()
