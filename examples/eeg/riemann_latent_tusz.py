"""Multiclass manifold-aware latent viz on TUSZ (seizure TYPES).

Same AIRM-vs-Euclidean SPD-latent viz as TUAB/TUEV, on TUSZ seizure-type events (19-ch,
shares the TUH montage; frozen TUAB-pretrained encoder). TUSZ annotations are per-channel
(channel,start_time,stop_time,label,confidence) with label = seizure type (fnsz/gnsz/...
/bckg); we collapse per-channel replication to ~one window per (5 s bucket, class) and keep
only classes with enough events. Distinct output file (riemann_latent_tusz.png).

Run (GPU): python -u -m examples.eeg.riemann_latent_tusz --tusz-root <TUSZ_PREPROCESSED/edf> --work $WORK
"""
import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder
from examples.eeg.tuev_probe import N_CH, SFREQ, WIN, _zscore
from examples.eeg.riemann_latent_tuev import event_spd, euclid_tangent, spd2_to_disk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Standard TUSZ seizure-type vocabulary (+ background); only the ones actually present
# with enough events are kept and remapped to contiguous indices for the viz.
TUSZ_CLASSES = ["fnsz", "gnsz", "spsz", "cpsz", "absz", "tnsz", "cnsz", "tcsz", "mysz", "bckg"]


def _tusz_events(csv_path, class_to_idx):
    """Deduped [(center_sec, class_idx)] — one per (label, 5 s bucket) per recording."""
    seen, out = set(), []
    try:
        with open(csv_path) as fh:
            for line in fh:
                if line.startswith(("#", "channel,")):
                    continue
                p = line.strip().split(",")
                if len(p) < 4:
                    continue
                lab = p[3].strip()
                if lab not in class_to_idx:
                    continue
                try:
                    start, stop = float(p[1]), float(p[2])
                except ValueError:
                    continue
                center = 0.5 * (start + stop)
                key = (lab, int(center // 5))
                if key in seen:
                    continue
                seen.add(key)
                out.append((center, class_to_idx[lab]))
    except Exception:
        return []
    return out


def build_tusz_split(root, split, per_class_cap, rng, class_to_idx):
    """root/split/**/*.edf with matching .csv -> [(edf, start_sample, class_idx)] capped/class."""
    edfs = sorted(glob.glob(os.path.join(root, split, "**", "*.edf"), recursive=True))
    if not edfs:
        raise FileNotFoundError(f"No .edf under {os.path.join(root, split)}")
    half = WIN // 2
    by_label = defaultdict(list)
    for e in edfs:
        c = e[:-4] + ".csv"
        if not os.path.exists(c):
            continue
        for center, idx in _tusz_events(c, class_to_idx):
            by_label[idx].append((e, int(center * SFREQ) - half, idx))
    picked = []
    for idx, items in by_label.items():
        ii = rng.permutation(len(items))
        if per_class_cap > 0:
            ii = ii[:per_class_cap]
        picked.extend(items[i] for i in ii)
    return picked


def main():
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tusz-root", required=True, help="TUSZ_PREPROCESSED/edf (has train/dev/eval)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--per-class-cap", type=int, default=80)
    ap.add_argument("--min-events", type=int, default=15, help="drop classes with fewer events")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)

    class_to_idx = {c: i for i, c in enumerate(TUSZ_CLASSES)}
    raw = build_tusz_split(a.tusz_root, a.split, a.per_class_cap, rng, class_to_idx)
    cnt = Counter(lab for _, _, lab in raw)
    keep = sorted([k for k, v in cnt.items() if v >= a.min_events])
    if len(keep) < 2:
        raise SystemExit(f"[tusz] <2 classes with >= {a.min_events} events: {cnt}")
    remap = {old: i for i, old in enumerate(keep)}
    names = [TUSZ_CLASSES[k] for k in keep]
    items = [(p, s, remap[l]) for (p, s, l) in raw if l in remap]
    print(f"[tusz] classes kept: {names} | {len(items)} events "
          f"({', '.join(f'{names[remap[k]]}={cnt[k]}' for k in keep)})", flush=True)

    ENCS = [
        ("SIGReg-ambient (ours)", f"{a.work}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", False),
        ("Random-init (control)", f"{a.work}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar", True),
        ("SIGReg-tangent",        f"{a.work}/checkpoints/c2_sigreg_tangent_s1/latest.pth.tar", False),
        ("PEIRA-tangent",         f"{a.work}/checkpoints/c4_peira_tangent_s1/latest.pth.tar", False),
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
        print(f"[tusz] {label}: {mats.shape[0]} events SPD({mats.shape[-1]})", flush=True)
        tan = StandardScaler().fit_transform(euclid_tangent(mats))
        sil_euc = silhouette_score(tan, y)
        emb_euc = TSNE(n_components=2, perplexity=a.perplexity, init="pca", random_state=0).fit_transform(tan)
        sil_airm, disk = float("nan"), None
        try:
            from pyriemann.utils.distance import pairwise_distance
            D = pairwise_distance(mats, metric="riemann"); np.fill_diagonal(D, 0.0)
            sil_airm = silhouette_score(D, y, metric="precomputed")
        except Exception as e:  # noqa: BLE001
            print(f"[tusz] {label}: AIRM silhouette failed: {e}", flush=True)
        try:
            import rtsne
            Y = np.asarray(rtsne.RiemannianTSNE(perplexity=a.perplexity, max_iter=a.max_iter).fit_transform(mats))
            disk = np.stack(spd2_to_disk(Y), axis=1)
        except Exception as e:  # noqa: BLE001
            print(f"[tusz] {label}: rtsne failed: {e}", flush=True)
        print(f"[tusz] {label}: silhouette euclidean(tan)={sil_euc:.4f} AIRM(geodesic)={sil_airm:.4f}", flush=True)
        results.append((label, y, emb_euc, sil_euc, disk, sil_airm))

    cmap = plt.get_cmap("tab10")
    n = len(results)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 8.0), squeeze=False)
    for j, (label, y, emb_euc, sil_euc, disk, sil_airm) in enumerate(results):
        for k in range(len(names)):
            mk = y == k
            ax[0][j].scatter(emb_euc[mk, 0], emb_euc[mk, 1], s=8, alpha=0.6, color=cmap(k), label=names[k])
        ax[0][j].set_title(f"{label}\nEuclidean t-SNE  sil={sil_euc:.3f}", fontsize=9)
        ax[0][j].set_xticks([]); ax[0][j].set_yticks([])
        if disk is not None:
            th = np.linspace(0, 2 * np.pi, 200)
            ax[1][j].plot(np.cos(th), np.sin(th), color="#bbb", lw=0.8)
            for k in range(len(names)):
                mk = y == k
                ax[1][j].scatter(disk[mk, 0], disk[mk, 1], s=8, alpha=0.6, color=cmap(k))
            ax[1][j].set_aspect("equal"); ax[1][j].set_xlim(-1.05, 1.05); ax[1][j].set_ylim(-1.05, 1.05)
        ax[1][j].set_title(f"AIRM Riemannian t-SNE (Poincare)\nsil={sil_airm:.3f}", fontsize=9)
        ax[1][j].set_xticks([]); ax[1][j].set_yticks([])
    ax[0][0].legend(fontsize=7, markerscale=1.4, ncol=2, loc="upper right")
    fig.suptitle(f"Frozen SPD latents on TUSZ {a.split} — {len(names)} seizure types "
                 "(cross-task, same-site)\nEuclidean (tangent) t-SNE vs AIRM Riemannian t-SNE; "
                 "silhouette is full-dim", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "riemann", "riemann_latent_tusz.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}", flush=True)
    print("RIEMANN_TUSZ_DONE", flush=True)


if __name__ == "__main__":
    main()
