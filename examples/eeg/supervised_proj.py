"""Label-aware projection: does the decodable structure (probe BA) become VISIBLE?

For TUEV/TUSZ SPD latents, on held-out EVAL, compare:
  * UNSUPERVISED: t-SNE of the cov-tangent (silhouette = clustering) + k-means named by majority
    label (purity / NMI / ARI).
  * SUPERVISED:   tangent-space LDA FIT ON TRAIN, shown on EVAL (the linear probe's view).

Random-init is the control: a supervised projection should separate OURS (decodable) on held-out
eval but NOT random -- proving it reflects generalizable signal, not LDA overfitting. Fit-on-train
/ show-on-eval is what makes this honest (never fit+show on the same set).

Run (GPU): python -u -m examples.eeg.supervised_proj --dataset tuev --root <...> --work $WORK
"""
import argparse
import os
from collections import Counter

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder
from examples.eeg.tuev_probe import build_split, CLASSES
from examples.eeg.riemann_latent_tuev import event_spd, euclid_tangent
from examples.eeg.riemann_latent_tusz import build_tusz_split, TUSZ_CLASSES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_tuev(root, tcap, ecap, rng):
    return build_split(root, "train", tcap, rng), build_split(root, "eval", ecap, rng), list(CLASSES)


def load_tusz(root, tcap, ecap, rng, minev=15):
    c2i = {c: i for i, c in enumerate(TUSZ_CLASSES)}
    raw = build_tusz_split(root, "train", tcap, rng, c2i)
    cnt = Counter(l for _, _, l in raw)
    keep = sorted([k for k, v in cnt.items() if v >= minev])
    remap = {o: i for i, o in enumerate(keep)}
    names = [TUSZ_CLASSES[k] for k in keep]
    tr = [(p, s, remap[l]) for p, s, l in raw if l in remap]
    rev = build_tusz_split(root, "eval", ecap, rng, c2i)
    ev = [(p, s, remap[l]) for p, s, l in rev if l in remap]
    return tr, ev, names


def purity_nmi_ari(y, cl):
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    pur = sum(np.bincount(y[cl == c]).max() for c in set(cl) if (cl == c).any())
    return pur / len(y), normalized_mutual_info_score(y, cl), adjusted_rand_score(y, cl)


def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, balanced_accuracy_score
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["tuev", "tusz"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--train-cap", type=int, default=200)
    ap.add_argument("--eval-cap", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    tr, ev, names = (load_tuev if a.dataset == "tuev" else load_tusz)(a.root, a.train_cap, a.eval_cap, rng)
    K = len(names)
    print(f"[sup] {a.dataset}: train={len(tr)} eval={len(ev)} classes={names}", flush=True)

    ck = f"{a.work}/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar"
    state = torch.load(ck, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    rows = []
    for label, rand in [("SIGReg-ambient (ours)", False), ("Random-init (control)", True)]:
        enc = build_encoder(cfg.model).to(dev)
        if not rand:
            enc.load_state_dict(state["encoder"])
        enc.eval()
        Mtr, ytr = event_spd(enc, tr, dev)
        Mev, yev = event_spd(enc, ev, dev)
        sc = StandardScaler().fit(euclid_tangent(Mtr))
        Xtr, Xev = sc.transform(euclid_tangent(Mtr)), sc.transform(euclid_tangent(Mev))

        sil_un = silhouette_score(Xev, yev)
        emb_un = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(Xev)

        # shrinkage-regularized LDA (eigen solver supports transform + Ledoit-Wolf shrinkage)
        # -- plain svd-LDA overfits the 528-dim tangent and does not generalize to eval.
        lda = LDA(solver="eigen", shrinkage="auto",
                  n_components=min(K - 1, Xtr.shape[1])).fit(Xtr, ytr)
        Lev = lda.transform(Xev)
        ba = balanced_accuracy_score(yev, lda.predict(Xev))
        sil_sup = silhouette_score(Lev, yev)
        emb_sup = Lev[:, :2] if Lev.shape[1] >= 2 else np.c_[Lev[:, 0], np.zeros(len(Lev))]

        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xev)
        pur, nmi, ari = purity_nmi_ari(yev, km.labels_)
        print(f"[sup] {label}: unsup_sil={sil_un:.3f} | LDA eval BA={ba:.3f} sup_sil={sil_sup:.3f} | "
              f"kmeans purity={pur:.3f} NMI={nmi:.3f} ARI={ari:.3f}", flush=True)
        rows.append((label, yev, emb_un, sil_un, emb_sup, sil_sup, ba, pur, nmi))

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    for i, (label, yev, emb_un, sil_un, emb_sup, sil_sup, ba, pur, nmi) in enumerate(rows):
        for k in range(K):
            mk = yev == k
            ax[i][0].scatter(emb_un[mk, 0], emb_un[mk, 1], s=7, alpha=0.6, color=cmap(k), label=names[k])
            ax[i][1].scatter(emb_sup[mk, 0], emb_sup[mk, 1], s=7, alpha=0.6, color=cmap(k))
        ax[i][0].set_title(f"{label}\nUNSUPERVISED t-SNE  sil={sil_un:.3f}\n"
                           f"k-means: purity={pur:.2f} NMI={nmi:.2f}", fontsize=8)
        ax[i][1].set_title(f"{label}\nSUPERVISED LDA (train$\\to$eval)  sil={sil_sup:.3f}\n"
                           f"eval BA={ba:.3f}", fontsize=8)
        for j in (0, 1):
            ax[i][j].set_xticks([]); ax[i][j].set_yticks([])
    ax[0][0].legend(fontsize=6, ncol=2, markerscale=1.3, loc="upper right")
    fig.suptitle(f"{a.dataset.upper()} latent — unsupervised clustering vs label-aware projection "
                 "(held-out eval)\nsupervised separation that holds on EVAL = generalizable "
                 "decodability (random control should NOT separate)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "riemann", f"supervised_proj_{a.dataset}.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}", flush=True)
    print("SUP_PROJ_DONE", flush=True)


if __name__ == "__main__":
    main()
