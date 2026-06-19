"""Label-efficiency curves for the frozen EEG-JEPA probe — the honest "value of SSL" figure.

Freeze the encoder, extract recording-level features ONCE (train + eval), then for each
label fraction p re-fit a logistic-regression probe on p% of TRAIN recordings (stratified)
and score held-out eval patients. We plot the SAME curve for a RANDOM (untrained) encoder
of the same architecture: the GAP between the two curves is what the SSL actually buys.
(EEG abnormality is power-driven, so random conv features are a strong baseline — this
figure quantifies the SSL's marginal contribution honestly.) The probe's `C` is selected
ONCE on a patient-disjoint dev split carved from TRAIN — never on eval (no leakage).

Run on a GPU node:
  python -u -m examples.eeg.label_efficiency --ckpt <.../latest.pth.tar> --out results/label_eff
"""
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder

RIEMANN_BALACC = 0.761                 # our 0-param classical baseline
FT_BAND = (0.814, 0.829)               # fine-tuned foundation models on TUAB (LaBraM-Base .. CBraMod)
FRACS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]


@torch.no_grad()
def extract(encoder, split, device, data_cfg):
    cfg = EEGConfig(**(data_cfg or {}))
    cfg.split, cfg.mode = split, "probe"
    ds = EEGDataset(cfg)
    paths = [p for p, _ in ds.items]
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False,
                                         num_workers=cfg.num_workers, pin_memory=True)
    X, y, pid, gi = [], [], [], 0
    for wins, labels, ok in loader:
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        z = encoder.represent(flat).reshape(B, N, -1).mean(dim=1).cpu().numpy()
        for k in range(B):
            if bool(ok[k]):
                X.append(z[k]); y.append(int(labels[k]))
                pid.append(os.path.basename(paths[gi]).split("_")[0])
            gi += 1
    return np.stack(X), np.array(y), np.array(pid)


def fit_score(Xtr, ytr, Xev, yev, C):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C).fit(sc.transform(Xtr), ytr)
    pe = clf.predict(sc.transform(Xev)); se = clf.predict_proba(sc.transform(Xev))[:, 1]
    return balanced_accuracy_score(yev, pe), roc_auc_score(yev, se)


def curve(Xtr, ytr, pidtr, Xev, yev):
    """Patient-disjoint dev split -> select C (never on eval) -> label-fraction sweep (5 seeds)."""
    rng = np.random.default_rng(0)
    pats = np.unique(pidtr); rng.shuffle(pats)
    devset = set(pats[: max(1, int(0.15 * len(pats)))].tolist())
    dm = np.array([p in devset for p in pidtr])
    bestC, bestv = 1.0, -1.0
    for C in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        v = fit_score(Xtr[~dm], ytr[~dm], Xtr[dm], ytr[dm], C)[0]
        if v > bestv:
            bestv, bestC = v, C
    Xfit, yfit = Xtr[~dm], ytr[~dm]
    rows = []
    for p in FRACS:
        bas = []
        for s in range(5):
            r = np.random.default_rng(100 + s)
            idx = []
            for cls in (0, 1):
                ci = np.where(yfit == cls)[0]
                k = min(len(ci), max(2, int(round(p * len(ci)))))
                idx.extend(r.choice(ci, size=k, replace=False).tolist())
            bas.append(fit_score(Xfit[np.array(idx)], yfit[np.array(idx)], Xev, yev, bestC)[0])
        rows.append({"frac": p, "n": int(round(p * len(yfit))),
                     "mean": float(np.mean(bas)), "std": float(np.std(bas))})
    return bestC, rows


def main():
    a = sys.argv
    ckpt = a[a.index("--ckpt") + 1]
    out = a[a.index("--out") + 1] if "--out" in a else "results/label_eff"
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"]); dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    enc = build_encoder(cfg.model).to(dev); enc.load_state_dict(state["encoder"]); enc.eval()

    print("[lab] extract JEPA features...", flush=True)
    Xtr, ytr, pidtr = extract(enc, "train", dev, dcfg); Xev, yev, _ = extract(enc, "eval", dev, dcfg)
    print("[lab] extract RANDOM-encoder features...", flush=True)
    rnd = build_encoder(cfg.model).to(dev).eval()
    Rtr, ry, pidr = extract(rnd, "train", dev, dcfg); Rev, rey, _ = extract(rnd, "eval", dev, dcfg)

    Cj, jrows = curve(Xtr, ytr, pidtr, Xev, yev)
    Cr, rrows = curve(Rtr, ry, pidr, Rev, rey)
    print(f"[lab] JEPA  C={Cj}:", [(r['frac'], round(r['mean'], 3)) for r in jrows], flush=True)
    print(f"[lab] RAND  C={Cr}:", [(r['frac'], round(r['mean'], 3)) for r in rrows], flush=True)
    gain = jrows[-1]['mean'] - rrows[-1]['mean']
    print(f"[lab] SSL gain @100% labels = {gain:+.3f} (JEPA {jrows[-1]['mean']:.3f} vs random {rrows[-1]['mean']:.3f})", flush=True)
    json.dump({"jepa": jrows, "random": rrows, "Cj": Cj, "Cr": Cr, "riemann": RIEMANN_BALACC},
              open(f"{out}/label_eff.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [r["frac"] * 100 for r in jrows]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.axhspan(*FT_BAND, color="green", alpha=0.10, label="fine-tuned foundation models (cross-corpus)")
    ax.axhline(RIEMANN_BALACC, color="gray", ls="--", lw=1, label=f"classical Riemannian ({RIEMANN_BALACC:.2f})")
    ax.errorbar(xs, [r["mean"] for r in jrows], yerr=[r["std"] for r in jrows],
                marker="o", capsize=3, lw=2, color="C0", label="frozen JEPA probe")
    ax.errorbar(xs, [r["mean"] for r in rrows], yerr=[r["std"] for r in rrows],
                marker="s", capsize=3, lw=2, color="C3", ls="--", label="random-encoder probe")
    ax.set_xscale("log")
    ax.set_xlabel("% of TRAIN labels used to fit the probe")
    ax.set_ylabel("Balanced accuracy (held-out patients)")
    ax.set_title(f"Label efficiency on TUAB — SSL adds {gain:+.2f} over random conv features")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/label_efficiency.png", dpi=140)
    print(f"[lab] saved {out}/label_efficiency.png + label_eff.json", flush=True)


if __name__ == "__main__":
    main()
