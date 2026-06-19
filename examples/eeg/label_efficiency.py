"""Label-efficiency curve for the frozen EEG-JEPA probe — the "value of SSL" figure.

Freeze the encoder, extract recording-level features ONCE (train + eval), then for
each label fraction p re-fit a logistic-regression probe on p% of TRAIN recordings
(stratified by class) and score held-out eval patients. The probe's `C` is selected
ONCE on a patient-disjoint dev split carved from TRAIN — never on eval (no leakage).
Saves JSON + a figure with random-floor / classical-Riemannian / fine-tuned
foundation-model reference overlays.

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


@torch.no_grad()
def extract(encoder, split, device, data_cfg):
    """-> X [N_rec, D] mean-pooled features, y labels, pid patient ids (from filename)."""
    cfg = EEGConfig(**(data_cfg or {}))
    cfg.split, cfg.mode = split, "probe"
    ds = EEGDataset(cfg)
    paths = [p for p, _ in ds.items]   # deterministic order == loader order (shuffle=False)
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
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
    clf.fit(sc.transform(Xtr), ytr)
    pe = clf.predict(sc.transform(Xev))
    se = clf.predict_proba(sc.transform(Xev))[:, 1]
    return balanced_accuracy_score(yev, pe), roc_auc_score(yev, se)


def main():
    a = sys.argv
    ckpt = a[a.index("--ckpt") + 1]
    out = a[a.index("--out") + 1] if "--out" in a else "results/label_eff"
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    enc = build_encoder(cfg.model).to(dev)
    enc.load_state_dict(state["encoder"]); enc.eval()

    print("[lab] extract train...", flush=True); Xtr, ytr, pidtr = extract(enc, "train", dev, dcfg)
    print("[lab] extract eval...", flush=True);  Xev, yev, _ = extract(enc, "eval", dev, dcfg)
    print("[lab] random-encoder floor...", flush=True)
    rnd = build_encoder(cfg.model).to(dev).eval()
    Rtr, ry, _ = extract(rnd, "train", dev, dcfg); Rev, rey, _ = extract(rnd, "eval", dev, dcfg)
    floor = fit_score(Rtr, ry, Rev, rey, 1.0)[0]

    # patient-disjoint DEV split from TRAIN (15% of patients) -> select C here, NEVER on eval
    rng = np.random.default_rng(0)
    pats = np.unique(pidtr); rng.shuffle(pats)
    devset = set(pats[: max(1, int(0.15 * len(pats)))].tolist())
    dmask = np.array([p in devset for p in pidtr])
    bestC, bestv = 1.0, -1.0
    for C in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        v = fit_score(Xtr[~dmask], ytr[~dmask], Xtr[dmask], ytr[dmask], C)[0]
        if v > bestv:
            bestv, bestC = v, C
    print(f"[lab] C={bestC} selected on dev (dev BalAcc {bestv:.3f}); fit pool = train minus dev", flush=True)

    Xfit, yfit = Xtr[~dmask], ytr[~dmask]   # eval stays the untouched held-out test
    fracs = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    rows = []
    for p in fracs:
        bas, au = [], []
        for s in range(5):
            r = np.random.default_rng(100 + s)
            idx = []
            for cls in (0, 1):
                ci = np.where(yfit == cls)[0]
                k = min(len(ci), max(2, int(round(p * len(ci)))))
                idx.extend(r.choice(ci, size=k, replace=False).tolist())
            b, c = fit_score(Xfit[np.array(idx)], yfit[np.array(idx)], Xev, yev, bestC)
            bas.append(b); au.append(c)
        rows.append({"frac": p, "n": int(round(p * len(yfit))),
                     "balacc_mean": float(np.mean(bas)), "balacc_std": float(np.std(bas)),
                     "auroc_mean": float(np.mean(au))})
        print(f"[lab] p={p:.2f} (n~{rows[-1]['n']}) BalAcc {np.mean(bas):.4f} ± {np.std(bas):.4f}", flush=True)

    json.dump({"floor": float(floor), "C": bestC, "riemann": RIEMANN_BALACC, "rows": rows},
              open(f"{out}/label_eff.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [r["frac"] * 100 for r in rows]
    ms = [r["balacc_mean"] for r in rows]
    ss = [r["balacc_std"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhspan(*FT_BAND, color="green", alpha=0.12, label="fine-tuned foundation models\n(cross-corpus)")
    ax.axhline(RIEMANN_BALACC, color="gray", ls="--", lw=1, label=f"classical Riemannian ({RIEMANN_BALACC:.2f})")
    ax.axhline(floor, color="red", ls=":", lw=1.2, label=f"random-encoder floor ({floor:.2f})")
    ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, color="C0", lw=2, label="frozen JEPA probe (5 seeds)")
    ax.set_xscale("log")
    ax.set_xlabel("% of TRAIN labels used to fit the probe")
    ax.set_ylabel("Balanced accuracy (held-out patients)")
    ax.set_title("Label efficiency — frozen in-domain EEG-JEPA on TUAB")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/label_efficiency.png", dpi=140)
    print(f"[lab] saved {out}/label_efficiency.png + label_eff.json", flush=True)


if __name__ == "__main__":
    main()
