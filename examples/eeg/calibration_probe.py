"""Calibration & selective prediction of a FROZEN EEG representation.

Fit a linear probe on TRAIN features (frozen encoder), evaluate on patient-disjoint
EVAL and report, beyond accuracy:
  * balanced accuracy + AUROC  (to confirm accuracy parity across cells),
  * Expected Calibration Error (ECE), raw and after temperature scaling
    (T fit on a patient-disjoint dev split carved from TRAIN — never on eval),
  * selective prediction: risk-coverage AURC + accuracy at 80% coverage.

Question: do geometry/PEIRA reps calibrate better / abstain more reliably than
ambient/random at TIED accuracy? (The axis where the accuracy-null may flip positive.)

Run:  python -u -m examples.eeg.calibration_probe --ckpt <...> --data-root <TUAB> [--random]
"""
import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract(encoder, split, device, data_cfg):
    cfg = EEGConfig(**(data_cfg or {})); cfg.split, cfg.mode = split, "probe"
    ds = EEGDataset(cfg)
    paths = [p for p, _ in ds.items]
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=cfg.num_workers)
    X, y, pid, gi = [], [], [], 0
    for wins, labels, ok in loader:
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device)
        z = encoder.represent(flat).reshape(B, N, -1).mean(1).cpu().numpy()
        for k in range(B):
            if bool(ok[k]):
                X.append(z[k]); y.append(int(labels[k]))
                pid.append(os.path.basename(paths[gi]).split("_")[0])
            gi += 1
    return np.stack(X), np.array(y), np.array(pid)


def _ece(p1, y, n=15):
    conf = np.maximum(p1, 1 - p1); pred = (p1 >= 0.5).astype(int); acc = (pred == y).astype(float)
    b = np.linspace(0, 1, n + 1); e = 0.0
    for i in range(n):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def _risk_cov(p1, y):
    conf = np.maximum(p1, 1 - p1); correct = ((p1 >= 0.5).astype(int) == y).astype(float)
    correct = correct[np.argsort(-conf)]
    risk = 1 - np.cumsum(correct) / np.arange(1, len(correct) + 1)
    k = max(1, int(0.8 * len(correct)))
    return float(np.mean(risk)), float(np.mean(correct[:k]))


def _fit_T(logit_dev, y_dev):
    t = torch.ones(1, requires_grad=True)
    z = torch.tensor(logit_dev, dtype=torch.float32); y = torch.tensor(y_dev, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=60); bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad(); loss = bce(z / t.clamp_min(0.05), y); loss.backward(); return loss

    opt.step(closure)
    return float(t.detach().clamp_min(0.05).item())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--random", action="store_true", help="untrained encoder (random floor)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(dev)
    tag = "RANDOM" if a.random else f"{cfg.model.ssl.get('reg_type','?')}/{cfg.model.ssl.get('reg_space','?')}"
    if not a.random:
        enc.load_state_dict(state["encoder"])
    enc.eval()

    dcfg = {"data_root": a.data_root, "label_scheme": "tuab", "n_channels": int(cfg.model.n_channels),
            "sfreq": 200, "window_sec": 10.0, "n_windows": 16, "num_workers": 8}
    Xtr, ytr, pidtr = extract(enc, "train", dev, dcfg)
    Xev, yev, _ = extract(enc, "eval", dev, dcfg)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    rng = np.random.default_rng(0); pats = np.unique(pidtr); rng.shuffle(pats)
    devset = set(pats[: max(1, int(0.15 * len(pats)))].tolist())
    dm = np.array([p in devset for p in pidtr])
    sc = StandardScaler().fit(Xtr[~dm])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr[~dm]), ytr[~dm])

    pe = clf.predict_proba(sc.transform(Xev))[:, 1]
    bacc = balanced_accuracy_score(yev, (pe >= 0.5).astype(int)); auroc = roc_auc_score(yev, pe)
    ece_raw = _ece(pe, yev); aurc, acc80 = _risk_cov(pe, yev)
    T = _fit_T(clf.decision_function(sc.transform(Xtr[dm])), ytr[dm])
    pe_T = 1.0 / (1.0 + np.exp(-clf.decision_function(sc.transform(Xev)) / T))
    ece_T = _ece(pe_T, yev)
    print(f"[calib] enc={tag} | BA={bacc:.4f} AUROC={auroc:.4f} | "
          f"ECE_raw={ece_raw:.4f} ECE_T={ece_T:.4f} (T={T:.2f}) | "
          f"AURC={aurc:.4f} acc@80cov={acc80:.4f}", flush=True)
    print("CALIB_DONE", flush=True)


if __name__ == "__main__":
    main()
