"""TUEV cross-dataset frozen-probe (P2 bonus) — generality of the frozen encoder.

HONEST FRAMING: this is **cross-TASK, SAME-SITE** transfer. TUEV shares the TUH
montage (the first 19 channels are byte-for-byte the TUAB order, see TUEV README
§7), so it tests "does a frozen TUAB-pretrained encoder carry to a *different task*
(6-class event typing)". It is NOT cross-site / cross-distribution (that needs a
non-TUH set). Report it as such — never "cross-site".

DESIGN — "what is one sample" (TUEV README §9 leaves this open):
one sample = one 10 s window (all 19 channels) centred on a distinct annotated
event region, labelled by that event's class (1..6). The CSV annotates per channel
and per 1 s, so the same event appears up to 22x (channels) and as many consecutive
1 s rows; we collapse these to ~one window per (5 s bucket, class) per recording,
then cap per class for balance/compute. Patient-disjoint by construction (official
train/ vs eval/ folders).

Reuses the frozen-probe recipe (StandardScaler + balanced LogisticRegression on
mean-pooled encoder features). Metrics = TUEV-standard: balanced accuracy,
weighted-F1, macro-F1, Cohen's kappa.

Run (GPU node):
  python -m examples.eeg.tuev_probe --ckpt <.../latest.pth.tar> \
      --tuev-root <.../TUEV_PREPROCESSED> [--floor] [--riemann] \
      [--per-class-cap 1500]

Classical Riemannian event baseline only:
  python -m examples.eeg.tuev_probe --riemann-only \
      --tuev-root <.../TUEV_PREPROCESSED>
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder

try:
    import pyedflib
except ImportError:
    pyedflib = None

CLASSES = ["spsw", "gped", "pled", "eyem", "artf", "bckg"]  # label_code 1..6
N_CH = 19
SFREQ = 200
WIN_SEC = 10.0
WIN = int(WIN_SEC * SFREQ)   # 2000 samples (matches the TUAB-pretrained encoder)


def _zscore(x, axis):
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-6
    return (x - mu) / sd


def _events_from_csv(csv_path):
    """Deduped [(center_sec, label0)] — one per (5 s bucket, class) per recording."""
    seen, out = set(), []
    try:
        with open(csv_path) as fh:
            next(fh, None)  # header: montage_channel,start_sec,stop_sec,label_code,label
            for line in fh:
                p = line.strip().split(",")
                if len(p) < 5:
                    continue
                try:
                    start, stop, code = float(p[1]), float(p[2]), int(p[3])
                except ValueError:
                    continue
                if not (1 <= code <= 6):
                    continue
                center = 0.5 * (start + stop)
                key = (code, int(center // 5))     # collapse per-channel + 1 s replication
                if key in seen:
                    continue
                seen.add(key)
                out.append((center, code - 1))     # 1..6 -> 0..5
    except Exception:
        return []
    return out


def build_split(root, split, per_class_cap, rng):
    """-> list of (edf_path, start_sample, label) window candidates, capped per class."""
    edfs = sorted(glob.glob(os.path.join(root, split, "*", "*.edf")))
    if not edfs:
        raise FileNotFoundError(f"No .edf under {os.path.join(root, split)}")
    half = WIN // 2
    by_label = defaultdict(list)
    for e in edfs:
        c = e[:-4] + ".csv"
        if not os.path.exists(c):
            continue
        for center, lab in _events_from_csv(c):
            start = int(center * SFREQ) - half
            by_label[lab].append((e, start, lab))
    picked = []
    for lab, items in by_label.items():
        idx = rng.permutation(len(items))
        if per_class_cap > 0:
            idx = idx[:per_class_cap]
        picked.extend(items[i] for i in idx)
    return picked


@torch.no_grad()
def extract(encoder, items, device):
    """Frozen encoder -> [N, D] features + labels. One EDF open per recording."""
    by_path = defaultdict(list)
    for path, start, lab in items:
        by_path[path].append((start, lab))
    X, y, buf, blab = [], [], [], []

    def flush():
        if not buf:
            return
        t = torch.from_numpy(np.stack(buf)).to(device, non_blocking=True)
        z = encoder.represent(t).cpu().numpy()
        X.extend(list(z)); y.extend(blab)
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
    return np.asarray(X, dtype=np.float32), np.asarray(y)


def extract_covariances(items, estimator_name="oas"):
    """Event windows -> [N, C, C] covariance matrices + labels.

    This mirrors ``extract`` but skips the neural encoder. It keeps the same TUEV
    event definition, class balancing cap, and readable-window filtering.
    """
    from pyriemann.estimation import Covariances

    est = Covariances(estimator=estimator_name)
    by_path = defaultdict(list)
    for path, start, lab in items:
        by_path[path].append((start, lab))
    covs, y, buf, blab = [], [], [], []

    def flush():
        if not buf:
            return
        C = est.transform(np.stack(buf).astype(np.float64))
        C = 0.5 * (C + C.transpose(0, 2, 1))
        covs.extend(list(C))
        y.extend(blab)
        buf.clear()
        blab.clear()

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
                buf.append(_zscore(x, axis=1))
                blab.append(lab)
                if len(buf) >= 256:
                    flush()
        finally:
            f._close()
    flush()
    return np.asarray(covs, dtype=np.float64), np.asarray(y)


def run_probe(Xtr, ytr, Xev, yev, tag):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix, f1_score)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pe = clf.predict(sc.transform(Xev))
    res = {
        "balanced_acc": round(float(balanced_accuracy_score(yev, pe)), 4),
        "weighted_f1": round(float(f1_score(yev, pe, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(yev, pe, average="macro", zero_division=0)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(yev, pe)), 4),
        "acc": round(float(accuracy_score(yev, pe)), 4),
        "n_train": int(len(ytr)), "n_eval": int(len(yev)),
    }
    print(f"[tuev] {tag}: {res}", flush=True)
    cm = confusion_matrix(yev, pe, labels=list(range(6)))
    print(f"[tuev] {tag} confusion (rows=true {CLASSES}):\n{cm}", flush=True)
    return res


def run_riemann_probe(Ctr, ytr, Cev, yev, tag):
    from examples.eeg.baseline_riemann import fit_score_riemann

    res = fit_score_riemann(Ctr, ytr, Cev, yev, CLASSES)
    keep = {
        "balanced_acc": res["balanced_acc"],
        "macro_f1": res["f1"],
        "auroc": res["auroc"],
        "acc": res["acc"],
        "n_train": res["n_train"],
        "n_eval": res["n_eval"],
    }
    print(f"[tuev] {tag}: {keep}", flush=True)
    print(f"[tuev] {tag} confusion (rows=true {CLASSES}):\n"
          f"{np.asarray(res['confusion_matrix'])}", flush=True)
    return res


def _counts(y):
    c = np.bincount(y, minlength=6)
    return ", ".join(f"{CLASSES[i]}={int(c[i])}" for i in range(6))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt",
                    help="frozen encoder checkpoint; optional with --riemann-only")
    ap.add_argument("--tuev-root", required=True)
    ap.add_argument("--per-class-cap", type=int, default=1500,
                    help="max windows per class per split (0 = no cap)")
    ap.add_argument("--floor", action="store_true",
                    help="also probe a random (untrained) encoder of the same architecture")
    ap.add_argument("--riemann", action="store_true",
                    help="also run the classical Riemannian covariance event probe")
    ap.add_argument("--riemann-only", action="store_true",
                    help="run only the classical Riemannian covariance event probe")
    ap.add_argument("--cov-estimator", default="oas",
                    help="pyRiemann covariance estimator for --riemann, e.g. oas/lwf/scm")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if pyedflib is None:
        sys.exit("pyedflib is required (pip install pyedflib)")

    rng = np.random.default_rng(a.seed)
    tr = build_split(a.tuev_root, "train", a.per_class_cap, rng)
    ev = build_split(a.tuev_root, "eval", a.per_class_cap, rng)
    print(f"[tuev] candidate windows: train={len(tr)} eval={len(ev)}", flush=True)

    if a.riemann or a.riemann_only:
        Ctr, cytr = extract_covariances(tr, a.cov_estimator)
        Cev, cyev = extract_covariances(ev, a.cov_estimator)
        print(f"[tuev] covariances: Ctr={Ctr.shape} Cev={Cev.shape}", flush=True)
        print(f"[tuev] riemann train dist: {_counts(cytr)}", flush=True)
        print(f"[tuev] riemann eval  dist: {_counts(cyev)}", flush=True)
        if len(Ctr) == 0 or len(Cev) == 0:
            sys.exit("[tuev] no readable covariance windows -- check tuev-root / channel count")
        run_riemann_probe(Ctr, cytr, Cev, cyev, "RIEMANN covariance+LR")

    if a.riemann_only:
        print("TUEV_DONE", flush=True)
        return

    if not a.ckpt:
        ap.error("--ckpt is required unless --riemann-only is set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(device)
    enc.load_state_dict(state["encoder"]); enc.eval()

    Xtr, ytr = extract(enc, tr, device)
    Xev, yev = extract(enc, ev, device)
    print(f"[tuev] features: Xtr={Xtr.shape} Xev={Xev.shape}", flush=True)
    print(f"[tuev] train dist: {_counts(ytr)}", flush=True)
    print(f"[tuev] eval  dist: {_counts(yev)}", flush=True)
    if len(Xtr) == 0 or len(Xev) == 0:
        sys.exit("[tuev] no readable windows — check tuev-root / channel count")

    run_probe(Xtr, ytr, Xev, yev, "TRAINED (frozen SIGReg encoder)")

    if a.floor:
        rnd = build_encoder(cfg.model).to(device).eval()
        Rtr, ry = extract(rnd, tr, device)
        Rev, rey = extract(rnd, ev, device)
        run_probe(Rtr, ry, Rev, rey, "RANDOM floor")

    print("TUEV_DONE", flush=True)


if __name__ == "__main__":
    main()
