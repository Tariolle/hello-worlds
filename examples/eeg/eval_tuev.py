"""TUEV evaluation — segment-level 6-class balanced accuracy.

Frozen encoder -> linear probe on TUEV event segments.
Reports balanced accuracy (macro) + per-class accuracy.

Run:
  # random encoder floor only (no checkpoint needed):
  python -m examples.eeg.eval_tuev --floor

  # trained encoder + floor:
  python -m examples.eeg.eval_tuev --ckpt ./checkpoints/eeg_ambient_sigreg/latest.pth.tar --floor

  # custom data root:
  python -m examples.eeg.eval_tuev --ckpt ... --data_root /path/to/TUEV_PREPROCESSED
"""
import sys
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from eb_jepa.datasets.eeg.tuev_dataset import TUEVConfig, TUEVDataset, CLASSES
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract_tuev(encoder, split: str, device, data_root: Optional[str] = None):
    """Frozen encoder -> (X [N, D], y [N]) for TUEV segments."""
    kwargs = {"data_root": data_root} if data_root else {}
    cfg = TUEVConfig(split=split, mode="probe", **kwargs)
    ds = TUEVDataset(cfg)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    X, y = [], []
    total, skipped = 0, 0
    for wins, labels, ok in loader:   # wins: [B, 1, C, T]
        B = wins.shape[0]
        flat = wins.reshape(B, *wins.shape[2:]).to(device, non_blocking=True)
        z = encoder.represent(flat).cpu().numpy()   # [B, D]
        for k in range(B):
            total += 1
            if bool(ok[k]):
                X.append(z[k]); y.append(int(labels[k]))
            else:
                skipped += 1
    print(f"  [{split}] {len(X)} segments ({skipped} skipped / {total} total)", flush=True)
    return np.stack(X), np.array(y)


def run_probe(Xtr, ytr, Xev, yev, tag: str):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xev))
    bal = balanced_accuracy_score(yev, pred)
    acc = accuracy_score(yev, pred)
    # per-class breakdown
    per_class = {}
    for i, cls in enumerate(CLASSES):
        mask = yev == i
        if mask.sum() > 0:
            per_class[cls] = round(float((pred[mask] == yev[mask]).mean()), 4)
    print(f"\n[TUEV eval] {tag}")
    print(f"  balanced_acc = {bal:.4f}   acc = {acc:.4f}")
    print(f"  per-class:  " + "  ".join(f"{c}={v:.3f}" for c, v in per_class.items()))
    return bal


def main():
    argv = sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = argv[argv.index("--data_root") + 1] if "--data_root" in argv else None

    if "--ckpt" in argv:
        ckpt_path = argv[argv.index("--ckpt") + 1]
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(state["cfg"])
        encoder = build_encoder(cfg.model).to(device)
        encoder.load_state_dict(state["encoder"])
        encoder.eval()
        print("[TUEV] Extracting TRAIN features (trained encoder)...", flush=True)
        Xtr, ytr = extract_tuev(encoder, "train", device, data_root)
        print("[TUEV] Extracting EVAL features...", flush=True)
        Xev, yev = extract_tuev(encoder, "eval", device, data_root)
        run_probe(Xtr, ytr, Xev, yev, tag="TRAINED encoder")

    if "--floor" in argv:
        if "--ckpt" in argv:
            # reuse cfg from checkpoint to build same arch
            rnd = build_encoder(cfg.model).to(device).eval()
        else:
            # build default arch
            import sys as _sys
            _sys.path.insert(0, ".")
            from examples.eeg.encoder import EEGEncoder1D
            rnd = EEGEncoder1D().to(device).eval()

        print("\n[TUEV] Extracting TRAIN features (random encoder)...", flush=True)
        Rtr, ry = extract_tuev(rnd, "train", device, data_root)
        print("[TUEV] Extracting EVAL features (random encoder)...", flush=True)
        Rev, rey = extract_tuev(rnd, "eval", device, data_root)
        run_probe(Rtr, ry, Rev, rey, tag="RANDOM encoder floor")

    if "--ckpt" not in argv and "--floor" not in argv:
        print("Usage: python -m examples.eeg.eval_tuev [--ckpt PATH] [--floor] "
              "[--data_root PATH]")


if __name__ == "__main__":
    main()
