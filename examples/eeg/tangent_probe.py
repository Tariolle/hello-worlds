"""Audit follow-up: expose the LEARNED SPD-tangent to the frozen probe.

The standard probe reads encoder.represent (time-pooled d_model vector) and is
structurally BLIND to the tangent the geometry arm shapes during SSL. This script
probes the tangent DIRECTLY: per window, tangent_features(cov_features(x)) ->
[d_cov(d_cov+1)/2] Log-Euclidean tangent vector, mean-pooled over the N windows per
recording, then a standard patient-disjoint linear probe.

Reads the SAME corpus as eval.py (default TUAB). Compare the number against:
  * the represent baseline (~0.82) — the probe-visible first-order rep,
  * the classical Riemannian baseline (0.761) — channel-cov + AIRM tangent + LR,
  * the RANDOM-encoder tangent floor (--floor).

Interpretation: for a TANGENT-arm checkpoint (reg_space=tangent) cov_proj is trained;
for an AMBIENT-arm checkpoint it is at init, so its tangent ~= a random-cov-proj
tangent on a trained backbone. If trained-tangent ~= ambient-tangent ~= 0.76, the
geometry arm learned no extra probe-readable second-order signal.

Run (GPU node):
  python -u -m examples.eeg.tangent_probe --ckpt <.../latest.pth.tar> \
      --data-root <TUAB_PREPROCESSED> [--floor]
"""
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.eval import probe
from examples.eeg.geometry import tangent_features
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract_tangent(encoder, split, device, data_cfg):
    """Frozen encoder -> [N_rec, d_cov(d_cov+1)/2] mean-pooled tangent features."""
    cfg = EEGConfig(**(data_cfg or {}))
    cfg.split, cfg.mode = split, "probe"
    ds = EEGDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False,
                                         num_workers=cfg.num_workers, pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:                 # wins: [B, N, C, T]
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        cov = encoder.cov_features(flat)            # [B*N, d_cov, T']
        tan = tangent_features(cov)                 # [B*N, d_cov(d_cov+1)/2]
        z = tan.reshape(B, N, -1).mean(dim=1).cpu().numpy()
        for k in range(B):
            if bool(ok[k]):
                X.append(z[k]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--floor", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(dev)
    enc.load_state_dict(state["encoder"]); enc.eval()
    reg_space = str(cfg.model.ssl.get("reg_space", "?"))
    reg_type = str(cfg.model.ssl.get("reg_type", "?"))

    data_cfg = {"data_root": a.data_root, "label_scheme": "tuab",
                "n_channels": int(cfg.model.n_channels), "sfreq": 200,
                "window_sec": 10.0, "n_windows": 16, "num_workers": 8}
    Xtr, ytr = extract_tangent(enc, "train", dev, data_cfg)
    Xev, yev = extract_tangent(enc, "eval", dev, data_cfg)
    print(f"[tangent-probe] ckpt reg={reg_type}/{reg_space}  tangent-dim={Xtr.shape[1]}", flush=True)
    print(f"[tangent-probe] TRAINED-tangent probe: {probe(Xtr, ytr, Xev, yev)}", flush=True)

    if a.floor:
        rnd = build_encoder(cfg.model).to(dev).eval()
        Rtr, ry = extract_tangent(rnd, "train", dev, data_cfg)
        Rev, rey = extract_tangent(rnd, "eval", dev, data_cfg)
        print(f"[tangent-probe] RANDOM-tangent floor: {probe(Rtr, ry, Rev, rey)}", flush=True)
    print("TANGENT_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
