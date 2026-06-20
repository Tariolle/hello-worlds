"""Cross-domain frozen probe: load a (e.g. TUSZ-pretrained) encoder, freeze it, and
linear-probe a DIFFERENT corpus (e.g. TUAB normal/abnormal).

Why this exists: eval.py's `--data-root` CLI override did not take effect on the
cluster checkpoint config (the probe kept reading the pretrain corpus, TUSZ). This
script bypasses that path entirely by constructing the probe data-config directly,
reusing the trusted `extract_features` + `probe` helpers. Encoder weights come from
the checkpoint; the data root is whatever you pass.

Run (GPU node):
  python -u -m examples.eeg.eval_xdomain --ckpt <.../latest.pth.tar> \
      --data-root <TUAB_PREPROCESSED> [--floor]
"""
import argparse

import torch
from omegaconf import OmegaConf

from examples.eeg.eval import extract_features, probe
from examples.eeg.main import build_encoder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True, help="probe corpus root (e.g. TUAB_PREPROCESSED)")
    ap.add_argument("--label-scheme", default="tuab", choices=["tuab", "folders"])
    ap.add_argument("--floor", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(dev)
    enc.load_state_dict(state["encoder"]); enc.eval()

    # Build a CLEAN probe data-config pointed at the target corpus (no pretrain root,
    # no file_list). Encoder geometry (n_channels) must match the checkpoint.
    data_cfg = {
        "data_root": a.data_root,
        "label_scheme": a.label_scheme,
        "n_channels": int(cfg.model.n_channels),
        "sfreq": 200,
        "window_sec": 10.0,
        "n_windows": 16,
        "num_workers": 8,
    }
    Xtr, ytr = extract_features(enc, "train", dev, data_cfg)
    Xev, yev = extract_features(enc, "eval", dev, data_cfg)
    print(f"[xdomain] ckpt={a.ckpt}")
    print(f"[xdomain] TRAINED: {probe(Xtr, ytr, Xev, yev)}", flush=True)

    if a.floor:
        rnd = build_encoder(cfg.model).to(dev).eval()
        Rtr, ry = extract_features(rnd, "train", dev, data_cfg)
        Rev, rey = extract_features(rnd, "eval", dev, data_cfg)
        print(f"[xdomain] RANDOM floor: {probe(Rtr, ry, Rev, rey)}", flush=True)
    print("XDOMAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
