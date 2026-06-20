"""TCP-Graph-JEPA — interpretable anomaly visualisation for one recording.

Given a checkpoint and a single EDF (or a pre-extracted ``[N,C,T,F]`` tensor),
produces:

  1. ``<out>/heatmap.png``  — channel x time anomaly heatmap (y = TCP channels,
     x = time in seconds), averaged over the windows (or a continuous strip with
     ``--contiguous``);
  2. ``<out>/timeline.png`` — file-level anomaly timeline (window score vs time);
  3. ``<out>/anomalies.json`` + ``anomalies.csv`` — top anomalous channel-time
     cells and the file-level score.

Run:
    python -m examples.eeg.graph_jepa.visualize_graph_jepa \
        --ckpt ./checkpoints/graph_jepa/latest.pth.tar \
        --edf <path.edf> --output-dir ./results/graph_jepa_viz
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eb_jepa.graph_jepa.scoring import (score_recording, top_anomalies,
                                        window_error_maps, window_scores_from_heat)
from eb_jepa.graph_jepa.windows import read_edf_windows
from examples.eeg.graph_jepa.common import load_checkpoint


def _heat_png(heat, channels, frame_sec, path, title, x0=0.0):
    C, T = heat.shape
    fig, ax = plt.subplots(figsize=(10, 7))
    extent = [x0, x0 + T * frame_sec, C - 0.5, -0.5]
    im = ax.imshow(heat, aspect="auto", cmap="magma", extent=extent,
                   interpolation="nearest")
    ax.set_yticks(range(C)); ax.set_yticklabels(channels, fontsize=7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("TCP channel")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="JEPA latent error (anomaly)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _timeline_png(starts, win_scores, file_score, path):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(starts, win_scores, "-o", ms=4, color="#b5179e")
    ax.axhline(file_score, ls="--", color="gray",
               label=f"file score = {file_score:.4f}")
    ax.set_xlabel("window start time (s)"); ax.set_ylabel("window anomaly score")
    ax.set_title("File-level anomaly timeline"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--edf", default=None)
    ap.add_argument("--tensor", default=None, help="alt input: [N,C,T,F] .npy/.pt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--contiguous", action="store_true",
                    help="read back-to-back windows -> continuous channel x time strip")
    ap.add_argument("--top-n", type=int, default=15)
    a = ap.parse_args()
    if not a.edf and not a.tensor:
        ap.error("provide --edf or --tensor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_cfg, scoring_cfg, stats = load_checkpoint(a.ckpt, device)
    channels = model.cfg.channels
    fcfg = data_cfg.feature_cfg()
    os.makedirs(a.output_dir, exist_ok=True)

    if a.edf:
        x, cm, starts = read_edf_windows(a.edf, data_cfg, stats=stats,
                                         n_windows=a.n_windows, contiguous=a.contiguous)
    else:
        from eb_jepa.graph_jepa.windows import _load_array
        arr = _load_array(a.tensor).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[None]
        arr = stats.apply(arr)   # same normalisation space as training
        x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        cm = torch.ones(x.shape[1], dtype=torch.bool)
        starts = np.arange(x.shape[0]) * data_cfg.window_sec

    cm_b = cm.unsqueeze(0).expand(x.shape[0], -1)
    heat = window_error_maps(model, x, cm_b, scoring_cfg, device=device)  # [N,C,T]
    win_scores = window_scores_from_heat(heat, cm_b, scoring_cfg)         # [N]
    file_score, _, mean_heat = score_recording(model, x, cm, scoring_cfg, device=device)

    # heatmap: continuous strip if contiguous, else window-averaged
    if a.contiguous:
        strip = np.concatenate([heat[i].numpy() for i in range(heat.shape[0])], axis=1)
        _heat_png(strip, channels, fcfg.frame_sec,
                  os.path.join(a.output_dir, "heatmap.png"),
                  f"Channel x time anomaly (continuous)  file score={file_score:.4f}",
                  x0=float(starts[0]))
    else:
        _heat_png(mean_heat, channels, fcfg.frame_sec,
                  os.path.join(a.output_dir, "heatmap.png"),
                  f"Channel x time anomaly (window-averaged)  file score={file_score:.4f}")

    _timeline_png(starts[:len(win_scores)], win_scores, file_score,
                  os.path.join(a.output_dir, "timeline.png"))

    tops = top_anomalies(mean_heat, channels, fcfg.frame_sec,
                         channel_mask=cm.numpy(), top_n=a.top_n)
    payload = {"file_score": float(file_score),
               "window_scores": [float(s) for s in win_scores],
               "window_starts_sec": [float(s) for s in starts[:len(win_scores)]],
               "top_anomalies": tops, "channels": channels,
               "input": a.edf or a.tensor}
    with open(os.path.join(a.output_dir, "anomalies.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(a.output_dir, "anomalies.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rank", "channel", "time_sec", "score"])
        for r, d in enumerate(tops, 1):
            w.writerow([r, d["channel"], d["time_sec"], d["score"]])

    print(f"[graph-viz] file_score={file_score:.4f} -> {a.output_dir}/"
          f" (heatmap.png, timeline.png, anomalies.json/csv)", flush=True)
    print("GRAPH_JEPA_VIZ_DONE", flush=True)


if __name__ == "__main__":
    main()
