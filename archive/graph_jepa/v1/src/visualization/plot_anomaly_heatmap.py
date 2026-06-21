"""Visualization utilities for TCP-Graph-JEPA anomaly heatmaps."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from archive.graph_jepa.v1.src.graphs.tcp_graph import TCP_CHANNELS


def plot_heatmap(heatmap, channels=None, window_sec: float = 7.0, output_path="heatmap.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(heatmap, dtype=float)
    channels = list(TCP_CHANNELS if channels is None else channels)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="magma")
    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(channels, fontsize=8)
    ticks = np.linspace(0, max(0, arr.shape[1] - 1), num=8)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / max(1, arr.shape[1] - 1) * window_sec:.1f}" for t in ticks])
    ax.set_xlabel("seconds within window")
    ax.set_ylabel("TCP channel")
    ax.set_title("TCP-Graph-JEPA channel-time anomaly heatmap")
    fig.colorbar(im, ax=ax, label="latent prediction error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_timeline(starts_sec, window_scores, output_path="timeline.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(starts_sec, window_scores, marker="o", linewidth=1.5)
    ax.set_xlabel("file time (seconds)")
    ax.set_ylabel("window anomaly score")
    ax.set_title("TCP-Graph-JEPA anomaly timeline")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def top_anomalous_regions(heatmap, channels=None, quantile: float = 0.95, window_sec: float = 7.0):
    arr = np.asarray(heatmap, dtype=float)
    channels = list(TCP_CHANNELS if channels is None else channels)
    threshold = float(np.quantile(arr, quantile))
    rows = []
    for c in range(arr.shape[0]):
        active = arr[c] >= threshold
        start = None
        for t, flag in enumerate(active.tolist() + [False]):
            if flag and start is None:
                start = t
            elif not flag and start is not None:
                end = t
                score = float(arr[c, start:end].mean())
                rows.append({
                    "channel": channels[c],
                    "start_sec": start / arr.shape[1] * window_sec,
                    "end_sec": end / arr.shape[1] * window_sec,
                    "mean_error": score,
                })
                start = None
    return sorted(rows, key=lambda r: r["mean_error"], reverse=True)


def write_region_outputs(rows, output_dir: str | Path):
    output_dir = Path(output_dir)
    with open(output_dir / "top_anomalous_regions.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    with open(output_dir / "top_anomalous_regions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["channel", "start_sec", "end_sec", "mean_error"])
        writer.writeheader()
        writer.writerows(rows)
