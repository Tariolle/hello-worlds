"""Visualize TCP-Graph-JEPA anomaly maps for one EDF or tensor file."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from archive.graph_jepa.v1.src.data.eeg_windows import FeatureNormalizer, load_single_input_windows
from archive.graph_jepa.v1.src.data.features import PreprocessConfig
from archive.graph_jepa.v1.src.eval.anomaly_scoring import aggregate_scores, score_windows
from archive.graph_jepa.v1.src.eval.evaluate_graph_jepa import load_checkpoint
from archive.graph_jepa.v1.src.graphs.tcp_graph import TCP_CHANNELS
from archive.graph_jepa.v1.src.visualization.plot_anomaly_heatmap import (
    plot_heatmap,
    plot_timeline,
    top_anomalous_regions,
    write_region_outputs,
)


def _preprocess_from_config(cfg: dict) -> PreprocessConfig:
    allowed = PreprocessConfig.__dataclass_fields__.keys()
    return PreprocessConfig(**{k: v for k, v in dict(cfg or {}).items() if k in allowed})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--edf_or_tensor", required=True)
    parser.add_argument("--output_dir", default="archive/graph_jepa/v1/results/viz")
    parser.add_argument("--n-masks", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-mode", default="random")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model, normalizer = load_checkpoint(args.checkpoint, device)
    cfg = ckpt["config"]
    windows, meta = load_single_input_windows(
        args.edf_or_tensor,
        preprocess=_preprocess_from_config(cfg.get("preprocess", {})),
        normalizer=normalizer,
    )
    out = score_windows(
        model,
        windows.to(device),
        n_masks=args.n_masks,
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        error_norm=cfg.get("eval", {}).get("error_norm", "l1"),
    )
    heatmaps = out["heatmap"]
    window_scores = out["window_score"]
    top_idx = int(torch.argmax(window_scores).item())
    starts = meta.get("starts_sec") or list(range(len(window_scores)))
    channels = ckpt.get("graph", {}).get("channels") or TCP_CHANNELS
    window_sec = float(meta.get("window_sec", cfg.get("preprocess", {}).get("window_sec", 7.0)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(
        heatmaps[top_idx].numpy(),
        channels=channels,
        window_sec=window_sec,
        output_path=output_dir / "channel_time_heatmap.png",
    )
    plot_timeline(starts, window_scores.numpy(), output_path=output_dir / "anomaly_timeline.png")
    rows = top_anomalous_regions(
        heatmaps[top_idx].numpy(),
        channels=channels,
        window_sec=window_sec,
    )
    write_region_outputs(rows, output_dir)
    file_score = float(aggregate_scores(window_scores, method="top_k_mean", top_k=0.10).item())
    window_rows = [
        {"window_index": i, "start_sec": float(starts[i]), "window_score": float(s)}
        for i, s in enumerate(window_scores.tolist())
    ]
    with open(output_dir / "window_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_index", "start_sec", "window_score"])
        writer.writeheader()
        writer.writerows(window_rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(args.edf_or_tensor),
                "file_score": file_score,
                "top_window_index": top_idx,
                "top_window_start_sec": float(starts[top_idx]),
                "heatmap": "channel_time_heatmap.png",
                "timeline": "anomaly_timeline.png",
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"[graph-jepa-viz] wrote {output_dir}")


if __name__ == "__main__":
    main()
