"""Evaluate TCP-Graph-JEPA anomaly scores on a dataset split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from archive.graph_jepa.v1.src.data.eeg_windows import EEGWindowDataset, FeatureNormalizer, collate_eeg_windows
from archive.graph_jepa.v1.src.data.features import PreprocessConfig
from archive.graph_jepa.v1.src.eval.anomaly_scoring import (
    aggregate_file_scores,
    binary_metrics,
    score_windows,
    threshold_from_normal,
)
from archive.graph_jepa.v1.src.models.tcp_graph_jepa import TCPGraphJEPA, TCPGraphJEPAConfig


def _preprocess_from_config(cfg: dict) -> PreprocessConfig:
    allowed = PreprocessConfig.__dataclass_fields__.keys()
    return PreprocessConfig(**{k: v for k, v in dict(cfg or {}).items() if k in allowed})


def load_checkpoint(path: str | Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TCPGraphJEPA(TCPGraphJEPAConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    normalizer = FeatureNormalizer.from_state(ckpt.get("normalizer"))
    return ckpt, model, normalizer


def evaluate(
    checkpoint: str | Path,
    data_root: str | Path | None = None,
    split: str = "eval",
    output_dir: str | Path = "archive/graph_jepa/v1/results",
    threshold: float | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model, normalizer = load_checkpoint(checkpoint, device)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    root = data_root or data_cfg["root"]
    ds = EEGWindowDataset(
        root=root,
        split=split,
        input_kind=data_cfg.get("input_kind", "auto"),
        normal_only=False,
        preprocess=_preprocess_from_config(cfg.get("preprocess", {})),
        normalizer=normalizer,
        cache_in_memory=data_cfg.get("cache_in_memory", True),
        edf_windows_per_file=data_cfg.get("edf_windows_per_file", 32),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 16),
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate_eeg_windows,
    )
    eval_cfg = cfg.get("eval", {})
    window_rows = []
    win_file_ids, win_scores = [], []
    labels = {}
    for batch in loader:
        out = score_windows(
            model,
            batch["x"].to(device),
            n_masks=eval_cfg.get("n_masks", 8),
            mask_ratio=eval_cfg.get("mask_ratio", model.cfg.mask_ratio),
            mask_mode=eval_cfg.get("mask_mode", "random"),
            error_norm=eval_cfg.get("error_norm", "l1"),
        )
        for i, score in enumerate(out["window_score"].tolist()):
            file_id = batch["file_id"][i]
            label = int(batch["label"][i])
            labels.setdefault(file_id, label)
            win_file_ids.append(file_id)
            win_scores.append(float(score))
            window_rows.append({
                "file_id": file_id,
                "path": batch["path"][i],
                "window_index": int(batch["window_index"][i]),
                "start_sec": float(batch["start_sec"][i]),
                "label": label,
                "window_score": float(score),
            })
    file_scores = aggregate_file_scores(
        win_file_ids,
        torch.tensor(win_scores),
        method=eval_cfg.get("file_aggregation", "top_k_mean"),
        top_k=eval_cfg.get("top_k", 0.10),
    )
    file_rows = [
        {"file_id": fid, "label": labels.get(fid, -1), "file_score": score}
        for fid, score in sorted(file_scores.items())
    ]
    y = [row["label"] for row in file_rows]
    s = [row["file_score"] for row in file_rows]
    metrics = None
    if y and all(v >= 0 for v in y) and len(set(y)) == 2:
        metrics = binary_metrics(y, s, threshold=threshold)
    elif y and all(v in {-1, 0} for v in y):
        threshold = threshold if threshold is not None else threshold_from_normal(s, 0.95)
        metrics = {"threshold": threshold, "flagged_files": [r["file_id"] for r in file_rows if r["file_score"] >= threshold]}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "window_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
        writer.writeheader()
        writer.writerows(window_rows)
    with open(out_dir / "file_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(file_rows[0].keys()))
        writer.writeheader()
        writer.writerows(file_rows)
    summary = {"metrics": metrics, "n_windows": len(window_rows), "n_files": len(file_rows)}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--output-dir", default="archive/graph_jepa/v1/results")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)
    summary = evaluate(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
