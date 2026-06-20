"""Train TCP-Graph-JEPA.

Usage:
  python train_graph_jepa.py --config configs/graph_jepa.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.eeg_windows import EEGWindowDataset, FeatureNormalizer, collate_eeg_windows
from src.data.features import PreprocessConfig
from src.eval.anomaly_scoring import aggregate_file_scores, binary_metrics, score_windows
from src.graphs.tcp_graph import graph_metadata
from src.models.tcp_graph_jepa import TCPGraphJEPA, TCPGraphJEPAConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_from_config(cfg: dict) -> PreprocessConfig:
    data = dict(cfg or {})
    allowed = PreprocessConfig.__dataclass_fields__.keys()
    return PreprocessConfig(**{k: v for k, v in data.items() if k in allowed})


def dataset_from_config(cfg: dict, split: str, normalizer=None, normal_only=False):
    data_cfg = cfg["data"]
    return EEGWindowDataset(
        root=data_cfg["root"],
        split=split,
        input_kind=data_cfg.get("input_kind", "auto"),
        normal_only=normal_only,
        preprocess=preprocess_from_config(cfg.get("preprocess", {})),
        normalizer=normalizer,
        cache_in_memory=data_cfg.get("cache_in_memory", True),
        edf_windows_per_file=data_cfg.get("edf_windows_per_file", 32),
    )


def build_model(cfg: dict, sample_x: torch.Tensor) -> TCPGraphJEPA:
    model_cfg = dict(cfg["model"])
    model_cfg.setdefault("channels", int(sample_x.shape[0]))
    model_cfg.setdefault("time_steps", int(sample_x.shape[1]))
    model_cfg.setdefault("feature_dim", int(sample_x.shape[2]))
    return TCPGraphJEPA(TCPGraphJEPAConfig(**model_cfg))


def validation_loss(model, loader, device, max_batches: int | None = None) -> float | None:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["x"].to(device)
            loss, _logs = model.compute_loss(x)
            losses.append(float(loss.detach().cpu()))
            if max_batches is not None and i + 1 >= max_batches:
                break
    return float(np.mean(losses)) if losses else None


def anomaly_metrics(model, loader, device, eval_cfg: dict) -> dict | None:
    file_ids, labels, scores = [], {}, []
    win_file_ids = []
    for batch in loader:
        x = batch["x"].to(device)
        out = score_windows(
            model,
            x,
            n_masks=eval_cfg.get("n_masks", 4),
            mask_ratio=eval_cfg.get("mask_ratio", model.cfg.mask_ratio),
            mask_mode=eval_cfg.get("mask_mode", "random"),
            error_norm=eval_cfg.get("error_norm", "l1"),
        )
        for file_id, label in zip(batch["file_id"], batch["label"].tolist()):
            labels.setdefault(file_id, int(label))
            win_file_ids.append(file_id)
        scores.extend(out["window_score"].tolist())
    if not scores:
        return None
    file_scores = aggregate_file_scores(
        win_file_ids,
        torch.tensor(scores),
        method=eval_cfg.get("file_aggregation", "top_k_mean"),
        top_k=eval_cfg.get("top_k", 0.10),
    )
    file_ids = sorted(file_scores)
    y = [labels[f] for f in file_ids]
    if any(v < 0 for v in y) or len(set(y)) < 2:
        return {"file_scores": file_scores}
    metrics = binary_metrics(y, [file_scores[f] for f in file_ids])
    metrics["file_scores"] = file_scores
    return metrics


def save_checkpoint(path: Path, model, cfg, normalizer, epoch: int, optimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_config": model.cfg.__dict__,
            "config": cfg,
            "normalizer": normalizer.state_dict(),
            "graph": graph_metadata(model.channels),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def run(config_path: str, limit_batches: int | None = None) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 0)))
    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    train_ds_raw = dataset_from_config(
        cfg,
        split=cfg["data"].get("train_split", "train"),
        normalizer=None,
        normal_only=cfg["data"].get("normal_only", True),
    )
    normalizer = FeatureNormalizer().fit(
        train_ds_raw,
        max_items=cfg["data"].get("normalizer_max_items"),
    )
    train_ds_raw.normalizer = normalizer
    train_loader = torch.utils.data.DataLoader(
        train_ds_raw,
        batch_size=cfg["training"].get("batch_size", 16),
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate_eeg_windows,
    )

    val_loader = None
    try:
        val_ds = dataset_from_config(
            cfg,
            split=cfg["data"].get("val_split", "eval"),
            normalizer=normalizer,
            normal_only=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg["training"].get("batch_size", 16),
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 0),
            collate_fn=collate_eeg_windows,
        )
    except FileNotFoundError:
        val_loader = None

    sample_x = train_ds_raw[0]["x"]
    model = build_model(cfg, sample_x).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"].get("lr", 1e-3),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints/tcp_graph_jepa"))
    log_rows = []
    for epoch in range(int(cfg["training"].get("epochs", 10))):
        model.train()
        losses = []
        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            opt.zero_grad(set_to_none=True)
            loss, logs = model.compute_loss(x)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if limit_batches is not None and batch_idx + 1 >= limit_batches:
                break
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            row["val_loss"] = validation_loss(
                model,
                val_loader,
                device,
                max_batches=cfg["training"].get("val_max_batches"),
            )
            metrics = anomaly_metrics(model, val_loader, device, cfg.get("eval", {}))
            if metrics:
                row["anomaly_metrics"] = {k: v for k, v in metrics.items() if k != "file_scores"}
        log_rows.append(row)
        print("[tcp-graph-jepa]", json.dumps(row), flush=True)
        save_checkpoint(ckpt_dir / "latest.pth.tar", model, cfg, normalizer, epoch, opt)
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(log_rows, f, indent=2)
        f.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/graph_jepa.yaml")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="debug/smoke option: stop each epoch after N batches")
    args = parser.parse_args(argv)
    run(args.config, limit_batches=args.limit_batches)


if __name__ == "__main__":
    main()
