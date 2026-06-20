"""TUAB benchmark renderer and local evaluation harness.

Default usage is safe before the JEPA model is trained:

    python -m examples.eeg.benchmark

That command reads ``examples/eeg/cfgs/benchmark.yaml`` and writes pretty
benchmark artifacts with JEPA rows marked as pending. It does not start
pretraining. Add explicit flags later to run local baselines or evaluate
checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


METRIC_COLUMNS = ("acc", "balanced_acc", "auroc", "f1")
REPORT_COLUMNS = (
    "rank",
    "method",
    "family",
    "status",
    "protocol",
    "acc",
    "balanced_acc",
    "auroc",
    "f1",
    "frozen_probe",
    "comparable_to_local_jepa",
    "seed",
    "checkpoint",
    "metric_source",
    "notes",
)


@dataclass
class RunRequest:
    run_riemann: bool
    run_random_floor: bool
    checkpoint_overrides: dict[str, str]
    data_root: str | None
    label_scheme: str | None
    class_names: list[str] | None
    train_split: str
    eval_split: str
    riemann_classifier: str
    riemann_aggregation: str
    riemann_mean_metric: str
    riemann_distance_metric: str
    riemann_tangent_metric: str
    riemann_cov_estimator: str


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _row_from_method(method: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    row = {
        "id": method["id"],
        "method": method["display_name"],
        "family": method.get("family", ""),
        "status": method.get("status", ""),
        "protocol": method.get("protocol") or dataset.get("local_protocol", ""),
        "checkpoint": method.get("checkpoint", ""),
        "seed": method.get("seed", ""),
        "frozen_probe": method.get("frozen_probe", True),
        "comparable_to_local_jepa": method.get("comparable_to_local_jepa", False),
        "metric_source": method.get("metric_source", "local"),
        "notes": method.get("notes", ""),
        "reg_type": method.get("reg_type", ""),
        "reg_space": method.get("reg_space", ""),
    }
    for metric in METRIC_COLUMNS:
        row[metric] = method.get(metric)
    return row


def _local_rows(cfg: dict[str, Any], requests: RunRequest) -> list[dict[str, Any]]:
    dataset = cfg["dataset"]
    rows = [_row_from_method(method, dataset) for method in cfg.get("local_methods", [])]
    by_id = {row["id"]: row for row in rows}

    if requests.run_riemann:
        by_id["riemann_tangent_logreg"].update(_run_riemann(requests))

    if requests.run_random_floor:
        by_id["random_encoder_floor"].update(_run_random_floor(cfg, requests))

    for method_id, ckpt in requests.checkpoint_overrides.items():
        if method_id not in by_id:
            raise KeyError(f"Unknown local method id for checkpoint override: {method_id}")
        by_id[method_id].update(_run_checkpoint(cfg, ckpt, requests))

    return rows


def _published_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = cfg["dataset"]
    return [_row_from_method(method, dataset) for method in cfg.get("published_references", [])]


def _run_riemann(requests: RunRequest) -> dict[str, Any]:
    from examples.eeg.baseline_riemann import run_recording_riemann

    metrics = run_recording_riemann(
        data_root=requests.data_root,
        label_scheme=requests.label_scheme or "tuab",
        class_names=requests.class_names,
        train_split=requests.train_split,
        eval_split=requests.eval_split,
        estimator=requests.riemann_cov_estimator,
        aggregation=requests.riemann_aggregation,
        classifier=requests.riemann_classifier,
        mean_metric=requests.riemann_mean_metric,
        distance_metric=requests.riemann_distance_metric,
        tangent_metric=requests.riemann_tangent_metric,
    )
    return {
        "status": "measured_local",
        "metric_source": "local_run",
        **metrics,
    }


def _apply_dataset_overrides(data_cfg: dict[str, Any], requests: RunRequest) -> dict[str, Any]:
    data_cfg = dict(data_cfg)
    if requests.data_root:
        data_cfg["data_root"] = requests.data_root
    if requests.label_scheme:
        data_cfg["label_scheme"] = requests.label_scheme
    if requests.class_names:
        data_cfg["class_names"] = requests.class_names
    return data_cfg


def _run_random_floor(cfg: dict[str, Any], requests: RunRequest) -> dict[str, Any]:
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from examples.eeg.eval import extract_features, probe
    from examples.eeg.main import build_encoder

    train_cfg = OmegaConf.load("examples/eeg/cfgs/train.yaml")
    data_cfg = OmegaConf.to_container(train_cfg.data, resolve=True)
    data_cfg = _apply_dataset_overrides(data_cfg, requests)
    seed = int(train_cfg.meta.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(train_cfg.model).to(device).eval()
    x_train, y_train, label_names = extract_features(
        encoder, requests.train_split, device, data_cfg, return_label_names=True)
    data_cfg["class_names"] = label_names
    x_eval, y_eval = extract_features(encoder, requests.eval_split, device, data_cfg)
    metrics = probe(x_train, y_train, x_eval, y_eval, label_names)
    return {
        "status": "measured_local",
        "metric_source": "local_run",
        **metrics,
    }


def _run_checkpoint(cfg: dict[str, Any], checkpoint: str, requests: RunRequest) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    from examples.eeg.eval import extract_features, probe
    from examples.eeg.main import build_encoder

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_cfg = OmegaConf.create(state["cfg"])
    data_cfg = OmegaConf.to_container(train_cfg.data, resolve=True)
    data_cfg = _apply_dataset_overrides(data_cfg, requests)
    encoder = build_encoder(train_cfg.model).to(device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    x_train, y_train, label_names = extract_features(
        encoder, requests.train_split, device, data_cfg, return_label_names=True)
    data_cfg["class_names"] = label_names
    x_eval, y_eval = extract_features(encoder, requests.eval_split, device, data_cfg)
    metrics = probe(x_train, y_train, x_eval, y_eval, label_names)
    return {
        "status": "measured_local",
        "checkpoint": str(ckpt_path),
        "metric_source": "local_run",
        **metrics,
    }


def _rank_rows(rows: list[dict[str, Any]], primary_metric: str) -> list[dict[str, Any]]:
    sortable = []
    for idx, row in enumerate(rows):
        value = _metric_value(row, primary_metric)
        sortable.append((value is None, -value if value is not None else 0.0, idx, row))
    ranked = []
    rank = 1
    for missing, _neg_value, _idx, row in sorted(sortable):
        out = dict(row)
        out["rank"] = "" if missing else rank
        if not missing:
            rank += 1
        ranked.append(out)
    return ranked


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt_metric(row.get(key)) for key in REPORT_COLUMNS})


def _write_markdown(rows: list[dict[str, Any]], path: Path, cfg: dict[str, Any]) -> None:
    primary = cfg["meta"]["primary_metric"]
    lines = [
        f"# {cfg['meta']['benchmark_name']} benchmark",
        "",
        f"Primary metric: `{primary}`. Higher is better.",
        "",
        f"Local protocol: {cfg['dataset']['local_protocol']}",
        "",
        f"Caution: {cfg['dataset']['caution']}",
        "",
        "| Rank | Method | Status | Protocol | Acc | BalAcc | AUROC | F1 | Comparable? | Source |",
        "|---:|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {method} | {status} | {protocol} | {acc} | {balanced_acc} | "
            "{auroc} | {f1} | {comparable} | {source} |".format(
                rank=row.get("rank", ""),
                method=row.get("method", ""),
                status=row.get("status", ""),
                protocol=row.get("protocol", ""),
                acc=_fmt_metric(row.get("acc")),
                balanced_acc=_fmt_metric(row.get("balanced_acc")),
                auroc=_fmt_metric(row.get("auroc")),
                f1=_fmt_metric(row.get("f1")),
                comparable="yes" if row.get("comparable_to_local_jepa") else "reference only",
                source=row.get("metric_source", ""),
            )
        )
    lines += [
        "",
        "## Notes",
        "",
    ]
    for row in rows:
        if row.get("notes"):
            lines.append(f"- **{row['method']}**: {row['notes']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(rows: list[dict[str, Any]], path: Path, cfg: dict[str, Any]) -> None:
    def cell(text: Any) -> str:
        import html

        return html.escape(_fmt_metric(text))

    row_html = []
    for row in rows:
        cls = "local" if row.get("comparable_to_local_jepa") else "reference"
        if "pending" in str(row.get("status", "")):
            cls += " pending"
        row_html.append(
            "<tr class='{cls}'>"
            "<td>{rank}</td><td>{method}</td><td>{status}</td><td>{protocol}</td>"
            "<td>{acc}</td><td>{balanced_acc}</td><td>{auroc}</td><td>{f1}</td>"
            "<td>{comparable}</td><td>{source}</td>"
            "</tr>".format(
                cls=cls,
                rank=cell(row.get("rank", "")),
                method=cell(row.get("method", "")),
                status=cell(row.get("status", "")),
                protocol=cell(row.get("protocol", "")),
                acc=cell(row.get("acc")),
                balanced_acc=cell(row.get("balanced_acc")),
                auroc=cell(row.get("auroc")),
                f1=cell(row.get("f1")),
                comparable="yes" if row.get("comparable_to_local_jepa") else "reference only",
                source=cell(row.get("metric_source", "")),
            )
        )
    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>{cell(cfg['meta']['benchmark_name'])}</title>
<style>
body {{ font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; }}
h1 {{ font-size: 28px; margin-bottom: 4px; }}
.subtle {{ color: #5f6b7a; max-width: 960px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 14px; }}
th, td {{ border-bottom: 1px solid #d9dee7; padding: 10px 12px; text-align: left; vertical-align: top; }}
th {{ background: #eef2f6; color: #2d3748; position: sticky; top: 0; }}
td:nth-child(1), td:nth-child(5), td:nth-child(6), td:nth-child(7), td:nth-child(8) {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr.local {{ background: #f7fbff; }}
tr.reference {{ background: #fffaf0; }}
tr.pending td {{ color: #687385; }}
.legend {{ display: flex; gap: 16px; margin-top: 16px; color: #4a5568; font-size: 13px; }}
.swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; border: 1px solid #c7ced9; }}
.local-s {{ background: #f7fbff; }}
.ref-s {{ background: #fffaf0; }}
</style>
<body>
<h1>{cell(cfg['meta']['benchmark_name'])}</h1>
<p class="subtle">Primary metric: <b>{cell(cfg['meta']['primary_metric'])}</b>. {cell(cfg['dataset']['local_protocol'])}</p>
<p class="subtle"><b>Caution:</b> {cell(cfg['dataset']['caution'])}</p>
<div class="legend"><span><span class="swatch local-s"></span>local comparable rows</span><span><span class="swatch ref-s"></span>published reference rows</span></div>
<table>
<thead><tr><th>Rank</th><th>Method</th><th>Status</th><th>Protocol</th><th>Acc</th><th>BalAcc</th><th>AUROC</th><th>F1</th><th>Comparable?</th><th>Source</th></tr></thead>
<tbody>
{''.join(row_html)}
</tbody>
</table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_plot(rows: list[dict[str, Any]], path: Path, metric: str) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-eeg-benchmark")
    )
    import matplotlib.pyplot as plt

    plotted = [row for row in rows if _metric_value(row, metric) is not None]
    if not plotted:
        return
    plotted = sorted(plotted, key=lambda row: _metric_value(row, metric) or -1.0)
    labels = [row["method"] for row in plotted]
    values = [_metric_value(row, metric) for row in plotted]
    colors = ["#2f80ed" if row.get("comparable_to_local_jepa") else "#f2994a" for row in plotted]
    height = max(4.0, 0.42 * len(plotted) + 1.2)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(metric)
    ax.set_title(f"TUAB / EEG abnormality references by {metric}")
    for y, value in enumerate(values):
        ax.text(min(value + 0.012, 0.98), y, f"{value:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_checkpoint_overrides(items: list[str]) -> dict[str, str]:
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--checkpoint must be METHOD_ID=PATH")
        method_id, path = item.split("=", 1)
        overrides[method_id.strip()] = path.strip()
    return overrides


def _parse_classes(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/eeg/cfgs/benchmark.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--label-scheme", choices=["tuab", "folders"], default=None)
    parser.add_argument("--classes",
                        help="comma-separated class folder names/order for folder-labelled data")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--run-riemann", action="store_true", help="Run the CPU Riemannian baseline.")
    parser.add_argument("--riemann-classifier", default="tangent-logreg",
                        choices=["tangent-logreg", "mdm"])
    parser.add_argument("--riemann-aggregation", default="riemann",
                        choices=["riemann", "logeuclid", "euclid"])
    parser.add_argument("--riemann-mean-metric", default="riemann",
                        choices=["riemann", "logeuclid", "euclid"])
    parser.add_argument("--riemann-distance-metric", default="riemann",
                        choices=["riemann", "logeuclid", "euclid"])
    parser.add_argument("--riemann-tangent-metric", default="riemann",
                        choices=["riemann", "logeuclid", "euclid"])
    parser.add_argument("--riemann-cov-estimator", default="oas",
                        help="covariance estimator for --run-riemann")
    parser.add_argument("--run-random-floor", action="store_true", help="Evaluate the untrained encoder floor.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="METHOD_ID=PATH",
        help="Evaluate one trained JEPA checkpoint and attach metrics to METHOD_ID.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = _load_yaml(Path(args.config))
    output_dir = Path(args.output_dir or cfg["meta"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = RunRequest(
        run_riemann=args.run_riemann,
        run_random_floor=args.run_random_floor,
        checkpoint_overrides=_parse_checkpoint_overrides(args.checkpoint),
        data_root=args.data_root,
        label_scheme=args.label_scheme,
        class_names=_parse_classes(args.classes),
        train_split=args.train_split,
        eval_split=args.eval_split,
        riemann_classifier=args.riemann_classifier,
        riemann_aggregation=args.riemann_aggregation,
        riemann_mean_metric=args.riemann_mean_metric,
        riemann_distance_metric=args.riemann_distance_metric,
        riemann_tangent_metric=args.riemann_tangent_metric,
        riemann_cov_estimator=args.riemann_cov_estimator,
    )
    rows = _local_rows(cfg, requests) + _published_rows(cfg)
    ranked = _rank_rows(rows, cfg["meta"]["primary_metric"])

    _write_json(ranked, output_dir / "eeg_benchmark_rows.json")
    _write_csv(ranked, output_dir / "eeg_benchmark.csv")
    _write_markdown(ranked, output_dir / "eeg_benchmark.md", cfg)
    _write_html(ranked, output_dir / "eeg_benchmark.html", cfg)
    for metric in METRIC_COLUMNS:
        _write_plot(ranked, output_dir / f"eeg_benchmark_{metric}.png", metric)

    print(f"[benchmark] wrote artifacts to {output_dir}")
    print("[benchmark] no pretraining was launched")


if __name__ == "__main__":
    main()
