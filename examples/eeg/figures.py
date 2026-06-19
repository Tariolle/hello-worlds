"""EEG-JEPA — figure generation for the hackathon deck.

Three figures:
  fig1  2×2 factorial bar chart with 3-seed error bars
  fig2  2-panel "value of self-supervision"
          panel A: BalAcc vs % label fraction
          panel B: BalAcc vs % pretrain-data fraction
  fig3  Collapse dynamics: eff_rank + mean_std per epoch, per cell

Usage:
  python -m examples.eeg.figures fig1 [--csv results/2x2.csv] [--out results/fig_2x2.png]
  python -m examples.eeg.figures fig2 [--label-csv ...] [--data-csv ...] [--out ...]
  python -m examples.eeg.figures fig3 [--log logs/run.log] [--out ...]
  python -m examples.eeg.figures all  [--out-dir results/figures]

CSV formats expected by Florent (see --help for each subcommand):

  fig1  results/2x2.csv
        reg,space,bal_acc_mean,bal_acc_std
        sigreg,ambient,0.819,0.007
        ...

  fig2  results/label_eff.csv   (label-efficiency, 5-seed re-fits)
        results/data_eff.csv    (data-efficiency, optional)
        fraction,bal_acc_mean,bal_acc_std
        0.01,0.72,0.02
        ...

  fig3  logs/run.log  OR  results/collapse.csv
        log  format: lines printed by main.py:
          [eeg] epoch 0 loss=1.2345 {'eff_rank': 3.2, 'mean_std': 0.012, ...}
        csv  format:
          epoch,cell,eff_rank,mean_std
          0,sigreg_ambient,3.2,0.012
          ...
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-eeg-figures")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# --------------------------------------------------------------------------- #
# Reference constants (from tasks/todo.md — 3-seed results)
# --------------------------------------------------------------------------- #

RIEMANN_BAL_ACC = 0.761
# Random floor: run `python -m examples.eeg.benchmark --run-random-floor`
# and fill in below. Placeholder until measured.
RANDOM_FLOOR_BAL_ACC = None   # set once measured

# Fine-tuned foundation models band (comparable cross-corpus references)
FT_BAND_LOW = 0.796   # BIOT fine-tuned
FT_BAND_HIGH = 0.829  # CBraMod fine-tuned

# Frozen baselines from published works (for the SSL-value panels)
BIOT_FROZEN = 0.780
EEG2REP_FROZEN = 0.766

# Hardcoded 3-seed means — used as default when no CSV is supplied
_2X2_DEFAULTS: list[dict[str, Any]] = [
    {"reg": "VICReg", "space": "ambient", "bal_acc_mean": 0.814, "bal_acc_std": 0.006},
    {"reg": "SIGReg", "space": "ambient", "bal_acc_mean": 0.819, "bal_acc_std": 0.007},
    {"reg": "SIGReg", "space": "tangent", "bal_acc_mean": 0.820, "bal_acc_std": 0.005},
    {"reg": "PEIRA",  "space": "ambient", "bal_acc_mean": 0.815, "bal_acc_std": 0.006},
    {"reg": "PEIRA",  "space": "tangent", "bal_acc_mean": 0.807, "bal_acc_std": 0.008},
]

# Colours
_COL_AMBIENT = "#2f80ed"
_COL_TANGENT = "#9b51e0"
_COL_RIEMANN = "#f2994a"
_COL_RANDOM  = "#828282"
_COL_FT_BAND = "#eb5757"


# --------------------------------------------------------------------------- #
# CSV loaders
# --------------------------------------------------------------------------- #

def _load_2x2_csv(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "reg":          r["reg"].strip(),
                "space":        r["space"].strip(),
                "bal_acc_mean": float(r["bal_acc_mean"]),
                "bal_acc_std":  float(r["bal_acc_std"]),
            })
    return rows


def _load_eff_csv(path: str | Path) -> tuple[list[float], list[float], list[float]]:
    """Returns (fractions, means, stds) sorted by fraction."""
    fracs, means, stds = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fracs.append(float(r["fraction"]))
            means.append(float(r["bal_acc_mean"]))
            stds.append(float(r["bal_acc_std"]))
    order = np.argsort(fracs)
    return [fracs[i] for i in order], [means[i] for i in order], [stds[i] for i in order]


def _parse_log_collapse(log_path: str | Path) -> dict[str, list]:
    """Parse stdout lines from main.py into per-cell collapse metrics.

    Expected line format:
      [eeg] epoch 3 loss=0.1234 {'eff_rank': 27.3, 'mean_std': 0.042, ...}
    The cell name is inferred from the log file name or a nearby marker line.
    Returns {cell: {'epoch': [...], 'eff_rank': [...], 'mean_std': [...]}}
    """
    pattern = re.compile(
        r"\[eeg\]\s+epoch\s+(\d+).*?'eff_rank':\s*([\d.]+).*?'mean_std':\s*([\d.]+)"
    )
    cell = Path(log_path).stem  # e.g. "sigreg_ambient"
    epochs, eff_ranks, mean_stds = [], [], []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                eff_ranks.append(float(m.group(2)))
                mean_stds.append(float(m.group(3)))
    return {cell: {"epoch": epochs, "eff_rank": eff_ranks, "mean_std": mean_stds}}


def _load_collapse_csv(path: str | Path) -> dict[str, list]:
    """Load collapse CSV with columns: epoch,cell,eff_rank,mean_std."""
    data: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cell = r["cell"].strip()
            if cell not in data:
                data[cell] = {"epoch": [], "eff_rank": [], "mean_std": []}
            data[cell]["epoch"].append(int(r["epoch"]))
            data[cell]["eff_rank"].append(float(r["eff_rank"]))
            data[cell]["mean_std"].append(float(r["mean_std"]))
    return data


# --------------------------------------------------------------------------- #
# Figure 1 — 2×2 factorial bar chart
# --------------------------------------------------------------------------- #

def plot_2x2(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    """Grouped bar chart: reg (x-axis groups) × space (ambient|tangent bars).

    Horizontal reference lines: Riemann baseline, random floor (if known),
    fine-tuned foundation-model band.
    """
    regs  = ["VICReg", "SIGReg", "PEIRA"]
    by_key = {(r["reg"], r["space"]): r for r in rows}

    x = np.arange(len(regs))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))

    ambient_means, ambient_stds = [], []
    tangent_means, tangent_stds = [], []
    for reg in regs:
        a = by_key.get((reg, "ambient"))
        t = by_key.get((reg, "tangent"))
        ambient_means.append(a["bal_acc_mean"] if a else np.nan)
        ambient_stds.append(a["bal_acc_std"]  if a else 0.0)
        tangent_means.append(t["bal_acc_mean"] if t else np.nan)
        tangent_stds.append(t["bal_acc_std"]   if t else 0.0)

    bars_a = ax.bar(x - width / 2, ambient_means, width,
                    yerr=ambient_stds, capsize=4,
                    color=_COL_AMBIENT, label="ambient (Euclidean)", alpha=0.88)
    bars_t = ax.bar(x + width / 2, tangent_means, width,
                    yerr=tangent_stds, capsize=4,
                    color=_COL_TANGENT, label="tangent (SPD log-Euclidean)", alpha=0.88)

    # Value labels
    for bar, val in zip(list(bars_a) + list(bars_t),
                        ambient_means + tangent_means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Reference lines
    ymin = 0.75
    ax.axhline(RIEMANN_BAL_ACC, color=_COL_RIEMANN, lw=1.5, ls="--",
               label=f"Riemannian 0-param ({RIEMANN_BAL_ACC:.3f})")
    if RANDOM_FLOOR_BAL_ACC is not None:
        ax.axhline(RANDOM_FLOOR_BAL_ACC, color=_COL_RANDOM, lw=1.2, ls=":",
                   label=f"random encoder floor ({RANDOM_FLOOR_BAL_ACC:.3f})")
        ymin = min(ymin, RANDOM_FLOOR_BAL_ACC - 0.02)

    ax.axhspan(FT_BAND_LOW, FT_BAND_HIGH, color=_COL_FT_BAND, alpha=0.08,
               label=f"fine-tuned foundation models [{FT_BAND_LOW:.3f}–{FT_BAND_HIGH:.3f}]")
    ax.axhline(FT_BAND_LOW,  color=_COL_FT_BAND, lw=0.7, ls="--", alpha=0.5)
    ax.axhline(FT_BAND_HIGH, color=_COL_FT_BAND, lw=0.7, ls="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(regs, fontsize=11)
    ax.set_ylabel("Balanced accuracy (frozen linear probe)", fontsize=10)
    ax.set_title("TUAB 2×2 factorial: regulariser × representation space\n"
                 "(3-seed mean ± std, full 2717/276 patient-disjoint split)", fontsize=10)
    ax.set_ylim(ymin, max(max(ambient_means + tangent_means,
                              default=0.85), FT_BAND_HIGH) + 0.025)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[figures] fig1 -> {out_path}")


# --------------------------------------------------------------------------- #
# Figure 2 — 2-panel "value of self-supervision"
# --------------------------------------------------------------------------- #

def plot_value_of_ssl(
    label_data: tuple | None,
    data_data: tuple | None,
    out_path: str | Path,
) -> None:
    """2-panel plot.

    label_data / data_data: (fracs, means, stds) as returned by _load_eff_csv,
    or None to skip that panel.

    Reference overlays on both panels:
      - horizontal: Riemann 0-param, BIOT frozen, EEG2Rep frozen
      - shaded band: fine-tuned foundation models
      - dotted: random encoder floor (if known)
    """
    n_panels = int(label_data is not None) + int(data_data is not None)
    if n_panels == 0:
        print("[figures] fig2: no data supplied, skipping", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    panels = []
    if label_data is not None:
        panels.append(("% labelled train data", label_data))
    if data_data is not None:
        panels.append(("% pretrain data", data_data))

    for ax, (xlabel, (fracs, means, stds)) in zip(axes, panels):
        xs = np.array(fracs) * 100
        ys = np.array(means)
        es = np.array(stds)

        ax.fill_between(xs, ys - es, ys + es, color=_COL_AMBIENT, alpha=0.2)
        ax.plot(xs, ys, "o-", color=_COL_AMBIENT, lw=2,
                label="EEG-JEPA SIGReg (ours, frozen)")

        # Reference lines
        ax.axhline(RIEMANN_BAL_ACC, color=_COL_RIEMANN, lw=1.5, ls="--",
                   label=f"Riemannian 0-param ({RIEMANN_BAL_ACC:.3f})")
        ax.axhline(BIOT_FROZEN, color="#6fcf97", lw=1.2, ls="--",
                   label=f"BIOT frozen ({BIOT_FROZEN:.3f})")
        ax.axhline(EEG2REP_FROZEN, color="#56ccf2", lw=1.2, ls="--",
                   label=f"EEG2Rep frozen ({EEG2REP_FROZEN:.3f})")
        if RANDOM_FLOOR_BAL_ACC is not None:
            ax.axhline(RANDOM_FLOOR_BAL_ACC, color=_COL_RANDOM, lw=1.0, ls=":",
                       label=f"random encoder floor ({RANDOM_FLOOR_BAL_ACC:.3f})")
        ax.axhspan(FT_BAND_LOW, FT_BAND_HIGH, color=_COL_FT_BAND, alpha=0.08)
        ax.axhline(FT_BAND_LOW,  color=_COL_FT_BAND, lw=0.7, ls="--", alpha=0.4)
        ax.axhline(FT_BAND_HIGH, color=_COL_FT_BAND, lw=0.7, ls="--", alpha=0.4)
        ft_patch = mpatches.Patch(color=_COL_FT_BAND, alpha=0.2,
                                  label=f"fine-tuned FMs [{FT_BAND_LOW:.3f}–{FT_BAND_HIGH:.3f}]"
                                        "\n(cross-corpus, full fine-tune)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [ft_patch], labels + [ft_patch.get_label()],
                  fontsize=7.5, loc="lower right")

        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Balanced accuracy (frozen linear probe)", fontsize=10)
        ax.set_title(f"Value of self-supervision: BalAcc vs {xlabel}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_ylim(0.5, max(float(np.max(ys)), FT_BAND_HIGH) + 0.03)

    fig.suptitle("EEG-JEPA TUAB — how much supervision / data do we need?\n"
                 "(full 2717/276 patient-disjoint split, SIGReg-ambient best cell)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[figures] fig2 -> {out_path}")


# --------------------------------------------------------------------------- #
# Figure 3 — Collapse dynamics
# --------------------------------------------------------------------------- #

_CELL_COLORS = {
    "sigreg_ambient": _COL_AMBIENT,
    "sigreg_tangent": _COL_TANGENT,
    "peira_ambient":  "#27ae60",
    "peira_tangent":  "#e67e22",
    "vicreg_ambient": "#95a5a6",
}
_CELL_LABELS = {
    "sigreg_ambient": "SIGReg ambient",
    "sigreg_tangent": "SIGReg tangent",
    "peira_ambient":  "PEIRA ambient",
    "peira_tangent":  "PEIRA tangent",
    "vicreg_ambient": "VICReg ambient",
}


def plot_collapse_dynamics(collapse_data: dict[str, dict], out_path: str | Path) -> None:
    """Two-row subplot: eff_rank (top) and mean_std (bottom) vs epoch, one line per cell."""
    if not collapse_data:
        print("[figures] fig3: no collapse data, skipping", file=sys.stderr)
        return

    fig, (ax_rank, ax_std) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for cell, series in sorted(collapse_data.items()):
        epochs = series["epoch"]
        color  = _CELL_COLORS.get(cell, "#333333")
        label  = _CELL_LABELS.get(cell, cell)
        if series["eff_rank"]:
            ax_rank.plot(epochs, series["eff_rank"], "-", color=color, lw=1.8, label=label)
        if series["mean_std"]:
            ax_std.plot(epochs, series["mean_std"], "-", color=color, lw=1.8, label=label)

    ax_rank.set_ylabel("Effective rank", fontsize=10)
    ax_rank.set_title("Collapse dynamics per cell — higher is better (no collapse)",
                      fontsize=10)
    ax_rank.legend(fontsize=8, loc="upper left")
    ax_rank.grid(alpha=0.25)

    ax_std.set_ylabel("Mean per-dim std", fontsize=10)
    ax_std.set_xlabel("Epoch", fontsize=10)
    ax_std.set_title("Per-dimension std over training — should grow from ~0", fontsize=10)
    ax_std.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[figures] fig3 -> {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _add_common(p):
    p.add_argument("--out-dir", default="results/figures",
                   help="Output directory (default: results/figures)")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("fig1", help="2×2 factorial bar chart")
    p1.add_argument("--csv", default=None,
                    help="Path to 2x2.csv (reg,space,bal_acc_mean,bal_acc_std). "
                         "Falls back to hardcoded 3-seed means from todo.md.")
    p1.add_argument("--out", default=None, help="Output PNG path.")
    _add_common(p1)

    p2 = sub.add_parser("fig2", help="2-panel value-of-SSL efficiency curves")
    p2.add_argument("--label-csv", default=None,
                    help="CSV for label-efficiency curve (fraction,bal_acc_mean,bal_acc_std).")
    p2.add_argument("--data-csv", default=None,
                    help="CSV for pretrain-data-efficiency curve (same format). Optional.")
    p2.add_argument("--out", default=None, help="Output PNG path.")
    _add_common(p2)

    p3 = sub.add_parser("fig3", help="Collapse dynamics from training logs")
    p3.add_argument("--log", action="append", default=[],
                    help="Path to a stdout log file from main.py. Repeat for multiple cells. "
                         "File stem is used as cell name.")
    p3.add_argument("--csv", default=None,
                    help="Path to collapse CSV (epoch,cell,eff_rank,mean_std). "
                         "Alternative to --log.")
    p3.add_argument("--out", default=None, help="Output PNG path.")
    _add_common(p3)

    pa = sub.add_parser("all", help="Generate all available figures")
    pa.add_argument("--2x2-csv", dest="csv_2x2", default=None)
    pa.add_argument("--label-csv", default=None)
    pa.add_argument("--data-csv", default=None)
    pa.add_argument("--collapse-log", action="append", default=[], dest="collapse_logs")
    pa.add_argument("--collapse-csv", default=None)
    _add_common(pa)

    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd in ("fig1", "all"):
        csv_2x2 = getattr(args, "csv", None) or getattr(args, "csv_2x2", None)
        rows = _load_2x2_csv(csv_2x2) if csv_2x2 else _2X2_DEFAULTS
        out = getattr(args, "out", None) or str(out_dir / "fig1_2x2.png")
        plot_2x2(rows, out)

    if args.cmd in ("fig2", "all"):
        label_csv = getattr(args, "label_csv", None)
        data_csv  = getattr(args, "data_csv", None)
        if label_csv is None and data_csv is None:
            if args.cmd == "fig2":
                print("[figures] fig2: supply --label-csv and/or --data-csv", file=sys.stderr)
                sys.exit(1)
            else:
                print("[figures] fig2: skipped (no CSV supplied)", file=sys.stderr)
        else:
            label_data = _load_eff_csv(label_csv) if label_csv else None
            data_data  = _load_eff_csv(data_csv)  if data_csv  else None
            out = getattr(args, "out", None) or str(out_dir / "fig2_ssl_value.png")
            plot_value_of_ssl(label_data, data_data, out)

    if args.cmd in ("fig3", "all"):
        logs = getattr(args, "log", []) or getattr(args, "collapse_logs", [])
        collapse_csv = getattr(args, "csv", None) or getattr(args, "collapse_csv", None)
        collapse_data: dict = {}
        if collapse_csv:
            collapse_data = _load_collapse_csv(collapse_csv)
        for log_path in logs:
            collapse_data.update(_parse_log_collapse(log_path))
        out = getattr(args, "out", None) or str(out_dir / "fig3_collapse.png")
        if collapse_data:
            plot_collapse_dynamics(collapse_data, out)
        elif args.cmd == "fig3":
            print("[figures] fig3: supply --log or --csv", file=sys.stderr)
            sys.exit(1)
        else:
            print("[figures] fig3: skipped (no log/csv supplied)", file=sys.stderr)


if __name__ == "__main__":
    main()
