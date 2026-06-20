"""Generate precise TUAB Abnormal comparison figures vs Laya.

The figure is intentionally framed as an indicative comparison against published
numbers, not a controlled reproduction. It visualizes:
  * Laya v2 Table 2 clinical "Abnormal" frozen linear probe numbers.
  * Our Euclidean latent EEG JEPA v3 seeds.
  * A conservative panel using only isolated reruns completed during this sweep.
  * An "all available v3 runs" panel including the previous seed-1 v3 run.

Outputs:
  results/figures/fig_laya_compare.png
  results/figures/fig_laya_compare_conservative.png
  results/figures/laya_compare_source.csv
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


OUT = os.path.join(os.path.dirname(__file__), "../../results/figures")
os.makedirs(OUT, exist_ok=True)


@dataclass(frozen=True)
class PublishedResult:
    name: str
    bal_acc: float
    std: float
    note: str


LAYA_TABLE2_ABNORMAL = [
    PublishedResult("REVE", 0.500, 0.000, "Laya v2 Table 2"),
    PublishedResult("CBraMod", 0.605, 0.011, "Laya v2 Table 2"),
    PublishedResult("LUNA", 0.733, 0.018, "Laya v2 Table 2"),
    PublishedResult("Laya-B", 0.778, 0.006, "Laya v2 Table 2"),
    PublishedResult("LaBraM", 0.781, 0.013, "Laya v2 Table 2"),
    PublishedResult("Laya-S", 0.798, 0.007, "Laya v2 Table 2"),
]

# Completed isolated reruns from cluster/train_ijepa_v3_seed.sbatch.
OURS_CONSERVATIVE = [
    ("seed 10000", 0.8183, "isolated rerun"),
    ("seed 1000", 0.8235, "isolated rerun"),
]

# Previous v3 seed-1 run with the same config, before seed-specific script.
OURS_ALL_AVAILABLE = [
    ("seed 10000", 0.8183, "isolated rerun"),
    ("seed 1000", 0.8235, "isolated rerun"),
    ("seed 1", 0.8341, "previous v3 run"),
]

SOURCE_URL = "https://arxiv.org/html/2603.16281v2"


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def write_source_csv() -> None:
    path = os.path.join(OUT, "laya_compare_source.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "method_or_seed", "balanced_accuracy", "std", "note", "source"])
        for row in LAYA_TABLE2_ABNORMAL:
            w.writerow(["published", row.name, row.bal_acc, row.std, row.note, SOURCE_URL])
        for seed, score, note in OURS_ALL_AVAILABLE:
            w.writerow(["ours_all_available", seed, score, "", note, "local Dalia logs"])
        for seed, score, note in OURS_CONSERVATIVE:
            w.writerow(["ours_conservative", seed, score, "", note, "local Dalia logs"])
    print(f"Saved {path}")


def plot_panel(ax, ours_runs: list[tuple[str, float, str]], title: str, subtitle: str = "") -> None:
    published_names = [r.name for r in LAYA_TABLE2_ABNORMAL]
    published_vals = [r.bal_acc for r in LAYA_TABLE2_ABNORMAL]
    published_errs = [r.std for r in LAYA_TABLE2_ABNORMAL]

    ours_vals = [v for _, v, _ in ours_runs]
    ours_mu, ours_sd = mean_std(ours_vals)
    laya_s = next(r for r in LAYA_TABLE2_ABNORMAL if r.name == "Laya-S")
    delta = ours_mu - laya_s.bal_acc

    labels = published_names + ["Ours\nEEG JEPA"]
    vals = published_vals + [ours_mu]
    errs = published_errs + [ours_sd]
    colors = ["#B0BEC5"] * len(published_names) + ["#1565C0"]
    edges = ["white"] * len(published_names) + ["#0D47A1"]

    x = np.arange(len(labels))
    ax.bar(
        x,
        vals,
        yerr=errs,
        capsize=4,
        color=colors,
        edgecolor=edges,
        linewidth=1.0,
        zorder=3,
    )

    # Individual seed markers for ours.
    jitter = np.linspace(-0.12, 0.12, len(ours_runs)) if len(ours_runs) > 1 else [0.0]
    ours_x = x[-1]
    for (seed, score, note), dx in zip(ours_runs, jitter):
        marker = "o" if note == "isolated rerun" else "D"
        ax.scatter(
            ours_x + dx,
            score,
            s=58,
            marker=marker,
            color="white",
            edgecolor="#0D47A1",
            linewidth=1.5,
            zorder=5,
        )
        ax.text(
            ours_x + dx,
            score + 0.006,
            seed.replace("seed ", "s"),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#0D47A1",
        )

    # Value labels.
    for xi, v, e in zip(x, vals, errs):
        ax.text(
            xi,
            v + e + 0.010,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(0.5, color="#E53935", linestyle=":", linewidth=1.2, zorder=2)
    ax.axhline(laya_s.bal_acc, color="#455A64", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(
        len(labels) - 1.56,
        laya_s.bal_acc + 0.004,
        "Laya-S published",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#455A64",
    )

    ax.annotate(
        f"+{delta:.3f} vs Laya-S",
        xy=(ours_x, ours_mu),
        xytext=(ours_x - 1.55, ours_mu + 0.055),
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.4),
        fontsize=10,
        fontweight="bold",
        color="#1565C0",
    )

    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=14)
    if subtitle:
        ax.text(
            0.0,
            0.985,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#37474F",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        )
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.46, 0.88)
    ax.grid(axis="y", alpha=0.25, zorder=0)


def make_combined_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    plot_panel(
        axes[0],
        OURS_CONSERVATIVE,
        "A — Conservative",
        "Completed seed-specific reruns only: seeds 1000 and 10000.",
    )
    plot_panel(
        axes[1],
        OURS_ALL_AVAILABLE,
        "B — All available v3 runs",
        "Adds previous seed-1 v3 run; isolated seed-1 rerun still running.",
    )

    handles = [
        Patch(facecolor="#B0BEC5", edgecolor="white", label="Published Laya v2 Table 2"),
        Patch(facecolor="#1565C0", edgecolor="#0D47A1", label="Ours: Euclidean latent EEG JEPA"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="#0D47A1",
               label="Isolated rerun seed", markersize=7, linewidth=0),
        Line2D([0], [0], marker="D", color="white", markeredgecolor="#0D47A1",
               label="Previous v3 seed", markersize=7, linewidth=0),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.055))
    fig.suptitle(
        "TUAB Abnormal frozen linear probe — indicative comparison vs Laya",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.005,
        "Published baselines: Laya v2 Table 2, clinical 'Abnormal' task, balanced accuracy mean±std over 5 seeds. "
        "Ours: frozen logistic probe on TUAB eval, Euclidean latent JEPA pretrained on TUAB train + TUEV. "
        "This is an indicative published-number comparison, not a controlled reproduction of Laya.",
        ha="center",
        va="top",
        fontsize=8,
        color="#37474F",
        wrap=True,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    path = os.path.join(OUT, "fig_laya_compare.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def make_conservative_single() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    plot_panel(
        ax,
        OURS_CONSERVATIVE,
        "TUAB Abnormal frozen probe — conservative comparison vs Laya",
        "",
    )
    handles = [
        Patch(facecolor="#B0BEC5", edgecolor="white", label="Published Laya v2 Table 2"),
        Patch(facecolor="#1565C0", edgecolor="#0D47A1", label="Ours: Euclidean latent EEG JEPA"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="#0D47A1",
               label="Completed isolated rerun seed", markersize=7, linewidth=0),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.055))
    fig.text(
        0.5,
        0.01,
        "Strict claim: ours mean over two completed isolated reruns vs Laya-S published 0.798±0.007. "
        "Published-number comparison only; Laya was not rerun locally.",
        ha="center",
        va="top",
        fontsize=8,
        color="#37474F",
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.98])
    path = os.path.join(OUT, "fig_laya_compare_conservative.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    write_source_csv()
    make_combined_figure()
    make_conservative_single()


if __name__ == "__main__":
    main()
