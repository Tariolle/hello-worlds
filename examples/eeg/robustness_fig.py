"""Robustness degradation figure (3-seed) — frozen-probe BA vs corruption severity.

Two panels (channel-dropout, additive noise). Per 2x2 cell: mean +- std balanced
accuracy over 3 seeds {1,1000,10000} at each severity. Random-encoder floor shown
single-seed as a dashed reference. Honest read: AMBIENT (esp. SIGReg) is the most
dropout-robust; the SPD-tangent / PEIRA objective degrades MORE under dropout; under
additive noise the cells overlap (no clean geometry advantage at 3 seeds).

Run (local, no GPU):  python examples/eeg/robustness_fig.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# balanced accuracy, 3 seeds {1, 1000, 10000}
DATA = {
    "SIGReg/ambient": {
        "clean": [0.8329, 0.8137, 0.8089],
        "drop": {0.1: [0.8341, 0.8129, 0.8162], 0.25: [0.7968, 0.8081, 0.8329], 0.5: [0.7735, 0.7687, 0.7762]},
        "noise": {0.1: [0.8295, 0.8137, 0.8116], 0.25: [0.8302, 0.8097, 0.8222], 0.5: [0.7921, 0.7989, 0.7786]},
        "c": "#2f80ed"},
    "SIGReg/tangent": {
        "clean": [0.8176, 0.8183, 0.8230],
        "drop": {0.1: [0.7929, 0.8284, 0.8010], 0.25: [0.7594, 0.7929, 0.7722], 0.5: [0.7217, 0.6395, 0.7216]},
        "noise": {0.1: [0.8203, 0.8049, 0.8230], 0.25: [0.8124, 0.7876, 0.8098], 0.5: [0.8016, 0.7443, 0.8271]},
        "c": "#e67e22"},
    "PEIRA/ambient": {
        "clean": [0.8262, 0.8295, 0.7890],
        "drop": {0.1: [0.8073, 0.8268, 0.7810], 0.25: [0.7351, 0.8348, 0.7798], 0.5: [0.6181, 0.7621, 0.5908]},
        "noise": {0.1: [0.8249, 0.8262, 0.7890], 0.25: [0.8130, 0.8195, 0.7844], 0.5: [0.7997, 0.7740, 0.7552]},
        "c": "#16a085"},
    "PEIRA/tangent": {
        "clean": [0.8110, 0.8010, 0.8083],
        "drop": {0.1: [0.7808, 0.7792, 0.8141], 0.25: [0.7713, 0.7268, 0.8194], 0.5: [0.6906, 0.5567, 0.8054]},
        "noise": {0.1: [0.8070, 0.7970, 0.8070], 0.25: [0.8110, 0.7905, 0.8124], 0.5: [0.8022, 0.7900, 0.8151]},
        "c": "#8e44ad"},
}
# random-encoder floor (single seed, reference only)
RANDOM = {"clean": 0.7943, "drop": {0.1: 0.7625, 0.25: 0.7173, 0.5: 0.5522},
          "noise": {0.1: 0.7922, 0.25: 0.7668, 0.5: 0.7038}}
LEVELS = [0.1, 0.25, 0.5]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, kind, title in zip(axes, ("drop", "noise"),
                           ("(a) Channel dropout (zero-pad)", "(b) Additive noise")):
    xs = [0.0] + LEVELS
    for name, d in DATA.items():
        ys = [np.mean(d["clean"])] + [np.mean(d[kind][l]) for l in LEVELS]
        es = [np.std(d["clean"])] + [np.std(d[kind][l]) for l in LEVELS]
        ax.errorbar(xs, ys, yerr=es, marker="o", lw=2, capsize=3, color=d["c"], label=name)
    ry = [RANDOM["clean"]] + [RANDOM[kind][l] for l in LEVELS]
    ax.plot(xs, ry, marker="s", lw=1.5, ls="--", color="#7f8c8d", label="random floor (1 seed)")
    ax.set_xlabel("corruption severity"); ax.set_title(title)
    ax.grid(alpha=0.3); ax.set_xticks(xs)
axes[0].set_ylabel("Balanced accuracy (held-out patients)")
axes[1].legend(fontsize=8, loc="lower left")
fig.suptitle("Frozen-probe robustness on TUAB (3-seed mean ± std) — ambient SIGReg is the most dropout-robust")
fig.tight_layout()
out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "results", "robustness", "robustness_3seed.png"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
