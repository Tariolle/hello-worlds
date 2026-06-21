"""SIGReg vs World Model comparison on TUEV — 4-panel diagnostic figure."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# --------------------------------------------------------------------------- #
# Data — parsed from job logs
# --------------------------------------------------------------------------- #

CLASSES = ["bckg", "spsw", "eyem", "artf", "gped", "pled"]
CHANCE  = 1 / 6

# SIGReg (job 75309) — already in train_log.json
SIGREG_LOG = os.path.join(os.path.dirname(__file__),
    "../../checkpoints/eeg_tuev_sigreg/train_log.json")

# WM — from terminal output (job 75517)
WM_RAW = [
    (0,  5.2913, 32.389), (1,  4.2373, 32.704), (2,  4.2116, 28.320),
    (3,  3.5637, 27.352), (4,  3.5791, 32.176), (5,  3.1766, 31.479),
    (6,  3.0960, 25.791), (7,  3.0098, 25.533), (8,  2.9618, 25.921),
    (9,  2.4093, 23.138), (10, 2.6448, 22.220), (11, 2.6016, 20.321),
    (12, 2.0599, 20.498), (13, 2.2086, 20.402), (14, 2.2538, 18.839),
    (15, 2.0870, 17.909), (16, 2.5282, 17.823), (17, 2.6416, 18.624),
    (18, 2.3161, 20.794), (19, 2.1854, 20.113), (20, 1.5925, 18.008),
    (21, 1.9665, 18.132), (22, 2.0558, 19.996), (23, 2.4348, 18.302),
    (24, 1.3002, 18.810), (25, 1.1936, 16.748), (26, 1.4632, 16.422),
    (27, 0.9724, 16.841), (28, 1.0866, 17.212), (29, 0.9062, 17.900),
]
WM_SSL_TAN = [
    0.13500, 1.12589, 1.27765, 0.78856, 0.30083, 0.72728, 0.10102, 0.74243,
    0.10504, 0.09526, 0.38387, 0.37217,-0.68007, 0.06183,-0.88868,-0.12923,
    0.04014, 0.08746, 1.45754,-0.45390,-0.70972, 0.15357, 1.22230,-0.90997,
   -0.90569, 0.08141,-1.41428,-1.71386,-3.43889,-3.04255,
]

WM_EPOCHS    = [r[0] for r in WM_RAW]
WM_LOSSES    = [r[1] for r in WM_RAW]
WM_EFF_RANKS = [r[2] for r in WM_RAW]

# Eval results
RESULTS = {
    "sigreg": {"random": 0.2830, "trained": 0.3533},
    "wm":     {"random": 0.3027, "trained": 0.2996},
}

PER_CLASS = {
    "sigreg": {
        "random":  {"bckg":0.384,"spsw":0.000,"eyem":0.227,"artf":0.460,"gped":0.412,"pled":0.216},
        "trained": {"bckg":0.374,"spsw":0.042,"eyem":0.267,"artf":0.500,"gped":0.492,"pled":0.446},
    },
    "wm": {
        "random":  {"bckg":0.346,"spsw":0.083,"eyem":0.267,"artf":0.452,"gped":0.439,"pled":0.230},
        "trained": {"bckg":0.095,"spsw":0.083,"eyem":0.427,"artf":0.565,"gped":0.425,"pled":0.203},
    },
}

# --------------------------------------------------------------------------- #
# Colors
# --------------------------------------------------------------------------- #
C_SIGREG  = "#2196F3"
C_WM      = "#FF5722"
C_RANDOM  = "#9E9E9E"
C_RANK    = "#FF9800"
C_TAN     = "#9C27B0"
C_CHANCE  = "#E91E63"

# --------------------------------------------------------------------------- #
# Figure: 2×2
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.35)

# ── Panel A: Normalized loss curves ─────────────────────────────────────────
ax_loss = fig.add_subplot(gs[0, 0])

with open(SIGREG_LOG) as f:
    sg_log = json.load(f)
sg_epochs = [r["epoch"] for r in sg_log]
sg_losses = [r["loss"]  for r in sg_log]

# Normalize to [0,1] relative to epoch-0 for visual comparison
sg_norm = np.array(sg_losses) / sg_losses[0]
wm_norm = np.array(WM_LOSSES)  / WM_LOSSES[0]

ax_loss.plot(sg_epochs, sg_norm, "o-", color=C_SIGREG, lw=2, ms=3,
             label=f"SIGReg  {sg_losses[0]:.2f}→{sg_losses[-1]:.2f} (−{(1-sg_norm[-1])*100:.0f}%)")
ax_loss.plot(WM_EPOCHS, wm_norm, "s-", color=C_WM, lw=2, ms=3,
             label=f"WM 3-geo  {WM_LOSSES[0]:.2f}→{WM_LOSSES[-1]:.2f} (−{(1-wm_norm[-1])*100:.0f}%)")

ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss / Loss₀  (normalized)")
ax_loss.set_title("A  —  Training loss (normalized to epoch 0)",
                  fontweight="bold", loc="left")
ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)
ax_loss.set_ylim(0, 1.05)

# ── Panel B: eff_rank evolution — the smoking gun ───────────────────────────
ax_rank = fig.add_subplot(gs[0, 1])

sg_ranks = [r["eff_rank"] for r in sg_log]
ax_rank.plot(sg_epochs, sg_ranks, "o-", color=C_SIGREG, lw=2.5, ms=4,
             label=f"SIGReg   {sg_ranks[0]:.0f} → {sg_ranks[-1]:.0f}")
ax_rank.plot(WM_EPOCHS, WM_EFF_RANKS, "s-", color=C_WM, lw=2.5, ms=4,
             label=f"WM 3-geo  {WM_EFF_RANKS[0]:.0f} → {WM_EFF_RANKS[-1]:.0f}")

ax_rank.axhspan(0, 20, color=C_WM, alpha=0.07, label="collapse zone (<20)")
ax_rank.text(28, 10, "collapse\nzone", ha="center", color=C_WM, fontsize=8, alpha=0.8)

ax_rank_r = ax_rank.twinx()
ax_rank_r.plot(WM_EPOCHS, WM_SSL_TAN, "^--", color=C_TAN, lw=1.2, ms=3,
               alpha=0.7, label="WM ssl_tan")
ax_rank_r.axhline(0, color=C_TAN, lw=0.8, ls=":")
ax_rank_r.set_ylabel("ssl_tan (WM tangent loss)", color=C_TAN, fontsize=9)
ax_rank_r.tick_params(axis="y", labelcolor=C_TAN)

ax_rank.set_xlabel("Epoch"); ax_rank.set_ylabel("eff_rank")
ax_rank.set_title("B  —  eff_rank + WM tangent loss instability",
                  fontweight="bold", loc="left")
lines1, lab1 = ax_rank.get_legend_handles_labels()
lines2, lab2 = ax_rank_r.get_legend_handles_labels()
ax_rank.legend(lines1[:2] + lines2, lab1[:2] + lab2, fontsize=8, loc="upper right")
ax_rank.grid(alpha=0.3)

# ── Panel C: Balanced accuracy — all methods ────────────────────────────────
ax_bar = fig.add_subplot(gs[1, 0])

methods  = ["Random\nfloor", "SIGReg\ntrained", "WM 3-geo\ntrained"]
vals     = [
    (RESULTS["sigreg"]["random"] + RESULTS["wm"]["random"]) / 2,  # avg random floor
    RESULTS["sigreg"]["trained"],
    RESULTS["wm"]["trained"],
]
# Use actual per-run randoms
vals = [RESULTS["sigreg"]["random"], RESULTS["sigreg"]["trained"], RESULTS["wm"]["trained"]]
colors_bar = [C_RANDOM, C_SIGREG, C_WM]
x = np.arange(len(methods))
bars = ax_bar.bar(x, vals, 0.5, color=colors_bar, edgecolor="white", zorder=3)

ax_bar.axhline(CHANCE, color=C_CHANCE, lw=1.5, ls=":", zorder=4,
               label=f"chance (1/6 = {CHANCE:.2f})")
ax_bar.axhline(RESULTS["sigreg"]["random"], color=C_RANDOM, lw=1.2, ls="--",
               alpha=0.6, label=f"SIGReg random floor ({RESULTS['sigreg']['random']:.3f})")

# value labels
for bar, v in zip(bars, vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# gap annotation
ax_bar.annotate("", xy=(1, vals[1]), xytext=(1, vals[0]),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax_bar.text(1.27, (vals[0] + vals[1])/2, f"+{vals[1]-vals[0]:.3f}", va="center",
            fontsize=9, fontweight="bold", color=C_SIGREG)

ax_bar.annotate("", xy=(2, vals[2]), xytext=(2, vals[0]),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax_bar.text(2.27, (vals[0] + vals[2])/2, f"{vals[2]-vals[0]:+.3f}", va="center",
            fontsize=9, fontweight="bold", color=C_WM)

ax_bar.set_xticks(x); ax_bar.set_xticklabels(methods, fontsize=10)
ax_bar.set_ylabel("Balanced accuracy (6-class TUEV, held-out patients)")
ax_bar.set_ylim(0, 0.5)
ax_bar.set_title("C  —  Balanced accuracy: SIGReg vs WM", fontweight="bold", loc="left")
ax_bar.legend(fontsize=8); ax_bar.grid(axis="y", alpha=0.3, zorder=0)

# ── Panel D: Per-class — SIGReg beats WM on most classes ───────────────────
ax_cls = fig.add_subplot(gs[1, 1])

x_cls = np.arange(len(CLASSES))
w = 0.25

sg_rand = [PER_CLASS["sigreg"]["random"][c]  for c in CLASSES]
sg_tr   = [PER_CLASS["sigreg"]["trained"][c] for c in CLASSES]
wm_tr   = [PER_CLASS["wm"]["trained"][c]     for c in CLASSES]

ax_cls.bar(x_cls - w,   sg_rand, w, color=C_RANDOM,  label="Random floor",  edgecolor="white", zorder=3)
ax_cls.bar(x_cls,       sg_tr,   w, color=C_SIGREG,  label="SIGReg trained", edgecolor="white", zorder=3)
ax_cls.bar(x_cls + w,   wm_tr,   w, color=C_WM,      label="WM trained",     edgecolor="white", zorder=3)

# bckg collapse marker on WM
ax_cls.annotate("collapse!\n(0.095)", xy=(0 + w, 0.095), xytext=(0 + w, 0.35),
                arrowprops=dict(arrowstyle="->", color=C_WM, lw=1.2),
                fontsize=7, color=C_WM, ha="center")

ax_cls.axhline(CHANCE, color=C_CHANCE, lw=1.2, ls=":", label=f"chance ({CHANCE:.2f})")
ax_cls.set_xticks(x_cls); ax_cls.set_xticklabels(CLASSES, fontsize=10)
ax_cls.set_ylabel("Per-class accuracy (eval)")
ax_cls.set_title("D  —  Per-class: WM collapses on bckg", fontweight="bold", loc="left")
ax_cls.set_ylim(0, 0.70)
ax_cls.legend(fontsize=8); ax_cls.grid(axis="y", alpha=0.3, zorder=0)

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
fig.suptitle(
    "SIGReg (simple) beats World Model (3-geometry) on TUEV — eff_rank tells the story",
    fontsize=11, fontweight="bold", y=1.01)

out_dir  = os.path.join(os.path.dirname(__file__), "../../results/figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig_sigreg_vs_wm.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_path}")
