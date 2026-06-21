"""Comparison figures: TUAB vs TUEV — random floor, SSL gap, loss curve, per-class."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

# TUAB (from existing results)
TUAB = {
    "random": 0.7895,
    "ssl":    0.8185,   # 3-seed mean
    "riemann": 0.761,
}

# TUEV (from job 75309)
TUEV = {
    "random": 0.2830,
    "ssl":    0.3533,
}

TUEV_PER_CLASS = {
    "random":  {"bckg": 0.384, "spsw": 0.000, "eyem": 0.227,
                "artf": 0.460, "gped": 0.412, "pled": 0.216},
    "trained": {"bckg": 0.374, "spsw": 0.042, "eyem": 0.267,
                "artf": 0.500, "gped": 0.492, "pled": 0.446},
}

TUEV_COUNTS = {
    "train": {"bckg": 2121, "spsw": 22,  "eyem": 238, "artf": 489, "gped": 880,  "pled": 463},
    "eval":  {"bckg": 800,  "spsw": 24,  "eyem": 75,  "artf": 124, "gped": 374,  "pled": 74},
}

# TUEV loss log
LOG_PATH = os.path.join(os.path.dirname(__file__),
                        "../../checkpoints/eeg_tuev_sigreg/train_log.json")
loss_history = []
if os.path.exists(LOG_PATH):
    with open(LOG_PATH) as f:
        loss_history = json.load(f)

CLASSES = ["bckg", "spsw", "eyem", "artf", "gped", "pled"]
CHANCE = 1 / 6  # 0.167

# --------------------------------------------------------------------------- #
# Colors
# --------------------------------------------------------------------------- #
C_RANDOM  = "#aaaaaa"
C_SSL     = "#2196F3"
C_TUAB    = "#4CAF50"
C_TUEV    = "#FF5722"
C_CHANCE  = "#E91E63"
C_RIEMANN = "#9C27B0"

# --------------------------------------------------------------------------- #
# Figure layout: 2×2
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# ── Panel A: Gap comparison (TUAB vs TUEV) ─────────────────────────────────
ax_gap = fig.add_subplot(gs[0, 0])

datasets  = ["TUAB\n(binary)", "TUEV\n(6-class)"]
randoms   = [TUAB["random"], TUEV["random"]]
ssls      = [TUAB["ssl"],    TUEV["ssl"]]
x         = np.arange(len(datasets))
w         = 0.32

b1 = ax_gap.bar(x - w/2, randoms, w, label="random encoder floor",
                color=C_RANDOM, edgecolor="white", linewidth=0.5, zorder=3)
b2 = ax_gap.bar(x + w/2, ssls, w, label="SSL trained (frozen probe)",
                color=[C_TUAB, C_TUEV], edgecolor="white", linewidth=0.5, zorder=3)

# gap arrows
for i, (r, s) in enumerate(zip(randoms, ssls)):
    xi = x[i] + w/2
    ax_gap.annotate("", xy=(xi, s), xytext=(xi, r),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax_gap.text(xi + 0.07, (r + s) / 2, f"+{s-r:.3f}", va="center",
                fontsize=9, fontweight="bold")

ax_gap.axhline(CHANCE, color=C_CHANCE, lw=1.2, ls=":", label=f"chance (1/6 = {CHANCE:.2f})")
ax_gap.axhline(TUAB["riemann"], color=C_RIEMANN, lw=1.2, ls="--",
               label=f"Riemann 0-param ({TUAB['riemann']:.3f})")

ax_gap.set_xticks(x); ax_gap.set_xticklabels(datasets, fontsize=11)
ax_gap.set_ylabel("Balanced accuracy")
ax_gap.set_title("A  —  SSL gap: TUAB vs TUEV", fontweight="bold", loc="left")
ax_gap.set_ylim(0, 1.0)
ax_gap.legend(fontsize=8, loc="upper right")
ax_gap.grid(axis="y", alpha=0.3, zorder=0)

# relative improvement labels inside bars
for i, (r, s) in enumerate(zip(randoms, ssls)):
    rel = (s - r) / r * 100
    ax_gap.text(x[i], 0.02, f"rel. +{rel:.0f}%", ha="center", fontsize=8,
                color="white", fontweight="bold", va="bottom")

# ── Panel B: TUEV loss curve ────────────────────────────────────────────────
ax_loss = fig.add_subplot(gs[0, 1])

if loss_history:
    epochs    = [r["epoch"]    for r in loss_history]
    losses    = [r["loss"]     for r in loss_history]
    eff_ranks = [r["eff_rank"] for r in loss_history]

    ax_loss.plot(epochs, losses, "o-", color=C_TUEV, lw=2, ms=3, label="SSL loss")
    ax_loss.set_ylabel("SSL loss", color=C_TUEV)
    ax_loss.tick_params(axis="y", labelcolor=C_TUEV)

    ax2 = ax_loss.twinx()
    ax2.plot(epochs, eff_ranks, "s--", color="#FF9800", lw=1.5, ms=3,
             label="eff_rank")
    ax2.set_ylabel("eff_rank", color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    drop_pct = (losses[0] - losses[-1]) / losses[0] * 100
    ax_loss.set_title(
        f"B  —  TUEV training ({len(epochs)} epochs)\n"
        f"loss {losses[0]:.3f}→{losses[-1]:.3f}  (−{drop_pct:.0f}%)   "
        f"eff_rank {eff_ranks[0]:.0f}→{eff_ranks[-1]:.0f}",
        fontweight="bold", loc="left", fontsize=9)
    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
else:
    ax_loss.text(0.5, 0.5, "loss log not found\n(run on Dalia first)",
                 ha="center", va="center", transform=ax_loss.transAxes)
    ax_loss.set_title("B  —  TUEV training", fontweight="bold", loc="left")

ax_loss.set_xlabel("Epoch")
ax_loss.grid(alpha=0.3)

# ── Panel C: TUEV per-class accuracy ───────────────────────────────────────
ax_cls = fig.add_subplot(gs[1, 0])

x_cls = np.arange(len(CLASSES))
rand_vals    = [TUEV_PER_CLASS["random"][c]  for c in CLASSES]
trained_vals = [TUEV_PER_CLASS["trained"][c] for c in CLASSES]

ax_cls.bar(x_cls - w/2, rand_vals,    w, color=C_RANDOM, label="random encoder",
           edgecolor="white", zorder=3)
ax_cls.bar(x_cls + w/2, trained_vals, w, color=C_TUEV,   label="SSL trained",
           edgecolor="white", zorder=3)

# delta above bars
for i, (r, t) in enumerate(zip(rand_vals, trained_vals)):
    delta = t - r
    col = "green" if delta > 0 else "red"
    ax_cls.text(i, max(r, t) + 0.01, f"{delta:+.2f}", ha="center",
                fontsize=8, color=col, fontweight="bold")

ax_cls.axhline(CHANCE, color=C_CHANCE, lw=1.2, ls=":", label=f"chance ({CHANCE:.2f})")
ax_cls.set_xticks(x_cls)
ax_cls.set_xticklabels(
    [f"{c}\n(tr:{TUEV_COUNTS['train'][c]} ev:{TUEV_COUNTS['eval'][c]})"
     for c in CLASSES], fontsize=8)
ax_cls.set_ylabel("Per-class accuracy")
ax_cls.set_title("C  —  TUEV per-class breakdown (eval)", fontweight="bold", loc="left")
ax_cls.set_ylim(0, 0.65)
ax_cls.legend(fontsize=8)
ax_cls.grid(axis="y", alpha=0.3, zorder=0)

# ── Panel D: Summary table ──────────────────────────────────────────────────
ax_tbl = fig.add_subplot(gs[1, 1])
ax_tbl.axis("off")

tuab_gap     = TUAB["ssl"] - TUAB["random"]
tuev_gap     = TUEV["ssl"] - TUEV["random"]
tuab_rel     = tuab_gap / TUAB["random"] * 100
tuev_rel     = tuev_gap / TUEV["random"] * 100

rows = [
    ["Metric",             "TUAB",              "TUEV"],
    ["Task",               "binary (2 cls)",     "6-class events"],
    ["Chance level",       "0.500",              "0.167"],
    ["Random floor",       f"{TUAB['random']:.3f}",  f"{TUEV['random']:.3f}"],
    ["SSL trained",        f"{TUAB['ssl']:.3f}",     f"{TUEV['ssl']:.3f}"],
    ["Gap (abs)",          f"+{tuab_gap:.3f}",        f"+{tuev_gap:.3f}"],
    ["Gap (rel)",          f"+{tuab_rel:.1f}%",       f"+{tuev_rel:.1f}%"],
    ["eff_rank (final)",   "~60",                "67"],
    ["Collapse?",          "no",                 "no"],
]

colors_row = []
for i, row in enumerate(rows):
    if i == 0:
        colors_row.append(["#37474F"] * 3)
    elif i in (5, 6):
        colors_row.append(["#f5f5f5", "#C8E6C9", "#FFE0B2"])
    else:
        colors_row.append(["#f5f5f5"] * 3)

tbl = ax_tbl.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center",
                   cellColours=colors_row[1:],
                   colColours=colors_row[0],
                   bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_text_props(color="white", fontweight="bold")
    if c in (1, 2) and r in (5, 6):  # gap rows
        cell.set_text_props(fontweight="bold")

ax_tbl.set_title("D  —  Summary", fontweight="bold", loc="left")

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
out_dir = os.path.join(os.path.dirname(__file__), "../../results/figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig_tuab_vs_tuev.png")
fig.suptitle("TUAB (binary) vs TUEV (6-class events) — SSL JEPA SIGReg×ambient",
             fontsize=12, fontweight="bold", y=1.01)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_path}")
