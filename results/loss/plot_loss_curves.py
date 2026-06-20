"""Parse SSL training logs (raw_epochs.txt) and plot loss + anti-collapse health curves.

Each line: @@LABEL@@ [eeg] epoch N loss=X {metric dict}
Comparable-across-cells metrics: invariance_loss, eff_rank, feat_std, offdiag_cov.
Total `loss` is NOT comparable across objectives (PEIRA includes a log-det term -> can be
negative); read per-curve convergence shape, not absolute level.
"""
import ast
import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "raw_epochs.txt")
LINE = re.compile(r"@@(?P<lab>[^@]+)@@ \[eeg\] epoch (?P<ep>\d+) loss=(?P<loss>[-\d.]+) (?P<d>\{.*\})")

# label -> per-epoch records
runs = {}
for ln in open(RAW, encoding="utf-8"):
    m = LINE.search(ln)
    if not m:
        continue
    d = ast.literal_eval(m.group("d"))
    d["loss"] = float(m.group("loss"))
    d["ep"] = int(m.group("ep"))
    runs.setdefault(m.group("lab"), []).append(d)
for lab in runs:
    runs[lab].sort(key=lambda r: r["ep"])


def series(lab, key):
    rs = runs[lab]
    return [r["ep"] for r in rs], [r.get(key, np.nan) for r in rs]


# seed-1 cells: (label, display, color, lw)
CELLS = [
    ("VICReg-amb_c0_s1", "VICReg-amb (c0)",        "#7f8c8d", 1.8),
    ("SIGReg-amb_c1_s1", "SIGReg-amb (c1, ours)",  "#2f80ed", 3.0),
    ("SIGReg-tan_c2_s1", "SIGReg-tan (c2)",        "#27ae60", 1.8),
    ("PEIRA-amb_c3_s1",  "PEIRA-amb (c3)",         "#e67e22", 1.8),
    ("PEIRA-tan_c4_s1",  "PEIRA-tan (c4)",         "#c0392b", 1.8),
]
SEEDS = [  # c1 across seeds -> show seeds genuinely diverge
    ("SIGReg-amb_c1_s1",     "seed 1",     "#2f80ed"),
    ("SIGReg-amb_c1_s1000",  "seed 1000",  "#16a085"),
    ("SIGReg-amb_c1_s10000", "seed 10000", "#8e44ad"),
]

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

panels = [
    ("loss",        "Total SSL loss\n(objectives differ in scale/sign — read shape, not level)"),
    ("invariance_loss", "Invariance / prediction loss\n(shared JEPA term — comparable across cells)"),
    ("eff_rank",    "Effective rank  (anti-collapse: higher = more dims used)"),
    ("feat_std",    "Feature std  (collapse guard: ->0 = collapse, ~1 = healthy)"),
    ("offdiag_cov", "Off-diagonal covariance  (lower = more decorrelated)"),
]
for axi, (key, title) in zip(ax.flat[:5], panels):
    for lab, disp, col, lw in CELLS:
        x, y = series(lab, key)
        axi.plot(x, y, color=col, lw=lw, label=disp)
    axi.set_title(title, fontsize=10)
    axi.set_xlabel("epoch"); axi.grid(alpha=0.25)
ax.flat[0].legend(fontsize=8, loc="upper right")

# panel 6: seed divergence for c1
axs = ax.flat[5]
for lab, disp, col in SEEDS:
    x, y = series(lab, "loss")
    axs.plot(x, y, color=col, lw=2.0, label=disp)
axs.set_title("SIGReg-amb (c1) across seeds\n(init/data differ by seed; NB slice dirs are shared "
              "-> confirmed slice-seed bug)", fontsize=9)
axs.set_xlabel("epoch"); axs.grid(alpha=0.25); axs.legend(fontsize=8)

fig.suptitle("EEG-JEPA SSL training curves — 5 regularizer cells, 30 epochs, TUAB pretrain (seed 1)",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(HERE, "loss_curves.png")
fig.savefig(out, dpi=140)
print("saved", out)

# also dump a compact convergence summary
print("\n%-24s %8s %8s %10s %9s" % ("cell", "loss0", "loss29", "effR 0->29", "fstd29"))
for lab, disp, _, _ in CELLS:
    rs = runs[lab]
    print("%-24s %8.3f %8.3f  %5.1f->%5.1f %9.3f" % (
        disp, rs[0]["loss"], rs[-1]["loss"], rs[0]["eff_rank"], rs[-1]["eff_rank"], rs[-1]["feat_std"]))
