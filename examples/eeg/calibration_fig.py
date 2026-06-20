"""Calibration / selective-prediction figure — frozen-probe reliability on TUAB.

Two panels (both lower = better): Expected Calibration Error (ECE, raw) and
selective-prediction risk (AURC). Honest read: every SSL-pretrained encoder is
better-calibrated and lower-risk than a RANDOM encoder, but geometry/PEIRA does NOT
beat ambient SIGReg (the tangent even worsens AURC). The positive here is "SSL > random",
not "geometry wins". Single seed.

Run (local, no GPU):  python examples/eeg/calibration_fig.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (label, ECE_raw, AURC, is_random)
ROWS = [
    ("VICReg/amb", 0.0384, 0.0682, False),
    ("SIGReg/amb", 0.0434, 0.0674, False),
    ("SIGReg/tan", 0.0395, 0.0744, False),
    ("PEIRA/amb", 0.0570, 0.0694, False),
    ("PEIRA/tan", 0.0486, 0.0896, False),
    ("RANDOM", 0.0649, 0.1094, True),
]
labels = [r[0] for r in ROWS]
ece = [r[1] for r in ROWS]
aurc = [r[2] for r in ROWS]
colors = ["#c0392b" if r[3] else "#5b8def" for r in ROWS]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, vals, title in zip(axes, (ece, aurc),
                           ("(a) Expected Calibration Error  (lower = better)",
                            "(b) Selective-prediction risk AURC  (lower = better)")):
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Frozen-probe reliability on TUAB — SSL beats the random floor; geometry/PEIRA does not beat ambient (1 seed)")
fig.tight_layout()
out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "results", "calibration", "calibration.png"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
