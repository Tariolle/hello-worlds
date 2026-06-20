"""Frozen head-to-head on TUAB — the apples-to-apples comparison EEG-FM-Bench proposes.

EVERY bar is FROZEN linear-probe balanced accuracy (encoder frozen, only a linear
head is fit). This is the regime EEG-FM-Bench audits, where foundation models
*collapse* — and where our in-domain JEPA holds ~0.82. The fine-tuned FM band is a
DIFFERENT, easier setting (weights updated) and is shown shaded for context only;
it is NOT an apples-to-apples comparison to a frozen probe (this is the exact
mislabelling we are fixing — the old value_of_ssl figure only showed that band).

NUMBERS — verify against the cited sources before the jury:
  frozen FMs : todo.md "frozen linear probe (consistent protocol)" / EEG-FM-Bench
               frozen-backbone TUAB collapse (LaBraM 0.604, CBraMod 0.547,
               EEGPT 0.766, BIOT 0.780).
  ours/random/riemann : measured locally (ours = 3-seed mean, SIGReg-ambient).
  fine-tuned band     : LaBraM-Base .. CBraMod fine-tuned (benchmark.yaml).

Run (local, no GPU):  python examples/eeg/frozen_headtohead.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (label, balanced_acc, kind)   kind: ours | frozen_fm | floor
BARS = [
    ("CBraMod (frozen)", 0.547, "frozen_fm"),
    ("LaBraM (frozen)", 0.604, "frozen_fm"),
    ("Riemannian 0-param", 0.761, "floor"),
    ("EEGPT (frozen)", 0.766, "frozen_fm"),
    ("BIOT (frozen)", 0.780, "frozen_fm"),
    ("random-encoder floor", 0.790, "floor"),
    ("Ours — SIGReg (frozen)", 0.819, "ours"),
]
FT_BAND = (0.814, 0.829)  # fine-tuned FMs — easier setting, NOT a frozen probe

bars = sorted(BARS, key=lambda b: b[1])
labels = [b[0] for b in bars]
vals = [b[1] for b in bars]
cmap = {"ours": "#2f80ed", "frozen_fm": "#9aa5b1", "floor": "#cfd6df"}
colors = [cmap[b[2]] for b in bars]

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.axvspan(*FT_BAND, color="green", alpha=0.10,
           label="fine-tuned FMs (easier setting — NOT frozen, not apples-to-apples)")
ax.barh(labels, vals, color=colors)
for i, v in enumerate(vals):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0.5, 0.88)
ax.set_xlabel("Balanced accuracy — FROZEN linear probe (TUAB, held-out patients)")
ax.set_title("Frozen head-to-head: FMs collapse frozen (EEG-FM-Bench regime); our JEPA holds ~0.82")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()

out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "results", "benchmark", "frozen_headtohead.png"))
fig.savefig(out, dpi=160)
print(f"saved {out}")
