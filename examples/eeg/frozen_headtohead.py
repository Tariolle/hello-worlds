"""Frozen head-to-head on TUAB — apples-to-apples FROZEN linear-probe (jury figure).

All bars are FROZEN linear-probe balanced accuracy (encoder frozen, only a linear
head is fit). Foundation-model numbers are quoted VERBATIM from EEG-FM-Bench
(Cui et al., arXiv 2508.17742, Table 1, frozen strategy) — we cite, we did not
re-measure. Our SIGReg bar is IN-DOMAIN (pretrained on TUAB-train); the
apples-to-apples general-pretrain bar (TUSZ -> frozen TUAB) is added once it lands.
The fine-tuned FM band is a different, easier setting — shaded for context only.

Run (local, no GPU):  python examples/eeg/frozen_headtohead.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (label, balanced_acc, std, kind)   FM rows: EEG-FM-Bench Table 1 (frozen, TUAB)
BARS = [
    ("CBraMod  [FM-Bench]", 0.5473, 0.0124, "frozen_fm"),
    ("LaBraM  [FM-Bench]", 0.6040, 0.0464, "frozen_fm"),
    ("BENDR  [FM-Bench]", 0.6659, 0.0246, "frozen_fm"),
    ("Riemannian 0-param (ours)", 0.7610, None, "floor"),
    ("EEGPT  [FM-Bench]", 0.7664, 0.0104, "frozen_fm"),
    ("BIOT  [FM-Bench]", 0.7798, 0.0075, "frozen_fm"),
    ("random-encoder floor (ours)", 0.7900, None, "floor"),
    ("Ours — SIGReg in-domain", 0.8190, 0.0120, "ours"),
]
FT_BAND = (0.814, 0.829)  # fine-tuned FMs — easier setting, NOT a frozen probe

bars = sorted(BARS, key=lambda b: b[1])
labels = [b[0] for b in bars]
vals = [b[1] for b in bars]
errs = [b[2] if b[2] else 0.0 for b in bars]
cmap = {"ours": "#2f80ed", "frozen_fm": "#9aa5b1", "floor": "#cfd6df"}
colors = [cmap[b[3]] for b in bars]

fig, ax = plt.subplots(figsize=(9.2, 4.9))
ax.axvspan(*FT_BAND, color="green", alpha=0.10,
           label="fine-tuned FMs (easier setting — NOT frozen)")
ax.barh(labels, vals, xerr=errs, color=colors, capsize=3, error_kw=dict(alpha=0.45))
for i, v in enumerate(vals):
    ax.text(v + errs[i] + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0.5, 0.88)
ax.set_xlabel("Balanced accuracy — FROZEN linear probe, TUAB (held-out patients)")
ax.set_title("Frozen head-to-head on TUAB — FM rows: EEG-FM-Bench (arXiv 2508.17742, Table 1)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()

out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "results", "benchmark", "frozen_headtohead.png"))
fig.savefig(out, dpi=160)
print(f"saved {out}")
