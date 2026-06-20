"""Frozen cross-TASK head-to-head on TUEV — apples-to-apples FROZEN linear-probe.

Foundation-model numbers: EEG-FM-Bench (Cui et al., arXiv 2508.17742, Table 1,
frozen strategy), TUEV 6-class balanced accuracy — quoted verbatim, not re-measured.
Ours = TUAB-pretrained SIGReg encoder, FROZEN, cross-task linear-probe on TUEV (our
window design). Both sides are frozen and NEITHER pretrained on TUEV -> a fair
cross-task frozen comparison. The honest baseline the random-encoder floor alone
could not provide.

⚠️ CAVEAT TO VERIFY before the jury: our encoder pretrained on TUAB-train; some
TUEV-eval patients may overlap TUAB-train patients (TUH shares patients across
corpora) -> potential leakage. Check the TUEV-eval ∩ TUAB-train patient overlap.

Run (local, no GPU):  python examples/eeg/frozen_headtohead_tuev.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (label, balanced_acc, std, kind)   FM rows: EEG-FM-Bench Table 1 (frozen, TUEV 6-class)
BARS = [
    ("BENDR  [FM-Bench]", 0.1667, 0.0000, "frozen_fm"),
    ("CBraMod  [FM-Bench]", 0.2106, 0.0022, "frozen_fm"),
    ("LaBraM  [FM-Bench]", 0.3148, 0.0527, "frozen_fm"),
    ("random-encoder floor (ours)", 0.3258, None, "floor"),
    ("Ours — TUAB→TUEV (frozen, cross-task)", 0.4252, None, "ours"),
    ("BIOT  [FM-Bench]", 0.4369, 0.0213, "frozen_fm"),
    ("EEGPT  [FM-Bench]", 0.4983, 0.0960, "frozen_fm"),
]
CHANCE = 1.0 / 6.0

bars = sorted(BARS, key=lambda b: b[1])
labels = [b[0] for b in bars]
vals = [b[1] for b in bars]
errs = [b[2] if b[2] else 0.0 for b in bars]
cmap = {"ours": "#2f80ed", "frozen_fm": "#9aa5b1", "floor": "#cfd6df"}
colors = [cmap[b[3]] for b in bars]

fig, ax = plt.subplots(figsize=(9.4, 4.6))
ax.axvline(CHANCE, color="gray", ls=":", lw=1.2, label=f"chance (1/6 = {CHANCE:.3f})")
ax.barh(labels, vals, xerr=errs, color=colors, capsize=3, error_kw=dict(alpha=0.45))
for i, v in enumerate(vals):
    ax.text(v + errs[i] + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0.15, 0.63)
ax.set_xlabel("Balanced accuracy — FROZEN linear probe, TUEV 6-class (held-out patients)")
ax.set_title("Frozen cross-task on TUEV — FM rows: EEG-FM-Bench (arXiv 2508.17742, Table 1)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()

out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "results", "benchmark", "frozen_headtohead_tuev.png"))
fig.savefig(out, dpi=160)
print(f"saved {out}")
