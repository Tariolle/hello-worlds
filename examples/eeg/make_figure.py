"""Assemble the "value of self-supervision" figure.

Panel (a) label efficiency: frozen JEPA probe vs random-encoder probe, BalAcc vs %labels
  (reads results/.../label_eff.json, written by label_efficiency.py).
Panel (b) pretraining-data efficiency (OPT-IN, --with-pretrain): BalAcc (full labels)
  vs % of TRAIN recordings used for SSL. Single-seed and noise-dominated below 50%
  (frac-to-frac sigma ~0.012 > the trend), so it is DROPPED by default; the data
  lives in pretrain_data.json for whoever wants to re-run it multi-seed.

  python -u -m examples.eeg.make_figure --out results/label_eff   # clean single panel
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FT_BAND = (0.814, 0.829)

a = sys.argv
out = a[a.index("--out") + 1] if "--out" in a else "results/label_eff"
# Panel (b) pretraining-data efficiency is single-seed and noise-dominated below
# 50% (frac-to-frac sigma ~0.012 > the trend) -> DROPPED by default. The data is
# kept in pretrain_data.json; pass --with-pretrain to render the 2-panel anyway.
with_pretrain = "--with-pretrain" in a
le = json.load(open(f"{out}/label_eff.json"))
pdj = f"{out}/pretrain_data.json"
pd = (json.load(open(pdj)) if os.path.exists(pdj) else None) if with_pretrain else None

n = 2 if pd else 1
fig, axes = plt.subplots(1, n, figsize=(11 if pd else 6.4, 4.3), squeeze=False)

ax = axes[0][0]
xs = [r["frac"] * 100 for r in le["jepa"]]
ax.axhspan(*FT_BAND, color="green", alpha=0.10, label="fine-tuned FMs (cross-corpus, easier — NOT frozen)")
ax.errorbar(xs, [r["mean"] for r in le["jepa"]], yerr=[r["std"] for r in le["jepa"]],
            marker="o", lw=2, capsize=3, color="C0", label="frozen JEPA probe")
ax.errorbar(xs, [r["mean"] for r in le["random"]], yerr=[r["std"] for r in le["random"]],
            marker="s", lw=2, ls="--", capsize=3, color="C3", label="random-encoder probe")
ax.set_xscale("log"); ax.set_xlabel("% of TRAIN labels (probe fit)")
ax.set_ylabel("Balanced accuracy (held-out patients)")
ax.set_title("(a) Label efficiency" if pd else "Label efficiency")
ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)

if pd:
    ax2 = axes[0][1]
    rows = sorted(pd["rows"], key=lambda r: r["frac"])
    xs2 = [r["frac"] * 100 for r in rows]
    ys2 = [r["balacc"] for r in rows]
    es2 = [r.get("std", 0.0) for r in rows]
    ax2.axhspan(*FT_BAND, color="green", alpha=0.10)
    if any(e > 0 for e in es2):  # multi-seed -> show CIs (error bars)
        ax2.errorbar(xs2, ys2, yerr=es2, marker="o", lw=2, capsize=3, color="C0",
                     label="frozen JEPA probe")
    else:
        ax2.plot(xs2, ys2, marker="o", lw=2, color="C0", label="frozen JEPA probe")
    ax2.set_xscale("log"); ax2.set_xlabel("% of TRAIN recordings (SSL pretrain)")
    ax2.set_ylabel("Balanced accuracy (held-out, full labels)")
    n_seeds = rows[0].get("seeds")
    ax2.set_title("(b) Pretraining-data efficiency" + (f" ({n_seeds} seeds)" if n_seeds else ""))
    ax2.legend(fontsize=7, loc="lower right"); ax2.grid(alpha=0.3)

fig.suptitle("Value of self-supervision — frozen in-domain EEG-JEPA on TUAB")
fig.tight_layout()
fig.savefig(f"{out}/value_of_ssl.png", dpi=140)
print(f"saved {out}/value_of_ssl.png")
