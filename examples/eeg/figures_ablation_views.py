"""Ablation two-view vs single-view JEPA — figures de comparaison.

Run: python -m examples.eeg.figures_ablation_views
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "../../results/figures")
os.makedirs(OUT, exist_ok=True)

LOG_2V = os.path.join(os.path.dirname(__file__),
    "../../checkpoints/eeg_jepa_abl_2view_log.json")
LOG_1V = os.path.join(os.path.dirname(__file__),
    "../../checkpoints/eeg_jepa_abl_1view_log.json")

with open(LOG_2V) as f: log_2v = json.load(f)
with open(LOG_1V) as f: log_1v = json.load(f)

ep_2v       = [r["epoch"]     for r in log_2v]
loss_2v     = [r["loss"]      for r in log_2v]
rank_2v     = [r["eff_rank"]  for r in log_2v]
pred_2v     = [r["pred_loss"] for r in log_2v]

ep_1v       = [r["epoch"]     for r in log_1v]
loss_1v     = [r["loss"]      for r in log_1v]
rank_1v     = [r["eff_rank"]  for r in log_1v]
pred_1v     = [r["pred_loss"] for r in log_1v]

# Résultats eval (hardcodés depuis le log du job)
RESULTS = {
    "two_view":    {"bal": 0.8163, "auroc": 0.8998},
    "single_view": {"bal": 0.8222, "auroc": 0.8968},
    "random_2v":   {"bal": 0.7692, "auroc": 0.8561},
    "random_1v":   {"bal": 0.7584, "auroc": 0.8466},
}

C_2V   = "#1565C0"   # bleu foncé — two-view
C_1V   = "#E65100"   # orange     — single-view
C_RAND = "#9E9E9E"   # gris       — random floor

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ── Panel A : training loss ────────────────────────────────────────────────
ax = axes[0]
ax.plot(ep_2v, loss_2v, lw=2.2, color=C_2V,  label="Two-view JEPA")
ax.plot(ep_1v, loss_1v, lw=2.2, color=C_1V,  label="Single-view JEPA")
ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss")
ax.set_title("A — Courbe de loss SSL\n(échelles différentes : tâches non comparables)",
             fontweight="bold", loc="left", fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
note = ("Note: la loss single-view est\nnaturellement plus basse\n"
        "(prédire sa propre fenêtre\nvs une fenêtre augmentée différemment)")
ax.text(0.97, 0.97, note, transform=ax.transAxes, fontsize=7.5,
        ha="right", va="top", color="#555",
        bbox=dict(facecolor="white", edgecolor="#ccc", alpha=0.85, pad=3))

# ── Panel B : eff_rank ─────────────────────────────────────────────────────
ax = axes[1]
ax.plot(ep_2v, rank_2v, lw=2.2, color=C_2V,  label="Two-view JEPA")
ax.plot(ep_1v, rank_1v, lw=2.2, color=C_1V,  label="Single-view JEPA")
ax.set_xlabel("Epoch"); ax.set_ylabel("eff_rank")
ax.set_title("B — Rang effectif des représentations\n(eff_rank élevé = pas de collapse)",
             fontweight="bold", loc="left", fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.text(0.5, 0.08,
        f"Final: two-view={rank_2v[-1]:.1f}  single-view={rank_1v[-1]:.1f}",
        transform=ax.transAxes, ha="center", fontsize=9,
        color="#444", style="italic")

# ── Panel C : résultats eval downstream ────────────────────────────────────
ax = axes[2]

labels   = ["Random\nfloor\n(two-view)", "Two-view\nJEPA", "Random\nfloor\n(single-view)", "Single-view\nJEPA"]
vals_bal = [RESULTS["random_2v"]["bal"], RESULTS["two_view"]["bal"],
            RESULTS["random_1v"]["bal"], RESULTS["single_view"]["bal"]]
colors   = [C_RAND, C_2V, C_RAND, C_1V]

x = np.arange(len(labels))
bars = ax.bar(x, vals_bal, 0.6, color=colors, edgecolor="white", zorder=3)

for bar, v in zip(bars, vals_bal):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

delta = RESULTS["single_view"]["bal"] - RESULTS["two_view"]["bal"]
sign  = "+" if delta >= 0 else ""
ax.annotate("",
    xy=(3, RESULTS["single_view"]["bal"]),
    xytext=(1, RESULTS["two_view"]["bal"]),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(2.0, (RESULTS["two_view"]["bal"] + RESULTS["single_view"]["bal"])/2 + 0.002,
        f"δ = {sign}{delta:.4f}", ha="center", fontsize=10,
        fontweight="bold", color="black")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Balanced accuracy (TUAB eval, patient-disjoint)")
ax.set_ylim(0.72, 0.86)
ax.set_title("C — Score downstream : balanced accuracy\n(sonde linéaire froide, 2717/276 patients)",
             fontweight="bold", loc="left", fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)

handles = [
    mpatches.Patch(color=C_2V,  label=f"Two-view JEPA  (bal={RESULTS['two_view']['bal']:.4f}  auroc={RESULTS['two_view']['auroc']:.4f})"),
    mpatches.Patch(color=C_1V,  label=f"Single-view JEPA  (bal={RESULTS['single_view']['bal']:.4f}  auroc={RESULTS['single_view']['auroc']:.4f})"),
    mpatches.Patch(color=C_RAND, label="Random encoder floor"),
]
ax.legend(handles=handles, fontsize=8, loc="lower right")

fig.suptitle(
    "Ablation Two-view vs Single-view JEPA — TUAB 30 epochs, seed=42, avg pool",
    fontsize=12, fontweight="bold", y=1.01,
)
fig.tight_layout()
out_path = os.path.join(OUT, "fig_ablation_views.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.close(fig)
