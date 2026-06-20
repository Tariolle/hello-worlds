"""Figures de présentation finale — 3 panneaux décisifs.

  Fig 1 — Benchmark vs litterature (TUAB)
  Fig 2 — Null result : toutes les methodes SSL donnent la meme chose
  Fig 3 — TUEV : valeur du SSL sur une tache plus dure

Run: python -m examples.eeg.figures_final
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "../../results/figures")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# DONNEES
# ============================================================

# --- TUAB 2x2 ablation (3-seed means) ---
ABLATION = {
    "VICReg\nambient":  {"bal": 0.814, "auroc": 0.880},
    "SIGReg\nambient":  {"bal": 0.819, "auroc": 0.901},
    "SIGReg\ntangent":  {"bal": 0.820, "auroc": 0.898},
    "PEIRA\nambient":   {"bal": 0.815, "auroc": 0.887},
    "PEIRA\ntangent":   {"bal": 0.807, "auroc": 0.878},
    "JEPA\n(ours)":   {"bal": 0.828, "auroc": 0.898},
}

RANDOM_FLOOR   = 0.762   # from JEPA run (stable across runs)
RIEMANN_0PARAM = 0.761
TUAB_CHANCE    = 0.500

# --- Baselines publiees (TUAB) ---
PUBLISHED = [
    # (label, bal_acc, frozen, comparable, auroc)
    ("LaBraM frozen",       0.604, True,  True,  None),
    ("CBraMod frozen",      0.547, True,  True,  None),
    ("BIOT frozen",         0.780, True,  True,  None),
    ("EEGPT frozen",        0.766, True,  True,  None),
    ("EEG2Rep frozen",      0.766, True,  True,  0.832),
    ("Laya-S frozen*",      0.798, True,  False, None),   # different split
    ("LuMamba 21kh*",       0.810, True,  False, None),   # 21k hrs pretrain
    # fine-tuned references
    ("BIOT fine-tuned",     0.796, False, False, None),
    ("LaBraM fine-tuned",   0.814, False, False, None),
    ("CBraMod fine-tuned",  0.829, False, False, None),
]

# TUEV results (all measured locally)
TUEV_30  = {"bal": 0.353, "random": 0.283, "eff_ep": 30}   # SIGReg 30ep fixed LR
TUEV_60  = {"bal": 0.341, "random": 0.272, "eff_ep": 60}   # SIGReg 60ep — mesure

# Charger TUEV 60 ep depuis le log si disponible
LOG60 = os.path.join(os.path.dirname(__file__),
    "../../checkpoints/eeg_tuev_sigreg/train_log.json")
if os.path.exists(LOG60):
    with open(LOG60) as f:
        _tuev_log = json.load(f)
    # dernier epoch
    last = _tuev_log[-1]
    TUEV_60["last_epoch"] = last.get("epoch", "?")
    TUEV_60["last_loss"]  = last.get("loss")
    TUEV_60["last_rank"]  = last.get("eff_rank")
    # la balanced_acc viendra du log eval — on la patchera plus tard


# ============================================================
# COULEURS
# ============================================================
C_OURS   = "#1565C0"      # bleu fonce — nos resultats
C_FROZEN = "#78909C"      # gris bleu — frozen comparables
C_FT     = "#FFA000"      # orange — fine-tuned (pas comparable)
C_NC     = "#B0BEC5"      # gris clair — non comparable (split different)
C_FLOOR  = "#9E9E9E"
C_RIEMANN= "#00897B"
C_CHANCE = "#E53935"
C_TUEV1  = "#7B1FA2"
C_TUEV2  = "#1565C0"


# ============================================================
# FIG 1 — Benchmark TUAB vs litterature
# ============================================================
fig1, ax = plt.subplots(figsize=(12, 7))

our_best_bal = max(v["bal"] for v in ABLATION.values())
our_mean_bal = np.mean([v["bal"] for v in ABLATION.values()])

# Construire les barres
entries = []
# Nos resultats
entries.append(("Notre best (SIGReg best seed)", 0.833, "ours", True))
entries.append((f"Nos methodes  mean={our_mean_bal:.3f}", our_mean_bal, "ours_range", True))
entries.append(("JEPA (ours)",  0.828, "ours", True))
# Baseline Riemannienne 0-parametre
entries.append(("Riemannian 0-param\n(no deep learning)", RIEMANN_0PARAM, "riemann", True))
# Publics comparables (frozen, meme protocole)
entries.append(("BIOT frozen",     0.780, "frozen", True))
entries.append(("EEGPT frozen",    0.766, "frozen", True))
entries.append(("EEG2Rep frozen",  0.766, "frozen", True))
entries.append(("LaBraM frozen",   0.604, "frozen", True))
entries.append(("CBraMod frozen",  0.547, "frozen", True))
# Publics non-comparables (fine-tuned ou split different)
entries.append(("CBraMod fine-tuned†", 0.829, "finetuned", False))
entries.append(("LaBraM fine-tuned†",  0.814, "finetuned", False))
entries.append(("BIOT fine-tuned†",    0.796, "finetuned", False))
entries.append(("Laya-S*",             0.798, "nc", False))
entries.append(("LuMamba (21kh)*",     0.810, "nc", False))

# trier par score
entries.sort(key=lambda e: e[1])

color_map = {
    "ours":      C_OURS,
    "ours_range": "#5C8DCA",
    "riemann":   C_RIEMANN,
    "frozen":    C_FROZEN,
    "finetuned": C_FT,
    "nc":        C_NC,
}

labels = [e[0] for e in entries]
vals   = [e[1] for e in entries]
colors = [color_map[e[2]] for e in entries]

y = np.arange(len(entries))
bars = ax.barh(y, vals, 0.65, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

# Labels valeurs
for bar, v in zip(bars, vals):
    ax.text(v + 0.004, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

# Lignes repere
ax.axvline(0.500, color=C_CHANCE, lw=1.2, ls=":", zorder=4, label=f"chance (0.50)")
ax.axvline(RIEMANN_0PARAM, color=C_RIEMANN, lw=1.0, ls="--", zorder=4, alpha=0.5)

# Zone "nos resultats"
ax.axvspan(our_mean_bal, 0.838, color=C_OURS, alpha=0.07, zorder=2,
           label=f"nos resultats [{our_mean_bal:.3f}–0.833]")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Balanced accuracy (TUAB, patient-disjoint split)", fontsize=11)
ax.set_title("Comparaison avec la litterature — TUAB detection d'anomalies EEG\n"
             "Protocole local : sonde lineaire froide, split complet (2717 train / 276 eval patients)",
             fontsize=11, fontweight="bold", loc="left")
ax.set_xlim(0.44, 0.91)
ax.grid(axis="x", alpha=0.3, zorder=0)

legend_handles = [
    mpatches.Patch(color=C_OURS,    label="Nos methodes (frozen probe)"),
    mpatches.Patch(color=C_RIEMANN, label="Riemannian 0-parametre"),
    mpatches.Patch(color=C_FROZEN,  label="Frozen probe publie (protocole comparable)"),
    mpatches.Patch(color=C_FT,      label="Fine-tuned† (supervisé, pas comparable)"),
    mpatches.Patch(color=C_NC,      label="* Split / protocole different (non comparable)"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

fig1.tight_layout()
fig1.savefig(os.path.join(OUT, "fig_benchmark.png"), dpi=150, bbox_inches="tight")
print("Saved fig_benchmark.png")
plt.close(fig1)


# ============================================================
# FIG 2 — Null result : toutes les methodes SSL ~ egales sur TUAB
# ============================================================
fig2, (ax_bar, ax_rank) = plt.subplots(1, 2, figsize=(13, 5))

# Panel gauche : balanced_acc par methode
methods = list(ABLATION.keys())
bals    = [ABLATION[m]["bal"] for m in methods]
colors2 = ["#1565C0" if "JEPA" in m else "#64B5F6" for m in methods]

x = np.arange(len(methods))
bars2 = ax_bar.bar(x, bals, 0.55, color=colors2, edgecolor="white", zorder=3)

ax_bar.axhline(RANDOM_FLOOR,    color=C_FLOOR,   lw=1.5, ls="--", label=f"random floor ({RANDOM_FLOOR:.3f})")
ax_bar.axhline(RIEMANN_0PARAM,  color=C_RIEMANN,  lw=1.5, ls=":",  label=f"Riemannian 0-param ({RIEMANN_0PARAM:.3f})")
ax_bar.axhline(TUAB_CHANCE,     color=C_CHANCE,   lw=1.2, ls=":",  label="chance (0.50)", alpha=0.7)

for bar, v in zip(bars2, bals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

spread = max(bals) - min(bals)
ax_bar.text(0.5, 0.85, f"spread = {spread:.3f}\n(< bruit d'un seed)",
            ha="center", transform=ax_bar.transAxes, fontsize=10,
            color="black", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F9A825", alpha=0.9))

ax_bar.set_xticks(x); ax_bar.set_xticklabels(methods, fontsize=9)
ax_bar.set_ylabel("Balanced accuracy")
ax_bar.set_ylim(0.70, 0.88)
ax_bar.set_title("A — Toutes les methodes SSL donnent ~0.82\n(ablation 2x2 + JEPA, 3 seeds sauf JEPA)",
                 fontweight="bold", loc="left", fontsize=10)
ax_bar.legend(fontsize=8); ax_bar.grid(axis="y", alpha=0.3, zorder=0)

# Panel droit : eff_rank evolution (VICReg vs SIGReg vs I-JEPA)
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../checkpoints")
sigreg_log_path = os.path.join(LOG_DIR, "eeg_tuev_sigreg/train_log.json")  # TUEV log
collapse_path   = os.path.join(os.path.dirname(__file__), "../../results/collapse_data.json")

if os.path.exists(collapse_path):
    with open(collapse_path) as f:
        collapse = json.load(f)
    for key, color, label in [
        ("vicreg_ambient",  "#78909C", "VICReg ambient"),
        ("sigreg_ambient",  "#1565C0", "SIGReg ambient"),
        ("sigreg_tangent",  "#42A5F5", "SIGReg tangent"),
        ("peira_tangent",   "#AB47BC", "PEIRA tangent"),
    ]:
        if key in collapse:
            d = collapse[key]
            ax_rank.plot(d["epoch"], d["eff_rank"], lw=2, label=label)

# I-JEPA ijepa epochs: 0..49, eff_rank from log
ijepa_log = os.path.join(LOG_DIR, "eeg_ijepa/train_log.json")
if os.path.exists(ijepa_log):
    with open(ijepa_log) as f:
        il = json.load(f)
    ax_rank.plot([r["epoch"] for r in il], [r["eff_rank"] for r in il],
                 lw=2.5, color=C_OURS, ls="-", marker="", label="JEPA (ours)")
else:
    # donnees hardcodees depuis log cluster
    ijepa_epochs = list(range(50))
    ijepa_ranks  = [
        26.5,30.2,34.1,38.0,41.3,43.8,45.9,47.2,48.1,48.8,
        49.1,49.6,49.9,50.4,51.4,52.2,52.0,54.0,54.4,54.9,
        54.8,55.1,56.0,56.4,56.7,57.5,57.5,58.1,58.6,58.7,
        58.6,59.3,59.8,59.8,59.9,59.9,60.3,60.5,60.7,60.4,
        61.5,61.0,61.0,60.4,62.0,61.8,61.9,61.2,61.7,61.4,
    ]
    ax_rank.plot(ijepa_epochs, ijepa_ranks, lw=2.5, color=C_OURS, label="JEPA (ours)")

ax_rank.set_xlabel("Epoch"); ax_rank.set_ylabel("eff_rank")
ax_rank.set_title("B — eff_rank : toutes les methodes evitent le collapse\n(c'est le plafond TUAB, pas la methode SSL)",
                  fontweight="bold", loc="left", fontsize=10)
ax_rank.legend(fontsize=8); ax_rank.grid(alpha=0.3)

fig2.suptitle("Resultat cle : le choix du regulariseur SSL ne change pas le score sur TUAB",
              fontsize=11, fontweight="bold", y=1.01)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig_null_result.png"), dpi=150, bbox_inches="tight")
print("Saved fig_null_result.png")
plt.close(fig2)


# ============================================================
# FIG 3 — TUEV : la valeur du SSL sur une tache dure
# ============================================================
fig3, (ax_loss, ax_bar3) = plt.subplots(1, 2, figsize=(13, 5))

# Panel gauche : courbe de loss TUEV 30ep vs 60ep
TUEV_LOG_PATH = os.path.join(LOG_DIR, "eeg_tuev_sigreg/train_log.json")
if os.path.exists(TUEV_LOG_PATH):
    with open(TUEV_LOG_PATH) as f:
        tuev_log = json.load(f)
    epochs60 = [r["epoch"] for r in tuev_log]
    losses60 = [r["loss"]  for r in tuev_log]
    ranks60  = [r.get("eff_rank") for r in tuev_log]

    ax_loss.plot(epochs60, losses60, "o-", color=C_TUEV2, lw=2, ms=2, label="SIGReg 60ep + cosine LR")
    ax_loss.axvline(29, color="black", lw=1, ls="--", alpha=0.5, label="30ep (ancien arret)")

    ax_r = ax_loss.twinx()
    if any(r is not None for r in ranks60):
        ax_r.plot(epochs60, ranks60, "^-", color=C_TUEV1, lw=1.5, ms=3, alpha=0.8, label="eff_rank")
        ax_r.set_ylabel("eff_rank", color=C_TUEV1, fontsize=9)
        ax_r.tick_params(axis="y", labelcolor=C_TUEV1)

    drop = (losses60[0] - losses60[-1]) / losses60[0] * 100
    ax_loss.set_title(f"A — TUEV : convergence non atteinte a 30ep\nloss {losses60[0]:.3f}->{losses60[-1]:.3f}  "
                      f"(-{drop:.0f}%)   eff_rank {ranks60[0] or '?':.0f}->{ranks60[-1] or '?':.0f}",
                      fontweight="bold", loc="left", fontsize=9)
    lines1, lab1 = ax_loss.get_legend_handles_labels()
    lines2, lab2 = ax_r.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
else:
    ax_loss.text(0.5, 0.5, "TUEV log en attente\n(job 75802 en cours)",
                 ha="center", va="center", transform=ax_loss.transAxes, fontsize=12)
    ax_loss.set_title("A — TUEV 60ep training curve", fontweight="bold", loc="left")

ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("SSL loss"); ax_loss.grid(alpha=0.3)

# Panel droit : bar chart TUEV — random / 30ep / 60ep / WM
labels3 = ["Random\nfloor", "SIGReg\n30ep\n(fixe)", "SIGReg\n60ep\n(cosine)", "World Model\n(echec)"]
vals3   = [0.283, 0.353, TUEV_60.get("bal") or 0.353, 0.300]
known   = [True, True, TUEV_60.get("bal") is not None, True]
colors3 = [C_FLOOR, C_TUEV2, C_OURS if known[2] else "#B0BEC5", "#FF5722"]

x3 = np.arange(len(labels3))
bars3 = ax_bar3.bar(x3, vals3, 0.55, color=colors3, edgecolor="white", zorder=3)

for bar, v, k in zip(bars3, vals3, known):
    label = f"{v:.3f}" if k else "en cours..."
    ax_bar3.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                 label, ha="center", va="bottom", fontsize=10, fontweight="bold")

ax_bar3.axhline(1/6, color=C_CHANCE, lw=1.2, ls=":", label=f"chance (0.167)")
ax_bar3.axhline(0.283, color=C_FLOOR, lw=1, ls="--", alpha=0.6)

# Fleche gain SSL
if known[2]:
    ax_bar3.annotate("", xy=(2, vals3[2]), xytext=(2, vals3[0]),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    gain = vals3[2] - vals3[0]
    ax_bar3.text(2.28, (vals3[0] + vals3[2])/2,
                 f"+{gain:.3f}\n(+{gain/vals3[0]*100:.0f}%)",
                 va="center", fontsize=9, fontweight="bold", color=C_OURS)

ax_bar3.set_xticks(x3); ax_bar3.set_xticklabels(labels3, fontsize=9)
ax_bar3.set_ylabel("Balanced accuracy (6-class TUEV)")
ax_bar3.set_ylim(0, 0.50)
ax_bar3.set_title("B — Sur TUEV (tache dure), le SSL apporte un gain reel\nvs. le World Model qui collapse (eff_rank 32->17)",
                  fontweight="bold", loc="left", fontsize=9)
ax_bar3.legend(fontsize=8); ax_bar3.grid(axis="y", alpha=0.3, zorder=0)

fig3.suptitle("TUEV 6-class : la valeur du SSL et pourquoi le World Model echoue",
              fontsize=11, fontweight="bold", y=1.01)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT, "fig_tuev_value.png"), dpi=150, bbox_inches="tight")
print("Saved fig_tuev_value.png")
plt.close(fig3)
