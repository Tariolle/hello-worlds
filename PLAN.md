# PLAN — EEG-JEPA · team Hello Worlds

24h hackathon "Hack the World(s)", EEG track. This is the source of truth for
**where we go, how, and who does what.** Live task checkboxes live in
`tasks/todo.md`.

## 0. One-liner
Pretrain a two-view JEPA on unlabeled EEG, freeze the encoder, linear-probe
**normal vs abnormal** on TUAB. Contribution = a controlled **2×2 factorial**:
`{SIGReg, PEIRA} × {ambient Euclidean, SPD-tangent}`. **Hypothesis:** PEIRA
(distribution-free) beats SIGReg (isotropic-Gaussian target) *specifically in the
tangent space*, because the log-mapped distribution of EEG channel covariances is
non-Gaussian — SIGReg fights it, PEIRA does not.

## 1. The experiment

|              | **ambient** (Euclidean pooled rep) | **tangent** (SPD log-Euclidean) |
|--------------|------------------------------------|---------------------------------|
| **VICReg**   | C0 — reference (eb_jepa default)   | — (skip)                        |
| **SIGReg**   | C1 — Laya-like baseline / gate     | C2 — geometry-aware SIGReg      |
| **PEIRA**    | C3 — distribution-free, ambient    | **C4 — the hypothesis cell**    |

Core 2×2 = {C1, C2, C3, C4}. References: C0 + classical **Riemannian** (0-param,
CPU, ~0.86 acc) + **random-encoder floor**.

**Money readouts** (all on frozen linear-probe balanced accuracy):
- Geometry main effect: tangent − ambient.
- Regulariser main effect: PEIRA − SIGReg.
- **Interaction (headline):** `(C4−C2) − (C3−C1) > 0` → PEIRA's edge over SIGReg is
  larger on the manifold. Positive confirms the hypothesis; null/negative is an
  honest negative result (still scores).

**No cell is discarded** — the losing cells are the control that gives the winner
meaning. We stop *tuning* a dominated cell, but its number stays in the table.

## 2. Metric & figures
Primary: **frozen linear-probe balanced accuracy + AUROC, full 2717/276 split,
recording level** (held-out patients). Figures:
1. 2×2 bar chart with 3-seed error bars + Riemannian/random reference lines.
2. Collapse dynamics per cell (effective rank, per-dim std vs epoch).
3. Label-fraction efficiency curve (1/5/10/25/100% labels) — novel on TUAB.
4. Robustness: probe accuracy under injected noise / channel dropout.

## 3. Timeline (elapsed hours; hard ends: 17:30 code · 18:00 slides · 19:00 jury)

| Phase | Hours | What | Owner(s) | Gate |
|---|---|---|---|---|
| Setup | 0–1 | SSH+env on Dalia, confirm SLURM partition/account + TUAB path | Florent · Hippolyte (logistics) | env imports, GPU visible |
| Pipeline | 1–4 | `baseline_riemann` (data sanity + yardstick); smoke `main.py` 1 epoch; probe harness vs random encoder; wandb | Clément (riemann) · Florent (smoke) · Yoann (probe+wandb) | baseline ≈0.86; loss↓, eff_rank>1 |
| **Gate** | 4–8 | **C1 (SIGReg×ambient) full run → first probe number** | Florent · Clément (collapse watch) | **≥ random floor, no collapse → qualification insurance** |
| Factorial | 8–16 | fan out C0–C4 across 3 GPUs, 1 seed → 2×2 picture; then 3-seed the key comparison; one change at a time | Florent + Clément · Yoann (live figures) | full 2×2 table populated |
| Lock | 16–21 | freeze best cells; robustness + label-fraction curves; error bars; final figures | all | figures final |
| Write | 21–24 | 10-min deck + report + rehearsal | Hippolyte (lead) · all | code 17:30, slides 18:00 |

Fan-out rule: launch C1 first; **the moment it trains without collapsing (~30 min),
launch the rest** on the free GPUs — don't wait for C1 to finish. 3 GPUs = 3
concurrent (the fair-share ceiling). Tangent row (C2,C4): watch for `eigh` NaNs,
hypersphere-tangent fallback ready.

## 4. Who does what
- **Florent** (lead, JEPA/geometry): encoder/ssl/geometry code, runs + monitors all
  experiments on Dalia, owns the 2×2 sweep + checkpoints.
- **Clément** (maths): PEIRA correctness, the *"why collapse happens / why our reg
  avoids it"* explanation (a direct jury criterion), tangent-space formalism, the
  Riemannian baseline, collapse-metric interpretation, ablation design + stats.
- **Yoann** (product): probe/eval harness polish, wandb dashboards, ALL figures
  (the 4 above), the demo. Builds the probe against a random encoder first so it's
  ready before pretraining finishes.
- **Hippolyte** (commercial): from H2 — 10-min deck skeleton + storytelling, the
  literature/baseline slides (LaBraM table, our parents Laya/EEG-ReMinD), PM +
  timekeeper (enforce 17:30/18:00), Dalia/organizer logistics liaison.

Pairing: Florent+Clément on the core; Yoann+Hippolyte on eval-figures-story.

## 5. Risks & mitigations
1. **Collapse** (#1) — monitor eff_rank/std every epoch; kill flat runs; fix = more
   regularisation, not less. (Also a *jury asset*: "we visualised collapse".)
2. **Data/compute friction** — resolve in H0–1; don't debug SLURM alone (organizer channel).
3. **Tangent `eigh` numerics** on real EEG — eps clamp + small d_cov; hypersphere fallback.
4. **PEIRA needs r_max ≥ 2** (≥2 shared predictable view-modes) or collapse-instability
   weakens — watch `tr_P` rising; if not, strengthen the two augmented views.

## 6. The 10-minute deck (Data → Architecture → Training → Inference/Eval → Insight)
1. **Problem/Data** — TUAB, normal/abnormal, frozen-probe protocol; the saturation +
   label-noise framing (why we don't chase accuracy). [Hippolyte]
2. **Architecture** — 1D conv encoder; the SPD-tangent representation; where SIGReg vs
   PEIRA acts. [Florent]
3. **Training** — two-view JEPA; collapse monitoring; the 2×2 sweep on 3 B200. [Florent]
4. **Eval** — frozen probe BalAcc/AUROC vs Riemannian 0-param + random floor; label/data
   efficiency. [Yoann]
5. **Insight** — the 2×2 table + interaction; honest limits; future work
   (manifold-correct SIGReg target, intrinsic Riemannian CCA). [Clément]

## 7. Honesty rules (non-negotiable)
- Report **balanced accuracy on the full split** + the probe head. Never plain
  accuracy on a reduced subset (the EEG-VJEPA "83%" is not comparable).
- We are **not** a foundation model, **not** SOTA in 24h. Cite parents: **Laya**
  (SIGReg-EEG), **EEG-ReMinD** (Riemannian-SSL-EEG, reconstruction). Our claim is the
  **intersection**: geometry-aware, latent-predictive, distribution-free anti-collapse
  on the SPD tangent — framed as a *controlled study*, not a "first".
