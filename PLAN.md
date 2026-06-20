# PLAN — EEG-JEPA · team Hello Worlds

24h hackathon "Hack the World(s)", EEG track. Source of truth for **where we go,
how, and who does what.** Live task checkboxes → `tasks/todo.md`; honest jury
numbers → `docs/positioning.md`.

## 0. Status & direction — UPDATED (generalist reframe, honest)
Original headline hypothesis (**PEIRA + tangent beats SIGReg** on TUAB frozen
probe) is **FALSIFIED**: the 3-seed 2×2 is a **clean null** (~0.82 every cell,
ranges overlap). We pivot the **framing**, not the codebase.

- **Contribution axis = generality under a *frozen* encoder + label/data
  efficiency** — not "geometry wins".
- **We claim:** a frozen in-domain JEPA whose linear-probe (**0.819 BA / ~0.90
  AUROC, 3 seeds**) is competitive with the TUAB *fine-tuning* literature and
  **above typical frozen FM baselines** — which **collapse when frozen** (LaBraM
  0.604 / CBraMod 0.547 in our consistent linear-probe) while ours holds ~0.82.
- **We do NOT claim** a world model, a foundation model, or SOTA. Single-dataset
  specialization ≠ generality. Framed as a **controlled frozen-transfer study**;
  we say **"match while frozen"**, never "beat" (the FMs are fine-tuned).
  `docs/positioning.md` rules are non-negotiable.

**Regulariser decision — the 2×2 is DONE, so:**
> **LOCK SIGReg** (ambient = primary, 0.819; tangent = tied, tightest variance).
> The 2×2 **stays as the published ablation / honest null** (PEIRA & tangent don't
> beat it; mechanism = wrapped-Gaussian mis-specification of the naive tangent,
> de Surrel 2025). We **stop *tuning* dominated cells and do NOT re-run all 4
> across any new eval — carry only SIGReg downstream.**

## 1. One-liner (reframed)
Pretrain a two-view JEPA on unlabeled EEG → **freeze** → linear-probe. Ship three
honest things: (1) a strong **frozen** in-domain probe above typical frozen-FM
numbers; (2) a clean **3-seed null** on geometry/PEIRA with a geometric mechanism;
(3) **label/data-efficiency** curves quantifying the SSL increment over a
**random-encoder floor (~0.79)**.

## 2. Priority ladder — strict necessary → bonus
### 🔴 P0 — strict necessary (claim invalid without it)
- [x] Frozen in-domain probe on TUAB — **full 2717/276 split, recording-level,
  patient-disjoint** (BA/AUROC). → **0.819 ± ~0.012**.
- [x] **Random-encoder floor disclosed (~0.79)** — the honest denominator.
- [x] **Riemannian 0-param yardstick** (0.761 here; ~0.86 whole-recording).
- [ ] **LEAKAGE GATE — patient-disjoint TRAIN dev-split in `eval.py`** before ANY
  hyperparameter/probe selection. *(Florent; blocks P1)*
- [ ] 3-seed **means + 95% CIs** wired into `benchmark.py`.

### 🟠 P1 — competitive core (the in-domain FIGURES = the real deliverable)
> The 0.819 number is **done**; these figures are **not** — and they, not the bare
> number, carry the jury. **Do all of P1 BEFORE any cross-dataset attempt.**
- [ ] **FM checkpoints under OUR identical frozen probe** = the honest head-to-head
  (they collapse frozen, we hold). Finalize the LaBraM 0.604 / CBraMod 0.547 row,
  note the preprocessing-differs caveat.
- [ ] **Label-efficiency curve** (1/2/5/10/25/50/100% labels, ≥3 seeds) + random
  floor — the money plot ("JEPA @25% labels ≈ random @100%").
- [ ] 2×2 **null table** with error bars + the wrapped-Gaussian mechanism slide.

### 🟢 P2 — bonus (only AFTER P1 in-domain figures are locked; time-boxed)
- [ ] **Cross-dataset frozen transfer — TUEV first.** Take the FROZEN SIGReg encoder,
  linear-probe a second dataset, NO re-pretrain. **"No re-pretrain" ≠ cheap:** the cost
  is the new loader + channel/montage alignment + new head/metric — the most time-risky
  item left. **TUEV first** (same TUH montage → encoder-compatible; but **cross-TASK,
  same-site** → label it honestly, *not* "cross-site"). Non-TUH (Sleep-EDF / BCI /
  CHB-MIT) = the real cross-distribution claim but max friction → only if TUEV is trivial.
  **Time-box 45–60 min; if the number isn't sane → "future work".**
- [ ] Pretrain-data-efficiency curve (BA vs % pretrain data).
- [ ] Collapse-dynamics figure (eff_rank / per-dim std vs epoch) per cell.
- [ ] Robustness: probe under channel dropout / injected noise.

> Selection rule: **select on the train dev-split only**; the 276 eval is touched
> **once**, config locked — else "we hold while FMs collapse" is inflated by
> selection leakage.

## 3. The 2×2 (DONE → it's the ablation now, not the headline)

|              | **ambient** (Euclidean pooled rep)        | **tangent** (SPD log-Euclidean) |
|--------------|-------------------------------------------|---------------------------------|
| **VICReg**   | C0 — reference (eb_jepa default) · 0.814   | — (skip)                        |
| **SIGReg**   | **C1 — LOCKED primary · 0.819**           | C2 — tied, tightest var · 0.820 |
| **PEIRA**    | C3 — 0.815                                 | C4 — ex-hypothesis cell · 0.807 |

3-seed means {1, 1000, 10000}. **Clean null:** no effect of regulariser or space,
no interaction (`(C4−C2)−(C3−C1)` ≈ 0). Role: the control that gives the locked
cell meaning + the honest negative. **No cell discarded** — dominated cells just
stop being tuned, their number stays in the table.

## 4. Metric & figures
Primary: **frozen linear-probe balanced accuracy + AUROC, full 2717/276 split,
recording level** (held-out patients). Figures (headline first):
1. **Value-of-SSL 2-panel** — BA vs %labels | vs %pretrain-data; overlays: random
   floor, Riemann 0.761, FT-foundation-model band (labelled "fine-tuned, cross-corpus").
2. **Frozen head-to-head** — ours ~0.82 vs FM *frozen* (LaBraM/CBraMod collapse) +
   EEG2Rep / BIOT / EEGPT.
3. 2×2 null bar chart, 3-seed error bars + Riemannian/random reference lines.
4. Collapse dynamics per cell (effective rank, per-dim std vs epoch).
   *(+bonus)* cross-dataset frozen-probe bar if P2 lands.

## 5. Timeline (elapsed hours; hard ends: 17:30 code · 18:00 slides · 19:00 jury)

| Phase | Hours | What | Owner(s) | Status |
|---|---|---|---|---|
| Setup | 0–1 | env on Dalia, SLURM partition/account, TUAB path | Florent · Hippolyte | ✅ |
| Pipeline | 1–4 | riemann yardstick + smoke `main.py` + probe harness + wandb | Clément · Florent · Yoann | ✅ |
| Gate | 4–8 | C1 SIGReg×ambient → first probe number | Florent · Clément | ✅ 0.833 seed1, no collapse |
| Factorial | 8–16 | C0–C4 × 3 seeds → 2×2 picture | Florent · Clément · Yoann | ✅ **clean null** |
| **← WE ARE HERE** | 8–16 | **lock SIGReg · leakage gate · label-eff curve · FM-frozen row · (bonus) 1 extra-dataset probe** | Florent (gate+curve) · Clément (CIs+mechanism) · Yoann (figures) | in progress |
| Lock | 16–21 | freeze figures, error bars, robustness | all | — |
| Write | 21–24 | 10-min deck + report + rehearsal | Hippolyte (lead) · all | — |

Fan-out rule still holds for any re-runs: launch SIGReg first; 3 GPUs = 3 concurrent
(fair-share ceiling). Tangent cells (C2): watch `eigh` NaNs, hypersphere-tangent
fallback ready.

## 6. Who does what
- **Florent** (lead, JEPA/geometry): encoder/ssl/geometry code, runs + monitors all
  experiments on Dalia, owns the locked SIGReg checkpoint + the leakage gate +
  label-efficiency curve.
- **Clément** (maths): PEIRA correctness, the *"why collapse happens / why our reg
  avoids it"* explanation (a direct jury criterion), tangent-space formalism, the
  Riemannian baseline, collapse-metric interpretation, the 3-seed means + 95% CIs,
  the negative-result + wrapped-Gaussian future-work slides.
- **Yoann** (product): probe/eval harness polish, wandb dashboards, ALL figures
  (the 4 above), the demo. Builds the probe against a random encoder first so it's
  ready before pretraining finishes.
- **Hippolyte** (commercial): from H2 — 10-min deck skeleton + storytelling, the
  literature/baseline slides (LaBraM table, our parents Laya / EEG-ReMinD), the
  honest positioning + DO-NOT-claim list, PM + timekeeper (enforce 17:30/18:00),
  Dalia/organizer logistics liaison.

Pairing: Florent+Clément on the core; Yoann+Hippolyte on eval-figures-story.

## 7. Risks & mitigations
1. **Collapse** (#1) — monitor eff_rank/std every epoch; kill flat runs; fix = more
   regularisation, not less. (Also a *jury asset*: "we visualised collapse".)
2. **Data/compute friction** — resolve in H0–1; don't debug SLURM alone (organizer channel).
3. **Tangent `eigh` numerics** on real EEG — eps clamp + small d_cov; hypersphere fallback.
4. **PEIRA needs r_max ≥ 2** (≥2 shared predictable view-modes) or collapse-instability
   weakens — watch `tr_P` rising; if not, strengthen the two augmented views.
5. **Over-claim / scope creep** — the generalist reframe must NOT drift into "world
   model / SOTA / we beat the FMs". Stay on **controlled frozen-transfer study,
   match-while-frozen**. Do NOT start a multi-dataset re-pretrain in the last hours;
   the only generality add is a **frozen probe on one extra dataset** (P2, no re-pretrain).

## 8. The 10-minute deck (Data → Architecture → Training → Inference/Eval → Insight)
1. **Problem/Data** — TUAB, normal/abnormal, frozen-probe protocol; the saturation +
   label-noise framing (why we don't chase accuracy). [Hippolyte]
2. **Architecture** — 1D conv encoder; the SPD-tangent representation; where SIGReg vs
   PEIRA acts. [Florent]
3. **Training** — two-view JEPA; collapse monitoring; the 2×2 sweep on 3 B200. [Florent]
4. **Eval** — frozen probe BalAcc/AUROC vs Riemannian 0-param + random floor; label/data
   efficiency; FMs collapse frozen while we hold. [Yoann]
5. **Insight** — generality-under-freeze + the 2×2 null (honest negative) with the
   geometric mechanism; honest limits; future work (manifold-correct SIGReg target,
   intrinsic Riemannian CCA, cross-dataset transfer). [Clément]

## 9. Honesty rules (non-negotiable)
- Report **balanced accuracy on the full split** + the probe head. Never plain
  accuracy on a reduced subset (the EEG-VJEPA "83%" is not comparable).
- We are **not** a world model, **not** a foundation model, **not** SOTA in 24h.
  Cite parents: **Laya** (SIGReg-EEG), **EEG-ReMinD** (Riemannian-SSL-EEG,
  reconstruction). Our claim is the **intersection**: geometry-aware,
  latent-predictive, distribution-free anti-collapse on the SPD tangent — framed as
  a *controlled study*, not a "first".
- **Generality is shown, not asserted:** the evidence is frozen-transfer +
  label-efficiency + "FMs collapse frozen while we hold" — say **"match while
  frozen"**, never "beat" (they are fine-tuned). Disclose the **random-encoder floor
  (~0.79)**: our in-domain SSL adds ~0.04 over it plus ~4× label efficiency. See
  `docs/positioning.md`.
