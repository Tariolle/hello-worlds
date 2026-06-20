# Contribution skeleton — EEG-JEPA (jury deck / report)

> Honest, multi-axis, multi-seed. We do NOT claim a new SOTA model. We claim a
> rigorous controlled study + a frozen-evaluation critique for clinical EEG.
> Parents to cite: Laya (SIGReg/LeJEPA-EEG), EEG-ReMinD, MENDR; benchmark refs:
> EEG-FM-Bench (arXiv:2508.17742), NeuroAtlas, Beyond-Accuracy.

## One-line claim (boxed)
> On frozen-probe clinical EEG (TUAB), *where* anti-collapse acts (ambient vs
> SPD-tangent) and *which* regulariser (VICReg/SIGReg/PEIRA) do **not** improve
> balanced accuracy, calibration, or noise-robustness — and the SPD-tangent
> objective is **measurably worse** under channel dropout. Meanwhile a *random*
> conv encoder (0.79) beats every published frozen foundation model, exposing that
> these benchmarks are power-dominated. In-domain SSL adds a small, quantified,
> label-efficient increment over that floor, and holds even when pretrained on a
> general TUH corpus (TUSZ → frozen TUAB).

## What is novel (the wedge)
- First to **isolate the *space* of anti-collapse** (ambient vs SPD-tangent of a
  learned covariance) inside an EEG-JEPA, across **3 axes × 3 seeds**.
- First to report, for clinical-EEG frozen probes, a **tuned random-encoder floor**,
  **calibration/selective-prediction**, and **3-seed corruption-robustness** — the
  exact gaps left open by EEG-FM-Bench / NeuroAtlas / Beyond-Accuracy.

## Results (each = number + figure + honest caveat)

1. **Accuracy null (2×2 × 3 seeds).** ~0.82 BA everywhere (SIGReg-amb 0.819, tan
   0.820, PEIRA-amb 0.815, tan 0.807); ranges overlap, eval-276 bootstrap CI
   ±0.02–0.03 > inter-cell gaps. → *benchmark-resolution-limited null.*
   Fig: 2×2 bar (3-seed). Mechanism: `docs/geometry_tangent_analysis.md`.

2. **We MEASURED the geometry (not hidden).** Exposing the learned SPD-tangent to
   the probe → 0.79 BA — a real but weak 2nd-order signal (SIGReg-tan 0.790 > its
   random-tangent floor 0.776), **below** the 1st-order rep (0.82). → TUAB is
   1st-order/power-dominated; 2nd-order caps ~0.78–0.80 (cf. MENDR 0.78–0.80,
   classical Riemann 0.761). Audit: probe is structurally blind to the tangent.

3. **Robustness (3-seed).** Channel dropout: **ambient SIGReg most robust**
   (drop +0.046 ± 0.011 at p=0.5); SPD-tangent/PEIRA degrade 2.5–3.5× more
   (+0.12–0.16). Additive noise: all SSL cells overlap (no geometry win); the
   single-seed "tangent more noise-robust" did **not** replicate.
   → geometry does not buy robustness; it **hurts** dropout robustness.
   Fig: `results/robustness/robustness_3seed.png`.

4. **Calibration / selective prediction.** Every SSL encoder beats the random
   floor (ECE 0.038–0.057 vs 0.065; AURC 0.067–0.090 vs 0.109); geometry/PEIRA do
   **not** beat ambient (tangent worsens AURC). → positive is "SSL > random", not
   "geometry wins". Fig: `results/calibration/calibration.png`. (1 seed.)

5. **Frozen-collapse critique (the sharp insight).** A random conv encoder (0.79)
   **beats every published frozen FM** on TUAB (CBraMod 0.547 / LaBraM 0.604 /
   BENDR 0.666 / EEGPT 0.766 / BIOT 0.780, EEG-FM-Bench). FM tokenizers are
   probe-hostile to band power. Fig: `results/benchmark/frozen_headtohead.png`.

6. **Quantified SSL increment + generality.** In-domain SSL adds ~+0.03 over the
   random floor and ~4× label efficiency. Apples-to-apples: general TUH pretrain
   (TUSZ, patient-disjoint from TUAB-eval) → frozen TUAB **0.814 ≈ in-domain 0.819**
   → not an in-domain artifact. Cross-task: TUAB encoder → TUEV frozen 0.425, above
   LaBraM/CBraMod/BENDR frozen.

## DO / DON'T (jury)
- **DO:** rigorous multi-axis × 3-seed null on geometry/PEIRA; the random-floor /
  frozen-collapse critique; "SSL > random" on calibration; general-pretrain holds;
  everything floor-relative and honest.
- **DON'T:** "SOTA"; "we beat the FMs" (they're fine-tuned → "match/exceed *frozen*");
  "geometry helps robustness" (it does not — 3-seed); cherry-pick PEIRA-tangent's
  noise nugget; "TUAB saturated at X".

## Figures
- `results/benchmark/frozen_headtohead.png` (+ `_tuev.png`) — frozen head-to-head.
- `results/label_eff/value_of_ssl.png` — label/data efficiency.
- `results/robustness/robustness_3seed.png` — robustness (3-seed).
- `results/calibration/calibration.png` — calibration / selective prediction.

## Caveats to state up front
- Single seed for calibration + the per-corruption draw; 3 seeds for accuracy &
  robustness. eval-276 CI ±0.02–0.03 — sub-0.02 gaps are noise.
- TUSZ general-pretrain = same-site TUH, not multi-site like the FMs; 1 seed.
- TUEV cross-task: patient-disjointness vs TAUB-train pretrain unverified
  (anonymised eval IDs); the FMs saw TUEG⊇TUEV.
- FM numbers are **cited** from EEG-FM-Bench (their protocol), not re-measured.
