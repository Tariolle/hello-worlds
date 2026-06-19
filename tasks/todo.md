# EEG-JEPA — live checklist + findings

> Strategy, experiment matrix (2×2), owners, timeline, deck → **[../PLAN.md](../PLAN.md)**.
> This file = execution checklist + reference numbers.

## Headline (what we tell the jury) — UPDATED post-result
Controlled study: does making a frozen EEG-JEPA's anti-collapse **geometry-aware**
(SPD-tangent) and/or **distribution-free** (PEIRA) beat ambient SIGReg on TUAB?
**Pre-registered hypothesis (PEIRA + tangent win) is FALSIFIED** — at 3 seeds every
cell is ~0.82, no significant effect. We ship: (1) a **strong frozen in-domain probe
(~0.82, above frozen foundation-model numbers)**, (2) a **clean negative result** with a
geometric mechanism (wrapped-Gaussian mis-specification, de Surrel 2025), (3)
label/data-efficiency curves showing the value of the SSL. Honest negative > fragile positive.

Parents we explicitly cite (NOT claim as ours):
- **Laya** (arXiv 2603.16281): SIGReg/LeJEPA JEPA on EEG, TUAB frozen probe. = our ambient baseline.
- **EEG-ReMinD** (arXiv 2501.08139): Riemannian SPD SSL on EEG, but *reconstruction*, not JEPA.
- **PEIRA** (arXiv 2605.17671, Arbel/Terver/Ponce): the regulariser; verified, distribution-free.

## Live findings
- Riemannian 0-param baseline: BalAcc **0.761** / AUROC 0.810.
- **2×2 sweep, 3 seeds {1,1000,10000}, MEAN frozen-probe BalAcc:**

  | reg \ space | ambient | tangent |
  |---|---|---|
  | VICReg | 0.814 | — |
  | SIGReg | 0.819 | 0.820 (tightest var) |
  | PEIRA  | 0.815 | 0.807 |

  → **CLEAN NULL**: all cells ~0.82 ± seed noise, ranges overlap → no significant effect of regulariser OR space. Tangent doesn't help/hurt; PEIRA not > SIGReg; no interaction. (1-seed "tangent hurts" did NOT replicate.)
  → The seed-1 **0.833** (SIGReg-ambient) was that cell's best seed; 3-seed mean = **0.819**. **Report ~0.82 (best 0.833), NOT 0.833.** Still > BIOT-frozen 0.78 / EEG2Rep 0.766 / Riemann 0.761.
  → DECISION: **fold geometry → honest negative result.** Mechanism + citation (wrapped Gaussian, de Surrel 2025) → `docs/geometry_tangent_analysis.md`.
  → Clément's `benchmark.py` (ranked comparison + protocol tracking) **merged into main**.

## Targets (frozen linear-probe, balanced accuracy, FULL 2717/276 split)
- < 0.60 embarrassing (collapse / raw-feature regime; LaBraM/CBraMod fall here under a plain linear probe)
- 0.72–0.78 respectable (EEGPT/BIOT frozen level)
- >= 0.80 strong
- Complexity yardstick: classical Riemannian tangent+LR ~0.86 acc, ~0 deep params (Gemein 2020).
- TUAB is label-noise-capped ~0.85–0.87 / inter-rater ~0.90 — DO NOT chase the 2nd decimal.

## Baseline reference numbers (for the comparison slide)
- Fine-tuned BalAcc band 0.795–0.829 (CBraMod 0.829, LaBraM-Base 0.814, BIOT 0.796). We are NOT fine-tuning.
- Frozen linear probe (consistent protocol): BIOT 0.780, EEGPT 0.766; LaBraM 0.604 / CBraMod 0.547 collapse.
- EEG2Rep frozen 0.766 acc / 0.832 AUROC — best apples-to-apples frozen target.
- EEG-VJEPA "83.3%" = plain acc on a non-standard subset — NOT comparable, do not cite as if it were.

## Build status
- [x] Trimmed eb_jepa vendor (losses.py, eeg/dataset.py) + package scaffold
- [x] `encoder.py` EEGEncoder1D (represent / feature_map / cov_features)
- [x] `geometry.py` SPD log-Euclidean tangent + collapse metrics
- [x] `peira.py` SC-PEIRA (verified algorithm)
- [x] `main.py` build_encoder / build_ssl (vicreg|sigreg|peira × ambient|tangent)
- [x] `eval.py` patient-disjoint probe + random floor
- [x] `baseline_riemann.py` pyRiemann tangent + LR yardstick
- [x] configs / pyproject / .gitignore / README
- [x] Push to GitHub (hello-worlds → origin/main) + code on Dalia `$WORK/hello-worlds`
- [x] SSH access working (key fix: trailing newline + Windows OpenSSH — see CLUSTER.md)
- [x] SLURM recipe found (signal-53 fix: `--account`+`--nodes`+`--ntasks`; GPU via `--gpus-per-node`)
- [x] aarch64 venv built on compute node (torch 2.11+cu128) — `$WORK/venvs/hw_aarch64`
- [x] TUAB path confirmed + already in train.yaml (no download needed)
- [x] Smoke: forward+backward of all 6 cells on a B200, eigh-tangent on GPU OK (`smoke.py`)
- [x] Verify tangent numerics (eigh/logm finite on GPU at d_cov=32) — covered by smoke
- [x] `git pull` workflow on cluster (HTTPS + fine-grained PAT; SSH is proxy-blocked)
- [x] baseline_riemann on TUAB ✅ EDF pipeline OK + yardstick **BalAcc 0.761 / AUROC 0.810**
      (quick 16-window mean; ~0.86 reachable with whole-recording covariance if we want a harder bar)
- [x] C1 SIGReg×ambient -> **BalAcc 0.833 / AUROC 0.901** ✅ GATE CLEARED (no collapse, eff_rank 27→65, ~3 min)
- [x] train/eval patient-disjoint VERIFIED (2076 vs 253 patients, overlap 0) — 0.833 is leak-free

### NEXT — task distribution (post-result: fold geometry, ship the honest negative)
- [x] **Florent** — 2×2 × 3 seeds DONE (clean null). NEXT, in order: (1) **patient-disjoint TRAIN dev-split in `eval.py`** = LEAKAGE GATE before ANY hyperparameter selection; (2) **mean⊕max pooling + L2-norm** re-score on the existing SIGReg-ambient ckpt; (3) **label-efficiency curve** (re-fit probe on 1/2/5/10/25/50/100% train labels, 5 seeds) + random floor; (4) pretrain-data-efficiency curve if time.
- [x] **Clément** — ✅ `benchmark.py` (ranked comparison + protocol tracking, MERGED) + ✅ geometry/wrapped-Gaussian analysis (`docs/geometry_tangent_analysis.md`). NEXT: feed the benchmark with our 3-seed **means + 95% CIs**; own the negative-result + wrapped-Gaussian future-work slides.
- [ ] **Yoann** — the **single figure = 2-panel "value of self-supervision"** (BalAcc vs %labels | vs %pretrain-data, overlays: random floor, Riemann 0.761, FT-foundation-model band labelled "fine-tuned, cross-corpus"); collapse-dynamics figure from TensorBoard; populate Clément's benchmark plots; demo.
- [ ] **Hippolyte** — 10-min deck; **honest positioning** (~0.82 frozen ≈ FT foundation models *while frozen*; the clean null is the contribution; DO-NOT-claim list); run the Deep Research; PM/timekeeper.
- **DROP (adversarial review):** the tangent geometry fix, C/λ/projector hp-sweeps (test-leakage + within-noise), longer-pretrain, attention-pool. Don't touch `geometry.py`.

## 24h sequencing (deadlines: 17:30 code, 18:00 slides, 19:00 jury)
- ~~H0–1  install on Dalia + SSH/venv/smoke~~ ✅
- ~~H1–4  baseline_riemann (0.761) + smoke + probe harness~~ ✅
- ~~H4–8  Rung 0 GATE~~ ✅ cleared at 0.833
- **H8–16 ← WE ARE HERE: 3-seed 2×2 + error bars; decide negative-result vs geometry-fix; robustness + label-fraction**
- H16–21 lock model, figures, error bars
- H21–24 slides + report + demo

## Team division (2 tech, 1 product, 1 commercial)
- **Florent (lead):** encoder/ssl/geometry, run + monitor experiments on Dalia.
- **Clément (maths):** PEIRA correctness + the "why collapse / why our reg avoids it"
  explanation (jury criterion), tangent formalism, collapse metrics, ablation design.
- **Yoann (product):** eval harness polish, wandb, all figures, demo.
- **Hippolyte (commercial):** 10-min deck + storytelling from H2, literature/baseline
  slides, PM/timekeeper, Dalia logistics liaison.

## Risks
1. Collapse (#1) — monitor eff_rank/std every epoch; kill flat runs; fix = more reg, not less.
2. Data/compute access friction — resolve H0–1, don't debug SLURM alone.
3. Tangent eigh numerics — d_cov small + eps clamp; hypersphere-tangent fallback ready.
4. PEIRA needs >= 2 shared predictable view-modes (r_max>=2) or collapse-instability weakens — watch tr_P rising.
