# EEG-JEPA — live checklist + findings

> Strategy, experiment matrix (2×2), owners, timeline, deck → **[../PLAN.md](../PLAN.md)**.
> This file = execution checklist + reference numbers.

## Headline (what we tell the jury)
A controlled study: **when the latent lives on the SPD manifold of EEG channel
covariances, which anti-collapse wins — SIGReg (isotropic-Gaussian target, as in
Laya) or PEIRA (distribution-free)?** PEIRA is the principled choice because the
log-mapped EEG covariance distribution is non-Gaussian; SIGReg fights it, PEIRA
does not. We measure frozen linear-probe balanced accuracy, collapse dynamics,
robustness, and label/data efficiency — at a tiny param/data budget.

Parents we explicitly cite (NOT claim as ours):
- **Laya** (arXiv 2603.16281): SIGReg/LeJEPA JEPA on EEG, TUAB frozen probe. = our ambient baseline.
- **EEG-ReMinD** (arXiv 2501.08139): Riemannian SPD SSL on EEG, but *reconstruction*, not JEPA.
- **PEIRA** (arXiv 2605.17671, Arbel/Terver/Ponce): the regulariser; verified, distribution-free.

## Live findings
- Riemannian 0-param baseline: BalAcc **0.761** / AUROC 0.810.
- Gate C1 SIGReg×ambient: BalAcc **0.833** / AUROC 0.901, leak-free (patient-disjoint verified), no collapse.
- **2×2 sweep (seed 1), frozen-probe BalAcc:**

  | reg \ space | ambient | tangent |
  |---|---|---|
  | VICReg | 0.806 | — |
  | SIGReg | **0.833** | 0.818 |
  | PEIRA  | 0.826 | 0.811 |

  → At 1 seed the geometry/PEIRA hypothesis is **NOT supported**: tangent slightly *hurts* (~−0.015 both regs), PEIRA ≈ SIGReg, **no interaction**. Best = SIGReg×ambient. All beat plain VICReg.
  → Differences ~0.01–0.03 = likely seed noise. **Run 3 seeds + error bars** before concluding, then either (a) honest negative result, or (b) give geometry a fair shot with a manifold-correct tangent (channel-cov SPD + AIRM + correct target distribution).

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

### NEXT — task distribution (post-gate phase)
- [~] **Florent** — 2×2 sweep {C0..C4} seed 1 → table (RUNNING), then 3-seed {1,1000,10000} + error bars on the key comparison (C2 vs C4 + interaction)
- [ ] **Florent** — quick architecture/training upgrades if they raise frozen BalAcc (encoder, SIGReg λ, augments)
- [ ] **Clément** — random-encoder floor; strengthen Riemann baseline toward 0.86 (whole-recording cov); collapse-dynamics writeup + PEIRA theory + "why collapse / why our reg avoids it" (jury criterion); manifold-correctness of the tangent arm
- [ ] **Yoann** — label-fraction efficiency curve (1/5/10/25/100%, novel on TUAB); robustness curve (noise / channel-dropout at probe); all figures (2×2 bars+error bars, collapse curves, label-frac, robustness, vs-baselines); wandb/demo
- [ ] **Hippolyte** — 10-min deck + storytelling; honest positioning (frozen vs FT, in-domain vs cross-dataset); literature/baseline slides; **run the Deep Research** (prompt below); PM/timekeeper

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
