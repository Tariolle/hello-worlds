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
- [ ] Push to GitHub; clone on Dalia; `uv pip install -e .`
- [ ] CONFIRM TUAB_PREPROCESSED path on Dalia; set data.data_root
- [ ] Smoke test: baseline_riemann (validates EDF pipeline, gives ~0.86)
- [ ] Smoke test: main.py 1 epoch tiny epoch_size on 1 GPU (loss decreases, eff_rank > 1)
- [ ] Verify tangent arm numerics (eigh/logm stability at d_cov=32); fallback = hypersphere tangent
- [ ] C1 SIGReg×ambient full run -> first probe number (THE qualification gate)
- [ ] Fan out 2×2 {C1..C4} + C0 (VICReg ref) across 3 GPUs, 1 seed -> populate the table
- [ ] 3-seed {1,1000,10000} the key comparison (C2 vs C4 + the interaction)
- [ ] Label-fraction curve (1/5/10/25/100%) — novel, ~free
- [ ] Robustness curve (noise / channel-dropout at probe time)
- [ ] Figures + 10-min deck + report

## 24h sequencing (deadlines: 17:30 code, 18:00 slides, 19:00 jury)
- H0–1  install on Dalia, confirm SLURM partition/account + TUAB path
- H1–4  baseline_riemann (data sanity + yardstick); smoke main.py; build probe vs random encoder
- H4–8  Rung 0 trains without collapse -> first number  (GATE — must clear by ~H8)
- H8–16 Rung 1 + Rung 2; 3 GPUs = 3 parallel arms; one change at a time
- H16–21 lock model, robustness + label-fraction, figures, 3-seed error bars
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
