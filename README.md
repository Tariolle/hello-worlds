# hello-worlds — Geometry-Aware JEPA for EEG

**Hack the World(s) · EEG track · team Hello Worlds.** A controlled **frozen-transfer**
study on **TUAB** (TUH Abnormal EEG): pretrain a two-view self-supervised encoder on
unlabeled EEG, **freeze it**, and linear-probe *normal vs abnormal*.

> **Headline.** Domain-matched JEPA pretraining yields a frozen encoder whose linear
> probe (**0.819 balanced accuracy / ~0.89 AUROC, 3 seeds**) is competitive with the
> TUAB fine-tuning literature and above every foundation model evaluated *frozen*
> (EEG-FM-Bench: 0.55–0.78). In a controlled 2×2 ablation, **neither a distribution-free
> regulariser (PEIRA) nor a geometry-aware SPD-tangent variant beats the plain ambient
> SIGReg baseline** — yet the frozen SPD latent is visibly *organised*, and **de Surrel's
> AIRM Riemannian embedding sharpens it over the Euclidean view**, most clearly for SIGReg.

We say "JEPA" loosely (after LeJEPA): the model is a **symmetric Siamese**
augmentation-invariance + anti-collapse objective — no predictor, no EMA/target encoder,
no latent prediction. Built on a trimmed vendor of
[`eb_jepa`](https://github.com/facebookresearch/eb_jepa) (its EEG dataloader +
VICReg/SIGReg losses are reused intact under `eb_jepa/`).

## The question
Where should the anti-collapse regulariser act, and which mechanism wins? The EEG channel
covariance lives on a curved **SPD manifold**, so we test acting in its **tangent** vs
plain **ambient** Euclidean space, with **SIGReg** (isotropic-Gaussian target) vs **PEIRA**
(distribution-free — the principled fit for a non-Gaussian tangent).

|            | ambient (Euclidean)        | tangent (SPD)        |
|------------|----------------------------|----------------------|
| **VICReg** | `C0` reference             | —                    |
| **SIGReg** | `C1` Laya-like baseline    | `C2` geometry-aware  |
| **PEIRA**  | `C3`                       | `C4` ex-hypothesis   |

**A clean 3-seed null:** every cell lands ~0.82, inter-cell gaps ≤ 0.013 (below a plausible
eval bootstrap CI). Geometry/PEIRA help neither accuracy, calibration, nor robustness — the
payoff is the **latent geometry**, not the probe number.

## Results — frozen linear probe, TUAB 2717/276 patient-disjoint split
| Method | BA | AUROC |
|---|--:|--:|
| **Ours — SIGReg in-domain** (TUAB→TUAB, 3 seeds) | **0.819** | ~0.89 |
| Ours — SIGReg general-pretrain (TUSZ→TUAB, 1 seed) | 0.814 | 0.889 |
| random-init encoder (floor) | 0.790 | — |
| baseline — supervised-from-scratch (end-to-end / frozen) | 0.817 / 0.797 | 0.906 / 0.908 |
| baseline — Riemannian covariance + logistic | 0.761 | — |
| baseline — channel mean+std → linear probe | 0.553 | — |
| frozen foundation models (CBraMod → BIOT) | 0.55 – 0.78 | — |

Ours/baseline rows are measured locally; FM rows are quoted from **EEG-FM-Bench** (Xiong et
al., arXiv 2508.17742). Figure: `results/benchmark/frozen_headtohead.png`.

## Repo layout
```
eb_jepa/             trimmed vendor: EEG dataloader + VICReg/SIGReg losses
examples/eeg/        entry points:
  main.py              pretrain a JEPA cell        eval.py                frozen probe (+ --floor)
  baseline_riemann.py  Riemannian cov baseline     baseline_chanstats.py  channel mean+std baseline
  benchmark.py         render the benchmark table  frozen_headtohead.py   the FM comparison figure
  cfgs/                train / ablation / supervised configs
cluster/             Dalia (IDRIS) SLURM scripts        → see CLUSTER.md
presentation/        jury deck (main.tex → main.pdf)    → make
results/             measured numbers + figures (benchmark, latent, robustness, loss, …)
references/          literature summaries (LeJEPA, Laya, S-JEPA, EEG-VJEPA)
docs/                positioning, geometry analysis, benchmark tutorial
```

## Quickstart
```bash
uv venv && uv pip install -e .        # torch matching your CUDA (see pyproject); on the cluster, see CLUSTER.md
DATA=<TUAB_PREPROCESSED>

# 0) sanity-check the data + 0-parameter yardsticks (no GPU)
python -m examples.eeg.baseline_riemann   --data-root $DATA
python -m examples.eeg.baseline_chanstats --data-root $DATA --n-windows 8

# 1) pretrain a cell (set model.ssl.reg_type / reg_space in the config)
python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml

# 2) frozen linear probe on held-out patients, with the random-encoder floor
python -m examples.eeg.eval --ckpt ./checkpoints/<run>/latest.pth.tar --floor

# 3) render the benchmark table / figures (no training)
python -m examples.eeg.benchmark
```
Multi-diagnosis (folder-labelled EDFs) and TUEV event-level probes are supported via
`eval.py --label-scheme folders --classes …` and `tuev_probe.py`. **You own the
patient-disjoint split** for the `folders` scheme.

## Additional tracks
- **TCP-Graph-JEPA** (`train_graph_jepa.py`, `evaluate_graph_jepa.py`) — graph JEPA over the
  22 TCP bipolar derivations; anomaly = failure of spatio-temporal latent predictability.
  Oriented AUROC **0.791** (the direction inverts: abnormal EEG is *more* predictable).
- **Fourier-JEPA** (`examples/eeg/cfgs/train_fourier.yaml`) — STFT spectral-stem encoder ablation.

## Scope & honesty
Not a foundation model, not SOTA in 24h. SIGReg-for-EEG (**Laya**) and Riemannian-SSL-for-EEG
(**EEG-ReMinD**, **MENDR**) already exist — cited as parents. We report **balanced accuracy on
the full split** with the probe head stated, disclose the **~0.79 random-encoder floor**, and
never compare against reduced-subset accuracy. Full references in `presentation/main.tex`;
positioning in `docs/positioning.md`.

**Team Hello Worlds** — [Florent Tariolle](https://tariolle.github.io/) · [Clément Genninasca](https://github.com/Clems06) · [Yoann Frayce](https://github.com/Seveyus) · [Hippolyte du Pac de Marsoulies](https://github.com/hdupac).
