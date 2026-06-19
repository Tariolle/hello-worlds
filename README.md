# hello-worlds — geometry-aware JEPA for EEG (TUAB)

Hackathon "Hack the World(s)" — EEG track. We pretrain a two-view JEPA on
unlabeled EEG, freeze the encoder, and linear-probe **normal vs abnormal** on
TUAB. The contribution is a **controlled study of where the anti-collapse
regulariser should live** — ambient Euclidean vs the tangent space of the EEG
covariance SPD manifold — and **which mechanism wins there**: SIGReg (assumes an
isotropic-Gaussian target) vs PEIRA (distribution-free, the principled fit for a
non-Gaussian manifold tangent).

Built on a trimmed vendor of [`eb_jepa`](https://github.com/facebookresearch/eb_jepa)
(its EEG dataloader + VICReg/SIGReg losses are reused intact under `eb_jepa/`).

## The ladder (each rung is presentable on its own)
| Rung | `reg_type` | `reg_space` | what it is |
|---|---|---|---|
| 0 | `sigreg` | `ambient` | Laya-like baseline + qualification insurance |
| 1 | `sigreg` | `tangent` | geometry-aware SIGReg |
| 2 | `peira`  | `tangent` | distribution-free anti-collapse on the manifold |

Reference arms: `vicreg/ambient` (eb_jepa default), plus the 0-param classical
**Riemannian baseline** (`baseline_riemann.py`, ~0.86 acc on TUH Abnormal).

## Quickstart (on the cluster)
```bash
uv venv && uv pip install -e .        # install torch matching cluster CUDA (see pyproject)
# 0) sanity-check data + get the complexity yardstick (no GPU):
python -m examples.eeg.baseline_riemann --data_root <TUAB_PREPROCESSED>
# 1) pretrain (edit cfgs/train.yaml: data.data_root, model.ssl.reg_*):
python -m examples.eeg.main  --fname examples/eeg/cfgs/train.yaml
# 2) frozen probe (held-out patients), with random-encoder floor:
python -m examples.eeg.eval  --ckpt ./checkpoints/eeg_ambient_sigreg/latest.pth.tar --floor
```

## Benchmark (safe before training)
The benchmark renderer compares local JEPA cells, local TUAB baselines, and
published references such as Laya without launching pretraining:
```bash
python -m examples.eeg.benchmark
```
Artifacts are written under `results/benchmark/`. See
`docs/eeg_benchmark_tutorial.md` for the human and agent workflow, including how
to add trained checkpoints later with `--checkpoint METHOD_ID=PATH`.

## What we measure
Frozen **linear-probe balanced accuracy + AUROC on the full 2717/276 split**,
recording level — plus collapse diagnostics (effective rank, per-dim std,
off-diagonal covariance) logged every epoch, a label-fraction efficiency curve,
and robustness under injected noise / channel dropout.

## Honesty rules (read before talking to the jury)
- Report **balanced accuracy on the full split**, and state the probe head.
  Never plain accuracy on a reduced subset (that is the EEG-VJEPA number; it is
  not comparable to the LaBraM table).
- We are **not** a foundation model and **not** SOTA in 24h. SIGReg-for-EEG
  (Laya) and Riemannian-SSL-for-EEG (EEG-ReMinD, reconstruction) already exist —
  cite them as parents. Our claim is the **intersection**: geometry-aware,
  *latent-predictive*, distribution-free anti-collapse on the SPD tangent.

See `tasks/todo.md` for the full plan, baselines, targets, and division.
