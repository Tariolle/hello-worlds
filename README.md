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
python -m examples.eeg.baseline_riemann --data-root <TUAB_PREPROCESSED>
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

## TCP-Graph-JEPA anomaly detection
The repo also includes a graph-based JEPA anomaly detector for TUH/TUAB-style
normal/abnormal EEG:

- the 22 TCP bipolar derivations are graph nodes,
- graph edges combine shared-electrode continuity and contralateral symmetry,
- the model masks channel-time tokens and predicts latent embeddings, not raw EEG,
- anomaly score = failure of spatial-temporal latent predictability,
- outputs include file/window scores and channel x time heatmaps.

Expected model input is `[batch, 22, time_steps, feature_dim]`, defaulting to
about 70 frames for 7 s windows and five log-bandpower features
`delta/theta/alpha/beta/gamma`. Pre-extracted `.pt/.pth/.npy/.npz` tensors are
used directly. Raw EDFs are optionally preprocessed with MNE: EDF load, resample,
0.5-70 Hz filter, optional 60 Hz notch, TCP bipolar construction, 7 s windows,
and log-bandpower features.

Train self-supervised on normal windows:

```bash
python train_graph_jepa.py --config configs/graph_jepa.yaml
```

Evaluate file-level anomaly scores:

```bash
python evaluate_graph_jepa.py \
  --checkpoint checkpoints/tcp_graph_jepa/latest.pth.tar \
  --data-root <TUAB_OR_FEATURE_ROOT> \
  --split eval \
  --output-dir results/graph_jepa_eval
```

Visualize one EDF or tensor file:

```bash
python visualize_graph_jepa.py \
  --checkpoint checkpoints/tcp_graph_jepa/latest.pth.tar \
  --edf_or_tensor <file.edf_or_tensor> \
  --output_dir results/graph_jepa_viz
```

For a quick CPU debug run, point `configs/graph_jepa.yaml` at a tiny tensor
dataset and use `--limit-batches 2`.

## What we measure
Frozen **linear-probe balanced accuracy + AUROC on the full 2717/276 split**,
recording level — plus collapse diagnostics (effective rank, per-dim std,
off-diagonal covariance) logged every epoch, a label-fraction efficiency curve,
and robustness under injected noise / channel dropout.

## Multi-diagnosis probes
TUAB only provides the binary `normal` / `abnormal` label. To predict illness
type, import a labelled EDF dataset with one class folder per diagnosis:

```text
MY_DIAGNOSIS_DATASET/
  train/
    normal/**/*.edf
    seizure/**/*.edf
    dementia/**/*.edf
  eval/
    normal/**/*.edf
    seizure/**/*.edf
    dementia/**/*.edf
```

Then run the frozen probe with folder labels:

```bash
python -m examples.eeg.eval \
  --ckpt ./checkpoints/eeg_ambient_sigreg/latest.pth.tar \
  --data-root MY_DIAGNOSIS_DATASET \
  --label-scheme folders \
  --classes normal,seizure,dementia
```

The classical Riemannian covariance baseline supports the same folder-labelled
layout:

```bash
python -m examples.eeg.baseline_riemann \
  --data-root MY_DIAGNOSIS_DATASET \
  --label-scheme folders \
  --classes normal,seizure,dementia
```

The evaluator reports accuracy, balanced accuracy, macro-F1, one-vs-rest AUROC
when defined, per-class recall, and the confusion matrix.

**You are responsible for the patient-disjoint split here.** Unlike TUAB (disjoint
by construction), the `folders` scheme trusts whatever you put under `train/` and
`eval/`. If recordings from the same patient land in both, the probe number is
leaked and not comparable — keep every subject's recordings on one side only.
Classes are resolved from the `train/` folders (or `--classes`) and reused for
`eval/`: a diagnosis present only under `eval/` is silently ignored, so check the
`[eeg-eval] classes` line the evaluator prints.

For TUEV, use the event-level probe because the dataset is not recording-labelled
like TUAB:

```bash
python -m examples.eeg.tuev_probe \
  --riemann-only \
  --tuev-root <TUEV_PREPROCESSED>
```

## Honesty rules (read before talking to the jury)
- Report **balanced accuracy on the full split**, and state the probe head.
  Never plain accuracy on a reduced subset (that is the EEG-VJEPA number; it is
  not comparable to the LaBraM table).
- We are **not** a foundation model and **not** SOTA in 24h. SIGReg-for-EEG
  (Laya) and Riemannian-SSL-for-EEG (EEG-ReMinD, reconstruction) already exist —
  cite them as parents. Our claim is the **intersection**: geometry-aware,
  *latent-predictive*, distribution-free anti-collapse on the SPD tangent.

See `tasks/todo.md` for the full plan, baselines, targets, and division.
