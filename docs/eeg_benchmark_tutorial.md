# EEG-JEPA Benchmark Tutorial

This benchmark compares the repo's geometry-aware JEPA variants against local
TUAB baselines and published EEG references such as Laya. The JEPA model is not
trained yet, so the default command only renders the benchmark table with local
JEPA rows marked as pending.

## What This Benchmark Measures

Primary local task: TUAB normal vs abnormal EEG classification.

Primary local protocol:

- Use the full `TUAB_PREPROCESSED` patient-disjoint `train` / `eval` split.
- Freeze the encoder.
- Extract one recording-level embedding by mean-pooling evenly spaced windows.
- Fit a balanced logistic-regression probe on train patients only.
- Report evaluation recording-level balanced accuracy and AUROC.

Do not present subset accuracy, fine-tuned accuracy, or EEG-Bench numbers as the
same result. The benchmark table keeps these references because they are useful
targets, but it marks them as `reference only`.

## Files

- `examples/eeg/cfgs/benchmark.yaml`: benchmark registry for local methods and
  published references.
- `examples/eeg/benchmark.py`: renderer and optional local evaluation harness.
- `results/benchmark/`: generated CSV, JSON, Markdown, HTML, and plot outputs.
- `examples/eeg/eval.py`: single-checkpoint frozen-probe evaluator.
- `examples/eeg/baseline_riemann.py`: CPU classical Riemannian baseline.

## Safe First Command

Run this before any training exists:

```bash
python -m examples.eeg.benchmark
```

This writes:

- `results/benchmark/eeg_benchmark.csv`
- `results/benchmark/eeg_benchmark.md`
- `results/benchmark/eeg_benchmark.html`
- `results/benchmark/eeg_benchmark_rows.json`
- one bar chart per metric that has data: `eeg_benchmark_acc.png`,
  `eeg_benchmark_balanced_acc.png`, `eeg_benchmark_auroc.png`,
  `eeg_benchmark_f1.png`

It does not run JEPA pretraining.

## Human Workflow

1. Install the environment:

```bash
uv venv
uv pip install -e .
```

2. Render the benchmark skeleton:

```bash
python -m examples.eeg.benchmark
```

3. Run the CPU baseline once the TUAB path is available:

```bash
python -m examples.eeg.benchmark \
  --run-riemann \
  --data-root /path/to/TUAB_PREPROCESSED
```

For an imported diagnosis dataset with class folders, pass the same label
options used by `examples.eeg.eval`:

```bash
python -m examples.eeg.benchmark \
  --run-riemann \
  --data-root /path/to/MY_DIAGNOSIS_DATASET \
  --label-scheme folders \
  --classes normal,seizure,dementia
```

4. Run the random encoder floor:

```bash
python -m examples.eeg.benchmark \
  --run-random-floor \
  --data-root /path/to/TUAB_PREPROCESSED
```

5. After a JEPA checkpoint exists, attach it to its registered method id:

```bash
python -m examples.eeg.benchmark \
  --data-root /path/to/TUAB_PREPROCESSED \
  --checkpoint jepa_sigreg_ambient=checkpoints/eeg_ambient_sigreg/latest.pth.tar
```

Add more `--checkpoint METHOD_ID=PATH` arguments to evaluate multiple trained
cells in one command.

The cross-task TUEV dataset is event-labelled rather than recording-labelled, so
run its dedicated probe instead of the TUAB benchmark harness:

```bash
python -m examples.eeg.tuev_probe \
  --riemann-only \
  --tuev-root /path/to/TUEV_PREPROCESSED
```

To compare the frozen encoder and the classical Riemannian event covariance
baseline on the exact same sampled event windows:

```bash
python -m examples.eeg.tuev_probe \
  --ckpt checkpoints/eeg_ambient_sigreg/latest.pth.tar \
  --tuev-root /path/to/TUEV_PREPROCESSED \
  --floor \
  --riemann
```

## JEPA Cells To Populate

The benchmark registry tracks these local cells:

| Method id | Regularizer | Space | Role |
|---|---|---|---|
| `jepa_vicreg_ambient` | VICReg | ambient | eb_jepa reference arm |
| `jepa_sigreg_ambient` | SIGReg | ambient | Laya-like local baseline |
| `jepa_sigreg_tangent` | SIGReg | tangent | geometry-aware SIGReg |
| `jepa_peira_ambient` | PEIRA | ambient | factorial control |
| `jepa_peira_tangent` | PEIRA | tangent | hypothesis cell |

The intended headline statistic is the interaction:

```text
(PEIRA_tangent - SIGReg_tangent) - (PEIRA_ambient - SIGReg_ambient)
```

Use balanced accuracy for this computation unless the project explicitly changes
the primary metric.

## Published Reference Rows

The table includes published numbers for context:

- Laya-S: closest conceptual reference because it is LeJEPA/SIGReg for EEG.
- LuMamba: efficient topology-invariant EEG model with a strong TUAB balanced
  accuracy target.
- EEG-VJEPA: direct JEPA-style TUAB reference, but on a smaller subset.
- LaBraM, ChronoNet, BioSerenity-E1: strong clinical or TUAB references with
  different adaptation protocols.

These rows are not marked comparable to the local full-split frozen probe unless
they are reproduced under the same local protocol.

## Agent Instructions

Use this sequence when an agent is asked to update or run the benchmark:

1. Check branch and worktree:

```bash
git status --short --branch
```

2. Read the registry and tutorial:

```bash
sed -n '1,260p' examples/eeg/cfgs/benchmark.yaml
sed -n '1,260p' docs/eeg_benchmark_tutorial.md
```

3. Render-only command is safe:

```bash
python -m examples.eeg.benchmark
```

4. Do not run `examples.eeg.main` unless the user explicitly asks to train.

5. Only run local data commands when a TUAB path is provided or already known:

```bash
python -m examples.eeg.benchmark --run-riemann --data-root <TUAB_PREPROCESSED>
python -m examples.eeg.benchmark --run-random-floor --data-root <TUAB_PREPROCESSED>
```

6. Only evaluate checkpoints that exist:

```bash
python -m examples.eeg.benchmark \
  --checkpoint <method_id>=<checkpoint_path> \
  --data-root <TUAB_PREPROCESSED>
```

7. When reporting results, state:

- whether the row is local comparable or a published reference,
- the protocol,
- the split,
- the metric,
- whether the encoder was frozen or fine-tuned,
- whether the checkpoint was trained or pending.

## Adding A New Reference

Add a row under `published_references` in `examples/eeg/cfgs/benchmark.yaml`.
Include:

- `id`
- `display_name`
- `family`
- `status: published_reference`
- `protocol`
- metric values
- `metric_source`
- `comparable_to_local_jepa: false` unless reproduced locally
- a short caveat in `notes`

Then run:

```bash
python -m examples.eeg.benchmark
```

## Adding A New Local Checkpoint

Add a row under `local_methods` in `examples/eeg/cfgs/benchmark.yaml`, then run:

```bash
python -m examples.eeg.benchmark \
  --checkpoint new_method_id=path/to/latest.pth.tar \
  --data-root <TUAB_PREPROCESSED>
```

The script will load the checkpoint config, rebuild the encoder, extract frozen
features, fit the local probe, and update the generated artifacts.
