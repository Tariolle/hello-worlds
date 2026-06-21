# Archived Fourier-JEPA - STFT spectral-stem encoder

> Exploratory encoder ablation; not part of the final SIGReg/PEIRA/SPD-tangent
> conclusion. The retained core encoder is `examples/eeg/encoder.py`.

An encoder ablation for the two-view EEG-JEPA track: swap the strided-conv
`EEGEncoder1D` for a **Fourier / STFT spectral-stem** encoder, holding the SSL
objective, seed, data and frozen probe fixed. This isolates the effect of a
spectral front-end for short EEG windows.

## Motivation

TUAB abnormality is **band-power-driven** (see `examples/eeg/cfgs/benchmark.yaml`
— the broadband Riemannian baseline is power-blind and lands *below* the random
floor). For a short window, a single STFT exposes the full per-band spectral
content in one shot, whereas a conv stack has to grow a deep receptive field to
see the same global periodic structure. A Fourier front-end bakes that inductive
bias in directly.

## Architecture (`archive/fourier_jepa/encoder.py`)

`x [B, 19, 2000]` (10 s @ 200 Hz, per-channel z-scored)

1. **STFT stem** — `torch.stft` per channel (`n_fft=128`, `hop=32`, Hann),
   `log1p(|S|^2)` log-power spectrogram → `[B, C, F=65, T'=63]`.
2. **Learnable spectral mixing** — flatten `(C, F)` and apply a 1×1 conv over the
   `C*F=1235` axis → `[B, d_hidden=256, 63]` (BN + GELU). Learns a filterbank
   combining bands across channels.
3. **Temporal conv blocks** — 2× (Conv1d k=3, BN, GELU) over the frame axis.
4. **`head` / `cov_proj`** — 1×1 convs → `[B, d_model=256, 63]` / `[B, d_cov=32, 63]`.

Same contract as `EEGEncoder1D`: `represent` (mean over T'), `feature_map`,
`cov_features`. `T'=63 > d_cov=32` keeps the SPD tangent covariance full-rank, so
the geometry/tangent arm works unchanged too. Selected via
`model.encoder.type: fourier`; the default `conv1d` keeps the retained core
configs/checkpoints on the convolutional path.

## Reproduce (matched pair, seed 1)

Local sanity:

```bash
python -m pytest archive/fourier_jepa/tests -q        # shape/grad/SPD-rank contract
python -m archive.fourier_jepa.smoke                  # fwd+bwd, all reg cells
```

On Dalia (from the isolated `$WORK/fourierjepa/hello-worlds` checkout):

```bash
sbatch archive/fourier_jepa/cluster/smoke.sbatch                                # validate graph on B200
sbatch archive/fourier_jepa/cluster/train.sbatch                                # Fourier SIGReg-ambient
CFG=examples/eeg/cfgs/train.yaml sbatch --export=ALL,CFG archive/fourier_jepa/cluster/train.sbatch  # conv baseline
sbatch --dependency=afterok:<fourier_jobid>:<conv_jobid> archive/fourier_jepa/cluster/benchmark.sbatch  # comparison table
```

The benchmark render writes `results/benchmark/eeg_benchmark.{md,csv,html}` and
per-metric plots, with measured rows for the conv baseline, the Fourier encoder,
the random floor and the Riemannian baseline, next to the published references.
