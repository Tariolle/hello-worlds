# Riemannian EEG Approach

This repo now has a first-class covariance-manifold baseline in
`examples/eeg/riemannian.py`, exposed through the backward-compatible command
`python -m examples.eeg.baseline_riemann`.

## Literature Basis

Implementation-relevant sources:

- Barachant, Bonnet, Congedo, Jutten, **"Multiclass Brain-Computer Interface
  Classification by Riemannian Geometry"**, IEEE TBME 2012. Core MDM recipe:
  represent trials as SPD covariance matrices, estimate one class centroid per
  class, and classify by minimum Riemannian distance.
  https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.classification.MDM.html
- Barachant, Bonnet, Congedo, Jutten, **"Classification of covariance matrices
  using a Riemannian-based kernel for BCI applications"**, Neurocomputing 2013.
  Motivation for tangent-space projection before standard vector classifiers.
  https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html
- Barachant and Congedo, **"A Plug&Play P300 BCI Using Information Geometry"**,
  2014. Extends covariance-manifold decoding beyond SMR/motor imagery into ERP
  tasks, with calibration-efficient adaptation.
  https://arxiv.org/abs/1409.0107
- Kalunga, Chevallier, Barthelemy, **"Using Riemannian geometry for SSVEP-based
  Brain Computer Interface"**, 2015. Explicitly motivates treating EEG covariance
  matrices as SPD manifold points rather than Euclidean matrices.
  https://arxiv.org/abs/1501.03227
- Congedo, Barachant, Andreev, **"A New Generation of Brain-Computer Interface
  Based on Riemannian Geometry"**, 2013. Frames Riemannian BCI as a simple,
  low-calibration benchmark family across ERP, SMR, and SSVEP.
  https://arxiv.org/abs/1310.8115
- Tibermacine et al., **"Riemannian Geometry-Based EEG Approaches: A Literature
  Review"**, 2024. Recent survey; useful for positioning classical covariance
  methods against deep Riemannian and hybrid approaches.
  https://arxiv.org/abs/2407.20250

## Implemented Pipeline

For TUAB/folder-labelled recording classification:

1. Read `n_windows` evenly spaced windows per recording with `EEGDataset`.
2. Estimate a channel covariance matrix per window. Defaults to OAS shrinkage;
   SCM and Ledoit-Wolf are also available.
3. Aggregate window covariances into one recording covariance. Defaults to the
   affine-invariant Riemannian mean (`--aggregation riemann`), with
   log-Euclidean and Euclidean alternatives for ablation.
4. Fit one of two classifiers:
   - `--classifier mdm`: class Frechet means plus minimum Riemannian distance.
   - `--classifier tangent-logreg`: tangent map at the train Frechet mean,
     standardization, balanced logistic regression.

For TUEV event classification, the same classifiers operate directly on
event-window covariance matrices via `examples/eeg/tuev_probe.py --riemann`.

## Commands

TUAB tangent-space logistic probe:

```bash
python -m examples.eeg.baseline_riemann \
  --data-root <TUAB_PREPROCESSED> \
  --classifier tangent-logreg \
  --aggregation riemann \
  --cov-estimator oas
```

Pure MDM variant:

```bash
python -m examples.eeg.baseline_riemann \
  --data-root <TUAB_PREPROCESSED> \
  --classifier mdm \
  --aggregation riemann \
  --mean-metric riemann \
  --distance-metric riemann
```

TUEV event-only Riemannian probe:

```bash
python -m examples.eeg.tuev_probe \
  --riemann-only \
  --tuev-root <TUEV_PREPROCESSED> \
  --riemann-classifier mdm
```

Benchmark harness:

```bash
python -m examples.eeg.benchmark \
  --run-riemann \
  --data-root <TUAB_PREPROCESSED> \
  --riemann-classifier tangent-logreg \
  --riemann-aggregation riemann
```

## Notes

- AIRM means have no closed form, so the code uses a Karcher iteration initialized
  by the log-Euclidean mean.
- Tangent-space features use `log(R^-1/2 C R^-1/2)` and sqrt(2)-weighted
  upper-triangular vectorization, matching the pyRiemann-style tangent map.
- MDM is the cleaner zero-parameter Riemannian classifier; tangent-logreg is often
  stronger when a Euclidean discriminative head can exploit tangent coordinates.
