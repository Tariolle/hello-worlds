# Research Handoff: What the Hackathon Established

This note turns the 24-hour project into a safe starting point for later work.
It records evidence and constraints; it is not a claim of a finished research
direction.

## Evidence worth carrying forward

1. On the patient-disjoint TUAB frozen-probe protocol, in-domain two-view
   SIGReg reaches 0.819 balanced accuracy across three seeds. The random encoder
   floor is about 0.79, so small differences between SSL variants need strong
   controls.
2. The ambient/tangent and SIGReg/PEIRA 2x2 comparison is a null on this task:
   no variant has a reliable accuracy, calibration, or corruption-robustness
   advantage over ambient SIGReg.
3. AIRM-aware visualization of SPD covariances gives a more geometry-faithful
   view of the learned latent structure. Treat this as a diagnostic hypothesis,
   not evidence of clinical or downstream improvement.
4. TUSZ-to-TUAB transfer is encouraging but single-seed and same-site. TUEV is a
   secondary cross-task signal with patient-overlap uncertainty relative to the
   TUAB pretraining data.

## What must not be claimed

- Geometry, tangent-space regularization, or PEIRA improves TUAB frozen-probe
  performance.
- This project is the first JEPA for EEG or the first Riemannian EEG model.
- A visualization alone proves class structure, clinical validity, or absence of
  subject leakage.
- A prediction-energy anomaly score will be high for abnormalities. The retained
  Graph-JEPA exploration showed that score orientation can be task-dependent.

## September research protocol: anomaly detection first

The first scientific test should be patient-wise abnormality detection on TUAB.

1. Use only normal training recordings to fit representation and normality
   models. Hold out a patient-disjoint development partition from training for
   threshold and score-direction decisions; never use evaluation labels for those
   choices.
2. Evaluate held-out normal versus abnormal TUAB recordings with AUROC, AUPRC,
   sensitivity/specificity operating points, and recording-level aggregation.
3. Compare three fixed variants: Euclidean temporal-prediction JEPA,
   tangent-SIGReg, and tangent-PEIRA. Use the same encoder capacity, masking,
   normal-only data, optimizer budget, and patient split.
4. Score prediction residual, distance from normal latent statistics, and a
   JEPA-SCORE-style Jacobian diagnostic separately. Calibrate the direction on
   development data instead of assuming a larger residual is anomalous.
5. Run three seeds. Audit patient/session leakage and report confidence intervals,
   representation rank/variance, and subject-prediction diagnostics with every
   comparison.

Advance a geometry variant only if it gives a reproducible improvement on the
primary anomaly metric or a reproducible, task-relevant representation diagnostic.
Otherwise keep anomaly detection as the result and move next to cross-subject
robustness or latent diagnostics.

## Visualization policy

Before proposing a new embedding method, compare:

- UMAP on precomputed AIRM or Log-Euclidean SPD distances;
- ordinary Euclidean UMAP; and
- the existing AIRM Riemannian t-SNE view.

Report neighborhood preservation, trustworthiness, embedding stability, and
post-hoc labels. UMAP already accepts precomputed distances and is built around a
Riemannian-manifold model, so a bespoke "Riemannian UMAP" is not the first
research task. See the [UMAP metric documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html).

## Reuse boundary

`hello-worlds` is the archived hackathon artifact. Later research code should
port only audited primitives into Manifold-JEPA: EEG data/split utilities,
Log-Euclidean SPD features, PEIRA, and evaluation helpers. Each port needs a unit
test and a provenance note. Do not inherit unchecked hackathon assumptions,
figures, or claims.
