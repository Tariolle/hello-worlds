# SPD-tangent anti-collapse: a controlled negative result + the principled fix

## What we tested
A 2×2 (+ VICReg reference) factorial — anti-collapse **{VICReg, SIGReg, PEIRA} × {ambient Euclidean, SPD log-Euclidean tangent}** — with a frozen linear probe on TUAB (patient-disjoint, full 2717/276 split), **3 seeds** {1, 1000, 10000}.

## Result (3-seed mean balanced accuracy)
| reg \ space | ambient | tangent |
|---|---|---|
| VICReg | 0.814 | — |
| SIGReg | 0.819 | **0.820** (tightest variance) |
| PEIRA  | 0.815 | 0.807 |

Per-cell seed ranges overlap heavily (~0.01–0.04 wide). **No significant effect of regulariser or of space — every cell lands at ~0.82 ± seed noise.** The geometry-aware tangent neither helps nor clearly hurts; PEIRA does not beat SIGReg; there is no interaction.

> Honesty note: the single-seed **0.833** (SIGReg-ambient, seed 1) was the *top* of that cell's range; its 3-seed mean is **0.819**. Report **~0.82 (best seed 0.833)**, never 0.833 alone.

## Why naive tangent regularisation is mis-specified (mechanism + citation)
Our tangent arm (`examples/eeg/geometry.py`) takes the per-window feature covariance `C` (SPD), maps it **Log-Euclidean at the identity** (`logm(C)`), vectorises (√2 off-diagonal), and applies the *same* anti-collapse there. Per **de Surrel, Lotte, Chevallier & Yger, "Wrapped Gaussian on the manifold of Symmetric Positive Definite Matrices", arXiv:2502.01512 (2025)**, the principled Gaussian on SPD is a **wrapped Gaussian** — a **non-isotropic** Gaussian in the tangent **at the Fréchet (Riemannian) mean**, pushed to the manifold by the **affine-invariant (AIRM)** exponential map. Our arm is therefore triple-mis-specified:

| | our `geometry.py` | principled (de Surrel 2025) |
|---|---|---|
| target | **isotropic** Gaussian (SIGReg) | **wrapped, non-isotropic** |
| base point | **identity** | **Fréchet mean** |
| metric | **Log-Euclidean** | **affine-invariant (AIRM)** |

SIGReg forces *isotropy* where the geometry induces a *non-isotropic* covariance — it fights the structure that carries the signal. **PEIRA is distribution-free**, so it escapes the *target* mis-specification (only base point / metric apply); but empirically tangent did not help PEIRA either. At n=3 the tangent effect is null, so our data **cannot confirm the wrapped-Gaussian prediction** — it stays a theoretical argument, not a validated mechanism.

## The principled fix — future work, not a 24h move
A correct "geometry-aware SIGReg" would regularise toward a **wrapped Gaussian** (de Surrel 2025): use the **channel** covariance (19 electrodes, not learned features), the **AIRM tangent at the running Fréchet mean** (stop-grad on the base point), and a Gaussianity test against the **non-isotropic wrapped** target instead of the isotropic one. We did not pursue it: it is an all-or-nothing re-implementation **+ re-pretrain** whose best case is a *tie* with ambient — because the linear Riemannian probe (and our 0.761 cov+tangent+LR baseline) already re-derives second-order tangent structure, making tangent-space SSL largely redundant by construction.

## Takeaway for the jury
A clean, **pre-registered negative result**: *where* anti-collapse acts (ambient vs SPD-tangent) and *which* regulariser (VICReg/SIGReg/PEIRA) do **not** significantly change frozen-probe accuracy on TUAB — all reach ~0.82. We give a geometric reason the naive tangent is mis-specified (citing the wrapped-Gaussian construction) and flag the principled fix as future work. We found no prior work isolating the *space* in which anti-collapse acts for EEG SSL.
