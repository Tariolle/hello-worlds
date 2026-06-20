# SPD-tangent anti-collapse: a controlled negative result + the principled fix

## What we tested
A 2×2 (+ VICReg reference) factorial — anti-collapse **{VICReg, SIGReg, PEIRA} × {ambient Euclidean, SPD log-Euclidean tangent}** — with a frozen linear probe on TUAB (patient-disjoint, full 2717/276 split), **3 seeds** {1, 1000, 10000}.

## Result (3-seed mean balanced accuracy)
| reg \ space | ambient | tangent |
|---|---|---|
| VICReg | 0.814 | — |
| SIGReg | 0.819 | **0.820** (tightest variance†) |
| PEIRA  | 0.815 | 0.807 |

Per-cell seed ranges overlap heavily (~0.01–0.04 wide). **No significant effect of regulariser or of space — every cell lands at ~0.82 ± seed noise.** The geometry-aware tangent neither helps nor clearly hurts; PEIRA does not beat SIGReg; there is no interaction.

> †**Caveat (verified):** BCS/SIGReg seeds its random slice directions from `self.step`, not the run seed (`eb_jepa/losses.py`), so the slice schedule is identical across the 3 seeds — SIGReg's error bars do **not** include slicing randomness and its "tightest variance" is *partly an artifact*. Don't lean on it. Fix (draw slices from the run-seeded global RNG) must precede any re-run, as it shifts all SIGReg numbers. See `docs/audit_notes_for_slides.md` §3.

> Honesty note: the single-seed **0.833** (SIGReg-ambient, seed 1) was the *top* of that cell's range; its 3-seed mean is **0.819**. Report **~0.82 (best seed 0.833)**, never 0.833 alone.

## Why naive tangent regularisation is mis-specified (mechanism + citation)
Our tangent arm (`examples/eeg/geometry.py`) takes the per-window feature covariance `C` (SPD), maps it **Log-Euclidean at the identity** (`logm(C)`), vectorises (√2 off-diagonal), and applies the *same* anti-collapse there. Per **de Surrel, Lotte, Chevallier & Yger, "Wrapped Gaussian on the manifold of Symmetric Positive Definite Matrices", arXiv:2502.01512 (2025)**, the principled Gaussian on SPD is a **wrapped Gaussian** — a **non-isotropic** Gaussian in the tangent (Def. 4.1: `t ~ N(μ, Σ)` with `Σ` a *full* `d(d+1)/2`-dim SPD covariance, not `σ²·I`), pushed to the manifold by the **affine-invariant (AIRM)** exponential map; its **centred** submodel `WG(p;0,Σ)` has Fréchet (Riemannian) mean equal to the base point `p` (Prop. 4.8). Our arm is therefore **doubly mis-specified**:

| | our `geometry.py` | principled (de Surrel 2025) |
|---|---|---|
| target | **isotropic** Gaussian (SIGReg ⇒ N(0,I)) | **wrapped, non-isotropic** (Def. 4.1, full `Σ`) |
| base point | **identity** | **Fréchet mean** (centred model, Prop. 4.8) |

**Why two axes, not three — the metric is not independent at our operating point.** At the identity base point the AIRM and Log-Euclidean logarithms *coincide*: AIRM `Log_p(q) = p^½ log(p^-½ q p^-½) p^½` reduces to `log(C)` at `p = I`, which is exactly the Log-Euclidean chart of `C`; and the AIRM inner product at `I` is the Frobenius product, so our √2 vectorisation is the correct isometry for **both** metrics. (`geometry.py` thus already computes the AIRM-at-identity Log correctly.) LE and AIRM only diverge once the base point leaves the identity — AIRM conjugates by `p^{±½}`, and the correct vectorisation then needs a *parallel-transported* basis (de Surrel App. A), not raw `upper_tri_vec`. The mis-specification is therefore concentrated in **target isotropy + identity base point**, not in a "metric" label. (Scoping caveat: Prop. 4.8 is the *centred* `μ=0` model; the paper's general `μ≠0` model deliberately does **not** use the Riemannian mean as base-point estimator — App. H — but for an anti-collapse term, which wants spread *around the barycentre*, the centred model is the right object.)

SIGReg forces *isotropy* where the geometry induces a *non-isotropic* covariance — it fights the structure that carries the signal. **PEIRA is distribution-free**, so it escapes the *target* mis-specification (only the base point applies); but empirically tangent did not help PEIRA either. At n=3 the tangent effect is null, so our data **cannot confirm the wrapped-Gaussian prediction** — it stays a theoretical argument, not a validated mechanism.

## The principled fix — future work, not a 24h move
A correct "geometry-aware SIGReg" would regularise toward a **wrapped Gaussian** (de Surrel 2025): use the **channel** covariance (19 electrodes, not learned features), the **AIRM tangent at the running Fréchet mean** (stop-grad on the base point — note the AIRM barycentre has **no closed form**, so this needs a Karcher iteration, and vectorisation needs the parallel-transported basis of App. A), and a Gaussianity test against the **non-isotropic wrapped** target instead of the isotropic one. Two reasons beyond "no time in 24h":

- **The principled object is already our historical 0.761 baseline.** `examples/eeg/baseline_riemann.py` was the channel-covariance + AIRM tangent + LR probe; it now delegates to the fuller direct implementation in `examples/eeg/riemannian.py` and also exposes MDM. Evaluated directly as a probe feature, the original run scored **0.761 ≪ 0.82** — the very object the "principled fix" steers toward already *underperforms* ambient by ~6 points (the linear probe re-derives second-order tangent structure for free). Best case for tangent SSL is therefore a *tie*. (Caveat: the baseline uses *raw* channel covariances, not learned-feature covariances — an imperfect but sign-clear proxy.)
- **SIGReg/BCS structurally cannot target a non-isotropic wrapped Gaussian.** Epps–Pulley tests each random 1-D slice against `N(0,1)` — an *isotropic* target; matching a full `Σ` would require whitening by `Σ̂^{-½}` first, which is circular with anti-collapse (whitening destroys the covariance signal we want to keep). The coherent "principled" anti-collapse on an AIRM tangent is therefore **distribution-free PEIRA**, not wrapped-SIGReg — which is consistent with our PEIRA-tangent already being a (null) member of the factorial.

## Takeaway for the jury
A clean, **pre-registered negative result**: *where* anti-collapse acts (ambient vs SPD-tangent) and *which* regulariser (VICReg/SIGReg/PEIRA) do **not** significantly change frozen-probe accuracy on TUAB — all reach ~0.82. We give a geometric reason the naive tangent is mis-specified (citing the wrapped-Gaussian construction) and flag the principled fix as future work. We found no prior work isolating the *space* in which anti-collapse acts for EEG SSL.
