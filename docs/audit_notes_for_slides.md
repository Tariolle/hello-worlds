# Audit notes for the geometry / SSL slides — captured for later

> Three independent parallel-agent verifications (each adversarially checked) audited our
> claims before the deck. This file captures the **slide-ready, reusable** conclusions so we
> don't re-derive them when we return to the slides. **Our headline result is unchanged: a
> clean 3-seed null (~0.82), geometry FOLDED, random-floor 0.79 disclosed.** Nothing below
> reopens that — it hardens the *framing* and lists the *deferred* moves if we ever revisit.

Companion docs: `docs/geometry_tangent_analysis.md` (the mechanism + fix), `docs/positioning.md`
(jury numbers). Figures already built: `results/label_eff/value_of_ssl.png` (label efficiency,
SSL gain +0.036, ~4× label efficiency) and `results/latent/latent_space.png` (frozen latent space,
best JEPA vs random; silhouette ~4–5× higher for JEPA, both modest — corroborates +0.036).

---

## 1 · Wrapped-Gaussian capitalization (de Surrel et al. 2025, arXiv:2502.01512)

**Verdict: core bulletproof; two precision fixes already applied to `geometry_tangent_analysis.md`.**

- **BULLETPROOF (our best card):** the natural Gaussian on SPD is *non-isotropic*, so an isotropic
  anti-collapse target (SIGReg ⇒ N(0,I)) is mis-specified in the tangent. Multi-sourced: abstract, §2, §4, Def 4.1.
- **Fix 1 (done):** "triple" → **"double"** mis-specification. At the identity base point LE-log ≡ AIRM-log = `logm(C)`; the metric is **not** an independent axis. Mis-spec = *isotropic target + identity base point*.
- **Fix 2 (done):** scope "Fréchet mean" to the **centred** submodel `WG(p;0,Σ)` (Prop 4.8). The paper's general `μ≠0` model deliberately **rejects** the Riemannian mean as base point (App. H). For an anti-collapse term the centred model is the right object anyway.

**Slide one-liner (defensible):**
> On the SPD manifold the natural Gaussian is non-isotropic by construction (de Surrel et al. 2025, Def. 4.1) — so an isotropic anti-collapse target (SIGReg ⇒ N(0,I)) is, *by construction*, the wrong target in the tangent space.

Present as **future-work motivation, not a validated mechanism** (n=3 tangent null). That hedge is what makes the slide unattackable.

**Verbatim citation bank:**
- Non-isotropy — Abstract: *"a non-isotropic wrapped Gaussian by leveraging the exponential map"*; §2: *"a non-isotropic distribution with some preferred directions"*.
- Def 4.1: `X = Exp_p(Vect_p⁻¹(t)), t ~ N(μ,Σ)`, `Σ ∈ P_{d(d+1)/2}` (full SPD, not σ²·I).
- AIRM Eq (1): `⟨u,v⟩_p = tr(p⁻¹u p⁻¹v)`; Eq (3): `Log_p(q)=p^½ log(p^-½ q p^-½) p^½`. **"Log-Euclidean": 0 occurrences in the paper.**
- Prop 4.8: *"A mean of `WG(p;0,Σ)` is `p`"* (centred case only).
- App. H: *"Why does estimating `p` using the Riemannian mean fails in the general case?"*
- Prop 5.1: `μ̂_N = (1/N)Σ VLog_{p*}(x_i)`; no closed form for `p̂_N` (Riemannian Conjugate Gradient).

**Say out loud:** the paper gives the geometric *premise* (SPD law = non-isotropic); the *regulariser* conclusion is OUR inference. The paper has 0 occurrences of self-supervised / collapse / regularize / target distribution.

---

## 2 · Intrinsic Riemannian (AIRM, not just LE tangent) — feasibility

**Verdict: CONFIRM THE FOLD. P(beat ~0.82 significantly) ≈ 0.07.**

Probability mass: clean null (≈0.82) **0.52** · within-noise bump **0.34** (a *liability* — reads as p-hacking, kills the pre-registered-null narrative) · NaN eats the time **0.08** · clean defensible win **0.06**. You'd be trading 0.06 of real upside for 0.34 of narrative downside.

**Mechanism sentence (good for the slide):**
> The real AIRM delta vs LE-at-identity is a *non-commuting curvature term* that lives in a tangent space the frozen probe is **structurally blind to** (eval reads `encoder.represent` = first-order temporal mean, never the tangent), inside a regulariser family already measured null at 3 seeds, behind a BN projector that whitens metric/scale/isotropy.

**Minimal variant spec (future-work box, ~40 LOC + 1 re-pretrain; realistically several hrs to de-risk NaN, not 1–2h):**
- **SPD object:** FEATURE covariance (`d_cov=32`), **not** the 19 channels — channel-cov has 0 learnable params, so no SSL gradient flows (that's `baseline_riemann.py`, decorative for SSL). Only `temporal_covariance(cov_features(x))` has a gradient path.
- **Base point:** detached EMA **Fréchet mean**, 1 Karcher step (stop-grad). The *only* ingredient not absorbed by the StandardScaler — at a fixed shared base point, AIRM is a fixed linear reparam the linear probe absorbs.
- **Metric:** AIRM. **Prediction:** geodesic distance, **not** wrapped-Gaussian (the Linear+BN+GELU projector already learns an affine whitening that makes an isotropic target satisfiable; and PEIRA, being distribution-free, has no target mis-spec to fix).
- **Numerical risk (HIGH):** `eigh`-backward `1/(λi−λj)` — AIRM whitening compresses the eigen-gap ~10× (≈7.8e-2 → 8.1e-3), pushing each window an order of magnitude closer to the `gap=0` cliff. Needs float64 (cheap at 32×32) + gap-floored custom backward + jitter + isfinite-skip.

**Honest correction to our own argument:** do **not** cite tangent-PEIRA=0.807 as evidence against wrapped-Gaussian. PEIRA is distribution-free → immune to the isotropic-target mis-spec the wrapped-Gaussian fixes → it never tests that hypothesis. 0.807 vs 0.815 is within noise: neither for nor against.

---

## 3 · SIGReg / PEIRA exhaustiveness

**Leverage ranking (what could actually move the null):** almost all probability mass is on the two **cross-cutting** knobs, not per-cell ones.
1. **Probe READOUT** (mean → meanstd / second-order) — **HIGH**. The exact DoF the tangent manipulates. `meanstd` exists in `eval.py:45-47` but was never run across the 4 cells.
2. **Epochs / undertraining** (eff_rank still rising at epoch 30) — **MED-HIGH**.
3. Per-cell knobs (SIGReg λ=10, num_slices=256; PEIRA λ=0.1, eta, Cholesky; tangent d_cov/eps) — noise or leakage-prone. **Not** the right lever to attack a null.

**PEIRA gradient sign: NOT a bug (verified analytically).**
`∇_z1(½·term) = (1/B)(QΣQ·z1 − Q·z2) = ∇_z1(−½ tr(Σ(N+λI)⁻¹))`. Minimizing the loss ⇒ tr_P rises ⇒ anti-collapse (collapse ⇒ Σ,N→0 ⇒ tr_P→0). Sign is correct; a flip is excluded.
> ⚠️ **The real trap is interpretability, not sign:** PEIRA's loss is a *surrogate-gradient* value — it can be negative and is non-monotone. **Only `tr_P` (and `eff_rank`) are progress signals.** Anyone who read "loss went down" to judge PEIRA training read an empty number.
- **Deferred:** add `tests/test_peira.py` (grad-match + tr_P-rises finite-difference). Ready-to-paste code is in the session transcript.

**BCS (SIGReg) seed bug: CONFIRMED (real).**
`eb_jepa/losses.py`: `g.manual_seed(self.step)` → slice directions are **identical across the 3 run seeds** (`self.step`=0,1,2… regardless of the run seed; `main.run()`'s `manual_seed(cfg.meta.seed)` only touches the global RNG). So SIGReg runs share the exact slice schedule → artificially correlated → **SIGReg error bars don't include slice randomness**, and the *"SIGReg-tangent tightest variance"* claim is **partly an artifact**.
- One-line fix: draw `A` from the global (run-seeded) RNG instead of the step-seeded generator (`A = torch.randn((z1.size(1), self.num_slices), device=dev)` inside `no_grad`; drop `self.step`).
- **For the deck (now):** DISCLOSE the caveat — we will **not** re-run a folded arm, and fixing the seed without re-running would only invalidate existing numbers. The fix MUST precede any future re-run (it shifts all SIGReg numbers).

**Two cheap ZERO-LEAK experiments (recommended "when we come back"):**
- **Exp A — Readout invariance** (re-score, ~1 min/ckpt, no retrain): re-score the 4 existing checkpoints with `mean` vs `meanstd` readout, choice fixed a priori or on the TRAIN patient-disjoint dev split. If tangent ≈ ambient even under a 2nd-order readout → the null **hardens** (not a readout artifact). If `meanstd` lifts tangent → the null was under-explored.
- **Exp B — Bootstrap CI on eval at FIXED encoder** (free): resample the 276 eval recordings with replacement → BalAcc CI ≈ ±0.02–0.03, **wider** than the max inter-cell gap (0.820−0.807 = 0.013). Turns an "underpowered (n=3 seeds) null" into a **"benchmark-resolution-limited null"**: TUAB-276 frozen-probe cannot separate these cells, full stop. Publishable insight on its own.
- **Guardrail (free, do first):** per-cell non-degeneracy audit — confirm each cell actually trained (eff_rank high; PEIRA tr_P rose). tangent-PEIRA=0.807 is the outlier; if its tr_P plateaued low / eff_rank dropped, it's a *partial collapse*, not a fair null entry — flag it, don't naively average it.

**One sentence for the jury:** Exp A tests whether the null holds when the readout finally sees second order; Exp B shows the gaps are below the benchmark's sampling noise; the audit proves all 4 cells are healthy competitors — three moves with **zero tuning on the eval**.
