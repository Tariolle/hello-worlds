# Honest positioning for the jury (Deep Research, reconciled to our 3-seed numbers)

> Source: team Deep Research (`deep-research-report.md`), reconciled with our actual results.
> The report was written against the single-seed **0.833**; our honest headline is the
> **3-seed mean 0.819** (0.833 was the lucky seed). **Use the 3-seed numbers below — not 0.833.**

## The number to report
Frozen, in-domain SSL (TUAB-train, no labels) → freeze → linear probe on patient-disjoint eval:
**Balanced accuracy 0.819 ± ~0.012 (3 seeds; best seed 0.833) · AUROC ~0.90 (best seed 0.901).**
Not "0.833", not "SOTA".

## Critical caveat — the random-feature floor (DISCLOSE this)
In OUR pipeline, a **random (untrained) conv encoder + linear probe reaches ~0.79 BalAcc** on TUAB —
EEG abnormality is power-driven, and random conv filters preserve band power. That floor is **≈
LaBraM-linear (0.795) and ABOVE EEG2Rep / BIOT / EEGPT frozen**. So the strong frozen numbers are
largely the *architecture + the task*; **our in-domain SSL adds ~0.04 over random (0.82 vs 0.79) plus
~4× label efficiency** (JEPA at 25% labels ≈ random at 100%). Disclosing this is the honest, rigorous
move: it pre-empts "is the SSL doing anything?", reframes the contribution as a *quantified* SSL
increment (not SOTA), and is itself a sharp benchmark insight — TUAB frozen-probe is easy for simple
power features, so several published "foundation-model" frozen numbers barely clear a random baseline.

## Where it sits — TUAB standard 2717/276 split
| Setting | Protocol | BA | AUROC |
|---|---|---:|---:|
| **Ours** | **in-domain SSL, frozen, linear probe** | **0.819 (3-seed)** | **~0.90** |
| LaBraM-Base / -Huge | fine-tuned | 0.814 / 0.826 | 0.902 / 0.916 |
| CBraMod | fine-tuned | 0.829 | 0.923 |
| REVE-Base | fine-tuned | 0.832 | 0.925 |
| LaBraM-Base | **linear probe** | 0.795 | 0.884 |
| REVE | **linear probe** | 0.810–0.821 | — |
| EEGPT | frozen / linear-probe-style | 0.798 | 0.872 |
| EEG2Rep | linear probe | 0.766 (acc) | 0.832 |
| BIOT | linear probe (EEG2Rep bench) | 0.751 (acc) | 0.829 |
| classical Riemannian (Gemein 2020) | cov+tangent+classifier | ~0.86 (acc) | — |

**Honest read:** 0.819 is **at the top of the frozen linear-probe results** (≈ REVE-linear; clearly above LaBraM-linear / EEG2Rep / BIOT / EEGPT) and at the **lower edge of the fine-tuned BA band**. AUROC ~0.90 is above BIOT/EEGPT but **below** CBraMod/REVE (~0.92). Very competitive at the operating point, not top at ranking, **not SOTA**.

## Apples-to-apples: general-pretrain, STILL frozen — the upgrade (lead with this)
"You pretrained on TUAB, the FMs didn't — unfair" is the obvious attack on the in-domain number. So we
ran the fair version: pretrain SIGReg on **TUSZ** (TUH Seizure corpus = general TUH EEG, **TUAB-eval
patients excluded** → patient-disjoint), freeze, linear-probe TUAB — the FMs' own regime (general
pretrain → frozen TUAB), without our in-domain edge.
> **Result: 0.814 BA / 0.889 AUROC** (1 seed) — essentially equal to in-domain **0.819**. The in-domain
> advantage was NOT the driver; we hold general-pretrained too.

Caveats: TUSZ is general **same-site** TUH (not multi-site like the FMs); single seed (in-domain is 3-seed).

## Frozen head-to-head — EEG-FM-Bench numbers (cite, NOT re-measured)
Frozen linear probe, TUAB balanced accuracy, quoted from **EEG-FM-Bench (Cui et al., arXiv 2508.17742, Table 1)**:

| Frozen FM | BA | | reference | BA |
|---|---:|---|---|---:|
| CBraMod | 0.547 | | BENDR | 0.666 |
| LaBraM | 0.604 | | EEGPT | 0.766 |
| BIOT | 0.780 | | random floor (ours) | ~0.79 |
| | | | **ours — in-domain / general** | **0.819 / 0.814** |

Two honest reads: (1) our **random floor (0.79) already beats every frozen FM** — TUAB frozen rewards
band-power and FM tokenizers are probe-hostile to it; (2) the **inversion** — the more an FM is tuned for
fine-tuning SOTA, the harder it collapses frozen (CBraMod, recent SOTA, is the *most* collapsed at 0.547).
Cross-task echo on **TUEV** (6-class, frozen): ours **0.425 > LaBraM 0.315 / CBraMod 0.211 / BENDR 0.167**,
~BIOT 0.437, < EEGPT 0.498 (EEG-FM-Bench) — but ⚠️ TUEV-eval patient-disjointness vs our TUAB-train pretrain
is **unverified** (anonymised eval IDs); the FMs saw TUEG⊇TUEV, so they had *more* TUEV exposure, not us.
Figures: `results/benchmark/frozen_headtohead{,_tuev}.png`.

## The negative result — now 3-seed, with precedent
Our 2×2 × 3 seeds is a **clean null**: tangent-SPD and PEIRA do not beat ambient SIGReg (~0.82 everywhere; ranges overlap). The Deep Research found this has precedent **in spirit**:
- **No direct prior art** for "anti-collapse regulariser in the tangent-SPD of covariances inside a JEPA" → our ablation is novel.
- **MENDR** (first Riemannian-SPD EEG foundation model) itself reports it only *matches or slightly underperforms* baselines on benchmark accuracy.
- Literature asymmetry: SPD/tangent geometry helps **classical decoding, robustness, interpretability** — not reliably **frozen-probe accuracy**. So "more geometry ≠ more frozen-probe points" is a documented pattern, not a surprise.

(Note: the report assumed our negative was **one-seed/preliminary**; we now have **3 seeds**, so it is a firmer null — but still "no measurable benefit on TUAB frozen", not "geometry is dead".)

## Ceiling — soften the language
There is **no published TUAB-specific Bayes ceiling**. TUH claims 97–100% inter-rater for the abnormal corpus, but Gemein warns it is likely inflated; broader EEG inter-rater agreement is ~86–88%; SCORE-AI's Normal-category agreement was 0.737. → **Do NOT say "TUAB saturated at ~0.83."** Say: labels are noisy, the ceiling is unknown, and **sub-1-point differences without multi-seed CIs are not meaningful.**

## Jury-safe contribution (the boxed claim)
> Domain-matched JEPA pretraining on TUAB yields a frozen encoder whose linear-probe performance
> (**0.819 BA / ~0.90 AUROC, 3 seeds**) is competitive with the TUAB fine-tuning literature and
> **stronger than typical published frozen-probe baselines**. This is **not an in-domain artifact**:
> pretraining on a *general* TUH corpus (TUSZ, patient-disjoint from TUAB-eval) and probing TUAB frozen
> still gives **0.814 BA** — and both numbers sit far above every foundation model evaluated frozen
> (EEG-FM-Bench: 0.55–0.78). In a controlled 2×2 ablation (3 seeds),
> **neither PEIRA nor a geometry-aware tangent-SPD anti-collapse variant improved over an ambient
> SIGReg baseline** — on TUAB frozen abnormality detection, extra geometric sophistication is not
> automatically rewarded. We give a geometric reason (wrapped-Gaussian mis-specification of the naive
> tangent, de Surrel 2025) and flag the principled fix as future work.

## DO / DON'T claim
**DO:** strongest among published frozen/linear-probe TUAB baselines; in the FT BA band; clean 3-seed null on geometry/PEIRA; **hold ~0.81–0.82 frozen even when general-pretrained (TUSZ→TUAB 0.814), above every FM frozen (EEG-FM-Bench 0.55–0.78)**; the SSL increment over the random floor (~+0.03–0.04) is real and disclosed.
**DON'T:** "SOTA on TUAB" (AUROC below best; in-domain advantage; preprocessing differs); "we beat CBraMod/LaBraM" (they are fine-tuned → say "match while frozen"); "tangent-SPD is useless for EEG" (only "no benefit in our TUAB frozen pilot"); "TUAB is saturated at X".

## If we push geometry later (future work)
The payoff for geometry is **robustness/transfer/calibration**, not a BA race: calibration under threshold
shift, montage perturbation, channel dropout, cross-dataset abnormality transfer — plus the principled
wrapped-Gaussian tangent (channel-cov + AIRM @ Fréchet mean). See `docs/geometry_tangent_analysis.md`.
