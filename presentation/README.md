# Presentation — Geometry-Aware JEPA for EEG

10-minute jury deck for the **Hack the World(s)** hackathon (EEG track), team
**Hello Worlds**. Compiled deck: `main.pdf`; source: `main.tex`.

LaTeX scaffold (metropolis theme + TikZ idioms, the 4-colour palette, `booktabs`
tables, `allowframebreaks` references, appendix section) adapted from
[`Tariolle/sls-wm/presentation`](https://github.com/Tariolle/sls-wm/tree/main/presentation).

## Build
```bash
make            # pdflatex x2 (TeX Live; metropolis falls back cleanly under pdflatex)
# or:
pdflatex main.tex && pdflatex main.tex
```
Needs the `metropolis` beamer theme (`beamertheme-metropolis`) and `colortbl`.

## Narrative (Data → Architecture → Training → Eval → Insight)
The main track is deliberately terse for the 10-minute slot; substantive detail
lives in the **appendix** (SPD-tangent construction, collapse diagnostics, eval
protocol, label-efficiency table, latent-space figure, disclosed caveats).

1. **Motivation** — the frozen-transfer question + protocol.
2. **Data** — TUAB, frozen-probe protocol, two-view augmentation.
3. **Architecture** — 1D conv encoder; ambient vs SPD-tangent anti-collapse; the 2×2 ladder.
4. **Training** — two-view JEPA; collapse monitoring.
5. **Results** — frozen head-to-head, value-of-SSL, the random-encoder floor, where 0.819 sits.
6. **Insight** — the clean 3-seed null + the wrapped-Gaussian mechanism + DO/DON'T claims.
7. **Limitations / Conclusion / References / Appendix.**

## Figures
`figures/` holds copies of the result plots so the deck is self-contained:
- `frozen_headtohead.png` — frozen head-to-head (← `results/benchmark/`)
- `value_of_ssl.png` — label & pretrain-data efficiency (← `results/label_eff/`)
- `latent_space.png` — frozen latent space, JEPA vs random (← `results/latent/`)

`\graphicspath` also points back at `../results/*` so the deck rebuilds against
freshly regenerated plots.

## Honesty rules (baked into the deck — do not relax)
Report **0.819** (3-seed mean), not 0.833 (best seed). Say **"match while
frozen"**, never "beat" the fine-tuned FMs. Disclose the **~0.79 random-encoder
floor**. Cite parents (Laya, EEG-ReMinD, PEIRA). Present the wrapped-Gaussian
fix as **future-work motivation**, not a validated mechanism. Never claim SOTA.
See `../docs/positioning.md`.
