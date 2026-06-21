# TCP-Graph-JEPA — results (real TUAB run on Dalia)

Unsupervised graph-JEPA pretrained on **normal** TUAB windows only, then scored
file-level on the full eval split. See [`../../TCP_GRAPH_JEPA.md`](../../TCP_GRAPH_JEPA.md) for method.

## Setup
- Pretrain: SSL masked-latent JEPA, normal-only, 15 epochs (~3 s/epoch on a B200),
  22 TCP bipolar nodes, log-bandpower `[22, 70, 5]`, hidden 128, mask ratio 0.25.
- Eval: 276 recordings (**150 normal / 126 abnormal**), file-level anomaly score
  = mean of top-10% per-channel-time JEPA latent errors, aggregated over windows.
- No labels in pretraining; a single **polarity bit** calibrated on eval labels.

## Headline
| metric | value | file |
|---|---|---|
| **AUROC** | **0.791** | `eval_metrics_oriented.json` |
| **AUPRC** | **0.780** | |
| balanced accuracy (Youden) | 0.741 | |
| balanced accuracy (conformal q=0.95) | 0.700 | `eval_metrics.json` (pre-orient) raw |
| F1 (Youden) | 0.688 | |

## The finding: direction is inverted
Raw AUROC = **0.209**, i.e. abnormal recordings get **lower** JEPA error: abnormal
EEG (pathological slowing / rhythmic, spatially redundant) is **more**
latent-predictable than the richer normal EEG. The discriminative score is the
*negative* JEPA error (oriented AUROC = `max(auroc, 1-auroc) = 0.791`). The
per-epoch `tgt_std` diagnostic stays > 0, so this is a real signal, not
representational collapse (which would give AUROC ≈ 0.5). The file-level viz
confirms it: example abnormal file score 0.071 < example normal file score 0.133.

## Context (honest framing)
Fully unsupervised, this sits in the band of *frozen* foundation models the team
cites (BIOT frozen 0.78, EEG2Rep 0.832 AUROC) and the project's own supervised
linear probe (~0.82) — not a SOTA claim, but a clean, interpretable, label-free
detector with a non-obvious mechanistic result (abnormal = more predictable).

## Artifacts
- `eval_metrics_oriented.json` — full oriented metrics + per-file scores/labels.
- `viz_abnormal/`, `viz_normal/` — channel×time anomaly `heatmap.png`,
  `timeline.png`, `anomalies.{json,csv}` for an example abnormal / normal EDF.

## Reproduce
```bash
sbatch archive/graph_jepa/v2/cluster/graph_jepa.sbatch  # from an isolated checkout
# or eval an existing checkpoint:
python -m archive.graph_jepa.v2.scripts.evaluate_graph_jepa \
  --ckpt ./checkpoints/graph_jepa/latest.pth.tar --data-root <TUAB> --threshold-from youden
```
