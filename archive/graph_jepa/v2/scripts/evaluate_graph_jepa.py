"""TCP-Graph-JEPA — file-level anomaly evaluation on a labelled split.

Scores every recording of the eval split (normal=0 / abnormal=1), then reports
AUROC / AUPRC / balanced accuracy / F1 / confusion matrix. The decision
threshold is taken from validation *normal* scores at a configurable quantile
(conformal-style) when ``--threshold-from normal``, or from Youden's J on the
eval set when ``--threshold-from youden``.

Run:
    python -m archive.graph_jepa.v2.scripts.evaluate_graph_jepa \
        --ckpt ./checkpoints/graph_jepa/latest.pth.tar \
        --data-root <TUAB_PREPROCESSED>
"""
import argparse
import json
import os

import numpy as np
import torch

from archive.graph_jepa.v2.core.metrics import (evaluate, normal_quantile_threshold,
                                        orient_scores, separation,
                                        youden_threshold)
from archive.graph_jepa.v2.core.scoring import score_file_loader
from archive.graph_jepa.v2.core.windows import make_graph_loader
from archive.graph_jepa.v2.scripts.common import load_checkpoint


def _loader(data_cfg, split, stats, batch_size, workers):
    cfg = type(data_cfg)(**{**vars(data_cfg)})
    cfg.split, cfg.mode, cfg.batch_size, cfg.num_workers = split, "file", batch_size, workers
    return make_graph_loader(cfg, stats=stats, shuffle=False)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--threshold-from", choices=["normal", "youden"], default="normal")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_cfg, scoring_cfg, stats = load_checkpoint(a.ckpt, device)
    if a.data_root:
        data_cfg.data_root = a.data_root
    print(f"[graph-eval] device={device} root={data_cfg.data_root}", flush=True)

    ev = _loader(data_cfg, a.split, stats, a.batch_size, a.workers)
    scores, labels, paths = score_file_loader(model, ev, scoring_cfg, device=device,
                                              max_files=a.max_files)
    print(f"[graph-eval] scored {len(scores)} files "
          f"({int((labels==0).sum())} normal / {int((labels==1).sum())} abnormal)",
          flush=True)

    # calibrate score polarity on the labels (abnormal turns out MORE predictable
    # on TUAB, so the native "high error = anomaly" direction is inverted)
    oriented, direction, auroc_raw = orient_scores(scores, labels)

    # threshold (computed in the oriented space so higher = abnormal)
    if a.threshold_from == "normal":
        normal_scores = oriented[labels == 0]
        thr = normal_quantile_threshold(normal_scores, a.quantile) \
            if len(normal_scores) else None
    else:
        thr = youden_threshold(oriented, labels) if labels.min() != labels.max() else None

    m = evaluate(oriented, labels, threshold=thr)
    m["auroc_raw"] = auroc_raw
    m["separation"] = separation(auroc_raw) if np.isfinite(auroc_raw) else float("nan")
    m["direction"] = ("native: high latent-error = abnormal" if direction > 0 else
                      "inverted: abnormal is MORE latent-predictable (low error)")
    m["threshold_from"] = a.threshold_from
    m["quantile"] = a.quantile
    print("[graph-eval] " + json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v)
         for k, v in m.items() if k != "confusion_matrix"}), flush=True)
    print(f"[graph-eval] confusion (rows=true 0/1, cols=pred 0/1): {m.get('confusion_matrix')}",
          flush=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"metrics": m,
                       "scores": scores.tolist(), "labels": labels.tolist(),
                       "paths": paths}, fh, indent=2)
        print(f"[graph-eval] wrote {a.out}", flush=True)
    print("GRAPH_JEPA_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
