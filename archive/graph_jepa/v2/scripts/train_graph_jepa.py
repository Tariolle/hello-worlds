"""TCP-Graph-JEPA — self-supervised pretraining entrypoint.

Masked-latent JEPA on (by default) NORMAL EEG windows only. Each step masks
channel-time tokens of a window ``[B,22,T,F]``, predicts the masked latents from
the visible spatial-temporal graph context, and regresses them onto an EMA
target encoder's latents (stop-gradient). No labels are used in the SSL loss.

Run:
    python -m archive.graph_jepa.v2.scripts.train_graph_jepa \
        --config archive/graph_jepa/v2/config.yaml \
        data.data_root=<TUAB_PREPROCESSED>

A checkpoint with model weights, config, feature-normalisation stats and graph
metadata is written to ``meta.ckpt_dir``.
"""
import argparse
import os
import time

import numpy as np
import torch

from archive.graph_jepa.v2.core.config import apply_overrides, bind, load_yaml
from archive.graph_jepa.v2.core.features import fit_feature_stats, FeatureStats
from archive.graph_jepa.v2.core.masking import MaskConfig, make_mask
from archive.graph_jepa.v2.core.metrics import evaluate, separation
from archive.graph_jepa.v2.core.model import ModelConfig, build_model
from archive.graph_jepa.v2.core.scoring import ScoringConfig, score_file_loader
from archive.graph_jepa.v2.core.windows import (GraphEEGConfig, GraphEEGDataset,
                                        make_graph_loader)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_stats(data_cfg: GraphEEGConfig, n_windows: int = 512) -> FeatureStats:
    """Fit per-(channel,band) normalisation on raw (unstandardised) windows."""
    raw_cfg = GraphEEGConfig(**{**vars(data_cfg)})
    ds = GraphEEGDataset(raw_cfg, stats=None)
    feats = []
    n = min(n_windows, len(ds))
    for i in range(n):
        item = ds[i] if raw_cfg.mode == "ssl" else None
        x = item[0] if raw_cfg.mode == "ssl" else None
        if x is None:
            break
        feats.append(x.numpy())
    feats = np.stack(feats, axis=0)            # [n, C, T, F]
    return fit_feature_stats(feats)


def build_eval_loader(data_cfg: GraphEEGConfig, stats):
    """A small labelled eval loader (normal vs abnormal) for in-training AUROC."""
    ev = GraphEEGConfig(**{**vars(data_cfg)})
    ev.split, ev.mode, ev.batch_size = "eval", "file", 4
    ev.num_workers = min(4, data_cfg.num_workers)
    try:
        loader, _ = make_graph_loader(ev, stats=stats, shuffle=False)
        return loader
    except Exception as e:                      # eval split may be absent
        print(f"[graph-jepa] no eval loader ({e})", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="archive/graph_jepa/v2/config.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*", help="dotlist overrides e.g. model.hidden_dim=64")
    a = ap.parse_args()

    cfg = apply_overrides(load_yaml(a.config), a.overrides)
    meta = cfg.get("meta", {})
    seed = int(meta.get("seed", 1))
    set_seed(seed)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_cfg = bind(GraphEEGConfig, cfg.get("data", {}))
    data_cfg.mode = "ssl"
    model_cfg = bind(ModelConfig, cfg.get("model", {}))
    # keep feature_dim / channels consistent with the data config
    fc = data_cfg.feature_cfg()
    model_cfg.feature_dim = fc.n_bands
    model_cfg.channels = list(data_cfg.channels)
    model_cfg.max_time = max(model_cfg.max_time, fc.n_frames)
    mask_cfg = bind(MaskConfig, cfg.get("mask", {}))
    optim_cfg = cfg.get("optim", {})
    scoring_cfg = bind(ScoringConfig, cfg.get("scoring", {}))

    print(f"[graph-jepa] device={device} seed={seed} "
          f"C={len(model_cfg.channels)} T={fc.n_frames} F={fc.n_bands}", flush=True)

    # ---- feature normalisation (train-only) -------------------------------
    if data_cfg.source == "edf":
        print("[graph-jepa] fitting feature stats on normal windows ...", flush=True)
    stats = fit_stats(data_cfg, n_windows=int(meta.get("stat_windows", 512)))

    loader, _ = make_graph_loader(data_cfg, stats=stats)
    model = build_model(model_cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(optim_cfg.get("lr", 1e-3)),
                            weight_decay=float(optim_cfg.get("weight_decay", 1e-5)))
    n_param = sum(p.numel() for p in params)
    print(f"[graph-jepa] trainable params: {n_param/1e6:.2f}M", flush=True)

    eval_loader = build_eval_loader(data_cfg, stats) if meta.get("eval_during", True) else None
    ckpt_dir = meta.get("ckpt_dir", "./checkpoints/graph_jepa")
    os.makedirs(ckpt_dir, exist_ok=True)
    epochs = int(optim_cfg.get("epochs", 15))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ema_decay = float(model_cfg.ema_decay)

    for epoch in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for x, cm in loader:
            x, cm = x.to(device), cm.to(device)
            B, C, T, _ = x.shape
            mask = make_mask(B, C, T, mask_cfg, channel_mask=cm, generator=gen,
                             device="cpu", channels=model_cfg.channels).to(device)
            out = model(x, mask, channel_mask=cm)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            model.update_target(ema_decay)
            losses.append(out["loss"].detach().item())
        # latent-variance diagnostic (collapse -> 0): std of target latents on
        # the last batch's valid positions
        with torch.no_grad():
            v = out["valid"]
            tgt_std = float(out["target"][v].std()) if v.any() else 0.0
        msg = (f"[graph-jepa] epoch {epoch} loss={np.mean(losses):.4f} "
               f"tgt_std={tgt_std:.3f} ({time.time()-t0:.1f}s, {len(losses)} steps)")
        if eval_loader is not None and (epoch % int(meta.get("eval_every", 5)) == 0
                                        or epoch == epochs - 1):
            scores, labels, _ = score_file_loader(
                model, eval_loader, scoring_cfg, device=device,
                max_files=int(meta.get("eval_max_files", 80)))
            if len(scores) and labels.min() != labels.max():
                m = evaluate(scores, labels)
                msg += (f" | eval AUROC={m['auroc']:.4f} (separability="
                        f"{separation(m['auroc']):.4f}) AUPRC={m['auprc']:.4f} (n={m['n']})")
        print(msg, flush=True)

        torch.save({
            "model": model.state_dict(),
            "model_cfg": vars(model_cfg),
            "data_cfg": vars(data_cfg),
            "mask_cfg": vars(mask_cfg),
            "scoring_cfg": vars(scoring_cfg),
            "feature_stats": stats.to_dict(),
            "graph_meta": model.metadata()["graph"],
            "epoch": epoch,
        }, os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[graph-jepa] done -> {ckpt_dir}/latest.pth.tar\nGRAPH_JEPA_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
