"""EEG World Model — training entrypoint for TUEV.

3-geometry JEPA: for each 1-second EEG window, predict the NEXT second in
Euclidean ambient space, Log-Euclidean tangent space, and Riemannian
(affine-invariant) SPD space simultaneously.

Run:
  python -m examples.eeg.main_tuev_wm --fname examples/eeg/cfgs/train_tuev_wm.yaml
"""
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.tuev_dataset import TUEVConfig, TUEVDataset
from examples.eeg.main import build_encoder
from examples.eeg.world_model import EEGWorldModel


def make_wm_loader(cfg_data, cfg_model):
    fields = TUEVConfig.__dataclass_fields__
    tuev_cfg = TUEVConfig(**{k: v for k, v in OmegaConf.to_container(cfg_data, resolve=True).items()
                             if k in fields})
    tuev_cfg.mode = "wm_ssl"
    ds = TUEVDataset(tuev_cfg)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=tuev_cfg.batch_size,
        num_workers=tuev_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=tuev_cfg.num_workers > 0,
    )


def run(fname="examples/eeg/cfgs/train_tuev_wm.yaml", cfg=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(
                cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    loader = make_wm_loader(cfg.data, cfg.model)
    encoder = build_encoder(cfg.model).to(device)
    model = EEGWorldModel(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "train_log.json")
    history = []

    for epoch in range(cfg.optim.epochs):
        model.train()
        losses, t0 = [], time.time()

        for x_past, x_future in loader:
            x_past = x_past.to(device, non_blocking=True)
            x_future = x_future.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, logs = model.compute_loss((x_past, x_future))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        elapsed = time.time() - t0
        row = {"epoch": epoch, "loss": round(mean_loss, 5), "elapsed_s": round(elapsed, 1),
               **{k: (round(v.item(), 5) if torch.is_tensor(v) else round(v, 5) if isinstance(v, float) else v)
                  for k, v in logs.items()}}
        history.append(row)
        print(f"[wm] epoch {epoch:3d}  loss={mean_loss:.4f}  {elapsed:.0f}s  {logs}", flush=True)

        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        torch.save(
            {"epoch": epoch, "encoder": encoder.state_dict(),
             "cfg": OmegaConf.to_container(cfg, resolve=True)},
            os.path.join(ckpt_dir, "latest.pth.tar"),
        )

    print(f"[wm] done -> {ckpt_dir}/latest.pth.tar  |  log -> {log_path}")


if __name__ == "__main__":
    fname = (sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv
             else "examples/eeg/cfgs/train_tuev_wm.yaml")
    run(fname=fname)
