"""EEG-JEPA SSL pretraining on TUEV — same loop as main.py but uses TUEVDataset."""
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.tuev_dataset import TUEVConfig, TUEVDataset
from eb_jepa.losses import VICRegLoss, BCS
from examples.eeg.encoder import EEGEncoder1D
from examples.eeg.geometry import tangent_features, collapse_metrics
from examples.eeg.peira import PEIRALoss
from examples.eeg.main import build_encoder, Projector, SSLModule, build_ssl


def make_tuev_loader(cfg_data: dict, split: str = "train"):
    tuev_cfg = TUEVConfig(**{k: v for k, v in cfg_data.items()
                              if k in TUEVConfig.__dataclass_fields__})
    tuev_cfg.split = split
    tuev_cfg.mode = "ssl"
    ds = TUEVDataset(tuev_cfg)
    return torch.utils.data.DataLoader(
        ds, batch_size=tuev_cfg.batch_size, shuffle=True,
        num_workers=tuev_cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=tuev_cfg.num_workers > 0,
    )


def run(fname):
    import json, time
    cfg = OmegaConf.load(fname)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    loader = make_tuev_loader(data_dict, split="train")

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr,
                             weight_decay=cfg.optim.weight_decay)

    ckpt_dir = cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "train_log.json")
    history = []

    for epoch in range(cfg.optim.epochs):
        ssl.train()
        losses, t0 = [], time.time()
        for batch in loader:
            batch = [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
            losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        elapsed = time.time() - t0
        row = {"epoch": epoch, "loss": round(mean_loss, 5), "elapsed_s": round(elapsed, 1),
               **{k: round(v.item(), 5) if torch.is_tensor(v) else v
                  for k, v in logs.items()}}
        history.append(row)
        print(f"[tuev] epoch {epoch:3d}  loss={mean_loss:.4f}  {elapsed:.0f}s  {logs}",
              flush=True)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))

    print(f"[tuev] done -> {ckpt_dir}/latest.pth.tar  |  log -> {log_path}")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/train_tuev.yaml"
    run(fname)
