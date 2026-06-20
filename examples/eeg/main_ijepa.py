"""EEG I-JEPA — latent-predictive SSL with EMA target encoder.

Why this is different from the current SIGReg training:
  Current approach: two augmented views → encoder → projector → BCS regularizer.
    This is regularized contrastive learning, NOT a JEPA.
  I-JEPA: masked context → encoder → predictor → *predict target representation*.
    The target representation comes from a DIFFERENT (EMA) encoder applied to the
    lightly-augmented full signal. The predictor must fill in the masked future —
    this forces truly semantic representation, not just augmentation invariance.

Architecture (follows BYOL + I-JEPA ideas):
  Online network:  enc_online → proj_online → predictor → z_pred  (all trained via SGD)
  Target network:  enc_target → proj_target → z_tgt               (EMA of online, stop-grad)

  Loss = MSE(z_pred, z_tgt.detach())                              (latent prediction)
       + BCS(z_proj_online_v1, z_proj_online_v2)                  (anti-collapse safety net)

The EMA asymmetry alone prevents collapse (BYOL result), but BCS is kept as a
safety net since EEG batches are small (B=256) and BYOL can be finicky.

Masking: on top of the existing light augmentations (noise, scale, chan-drop) we
apply an *additional* heavy temporal mask (mask_ratio=0.4 by default) to the
context view only. The target view keeps only the light augmentations.

LR schedule: cosine decay with linear warmup.
EMA momentum: annealed from ema_start (0.996) toward 1.0 following a cosine
schedule (so it moves slowly early, then barely at all late in training).
"""
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader
from eb_jepa.losses import BCS
from examples.eeg.combined_dataset import make_combined_loader
from examples.eeg.encoder import EEGEncoder1D
from examples.eeg.freq_aug import random_freq_aug
from examples.eeg.geometry import collapse_metrics
from examples.eeg.main import Projector, build_encoder


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class Predictor(nn.Module):
    """Small MLP: proj output (online) → predicted target proj output."""

    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


def apply_temporal_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Zero out a random contiguous block covering ~mask_ratio of the time axis.

    x: [B, C, T] — already on the right device.
    Returns a new tensor (does not modify x in-place).
    """
    B, C, T = x.shape
    n_mask = int(T * mask_ratio)
    if n_mask == 0:
        return x
    out = x.clone()
    starts = torch.randint(0, max(1, T - n_mask + 1), (B,))
    for b, s in enumerate(starts.tolist()):
        out[b, :, s:s + n_mask] = 0.0
    return out


def cosine_lr(base_lr, epoch, total_epochs, warmup_epochs=5, min_lr=1e-6):
    """Cosine schedule with linear warmup. Returns the lr for this epoch."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    t = epoch - warmup_epochs
    T = total_epochs - warmup_epochs
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t / T))


def cosine_ema_momentum(m_start, epoch, total_epochs):
    """Anneal EMA momentum from m_start toward 1.0 over training."""
    return 1.0 - (1.0 - m_start) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2


# --------------------------------------------------------------------------- #
# I-JEPA module
# --------------------------------------------------------------------------- #

class EEGIJEPAModule(nn.Module):
    """Online + target network with EMA. Loss = latent prediction + BCS anti-collapse."""

    def __init__(self, encoder, cfg):
        super().__init__()
        s = cfg.ssl
        self.mask_ratio      = s.get("mask_ratio", 0.4)
        self.freq_mask_p     = s.get("freq_mask_p", 0.0)
        self.freq_mask_bands = s.get("freq_mask_bands", 1)
        self.single_view     = s.get("single_view", False)

        d = encoder.out_dim
        proj_spec = s.get("proj", "512-512-256")

        # ── Online network ────────────────────────────────────────────────────
        self.enc_online = encoder
        self.proj_online = Projector(d, proj_spec)
        p = self.proj_online.out_dim
        self.predictor = Predictor(p, hidden=s.get("pred_hidden", 512))

        # ── Target network (EMA, no grad) ─────────────────────────────────────
        self.enc_target  = copy.deepcopy(encoder)
        self.proj_target = copy.deepcopy(self.proj_online)
        for param in list(self.enc_target.parameters()) + list(self.proj_target.parameters()):
            param.requires_grad_(False)

        # ── Anti-collapse safety net on online projector outputs ──────────────
        self.bcs = BCS(
            num_slices=s.get("num_slices", 256),
            lmbd=s.get("lmbd", 10.0),
        )
        self.bcs_weight = s.get("bcs_weight", 1.0)

    @torch.no_grad()
    def ema_update(self, momentum: float):
        """Update target network as EMA of online network."""
        for p_o, p_t in zip(self.enc_online.parameters(), self.enc_target.parameters()):
            p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)
        for p_o, p_t in zip(self.proj_online.parameters(), self.proj_target.parameters()):
            p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)

    def compute_loss(self, batch, return_extras=False):
        v1, v2 = batch  # both lightly augmented by the dataloader

        # Context view: temporal masking + optional frequency-band masking
        v_ctx = apply_temporal_mask(v1, self.mask_ratio)
        if self.freq_mask_p > 0:
            v_ctx = random_freq_aug(v_ctx, sfreq=200.0,
                                    p_mask=self.freq_mask_p,
                                    n_bands=self.freq_mask_bands)

        if self.single_view:
            # ── Single-view: predict clean v1 from masked v1 (same window).
            # No second augmented view; collapse prevention from EMA asymmetry only.
            z_ctx  = self.proj_online(self.enc_online.represent(v_ctx))   # [B, p]
            z_pred = self.predictor(z_ctx)
            with torch.no_grad():
                z_tgt = self.proj_target(self.enc_target.represent(v1))   # clean v1

            pred_loss = F.mse_loss(z_pred, z_tgt.detach())
            loss = pred_loss
            logs = {
                "pred_loss": round(pred_loss.item(), 5),
                "bcs_loss":  0.0,
                **collapse_metrics(z_ctx.detach()),
            }
        else:
            # ── Two-view (default): predict EMA(v2) from masked v1.
            # BCS on (z_ctx, z_v2) as anti-collapse safety net.
            z1 = self.proj_online(self.enc_online.represent(v_ctx))   # [B, p]
            z2 = self.proj_online(self.enc_online.represent(v2))      # [B, p]
            z_pred = self.predictor(z1)
            with torch.no_grad():
                z_tgt = self.proj_target(self.enc_target.represent(v2))

            pred_loss = F.mse_loss(z_pred, z_tgt.detach())
            bcs_out   = self.bcs(z1, z2)
            bcs_loss  = bcs_out["loss"]
            loss = pred_loss + self.bcs_weight * bcs_loss
            logs = {
                "pred_loss": round(pred_loss.item(), 5),
                "bcs_loss":  round(bcs_loss.item(), 5),
                **{k: round(v.item() if torch.is_tensor(v) else v, 5)
                   for k, v in bcs_out.items() if k != "loss"},
                **collapse_metrics(z1.detach()),
            }

        return loss, logs


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def run(fname="examples/eeg/cfgs/train_ijepa.yaml", cfg=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(
                [f"{k}={v}" for k, v in overrides.items()]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    if cfg.data.get("tuev_root", None):
        loader = make_combined_loader(cfg)
    else:
        dcfg = EEGConfig(**OmegaConf.to_container(cfg.data, resolve=True))
        dcfg.mode = "ssl"
        loader = make_loader(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    model   = EEGIJEPAModule(encoder, cfg.model).to(device)

    # Only online parameters require grad
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr,
                             weight_decay=cfg.optim.weight_decay)

    ckpt_dir = cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "train_log.json")

    total_epochs   = cfg.optim.epochs
    warmup_epochs  = cfg.optim.get("warmup_epochs", 5)
    ema_start      = cfg.optim.get("ema_momentum", 0.996)

    history = []
    for epoch in range(total_epochs):
        model.train()

        # LR schedule
        lr = cosine_lr(cfg.optim.lr, epoch, total_epochs, warmup_epochs)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # EMA momentum schedule
        ema_m = cosine_ema_momentum(ema_start, epoch, total_epochs)

        epoch_losses, t0 = [], time.time()
        for batch in loader:
            batch = [b.to(device, non_blocking=True) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = model.compute_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            model.ema_update(ema_m)
            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        elapsed   = time.time() - t0

        row = {
            "epoch": epoch,
            "loss":  round(mean_loss, 5),
            "lr":    round(lr, 7),
            "ema_m": round(ema_m, 6),
            "elapsed_s": round(elapsed, 1),
            **{k: (round(v.item(), 5) if torch.is_tensor(v) else
                   round(v, 5) if isinstance(v, float) else v)
               for k, v in logs.items()},
        }
        history.append(row)
        print(f"[ijepa] epoch {epoch:3d}  loss={mean_loss:.4f}  "
              f"lr={lr:.2e}  ema_m={ema_m:.4f}  {elapsed:.0f}s  "
              f"eff_rank={logs.get('eff_rank', '?')}  pred={logs.get('pred_loss','?')}",
              flush=True)

        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        }, os.path.join(ckpt_dir, "latest.pth.tar"))

    print(f"[ijepa] done -> {ckpt_dir}/latest.pth.tar  |  log -> {log_path}")


if __name__ == "__main__":
    fname = (sys.argv[sys.argv.index("--fname") + 1]
             if "--fname" in sys.argv
             else "examples/eeg/cfgs/train_ijepa.yaml")
    run(fname=fname)
