"""TCP-Graph-JEPA model for EEG anomaly detection.

The model predicts masked latent embeddings from visible channel-time graph
context. It does not reconstruct raw EEG.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from archive.graph_jepa.v1.src.graphs.tcp_graph import CONTRALATERAL_PAIRS, TCP_CHANNELS, dense_adjacency


@dataclass
class TCPGraphJEPAConfig:
    channels: int = 22
    time_steps: int = 70
    feature_dim: int = 5
    hidden_dim: int = 128
    temporal_layers: int = 2
    temporal_heads: int = 4
    graph_layers: int = 2
    dropout: float = 0.1
    mask_ratio: float = 0.25
    mask_mode: str = "mixed"
    loss: str = "smooth_l1"
    max_time_steps: int = 256
    tcp_channels: list[str] | None = None


def _pair_indices(channels: list[str]) -> list[tuple[int, int]]:
    idx = {name: i for i, name in enumerate(channels)}
    return [(idx[a], idx[b]) for a, b in CONTRALATERAL_PAIRS if a in idx and b in idx]


class ChannelTimeMasker:
    """Create BoolTensor masks over ``[B, C, T]`` channel-time tokens."""

    def __init__(
        self,
        channels: list[str] | None = None,
        mask_ratio: float = 0.25,
        mode: str = "mixed",
        span_min: int = 4,
        span_max: int = 16,
    ):
        self.channels = list(TCP_CHANNELS if channels is None else channels)
        self.mask_ratio = float(mask_ratio)
        self.mode = mode
        self.span_min = int(span_min)
        self.span_max = int(span_max)
        self.pairs = _pair_indices(self.channels)

    def __call__(
        self,
        batch_size: int,
        channels: int,
        time_steps: int,
        device: torch.device | str,
        mode: str | None = None,
        mask_ratio: float | None = None,
    ) -> torch.BoolTensor:
        mode = self.mode if mode is None else mode
        ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        if mode == "mixed":
            mode = random.choice(["random", "channel_span", "contralateral"])
        if mode == "random":
            mask = torch.rand(batch_size, channels, time_steps, device=device) < ratio
        elif mode == "channel_span":
            mask = self._channel_span(batch_size, channels, time_steps, device, ratio)
        elif mode == "contralateral":
            mask = self._contralateral(batch_size, channels, time_steps, device, ratio)
        elif mode == "all":
            mask = torch.ones(batch_size, channels, time_steps, device=device, dtype=torch.bool)
        else:
            raise ValueError(f"unknown mask mode: {mode!r}")
        if not mask.any():
            mask[:, 0, 0] = True
        return mask

    def _span(self, time_steps: int) -> tuple[int, int]:
        hi = max(1, min(self.span_max, time_steps))
        lo = max(1, min(self.span_min, hi))
        length = random.randint(lo, hi)
        start = random.randint(0, max(0, time_steps - length))
        return start, start + length

    def _channel_span(self, batch_size, channels, time_steps, device, ratio):
        mask = torch.zeros(batch_size, channels, time_steps, device=device, dtype=torch.bool)
        target = max(1, int(round(channels * time_steps * ratio)))
        for b in range(batch_size):
            filled = 0
            while filled < target:
                ch = random.randrange(channels)
                s, e = self._span(time_steps)
                before = int(mask[b].sum().item())
                mask[b, ch, s:e] = True
                filled += int(mask[b].sum().item()) - before
        return mask

    def _contralateral(self, batch_size, channels, time_steps, device, ratio):
        mask = torch.zeros(batch_size, channels, time_steps, device=device, dtype=torch.bool)
        pairs = self.pairs or [(i, min(i + 1, channels - 1)) for i in range(0, channels, 2)]
        target = max(1, int(round(channels * time_steps * ratio)))
        for b in range(batch_size):
            filled = 0
            while filled < target:
                left, right = random.choice(pairs)
                ch = left if random.random() < 0.5 else right
                s, e = self._span(time_steps)
                before = int(mask[b].sum().item())
                mask[b, ch, s:e] = True
                filled += int(mask[b].sum().item()) - before
        return mask


class GraphBlock(nn.Module):
    """Dense adjacency graph message passing over channels at every time step."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("ij,bjth->bith", adj.to(h.device), h)
        update = self.msg(self.norm(h + agg))
        return h + self.dropout(update)


class TCPGraphJEPA(nn.Module):
    def __init__(self, cfg: TCPGraphJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.channels = list(cfg.tcp_channels or TCP_CHANNELS)
        if len(self.channels) != cfg.channels:
            raise ValueError(f"cfg.channels={cfg.channels} but got {len(self.channels)} names")
        self.feature_proj = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, cfg.max_time_steps, cfg.hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.temporal_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=cfg.temporal_layers)
        self.graph_blocks = nn.ModuleList(
            [GraphBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.graph_layers)]
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.masker = ChannelTimeMasker(
            channels=self.channels,
            mask_ratio=cfg.mask_ratio,
            mode=cfg.mask_mode,
        )
        self.register_buffer("adj", dense_adjacency(self.channels), persistent=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    @classmethod
    def from_dict(cls, data: dict) -> "TCPGraphJEPA":
        cfg = TCPGraphJEPAConfig(**data)
        return cls(cfg)

    def make_mask(
        self,
        batch_size: int,
        time_steps: int,
        device: torch.device | str,
        mode: str | None = None,
        mask_ratio: float | None = None,
    ) -> torch.BoolTensor:
        return self.masker(
            batch_size,
            self.cfg.channels,
            time_steps,
            device=device,
            mode=mode,
            mask_ratio=mask_ratio,
        )

    def encode(self, x: torch.Tensor, mask: torch.BoolTensor | None = None) -> torch.Tensor:
        """Encode feature windows into latent ``[B,C,T,H]`` tokens."""
        if x.ndim != 4:
            raise ValueError(f"expected x [B,C,T,F], got {tuple(x.shape)}")
        bsz, channels, time_steps, feat_dim = x.shape
        if channels != self.cfg.channels or feat_dim != self.cfg.feature_dim:
            raise ValueError(
                f"expected [B,{self.cfg.channels},T,{self.cfg.feature_dim}], got {tuple(x.shape)}"
            )
        if time_steps > self.cfg.max_time_steps:
            raise ValueError(
                f"time_steps={time_steps} exceeds max_time_steps={self.cfg.max_time_steps}"
            )
        h = self.feature_proj(x)
        h = h + self.pos_embed[:, :, :time_steps, :]
        if mask is not None:
            h = h.clone()
            h[mask] = self.mask_token.to(h.dtype)
        h = h.reshape(bsz * channels, time_steps, self.cfg.hidden_dim)
        h = self.temporal(h)
        h = h.reshape(bsz, channels, time_steps, self.cfg.hidden_dim)
        for block in self.graph_blocks:
            h = block(h, self.adj)
        return h

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x, mask=None)

    def predict_context(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return self.predictor(self.encode(x, mask=mask))

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor | None = None) -> dict:
        if mask is None:
            mask = self.make_mask(x.shape[0], x.shape[2], x.device)
        with torch.no_grad():
            target = self.encode_target(x)
        pred = self.predict_context(x, mask)
        return {"pred": pred, "target": target.detach(), "mask": mask}

    def compute_loss(self, x: torch.Tensor, mask: torch.BoolTensor | None = None) -> tuple[torch.Tensor, dict]:
        out = self.forward(x, mask=mask)
        pred = out["pred"][out["mask"]]
        target = out["target"][out["mask"]]
        if pred.numel() == 0:
            raise RuntimeError("empty JEPA mask")
        if self.cfg.loss == "l1":
            loss = F.l1_loss(pred, target)
        elif self.cfg.loss == "l2":
            loss = F.mse_loss(pred, target)
        elif self.cfg.loss == "smooth_l1":
            loss = F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"unknown loss: {self.cfg.loss!r}")
        logs = {
            "loss": float(loss.detach().cpu()),
            "mask_ratio": float(out["mask"].float().mean().detach().cpu()),
        }
        return loss, logs
