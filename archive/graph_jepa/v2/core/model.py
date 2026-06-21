"""TCP-Graph-JEPA — masked-latent prediction over the TCP montage graph.

Pipeline (all latent, **no raw-signal reconstruction**):

    x [B, C, T, F]
      -> feature projection      Linear(F -> H)
      -> (context only) replace masked (c,t) tokens by a learnable mask token
      -> temporal encoder        per-channel sequence model over time   [B,C,T,H]
      -> graph encoder           dense relational message passing over the C
                                 nodes at each time step                 [B,C,T,H]

The **context encoder** sees the masked input; a small **predictor** MLP maps its
output at the masked positions to the latent the **target encoder** produces from
the *unmasked* input. The target encoder is an EMA copy (stop-gradient), so the
objective is JEPA-style latent prediction:

    loss = smooth_l1( predictor(context(x_masked))[M] , sg(target(x))[M] )

evaluated only on masked AND available channel-time positions. The same
per-position error is the anomaly signal at inference (see ``scoring.py``).

Dense adjacency message passing is used throughout (no PyTorch-Geometric
dependency); the graph comes from ``tcp_graph.build_dense_adjacency``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcp_graph import TCP_CHANNELS, build_dense_adjacency, graph_metadata


@dataclass
class ModelConfig:
    channels: List[str] = field(default_factory=lambda: list(TCP_CHANNELS))
    feature_dim: int = 5
    hidden_dim: int = 128
    temporal: str = "transformer"   # transformer | gru | tcn
    n_temporal_layers: int = 2
    n_heads: int = 4
    n_graph_layers: int = 2
    dropout: float = 0.1
    max_time: int = 256             # positional-embedding capacity (>= T)
    predictor_hidden: int = 128
    target_mode: str = "ema"        # ema | shared
    ema_decay: float = 0.996
    loss: str = "smooth_l1"         # smooth_l1 | l1 | mse
    use_self_loops: bool = True
    use_contralateral: bool = True
    use_shared: bool = True


# --------------------------------------------------------------------------- #
# Temporal encoder (per channel, over time)
# --------------------------------------------------------------------------- #
class TemporalEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H = cfg.hidden_dim
        self.kind = cfg.temporal
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_time, H))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if cfg.temporal == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=H, nhead=cfg.n_heads, dim_feedforward=2 * H,
                dropout=cfg.dropout, activation="gelu", batch_first=True,
                norm_first=True)
            self.net = nn.TransformerEncoder(layer, num_layers=cfg.n_temporal_layers)
        elif cfg.temporal == "gru":
            self.net = nn.GRU(H, H, num_layers=cfg.n_temporal_layers,
                              batch_first=True, dropout=cfg.dropout
                              if cfg.n_temporal_layers > 1 else 0.0)
        elif cfg.temporal == "tcn":
            convs = []
            for i in range(cfg.n_temporal_layers):
                d = 2 ** i
                convs.append(nn.Conv1d(H, H, kernel_size=3, padding=d, dilation=d))
                convs.append(nn.GELU())
                convs.append(nn.Dropout(cfg.dropout))
            self.net = nn.Sequential(*convs)
        else:
            raise ValueError(f"unknown temporal encoder: {cfg.temporal!r}")

    def forward(self, h):                      # h: [N, T, H]
        T = h.shape[1]
        h = h + self.pos[:, :T, :]
        if self.kind == "transformer":
            return self.net(h)
        if self.kind == "gru":
            out, _ = self.net(h)
            return out
        # tcn: [N,T,H] -> [N,H,T] -> conv -> back
        return self.net(h.transpose(1, 2)).transpose(1, 2)


# --------------------------------------------------------------------------- #
# Relational dense graph layer (one step of message passing over C nodes)
# --------------------------------------------------------------------------- #
class RelationalGraphLayer(nn.Module):
    """h' = LN(h + sum_r A_r @ (h W_r));  then FFN with residual."""

    def __init__(self, H: int, relations: List[str], dropout: float):
        super().__init__()
        self.relations = relations
        self.lin = nn.ModuleDict({r: nn.Linear(H, H, bias=False) for r in relations})
        self.norm1 = nn.LayerNorm(H)
        self.norm2 = nn.LayerNorm(H)
        self.ffn = nn.Sequential(
            nn.Linear(H, 2 * H), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * H, H))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, adj: Dict[str, torch.Tensor]):  # h: [N, C, H]
        msg = 0.0
        for r in self.relations:
            # A_r: [C,C]  ;  (h W_r): [N,C,H]  ;  A_r @ . over the C axis
            msg = msg + torch.einsum("ij,njh->nih", adj[r], self.lin[r](h))
        h = self.norm1(h + self.drop(msg))
        h = self.norm2(h + self.drop(self.ffn(h)))
        return h


class GraphEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        relations = []
        if cfg.use_shared:
            relations.append("shared")
        if cfg.use_contralateral:
            relations.append("contra")
        if cfg.use_self_loops:
            relations.append("self")
        if not relations:
            relations = ["self"]
        self.relations = relations
        self.layers = nn.ModuleList(
            [RelationalGraphLayer(cfg.hidden_dim, relations, cfg.dropout)
             for _ in range(cfg.n_graph_layers)])

    def forward(self, h, adj):                 # h: [B, C, T, H]
        B, C, T, H = h.shape
        # fold time into batch: per-timestep graph propagation
        h = h.permute(0, 2, 1, 3).reshape(B * T, C, H)
        for layer in self.layers:
            h = layer(h, adj)
        return h.reshape(B, T, C, H).permute(0, 2, 1, 3).contiguous()


# --------------------------------------------------------------------------- #
# Encoder = projection -> temporal -> graph
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.temporal = TemporalEncoder(cfg)
        self.graph = GraphEncoder(cfg)
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x, adj, mask=None, mask_token=None):  # x: [B,C,T,F]
        B, C, T, Fdim = x.shape
        h = self.proj(x)                              # [B,C,T,H]
        if mask is not None and mask_token is not None:
            h = torch.where(mask.unsqueeze(-1), mask_token.to(h.dtype), h)
        h = h.reshape(B * C, T, -1)
        h = self.temporal(h).reshape(B, C, T, -1)     # [B,C,T,H]
        h = self.graph(h, adj)                        # [B,C,T,H]
        return self.out_norm(h)


# --------------------------------------------------------------------------- #
# Full JEPA model
# --------------------------------------------------------------------------- #
class TCPGraphJEPA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        adj = build_dense_adjacency(
            cfg.channels, add_self_loops=cfg.use_self_loops,
            contralateral=cfg.use_contralateral, shared=cfg.use_shared)
        # register adjacency as buffers so .to(device)/state_dict carry them
        for k in ("shared", "contra", "self", "combined"):
            self.register_buffer(f"adj_{k}", adj[k], persistent=False)

        self.context_encoder = Encoder(cfg)
        self.mask_token = nn.Parameter(torch.zeros(cfg.hidden_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.predictor_hidden), nn.GELU(),
            nn.LayerNorm(cfg.predictor_hidden),
            nn.Linear(cfg.predictor_hidden, cfg.hidden_dim))

        if cfg.target_mode == "ema":
            self.target_encoder = copy.deepcopy(self.context_encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.target_encoder = None  # "shared": reuse context encoder, no grad

    # -- adjacency dict view -------------------------------------------------
    @property
    def adj(self) -> Dict[str, torch.Tensor]:
        return {"shared": self.adj_shared, "contra": self.adj_contra,
                "self": self.adj_self, "combined": self.adj_combined}

    # -- target embeddings (stop-gradient) ----------------------------------
    @torch.no_grad()
    def target(self, x):
        enc = self.target_encoder if self.target_encoder is not None \
            else self.context_encoder
        was_training = enc.training
        enc.eval()
        z = enc(x, self.adj, mask=None, mask_token=None)
        if was_training and self.target_encoder is None:
            enc.train()
        return z

    # -- context predictions ------------------------------------------------
    def predict(self, x, mask):
        ctx = self.context_encoder(x, self.adj, mask=mask, mask_token=self.mask_token)
        return self.predictor(ctx)

    def forward(self, x, mask, channel_mask=None):
        """Return dict(loss, pred, target, err[B,C,T], valid[B,C,T])."""
        pred = self.predict(x, mask)             # [B,C,T,H]
        tgt = self.target(x).detach()            # [B,C,T,H]

        valid = mask.clone()
        if channel_mask is not None:
            cm = channel_mask.to(device=mask.device, dtype=torch.bool)
            if cm.dim() == 1:
                cm = cm.unsqueeze(0).expand(mask.shape[0], -1)
            valid = valid & cm.unsqueeze(-1)

        err = self._err(pred, tgt)               # [B,C,T] per-position error
        sel = valid
        if sel.any():
            loss = err[sel].mean()
        else:                                    # degenerate batch (no targets)
            loss = (pred.sum() * 0.0)
        return {"loss": loss, "pred": pred, "target": tgt, "err": err, "valid": valid}

    def _err(self, pred, tgt):
        if self.cfg.loss == "l1":
            return (pred - tgt).abs().mean(dim=-1)
        if self.cfg.loss == "mse":
            return ((pred - tgt) ** 2).mean(dim=-1)
        # smooth_l1 (Huber), reduced over the hidden dim
        return F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=-1)

    # -- EMA update of the target encoder -----------------------------------
    @torch.no_grad()
    def update_target(self, decay: Optional[float] = None):
        if self.target_encoder is None:
            return
        d = self.cfg.ema_decay if decay is None else decay
        for pt, pc in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            pt.mul_(d).add_(pc.detach(), alpha=1.0 - d)
        for bt, bc in zip(self.target_encoder.buffers(),
                          self.context_encoder.buffers()):
            bt.copy_(bc)

    # -- serialisable description -------------------------------------------
    def metadata(self) -> dict:
        return {"graph": graph_metadata(self.cfg.channels),
                "config": vars(self.cfg).copy()}


def build_model(cfg: ModelConfig) -> TCPGraphJEPA:
    return TCPGraphJEPA(cfg)
