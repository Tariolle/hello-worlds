"""1D EEG encoder for the EEG-JEPA track.

Maps an EEG window ``[B, C=19, T]`` to a representation ``[B, D]`` via a strided
``Conv1d`` stack (kernel 7, stride 2, BatchNorm + GELU) that downsamples time,
followed by pooling. This pooled vector is what the downstream probe reads.

Pooling modes (set via pool= argument or pool_type config):
  "mean"   — global average pool (default, baseline)
  "attn"   — learned attention pooler: the model weights time steps by relevance.
              For EEG, this lets the encoder focus on high-activity intervals
              (e.g., spike-wave bursts) rather than averaging them with baseline.

For the geometry-aware (tangent-SPD) regulariser we also expose:
  * feature_map(x) -> [B, d_model, T'] — the pre-pool features
  * cov_features(x) -> [B, d_cov, T']  — low-dim projection for SPD covariance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooler(nn.Module):
    """Learned attention over T' time steps -> [B, D] summary.

    Scores each time step with a small MLP, softmax over time, weighted sum.
    Adds ~D*hidden_dim parameters (tiny vs encoder).
    """
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, kernel_size=1),   # [B, 1, T']
        )

    def forward(self, feat):              # [B, D, T'] -> [B, D]
        w = F.softmax(self.score(feat), dim=-1)   # [B, 1, T']
        return (feat * w).sum(dim=-1)             # [B, D]


class EEGEncoder1D(nn.Module):
    def __init__(self, n_channels=19, widths=(64, 128, 128, 256), d_model=256,
                 d_cov=32, kernel=7, stride=2, pool_type="mean"):
        super().__init__()
        chans = [n_channels, *widths]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [
                nn.Conv1d(cin, cout, kernel, stride=stride, padding=kernel // 2),
                nn.BatchNorm1d(cout),
                nn.GELU(),
            ]
        self.backbone = nn.Sequential(*blocks)
        self.head     = nn.Conv1d(widths[-1], d_model, kernel_size=1)
        self.cov_proj = nn.Conv1d(d_model, d_cov, kernel_size=1)
        self.out_dim  = d_model
        self.d_cov    = d_cov
        self.pool_type = pool_type
        if pool_type == "attn":
            self.pooler = AttentionPooler(d_model)

    def feature_map(self, x):            # [B, C, T] -> [B, d_model, T']
        return self.head(self.backbone(x))

    def represent(self, x):              # [B, C, T] -> [B, d_model]
        fm = self.feature_map(x)
        if self.pool_type == "attn":
            return self.pooler(fm)
        return fm.mean(dim=-1)           # default: global avg pool

    def cov_features(self, x):           # [B, C, T] -> [B, d_cov, T']
        return self.cov_proj(self.feature_map(x))
