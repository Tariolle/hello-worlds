"""1D EEG encoder for the EEG-JEPA track.

Maps an EEG window ``[B, C=19, T]`` to a representation ``[B, D]`` via a strided
``Conv1d`` stack (kernel 7, stride 2, BatchNorm + GELU) that downsamples time,
followed by global average pooling. This pooled vector is what the downstream
probe reads (``represent``).

For the geometry-aware (tangent-SPD) regulariser we also expose:
  * ``feature_map(x) -> [B, d_model, T']`` — the pre-pool features,
  * ``cov_features(x) -> [B, d_cov, T']``  — a low-dim projection from which the
    per-window temporal feature *covariance* (an SPD matrix) is formed in
    ``geometry.tangent_features``. ``d_cov < T'`` keeps that covariance full-rank.
"""
import torch.nn as nn


class EEGEncoder1D(nn.Module):
    def __init__(self, n_channels=19, widths=(64, 128, 128, 256), d_model=256,
                 d_cov=32, kernel=7, stride=2):
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
        self.head = nn.Conv1d(widths[-1], d_model, kernel_size=1)
        self.cov_proj = nn.Conv1d(d_model, d_cov, kernel_size=1)
        self.out_dim = d_model
        self.d_cov = d_cov

    def feature_map(self, x):           # [B, C, T] -> [B, d_model, T']
        return self.head(self.backbone(x))

    def represent(self, x):             # [B, C, T] -> [B, d_model]
        return self.feature_map(x).mean(dim=-1)

    def cov_features(self, x):          # [B, C, T] -> [B, d_cov, T']
        return self.cov_proj(self.feature_map(x))
