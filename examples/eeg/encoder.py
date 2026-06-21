"""1D EEG encoder for the geometry-aware EEG-JEPA study.

The encoder maps an EEG window [B, C, T] through a strided Conv1d stack and
global average pooling. The pre-pool feature maps also support the SPD-tangent
regularizer through temporal feature covariances.
"""
import torch.nn as nn


class EEGEncoder1D(nn.Module):
    def __init__(self, n_channels=19, widths=(64, 128, 128, 256), d_model=256,
                 d_cov=32, kernel=7, stride=2):
        super().__init__()
        channels = [n_channels, *widths]
        blocks = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            blocks += [
                nn.Conv1d(input_channels, output_channels, kernel, stride=stride,
                          padding=kernel // 2),
                nn.BatchNorm1d(output_channels),
                nn.GELU(),
            ]
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv1d(widths[-1], d_model, kernel_size=1)
        self.cov_proj = nn.Conv1d(d_model, d_cov, kernel_size=1)
        self.out_dim = d_model
        self.d_cov = d_cov

    def feature_map(self, x):
        """Return pre-pool features with shape [B, d_model, T']."""
        return self.head(self.backbone(x))

    def represent(self, x):
        """Return global-average-pooled representations with shape [B, d_model]."""
        return self.feature_map(x).mean(dim=-1)

    def cov_features(self, x):
        """Return covariance features with shape [B, d_cov, T']."""
        return self.cov_proj(self.feature_map(x))
