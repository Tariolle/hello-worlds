"""Fourier (spectral) 1D EEG encoder for the EEG-JEPA track.

A drop-in replacement for ``EEGEncoder1D`` (see encoder.py) that front-ends the
network with a Short-Time Fourier Transform instead of a strided-conv time
stack. Motivation: TUAB abnormality is *band-power-driven* (see
``cfgs/benchmark.yaml`` and the Riemannian-baseline note there), and for a short
window a single STFT exposes the full per-band spectral content in one shot,
whereas a conv stack has to grow a deep receptive field to see the same global
periodic structure. This makes a Fourier front-end a natural fit for short
timed signals.

Same downstream contract as ``EEGEncoder1D`` so it slots into the existing SSL
loop, probe and benchmark unchanged:
  * ``feature_map(x) -> [B, d_model, T']`` — pre-pool time-frequency features,
  * ``represent(x)   -> [B, d_model]``     — global-average-pooled over T'
    (this is what the frozen probe reads),
  * ``cov_features(x)-> [B, d_cov, T']``    — low-dim projection from which the
    per-window temporal feature *covariance* (an SPD matrix) is formed in
    ``geometry.tangent_features``. ``d_cov < T'`` keeps that covariance
    full-rank, so the default STFT (n_fft=128, hop=32 over T=2000 -> T'=63)
    leaves headroom over the default ``d_cov=32``.

Pipeline:  x [B, C, T]
  --STFT--> log1p power spectrogram        [B, C, F, T_frames]
  --flatten (C, F)-->                      [B, C*F, T_frames]
  --1x1 conv (learnable spectral mixing)-> [B, d_hidden, T_frames]   (BN + GELU)
  --temporal conv blocks (kernel 3)-->     [B, d_hidden, T']
  --head / cov_proj 1x1-->                 [B, d_model, T'] / [B, d_cov, T']
``T' = T_frames`` (no temporal striding) so the spectral resolution is preserved
for the probe and the SPD tangent path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEEGEncoder1D(nn.Module):
    def __init__(self, n_channels=19, n_fft=128, hop=32, d_model=256, d_cov=32,
                 d_hidden=256, n_temporal_layers=2, kernel=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_fft = n_fft
        self.hop = hop
        # Hann window kept as a buffer so .to(device) / state_dict move it along.
        self.register_buffer("window", torch.hann_window(n_fft))
        n_freq = n_fft // 2 + 1
        # Learnable mixing over (channel, frequency) pairs at each STFT frame: a
        # 1x1 conv over the flattened C*F axis learns a spectral filterbank that
        # combines bands across channels.
        self.spectral_mix = nn.Conv1d(n_channels * n_freq, d_hidden, kernel_size=1)
        self.stem_norm = nn.BatchNorm1d(d_hidden)
        blocks = []
        for _ in range(n_temporal_layers):
            blocks += [
                nn.Conv1d(d_hidden, d_hidden, kernel, padding=kernel // 2),
                nn.BatchNorm1d(d_hidden),
                nn.GELU(),
            ]
        self.temporal = nn.Sequential(*blocks)
        self.head = nn.Conv1d(d_hidden, d_model, kernel_size=1)
        self.cov_proj = nn.Conv1d(d_model, d_cov, kernel_size=1)
        self.out_dim = d_model
        self.d_cov = d_cov

    def _spectrogram(self, x):          # [B, C, T] -> [B, C*F, T_frames]
        b, c, t = x.shape
        xf = x.reshape(b * c, t)
        spec = torch.stft(
            xf, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft,
            window=self.window, center=True, return_complex=True)  # [b*c, F, frames]
        # |S|^2 via real/imag (no sqrt) -> stable backward, then log1p compresses
        # the heavy-tailed power scale into a probe-friendly range.
        power = spec.real ** 2 + spec.imag ** 2
        logp = torch.log1p(power)
        n_freq, frames = logp.shape[-2], logp.shape[-1]
        return logp.reshape(b, c * n_freq, frames)

    def feature_map(self, x):           # [B, C, T] -> [B, d_model, T']
        h = self._spectrogram(x)
        h = F.gelu(self.stem_norm(self.spectral_mix(h)))
        h = self.temporal(h)
        return self.head(h)

    def represent(self, x):             # [B, C, T] -> [B, d_model]
        return self.feature_map(x).mean(dim=-1)

    def cov_features(self, x):          # [B, C, T] -> [B, d_cov, T']
        return self.cov_proj(self.feature_map(x))
