"""Time-resolved log-bandpower features for TCP-Graph-JEPA.

A bipolar EEG window ``[C, W]`` (C channels, W samples) is turned into a
channel x time x band feature tensor ``[C, T, F]``:

  * the window is split into ``T`` short frames (default 0.1 s -> 70 frames for a
    7 s window), and for each frame and each frequency band we store
    ``log(mean band power in that frame)``.

Band power is computed with the **filter-envelope** method by default: the whole
window is band-pass filtered once per band (zero-phase Butterworth), squared to
get instantaneous power, then averaged inside each frame. This is robust at a
0.1 s frame resolution where a per-frame FFT would be hopeless for the delta band
(a 20-sample frame cannot resolve 0.5 Hz). A ``welch`` method is also provided.

Only NumPy + SciPy are required (both in the cluster venv and locally).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, iirnotch, sosfiltfilt, welch, tf2sos
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    _HAVE_SCIPY = False


# Canonical band edges (Hz). gamma capped at 70 Hz (the configured bandpass top).
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 70.0),
}


@dataclass
class FeatureConfig:
    sfreq: int = 200
    window_sec: float = 7.0
    frame_sec: float = 0.1
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BANDS))
    method: str = "filter"          # filter (envelope) | welch
    notch_hz: Optional[float] = 60.0  # line-noise notch (US data); None to disable
    log_eps: float = 1e-8
    filter_order: int = 4

    @property
    def window(self) -> int:
        return int(round(self.window_sec * self.sfreq))

    @property
    def frame(self) -> int:
        return max(1, int(round(self.frame_sec * self.sfreq)))

    @property
    def n_frames(self) -> int:
        return self.window // self.frame

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def band_names(self) -> List[str]:
        return list(self.bands.keys())


def _bandpass_sos(lo: float, hi: float, sfreq: int, order: int):
    nyq = 0.5 * sfreq
    hi = min(hi, nyq * 0.999)
    lo = max(lo, 1e-3)
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


def _notch_sos(freq: float, sfreq: int, q: float = 30.0):
    b, a = iirnotch(freq / (0.5 * sfreq), q)
    return tf2sos(b, a)


def log_bandpower(x: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """``x: [C, W]`` (float) -> ``[C, T, F]`` log mean band-power per frame.

    The window is right-trimmed to ``T*frame`` samples so it reshapes cleanly.
    """
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for log_bandpower (pip install scipy)")
    x = np.asarray(x, dtype=np.float64)
    # Sanitize: a single NaN/Inf sample is otherwise smeared across the whole
    # channel by the zero-phase filter (real TUH/TUAB EDFs occasionally contain
    # NaN / clipped segments). Replace with 0 so the channel degrades to ~log(eps).
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    C = x.shape[0]
    T, fr = cfg.n_frames, cfg.frame
    usable = T * fr
    if x.shape[1] < usable:  # pad short windows with edge values (rare)
        pad = usable - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad)), mode="edge")
    x = x[:, :usable]

    if cfg.notch_hz is not None and cfg.notch_hz < 0.5 * cfg.sfreq:
        sos_n = _notch_sos(cfg.notch_hz, cfg.sfreq)
        x = sosfiltfilt(sos_n, x, axis=1)

    out = np.empty((C, T, cfg.n_bands), dtype=np.float32)

    if cfg.method == "filter":
        for bi, (lo, hi) in enumerate(cfg.bands.values()):
            sos = _bandpass_sos(lo, hi, cfg.sfreq, cfg.filter_order)
            filt = sosfiltfilt(sos, x, axis=1)              # [C, usable]
            power = (filt ** 2).reshape(C, T, fr).mean(axis=2)  # [C, T]
            out[:, :, bi] = np.log(power + cfg.log_eps)
    elif cfg.method == "welch":
        # Per-frame Welch PSD then integrate each band. Slower; for completeness.
        frames = x.reshape(C, T, fr)
        nper = min(fr, 64)
        for t in range(T):
            f_, pxx = welch(frames[:, t, :], fs=cfg.sfreq, nperseg=nper, axis=1)
            for bi, (lo, hi) in enumerate(cfg.bands.values()):
                m = (f_ >= lo) & (f_ < hi)
                bp = pxx[:, m].mean(axis=1) if m.any() else np.full(C, cfg.log_eps)
                out[:, t, bi] = np.log(bp + cfg.log_eps)
    else:
        raise ValueError(f"unknown feature method: {cfg.method!r}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Feature normalisation (training statistics only -> no leakage)
# --------------------------------------------------------------------------- #
@dataclass
class FeatureStats:
    """Per-(channel, band) mean/std, broadcast over time. Shapes ``[C, 1, F]``."""
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x):
        """Standardise ``[C, T, F]`` (numpy or torch) by the stored stats."""
        if hasattr(x, "detach"):  # torch tensor
            import torch
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return (x - mean) / std
        return (x - self.mean) / self.std

    def to_dict(self):
        return {"mean": np.asarray(self.mean).tolist(),
                "std": np.asarray(self.std).tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"], dtype=np.float32))


def fit_feature_stats(feats: np.ndarray, eps: float = 1e-6) -> FeatureStats:
    """``feats: [N, C, T, F]`` -> per-(channel,band) mean/std over N and T."""
    feats = np.nan_to_num(np.asarray(feats, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
    mean = feats.mean(axis=(0, 2), keepdims=True)[0]  # [C, 1, F]
    std = feats.std(axis=(0, 2), keepdims=True)[0]
    # guard degenerate / non-finite stats so normalisation never emits NaN
    std = np.where(np.isfinite(std) & (std > eps), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return FeatureStats(mean=mean.astype(np.float32), std=std.astype(np.float32))
