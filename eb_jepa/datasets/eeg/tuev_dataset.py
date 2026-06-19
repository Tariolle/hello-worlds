"""EEG dataset — TUH EEG Events Corpus (TUEV_PREPROCESSED).

6-class segment-level classification:
  0=BCKG (background)  1=SPSW (spike+sharp wave)  2=EYEM (eye movement)
  3=ARTF (artifact)    4=GPED (gen. periodic epileptiform)  5=PLED (periodic lateralized)

Expected layout (mirrors TUAB_PREPROCESSED):
  <data_root>/
    train/  {bckg,spsw,eyem,artf,gped,pled}/  *.edf
    eval/   ...

If your TUEV is in raw format (EDF + .lbl_bi annotation files), set
raw_format=True in TUEVConfig — not yet implemented, contact team.
"""
import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import pyedflib
except ImportError:
    pyedflib = None

CLASSES = ["bckg", "spsw", "eyem", "artf", "gped", "pled"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


@dataclass
class TUEVConfig:
    data_root: str = (
        "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUEV_PREPROCESSED"
    )
    split: str = "train"
    mode: str = "ssl"           # ssl | probe
    n_channels: int = 19
    sfreq: int = 200
    window_sec: float = 4.0     # TUEV events are typically 1-4 s; 4 s = 800 samples
    epoch_size: int = 20000
    n_windows: int = 1          # TUEV: one segment = one item (no multi-window pooling)
    batch_size: int = 128
    num_workers: int = 8
    aug_noise_std: float = 0.1
    aug_scale_jitter: float = 0.2
    aug_chan_drop_p: float = 0.2
    aug_time_mask_frac: float = 0.2


def _list_labelled_tuev(root: str, split: str) -> List[Tuple[str, int]]:
    items = []
    for cls, idx in CLASS_TO_IDX.items():
        pattern = os.path.join(root, split, cls, "**", "*.edf")
        for p in sorted(glob.glob(pattern, recursive=True)):
            items.append((p, idx))
    if not items:
        raise FileNotFoundError(
            f"No TUEV .edf files under {os.path.join(root, split)}. "
            f"Expected subdirs: {CLASSES}. "
            f"Check data_root and that TUEV_PREPROCESSED exists."
        )
    return items


def _list_edf_tuev(root: str, split: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, split, "**", "*.edf"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .edf under {os.path.join(root, split)}")
    return files


def _zscore(x: np.ndarray, axis: int) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-6
    return (x - mu) / sd


class TUEVDataset(torch.utils.data.Dataset):
    """SSL mode: random windows from any class. Probe mode: one segment per item."""

    def __init__(self, cfg: TUEVConfig):
        if pyedflib is None:
            raise ImportError("pyedflib required (pip install pyedflib)")
        self.cfg = cfg
        self.window = int(cfg.window_sec * cfg.sfreq)
        if cfg.mode == "ssl":
            self.files = _list_edf_tuev(cfg.data_root, cfg.split)
            self.items = None
        else:
            self.files = None
            self.items = _list_labelled_tuev(cfg.data_root, cfg.split)
        self._rng = np.random.default_rng()

    def __len__(self):
        return self.cfg.epoch_size if self.cfg.mode == "ssl" else len(self.items)

    def _read_window(self, path: str, start: Optional[int] = None) -> Optional[np.ndarray]:
        cfg = self.cfg
        try:
            f = pyedflib.EdfReader(path)
        except Exception:
            return None
        try:
            n_ch = min(f.signals_in_file, cfg.n_channels)
            if n_ch < cfg.n_channels:
                return None
            nsamp = int(min(f.getNSamples()[:cfg.n_channels]))
            if nsamp < self.window:
                # pad short segments rather than skip — TUEV events can be short
                x = np.zeros((cfg.n_channels, self.window), dtype=np.float32)
                for c in range(cfg.n_channels):
                    sig = f.readSignal(c, 0, nsamp)
                    x[c, :len(sig)] = sig
                return _zscore(x, axis=1)
            if start is None:
                start = int(self._rng.integers(0, nsamp - self.window + 1))
            start = min(start, nsamp - self.window)
            x = np.empty((cfg.n_channels, self.window), dtype=np.float32)
            for c in range(cfg.n_channels):
                x[c] = f.readSignal(c, start, self.window)
            return _zscore(x, axis=1)
        except Exception:
            return None
        finally:
            f._close()

    def _augment(self, x: np.ndarray) -> np.ndarray:
        cfg, rng = self.cfg, self._rng
        x = x.copy()
        if cfg.aug_scale_jitter > 0:
            scale = 1.0 + rng.uniform(-cfg.aug_scale_jitter, cfg.aug_scale_jitter,
                                      size=(cfg.n_channels, 1)).astype(np.float32)
            x *= scale
        if cfg.aug_noise_std > 0:
            x += rng.normal(0, cfg.aug_noise_std, size=x.shape).astype(np.float32)
        if cfg.aug_chan_drop_p > 0:
            mask = (rng.random(cfg.n_channels) > cfg.aug_chan_drop_p).astype(np.float32)
            x *= mask[:, None]
        if cfg.aug_time_mask_frac > 0:
            mlen = int(rng.uniform(0, cfg.aug_time_mask_frac) * self.window)
            if mlen > 0:
                s = int(rng.integers(0, self.window - mlen))
                x[:, s:s + mlen] = 0.0
        return x

    def __getitem__(self, i):
        self._rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        if self.cfg.mode == "ssl":
            for _ in range(8):
                path = self.files[self._rng.integers(len(self.files))]
                x = self._read_window(path)
                if x is not None:
                    break
            else:
                x = np.zeros((self.cfg.n_channels, self.window), dtype=np.float32)
            return torch.from_numpy(self._augment(x)), torch.from_numpy(self._augment(x))
        # probe: one segment -> [1, C, T], label, ok
        path, label = self.items[i]
        w = self._read_window(path, start=0)
        ok = w is not None
        if not ok:
            w = np.zeros((self.cfg.n_channels, self.window), dtype=np.float32)
        # shape [1, C, T] to match TUAB probe convention (n_windows=1 here)
        return torch.from_numpy(w[None]), int(label), ok
