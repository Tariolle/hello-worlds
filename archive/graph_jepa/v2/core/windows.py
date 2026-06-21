"""Windowed [C, T, F] EEG dataset for TCP-Graph-JEPA.

Produces fixed-shape ``x: [channels=22, time_steps, feature_dim]`` tensors plus a
per-sample **channel availability mask** (TUAB's ``01_tcp_ar`` files have no
A1/A2, so ``A1-T3`` and ``T4-A2`` are not constructible and are zero-filled +
flagged unavailable).

Three sources, selected by ``GraphEEGConfig.source``:

  * ``edf``       — read TUH/TUAB EDFs, build the 22 TCP bipolar derivations from
    referential electrodes (``EEG FP1-REF`` ...), then log-bandpower features.
  * ``tensor``    — load a pre-extracted ``[N, C, T, F]`` array (``.npy`` / ``.pt``)
    with optional ``[N]`` labels; reuse existing features instead of recomputing.
  * ``synthetic`` — random tensors (for CPU smoke tests, no EDF/MNE needed).

Two access modes:

  * ``ssl``  — one random window per ``__getitem__`` -> ``(x, channel_mask)``;
    label-free, optionally restricted to the ``normal`` class for normal-only
    pretraining.
  * ``file`` — one *recording* -> ``(x[N,C,T,F], label, channel_mask, ok, path)``
    for file-level anomaly scoring / evaluation.

EDF reading prefers ``pyedflib`` (the cluster venv has it on compute nodes) and
falls back to ``mne``; both are optional so the synthetic/tensor paths import
cleanly on any machine.
"""
from __future__ import annotations

import glob
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from .features import FeatureConfig, FeatureStats, log_bandpower
from .tcp_graph import TCP_CHANNELS, split_electrodes

try:
    import pyedflib
except Exception:
    pyedflib = None
try:
    import mne
except Exception:
    mne = None


# --------------------------------------------------------------------------- #
@dataclass
class GraphEEGConfig:
    data_root: str = ""
    split: str = "train"               # train | eval
    mode: str = "ssl"                  # ssl | file
    source: str = "edf"               # edf | tensor | synthetic
    channels: List[str] = field(default_factory=lambda: list(TCP_CHANNELS))
    # feature / windowing
    sfreq: int = 200
    window_sec: float = 7.0
    frame_sec: float = 0.1
    bands: Optional[dict] = None        # None -> DEFAULT_BANDS
    feature_method: str = "filter"
    notch_hz: Optional[float] = 60.0
    zscore_raw: bool = True             # z-score bipolar signal before bandpower
    # sampling
    epoch_size: int = 4000              # virtual SSL windows per epoch
    n_windows: int = 8                  # evenly-spaced windows per recording (file mode)
    batch_size: int = 64
    num_workers: int = 8
    frac: float = 1.0                  # SSL: fraction of files used
    frac_seed: int = 0
    # labels
    label_scheme: str = "tuab"         # tuab (normal=0/abnormal=1) | folders
    class_names: Optional[List[str]] = None
    ssl_classes: Optional[List[str]] = None   # SSL restriction; default ["normal"]
    # tensor / synthetic
    tensor_path: Optional[str] = None
    tensor_label_path: Optional[str] = None
    synthetic_n: int = 256
    seed: int = 0

    def feature_cfg(self) -> FeatureConfig:
        kw = dict(sfreq=self.sfreq, window_sec=self.window_sec,
                  frame_sec=self.frame_sec, method=self.feature_method,
                  notch_hz=self.notch_hz)
        if self.bands:
            kw["bands"] = {k: tuple(v) for k, v in self.bands.items()}
        return FeatureConfig(**kw)


# --------------------------------------------------------------------------- #
# EDF helpers (label-based bipolar montage construction)
# --------------------------------------------------------------------------- #
_LABEL_RE = re.compile(r"^\s*(?:EEG\s+)?([A-Za-z0-9]+?)\s*-?\s*(?:REF|LE)?\s*$",
                       re.IGNORECASE)


def normalize_electrode(label: str) -> str:
    """``"EEG FP1-REF"`` -> ``"FP1"``; ``"T3"`` -> ``"T3"`` (upper-cased)."""
    m = _LABEL_RE.match(label)
    name = m.group(1) if m else label
    return name.strip().upper()


def electrode_index_map(labels: List[str]) -> dict:
    return {normalize_electrode(lab): i for i, lab in enumerate(labels)}


def _zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-6
    return (x - mu) / sd


def build_bipolar(ref: np.ndarray, emap: dict, channels: List[str]
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Referential signals ``ref[n_ref, W]`` + electrode->row map -> bipolar
    ``[C, W]`` and a boolean ``channel_mask[C]`` (True = constructible)."""
    C, W = len(channels), ref.shape[1]
    out = np.zeros((C, W), dtype=np.float32)
    avail = np.zeros(C, dtype=bool)
    for ci, ch in enumerate(channels):
        a, b = split_electrodes(ch)
        ia, ib = emap.get(a.upper()), emap.get(b.upper())
        if ia is not None and ib is not None:
            out[ci] = ref[ia] - ref[ib]
            avail[ci] = True
    return out, avail


class _EdfHandle:
    """Minimal uniform reader over pyedflib / mne for partial window reads."""

    def __init__(self, path: str, sfreq: int):
        self.path, self.target_sfreq = path, sfreq
        self.backend = None
        if pyedflib is not None:
            self._r = pyedflib.EdfReader(path)
            self.labels = self._r.getSignalLabels()
            self.fs = [self._r.getSampleFrequency(i) for i in range(self._r.signals_in_file)]
            self.nsamp = [self._r.getNSamples()[i] for i in range(self._r.signals_in_file)]
            self.backend = "pyedflib"
        elif mne is not None:
            self._raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            self.labels = list(self._raw.ch_names)
            self.fs = [float(self._raw.info["sfreq"])] * len(self.labels)
            self.nsamp = [self._raw.n_times] * len(self.labels)
            self.backend = "mne"
        else:
            raise ImportError("need pyedflib or mne to read EDF files")
        self.emap = electrode_index_map(self.labels)

    def n_samples_target(self) -> int:
        # length of the recording in target-sfreq samples (use the min over signals)
        secs = min(n / f for n, f in zip(self.nsamp, self.fs) if f > 0)
        return int(secs * self.target_sfreq)

    def read_window(self, start_sec: float, n_target: int, channels: List[str]):
        """Read [n_ref, n_target] referential block at target sfreq, then bipolar."""
        ref = np.zeros((len(self.labels), n_target), dtype=np.float32)
        for i, fs in enumerate(self.fs):
            n_src = int(round(n_target * fs / self.target_sfreq))
            s = int(round(start_sec * fs))
            sig = self._read_signal(i, s, n_src)
            if fs != self.target_sfreq:
                sig = _resample_to(sig, n_target)
            ref[i, :len(sig)] = sig[:n_target]
        return build_bipolar(ref, self.emap, channels)

    def _read_signal(self, i, start, n):
        if self.backend == "pyedflib":
            n = min(n, self.nsamp[i] - start)
            return self._r.readSignal(i, start, n).astype(np.float32)
        data, _ = self._raw[i, start:start + n]
        return data[0].astype(np.float32)

    def close(self):
        if self.backend == "pyedflib":
            self._r._close()


def _resample_to(sig: np.ndarray, n: int) -> np.ndarray:
    if len(sig) == n:
        return sig
    try:
        from scipy.signal import resample
        return resample(sig, n).astype(np.float32)
    except Exception:
        idx = np.linspace(0, len(sig) - 1, n)
        return np.interp(idx, np.arange(len(sig)), sig).astype(np.float32)


# --------------------------------------------------------------------------- #
# File listing
# --------------------------------------------------------------------------- #
def _glob_edf(root: str, split: str, cls: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, split, cls, "**", "*.edf"),
                            recursive=True))


def _tuab_classes(class_names) -> List[str]:
    if class_names:
        return list(class_names)
    return ["normal", "abnormal"]


def list_ssl_files(cfg: GraphEEGConfig) -> List[str]:
    classes = cfg.ssl_classes or ["normal"]
    files: List[str] = []
    for cls in classes:
        files += _glob_edf(cfg.data_root, cfg.split, cls)
    if not files:
        raise FileNotFoundError(
            f"No SSL .edf under {cfg.data_root}/{cfg.split} for classes {classes}")
    if cfg.frac < 1.0:
        r = np.random.default_rng(cfg.frac_seed)
        k = max(1, int(cfg.frac * len(files)))
        files = sorted(np.array(files)[r.choice(len(files), k, replace=False)].tolist())
    return files


def list_labelled_files(cfg: GraphEEGConfig) -> Tuple[List[Tuple[str, int]], List[str]]:
    classes = _tuab_classes(cfg.class_names)
    per_class = [[(p, label) for p in _glob_edf(cfg.data_root, cfg.split, cls)]
                 for label, cls in enumerate(classes)]
    # round-robin interleave so any prefix (e.g. an eval cap or the in-training
    # AUROC subset) stays class-balanced rather than all-normal-then-all-abnormal.
    items = []
    for i in range(max((len(c) for c in per_class), default=0)):
        for c in per_class:
            if i < len(c):
                items.append(c[i])
    if not items:
        raise FileNotFoundError(
            f"No labelled .edf under {cfg.data_root}/{cfg.split} for {classes}")
    return items, classes


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class GraphEEGDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: GraphEEGConfig, stats: Optional[FeatureStats] = None):
        self.cfg = cfg
        self.fcfg = cfg.feature_cfg()
        self.stats = stats
        self.label_names = None
        self._rng = np.random.default_rng(cfg.seed)
        self._rng_seed = None

        if cfg.source == "synthetic":
            self._init_synthetic()
        elif cfg.source == "tensor":
            self._init_tensor()
        elif cfg.source == "edf":
            if pyedflib is None and mne is None:
                raise ImportError("source='edf' needs pyedflib or mne")
            if cfg.mode == "ssl":
                self.files = list_ssl_files(cfg)
                self.items = None
            else:
                self.items, self.label_names = list_labelled_files(cfg)
                self.files = None
        else:
            raise ValueError(f"unknown source: {cfg.source!r}")

    # -- synthetic / tensor --------------------------------------------------
    def _feat_shape(self):
        return len(self.cfg.channels), self.fcfg.n_frames, self.fcfg.n_bands

    def _init_synthetic(self):
        C, T, Fdim = self._feat_shape()
        rng = np.random.default_rng(self.cfg.seed)
        self._syn = rng.standard_normal((self.cfg.synthetic_n, C, T, Fdim)).astype(np.float32)
        self._syn_labels = np.zeros(self.cfg.synthetic_n, dtype=np.int64)
        self._avail = np.ones(C, dtype=bool)
        self.label_names = ["normal", "abnormal"]

    def _init_tensor(self):
        arr = _load_array(self.cfg.tensor_path)
        if arr.ndim != 4:
            raise ValueError(f"tensor must be [N,C,T,F], got {arr.shape}")
        self._syn = arr.astype(np.float32)
        if self.cfg.tensor_label_path:
            self._syn_labels = _load_array(self.cfg.tensor_label_path).astype(np.int64).reshape(-1)
        else:
            self._syn_labels = np.zeros(len(arr), dtype=np.int64)
        self._avail = np.ones(arr.shape[1], dtype=bool)
        self.label_names = ["normal", "abnormal"]

    # -- length --------------------------------------------------------------
    def __len__(self):
        if self.cfg.source in ("synthetic", "tensor"):
            return len(self._syn)
        if self.cfg.mode == "ssl":
            return self.cfg.epoch_size
        return len(self.items)

    def _ensure_rng(self):
        worker = torch.utils.data.get_worker_info()
        seed = worker.seed if worker is not None else (self.cfg.seed + 12345)
        seed = int(seed % 2**32)
        if self._rng_seed != seed:
            self._rng = np.random.default_rng(seed)
            self._rng_seed = seed

    # -- feature post-processing --------------------------------------------
    def _finalize(self, feats: np.ndarray) -> torch.Tensor:
        if self.stats is not None:
            feats = self.stats.apply(feats)
        return torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))

    # -- EDF window -> features ---------------------------------------------
    def _edf_window_features(self, h: "_EdfHandle", start_sec: float):
        n_target = self.fcfg.window
        bip, avail = h.read_window(start_sec, n_target, self.cfg.channels)
        # a channel with any non-finite raw sample is treated as unavailable
        # (excluded from stats, masking and scoring) rather than silently zeroed
        avail = avail & np.isfinite(bip).all(axis=1)
        if self.cfg.zscore_raw and avail.any():
            bip[avail] = _zscore(bip[avail], axis=1)
        feats = log_bandpower(bip, self.fcfg)        # [C,T,F]
        feats[~avail] = 0.0
        return feats, avail

    # -- item ----------------------------------------------------------------
    def __getitem__(self, i):
        self._ensure_rng()
        if self.cfg.source in ("synthetic", "tensor"):
            x = self._finalize(self._syn[i].copy())
            cm = torch.from_numpy(self._avail.copy())
            if self.cfg.mode == "ssl":
                return x, cm
            return x.unsqueeze(0), int(self._syn_labels[i]), cm, True, f"idx{i}"

        if self.cfg.mode == "ssl":
            return self._getitem_ssl()
        return self._getitem_file(i)

    def _getitem_ssl(self):
        C, T, Fdim = self._feat_shape()
        for _ in range(8):
            path = self.files[self._rng.integers(len(self.files))]
            try:
                h = _EdfHandle(path, self.cfg.sfreq)
            except Exception:
                continue
            try:
                nT = h.n_samples_target()
                if nT <= self.fcfg.window + 1:
                    continue
                start = float(self._rng.integers(0, nT - self.fcfg.window)) / self.cfg.sfreq
                feats, avail = self._edf_window_features(h, start)
            except Exception:
                continue
            finally:
                h.close()
            return self._finalize(feats), torch.from_numpy(avail)
        # fallback: zeros (rare unreadable streak)
        return (torch.zeros(C, T, Fdim), torch.zeros(C, dtype=torch.bool))

    def _getitem_file(self, i):
        path, label = self.items[i]
        C, T, Fdim = self._feat_shape()
        N = self.cfg.n_windows
        try:
            h = _EdfHandle(path, self.cfg.sfreq)
            nT = h.n_samples_target()
            if nT <= self.fcfg.window + 1:
                raise ValueError("recording shorter than one window")
            starts = np.linspace(0, nT - self.fcfg.window, N) / self.cfg.sfreq
            feats = np.zeros((N, C, T, Fdim), dtype=np.float32)
            avail = None
            for j, s in enumerate(starts):
                f_j, a_j = self._edf_window_features(h, float(s))
                feats[j] = f_j
                avail = a_j
            h.close()
            x = self._finalize(feats)
            cm = torch.from_numpy(avail if avail is not None else np.ones(C, bool))
            return x, int(label), cm, True, path
        except Exception:
            return (torch.zeros(N, C, T, Fdim), int(label),
                    torch.zeros(C, dtype=torch.bool), False, path)


def read_edf_windows(path: str, cfg: GraphEEGConfig, stats: Optional[FeatureStats] = None,
                     n_windows: int = 8, contiguous: bool = False,
                     start_sec: float = 0.0):
    """Read one EDF (any path) into ``[N, C, T, F]`` features for visualisation.

    With ``contiguous=True`` the windows are read back-to-back from ``start_sec``
    (a continuous channel x time strip); otherwise they are evenly spaced over the
    recording. Returns ``(x, channel_mask, starts_sec)``.
    """
    fcfg = cfg.feature_cfg()
    win = fcfg.window
    h = _EdfHandle(path, cfg.sfreq)
    try:
        nT = h.n_samples_target()
        if contiguous:
            starts = start_sec + np.arange(n_windows) * cfg.window_sec
            starts = starts[(starts * cfg.sfreq + win) <= nT]
            if len(starts) == 0:
                starts = np.array([0.0])
        else:
            starts = np.linspace(0, max(0, nT - win), n_windows) / cfg.sfreq
        feats, avail = [], None
        for s in starts:
            bip, a = h.read_window(float(s), win, cfg.channels)
            a = a & np.isfinite(bip).all(axis=1)
            if cfg.zscore_raw and a.any():
                bip[a] = _zscore(bip[a], axis=1)
            f = log_bandpower(bip, fcfg)
            f[~a] = 0.0
            if stats is not None:
                f = stats.apply(f)
            feats.append(f.astype(np.float32))
            avail = a
    finally:
        h.close()
    x = torch.from_numpy(np.stack(feats, axis=0))
    cm = torch.from_numpy(avail if avail is not None else np.ones(len(cfg.channels), bool))
    return x, cm, np.asarray(starts, dtype=np.float64)


def _load_array(path: str) -> np.ndarray:
    if path is None:
        raise ValueError("tensor source requires tensor_path")
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith((".pt", ".pth")):
        return torch.load(path, map_location="cpu").numpy()
    if path.endswith(".npz"):
        z = np.load(path)
        return z[z.files[0]]
    raise ValueError(f"unsupported tensor file: {path}")


def make_graph_loader(cfg: GraphEEGConfig, stats: Optional[FeatureStats] = None,
                      shuffle: Optional[bool] = None):
    ds = GraphEEGDataset(cfg, stats=stats)
    is_train = cfg.mode == "ssl"
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        num_workers=cfg.num_workers, pin_memory=False,
        drop_last=is_train,
        persistent_workers=cfg.num_workers > 0), ds
