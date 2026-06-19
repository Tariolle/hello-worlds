"""TUEV dataset — raw HTK format (TUEV_RAW_DATA on Dalia).

Signal:  {patient}/{patient}_{segment}_ch{channel:03d}.htk
         float32 samples at 200 Hz, 12-byte HTK header to skip.
Labels:  {patient}/{patient}_{segment}_ch{channel:03d}.lab
         lines: <start_100kHz>  <end_100kHz>  <class>
         time unit: 1/100 000 s   →   200 samples = 1 second.

6 classes: bckg spsw eyem artf gped pled
Strategy: each annotated 1-second window = one probe sample.
          SSL mode reads random 1-second windows from all HTK files.
"""
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

CLASSES = ["bckg", "spsw", "eyem", "artf", "gped", "pled"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

_LAB_BASE_HZ = 100_000      # .lab time stamps base rate
_SIGNAL_HZ   = 200          # HTK files sampling rate
_SAMPLES_PER_SECOND = 200   # = _SIGNAL_HZ
_UNIT_TO_SAMPLE = _SIGNAL_HZ / _LAB_BASE_HZ   # 0.002 → multiply lab units by this

# pattern:  {anything}_{segment}_ch{NNN}.htk
_HTK_RE = re.compile(r"^(.+)_ch(\d{3})\.htk$", re.IGNORECASE)
_LAB_RE = re.compile(r"^(.+)_ch(\d{3})\.lab$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Low-level HTK I/O
# --------------------------------------------------------------------------- #

def _htk_n_samples(path: str) -> int:
    return (os.path.getsize(path) - 12) // 4


def _read_htk_slice(path: str, start: int, n: int) -> np.ndarray:
    """Read n float32 samples starting at sample `start`. Zero-pads if needed.

    HTK standard format is big-endian (IEEE). NEDC/TUH raw data follows this.
    """
    total = _htk_n_samples(path)
    start = max(0, min(start, total))
    n_avail = min(n, total - start)
    buf = np.zeros(n, dtype=np.float32)
    if n_avail > 0:
        with open(path, "rb") as f:
            f.seek(12 + start * 4)
            raw = f.read(n_avail * 4)
        arr = np.frombuffer(raw, dtype=">f4").astype(np.float32)  # big-endian
        buf[:len(arr)] = arr
    return buf


# --------------------------------------------------------------------------- #
# Index building
# --------------------------------------------------------------------------- #

@dataclass
class _Item:
    patient: str
    ch_paths: Dict[int, str]   # channel → .htk path
    start_sample: int
    label: int


def _parse_lab(lab_path: str) -> List[Tuple[int, int]]:
    """Parse .lab → list of (start_sample, class_idx)."""
    items = []
    with open(lab_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start_u = int(parts[0])
                cls = parts[2].lower()
            except (ValueError, IndexError):
                continue
            if cls not in CLASS_TO_IDX:
                continue
            start_s = int(start_u * _UNIT_TO_SAMPLE)
            items.append((start_s, CLASS_TO_IDX[cls]))
    return items


def _build_index(root: str, split: str, n_ch: int) -> List[_Item]:
    """
    Scan split_dir for HTK + LAB files. Group channels by
    (patient, segment_key). Parse LAB from ch000 (or lowest available)
    to get annotated time windows → one _Item per annotated window.
    """
    split_dir = os.path.join(root, split)

    # Collect: segment_key → {channel: htk_path}
    htk_by_seg: Dict[str, Dict[int, str]] = defaultdict(dict)
    lab_by_seg: Dict[str, Dict[int, str]] = defaultdict(dict)

    for dirpath, _, fnames in os.walk(split_dir):
        # patient = last component of dirpath (may be nested)
        patient = os.path.basename(dirpath)
        for fname in fnames:
            mh = _HTK_RE.match(fname)
            if mh:
                stem, ch_str = mh.group(1), mh.group(2)
                ch = int(ch_str)
                if ch >= n_ch:
                    continue
                key = (patient, stem)
                htk_by_seg[key][ch] = os.path.join(dirpath, fname)
            ml = _LAB_RE.match(fname)
            if ml:
                stem, ch_str = ml.group(1), ml.group(2)
                ch = int(ch_str)
                key = (patient, stem)
                lab_by_seg[key][ch] = os.path.join(dirpath, fname)

    items: List[_Item] = []
    for key, ch_paths in htk_by_seg.items():
        patient = key[0]
        lab_ch_paths = lab_by_seg.get(key, {})
        # use lowest-channel LAB as event list
        if not lab_ch_paths:
            continue
        lab_path = lab_ch_paths[min(lab_ch_paths)]
        annotations = _parse_lab(lab_path)
        for start_s, label in annotations:
            items.append(_Item(
                patient=patient,
                ch_paths=ch_paths,
                start_sample=start_s,
                label=label,
            ))

    return items


def _build_ssl_segments(root: str, split: str, n_ch: int
                        ) -> List[Dict[int, str]]:
    """
    Return list of {channel: htk_path} dicts, one per (patient, segment).
    Each dict has at least one channel. SSL reads random windows across channels.
    """
    by_seg: Dict[Tuple, Dict[int, str]] = defaultdict(dict)
    split_dir = os.path.join(root, split)
    for dirpath, _, fnames in os.walk(split_dir):
        patient = os.path.basename(dirpath)
        for fname in fnames:
            m = _HTK_RE.match(fname)
            if not m:
                continue
            stem, ch_str = m.group(1), m.group(2)
            ch = int(ch_str)
            if ch >= n_ch:
                continue
            by_seg[(patient, stem)][ch] = os.path.join(dirpath, fname)
    return [v for v in by_seg.values() if v]


# --------------------------------------------------------------------------- #
# Config & Dataset
# --------------------------------------------------------------------------- #

@dataclass
class TUEVConfig:
    data_root: str = (
        "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUEV_RAW_DATA"
    )
    split: str = "train"
    mode: str = "ssl"           # ssl | probe
    n_channels: int = 19        # use first 19 of 21 (matches TUAB)
    sfreq: int = 200
    window_sec: float = 1.0     # TUEV events are 1-second windows
    epoch_size: int = 20000
    n_windows: int = 1          # probe: 1 window per item (already 1-second)
    batch_size: int = 128
    num_workers: int = 8
    aug_noise_std: float = 0.1
    aug_scale_jitter: float = 0.2
    aug_chan_drop_p: float = 0.2
    aug_time_mask_frac: float = 0.2


def _zscore(x: np.ndarray, axis: int) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-6
    return (x - mu) / sd


class TUEVDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: TUEVConfig):
        self.cfg = cfg
        self.window = int(cfg.window_sec * cfg.sfreq)   # 200 samples
        self._rng = np.random.default_rng()

        if cfg.mode in ("ssl", "wm_ssl"):
            self._ssl_segs = _build_ssl_segments(
                cfg.data_root, cfg.split, cfg.n_channels)
            self._items = None
            if not self._ssl_segs:
                raise FileNotFoundError(
                    f"No .htk files under {cfg.data_root}/{cfg.split}")
            print(f"[TUEV-{cfg.mode}] {cfg.split}: {len(self._ssl_segs)} segments",
                  flush=True)
        else:
            self._ssl_files = None
            self._items = _build_index(cfg.data_root, cfg.split, cfg.n_channels)
            if not self._items:
                raise FileNotFoundError(
                    f"No annotated TUEV events under {cfg.data_root}/{cfg.split}. "
                    "Expected .lab files alongside .htk files.")
            counts = [0] * NUM_CLASSES
            for it in self._items:
                counts[it.label] += 1
            print(f"[TUEV-probe] {cfg.split}: {len(self._items)} events  "
                  + "  ".join(f"{CLASSES[i]}={counts[i]}" for i in range(NUM_CLASSES)),
                  flush=True)

    def __len__(self):
        return self.cfg.epoch_size if self.cfg.mode in ("ssl", "wm_ssl") else len(self._items)

    def _read_channels(self, ch_paths: Dict[int, str], start: int) -> np.ndarray:
        x = np.zeros((self.cfg.n_channels, self.window), dtype=np.float32)
        for ch, path in ch_paths.items():
            if ch < self.cfg.n_channels:
                x[ch] = _read_htk_slice(path, start, self.window)
        return _zscore(x, axis=1)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        cfg, rng = self.cfg, self._rng
        x = x.copy()
        if cfg.aug_scale_jitter > 0:
            x *= 1.0 + rng.uniform(-cfg.aug_scale_jitter, cfg.aug_scale_jitter,
                                   (cfg.n_channels, 1)).astype(np.float32)
        if cfg.aug_noise_std > 0:
            x += rng.normal(0, cfg.aug_noise_std, x.shape).astype(np.float32)
        if cfg.aug_chan_drop_p > 0:
            x *= (rng.random(cfg.n_channels) > cfg.aug_chan_drop_p).astype(np.float32)[:, None]
        if cfg.aug_time_mask_frac > 0:
            mlen = int(rng.uniform(0, cfg.aug_time_mask_frac) * self.window)
            if mlen > 0:
                s = int(rng.integers(0, max(1, self.window - mlen)))
                x[:, s:s + mlen] = 0.0
        return x

    def __getitem__(self, i):
        self._rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())

        if self.cfg.mode == "ssl":
            ch_paths = self._ssl_segs[self._rng.integers(len(self._ssl_segs))]
            # use any channel to determine segment length
            ref_path = next(iter(ch_paths.values()))
            n = _htk_n_samples(ref_path)
            start = int(self._rng.integers(0, max(1, n - self.window + 1)))
            x = self._read_channels(ch_paths, start)
            v1 = torch.from_numpy(self._augment(x))
            v2 = torch.from_numpy(self._augment(x))
            return v1, v2

        if self.cfg.mode == "wm_ssl":
            # World model: return two CONSECUTIVE windows (past, future).
            # Retry up to 20 times to find a segment long enough for 2 windows.
            for _ in range(20):
                ch_paths = self._ssl_segs[self._rng.integers(len(self._ssl_segs))]
                ref_path = next(iter(ch_paths.values()))
                n = _htk_n_samples(ref_path)
                if n >= 2 * self.window:
                    break
            max_start = max(1, n - 2 * self.window + 1)
            start = int(self._rng.integers(0, max_start))
            x_past = self._read_channels(ch_paths, start)
            x_future = self._read_channels(ch_paths, start + self.window)
            v1 = torch.from_numpy(self._augment(x_past))
            v2 = torch.from_numpy(self._augment(x_future))
            return v1, v2

        # probe: one annotated 1-second window
        item = self._items[i]
        x = self._read_channels(item.ch_paths, item.start_sample)
        # returns [1, C, T] to be compatible with TUAB probe harness
        return torch.from_numpy(x[None]), int(item.label), True
