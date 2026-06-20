"""Combined TUAB + TUEV SSL dataloader for cross-corpus pretraining.

Motivation:
  TUAB alone (~2700 subjects) saturates at ~0.83 balanced_acc. TUEV adds 600+
  long clinical recordings with diverse pathology (spikes, eye movements,
  artifacts, focal discharges) — exactly the diversity that should help TUAB eval.

Both datasets use 10-second windows at 200 Hz (2000 samples, 19 channels).
TUEV segments are 7-40 min long, so 10s random windows are well-defined.

Usage in config:
  data:
    data_root: /path/to/TUAB_PREPROCESSED
    tuev_root:  /path/to/TUEV_RAW_DATA      # enables combined mode
    epoch_size: 20000                        # TUAB virtual size
    tuev_epoch_size: 5000                    # TUEV virtual size
"""
from omegaconf import OmegaConf
import torch
from torch.utils.data import ConcatDataset, DataLoader

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from eb_jepa.datasets.eeg.tuev_dataset import TUEVConfig, TUEVDataset


def make_combined_loader(cfg):
    """Build a DataLoader over TUAB + TUEV.

    cfg must be the top-level config (with cfg.data having tuev_root).
    Returns a DataLoader yielding (v1, v2) pairs of shape [B, 19, 2000].
    """
    d = cfg.data
    tuab_kwargs = OmegaConf.to_container(d, resolve=True)
    # strip TUEV-specific keys before passing to EEGConfig
    tuev_root       = tuab_kwargs.pop("tuev_root", None)
    tuev_epoch_size = tuab_kwargs.pop("tuev_epoch_size", 5000)

    tuab_cfg = EEGConfig(**tuab_kwargs)
    tuab_cfg.mode = "ssl"
    ds_tuab = EEGDataset(tuab_cfg)
    print(f"[combined] TUAB ssl: {len(ds_tuab)} virtual samples, "
          f"{len(ds_tuab.files)} recordings", flush=True)

    tuev_cfg = TUEVConfig(
        data_root=tuev_root,
        split=d.get("split", "train"),
        mode="ssl",
        n_channels=tuab_cfg.n_channels,
        sfreq=tuab_cfg.sfreq,
        window_sec=tuab_cfg.window_sec,     # 10s — matches TUAB
        epoch_size=tuev_epoch_size,
        batch_size=tuab_cfg.batch_size,
        num_workers=0,                       # parent loader handles workers
        aug_noise_std=tuab_cfg.aug_noise_std,
        aug_scale_jitter=tuab_cfg.aug_scale_jitter,
        aug_chan_drop_p=tuab_cfg.aug_chan_drop_p,
        aug_time_mask_frac=tuab_cfg.aug_time_mask_frac,
    )
    ds_tuev = TUEVDataset(tuev_cfg)
    print(f"[combined] TUEV ssl: {tuev_epoch_size} virtual samples, "
          f"{len(ds_tuev._ssl_segs)} segments", flush=True)

    combined = ConcatDataset([ds_tuab, ds_tuev])
    print(f"[combined] total epoch size: {len(combined)}", flush=True)

    return DataLoader(
        combined,
        batch_size=tuab_cfg.batch_size,
        shuffle=True,
        num_workers=tuab_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=tuab_cfg.num_workers > 0,
    )
