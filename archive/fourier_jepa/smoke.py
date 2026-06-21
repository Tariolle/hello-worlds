"""Forward+backward smoke test for the Fourier-STFT encoder (every reg cell).

Mirrors the core smoke path but for FourierEEGEncoder1D: builds the encoder from
train.yaml and runs encoder -> SSL objective -> loss.backward() on a
random batch (no data, no EDF read). Checks the model graph is sound on the
target hardware and that the STFT front-end backprops. Run on a GPU node via
the archived cluster launcher's smoke path, or locally on CPU.
"""
from pathlib import Path

import torch
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder, build_ssl

cfg = OmegaConf.load(Path(__file__).with_name("train.yaml")).model
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", dev, "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")

B = 32
v1 = torch.randn(B, 19, 2000, device=dev)
v2 = torch.randn(B, 19, 2000, device=dev)

enc = build_encoder(cfg).to(dev)
fm, rep = enc.feature_map(v1), enc.represent(v1)
print(f"encoder={type(enc).__name__} feature_map={tuple(fm.shape)} "
      f"represent={tuple(rep.shape)} (T'={fm.shape[-1]} > d_cov={enc.d_cov})")

for rt in ["vicreg", "sigreg", "peira"]:
    for rs in ["ambient", "tangent"]:
        cfg.ssl.reg_type, cfg.ssl.reg_space = rt, rs
        enc = build_encoder(cfg).to(dev)
        ssl = build_ssl(enc, cfg).to(dev)
        loss, logs = ssl.compute_loss((v1, v2))
        loss.backward()
        assert torch.isfinite(loss), f"non-finite loss for {rt} x {rs}"
        print(f"  {rt:7s} x {rs:8s} -> loss {loss.item():.4f} | {logs}")
print("FOURIER_SMOKE_DONE")
