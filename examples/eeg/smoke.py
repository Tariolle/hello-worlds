"""Forward+backward smoke test for every (reg_type x reg_space) cell.

Runs the encoder -> SSL objective -> loss.backward() on a random batch (no data,
no EDF read), so it only checks that the model graph is sound on the target
hardware. Run on a GPU node via cluster/smoke.sbatch, or locally on CPU.
"""
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from examples.eeg.main import build_encoder, build_ssl  # noqa: E402

cfg = OmegaConf.load("examples/eeg/cfgs/train.yaml").model
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", dev, "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")

B = 64
v1 = torch.randn(B, 19, 2000, device=dev)
v2 = torch.randn(B, 19, 2000, device=dev)
for rt in ["vicreg", "sigreg", "peira"]:
    for rs in ["ambient", "tangent"]:
        cfg.ssl.reg_type, cfg.ssl.reg_space = rt, rs
        enc = build_encoder(cfg).to(dev)
        ssl = build_ssl(enc, cfg).to(dev)
        loss, logs = ssl.compute_loss((v1, v2))
        loss.backward()
        print(f"  {rt:7s} x {rs:8s} -> loss {loss.item():.4f} | {logs}")
print("SMOKE_DONE")
