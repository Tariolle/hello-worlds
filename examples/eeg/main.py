"""EEG — SSL pretraining entrypoint (self-supervised representation learning).

Two-view JEPA on unlabeled EEG: encode two augmented views of a 10 s window,
project, and apply an anti-collapse regulariser. The experiment knobs are the
regulariser TYPE (``vicreg`` | ``sigreg`` | ``peira``) and the SPACE it acts in:
  * ``ambient`` -> the pooled Euclidean representation        ``[B, D]``
  * ``tangent`` -> the Log-Euclidean tangent vector of the per-window temporal
                   feature covariance (an SPD matrix)          ``[B, d(d+1)/2]``
The probe (eval.py) ALWAYS reads the pooled representation, so this isolates the
effect of *where* anti-collapse is enforced (ambient Euclidean vs SPD tangent).

Ladder: ambient-SIGReg (Laya-like baseline) -> tangent-SIGReg -> tangent-PEIRA.

Run:  python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml
"""
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader
from eb_jepa.losses import VICRegLoss, BCS
from examples.eeg.encoder import EEGEncoder1D
from examples.eeg.geometry import tangent_features, collapse_metrics
from examples.eeg.peira import PEIRALoss


class Projector(nn.Module):
    """MLP projector from a dash-spec like '512-512-256' (BN + GELU between layers)."""

    def __init__(self, in_dim, spec="512-512-256"):
        super().__init__()
        dims = [in_dim] + [int(x) for x in str(spec).split("-")]
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers += [nn.BatchNorm1d(b), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# 1) ENCODER
# --------------------------------------------------------------------------- #
def build_encoder(cfg):
    """1D EEG encoder mapping [B, C=19, T] -> [B, D]. See encoder.py."""
    e = cfg.encoder
    return EEGEncoder1D(
        n_channels=cfg.get("n_channels", 19),
        widths=tuple(e.widths),
        d_model=e.d_model,
        d_cov=e.d_cov,
        kernel=e.get("kernel", 7),
        stride=e.get("stride", 2),
    )


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE
# --------------------------------------------------------------------------- #
class SSLModule(nn.Module):
    """Two-view JEPA. Holds the encoder so the provided loop optimises it."""

    def __init__(self, encoder, cfg):
        super().__init__()
        s = cfg.ssl
        self.encoder = encoder
        self.reg_space = s.reg_space  # ambient | tangent
        self.reg_type = s.reg_type    # vicreg | sigreg | peira
        in_dim = (encoder.out_dim if self.reg_space == "ambient"
                  else encoder.d_cov * (encoder.d_cov + 1) // 2)
        self.projector = Projector(in_dim, s.get("proj", "512-512-256"))
        p = self.projector.out_dim
        if self.reg_type == "vicreg":
            self.reg = VICRegLoss(std_coeff=s.get("std_coeff", 1.0),
                                  cov_coeff=s.get("cov_coeff", 1.0))
        elif self.reg_type == "sigreg":
            self.reg = BCS(num_slices=s.get("num_slices", 256), lmbd=s.get("lmbd", 10.0))
        elif self.reg_type == "peira":
            self.reg = PEIRALoss(dim=p, lam=s.get("lam", 0.1))
        else:
            raise ValueError(f"unknown reg_type: {self.reg_type}")

    def _embed(self, x):
        if self.reg_space == "ambient":
            h = self.encoder.represent(x)
        else:
            h = tangent_features(self.encoder.cov_features(x))
        return self.projector(h)

    def compute_loss(self, batch, with_metrics=True):
        v1, v2 = batch
        z1, z2 = self._embed(v1), self._embed(v2)
        out = self.reg(z1, z2)
        loss = out["loss"]
        logs = {k: (round(v.item(), 4) if torch.is_tensor(v) else v)
                for k, v in out.items() if k != "loss"}
        if with_metrics:
            logs.update(collapse_metrics(z1.detach()))
        return loss, logs


def build_ssl(encoder, cfg):
    """Two-view SSL module exposing compute_loss(batch) -> (loss, logs)."""
    return SSLModule(encoder, cfg)


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided by eb_jepa (unchanged)
# --------------------------------------------------------------------------- #
def run(fname="examples/eeg/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = EEGConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        last_batch_idx = len(loader) - 1
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch, with_metrics=batch_idx == last_batch_idx)
            loss.backward(); opt.step()
        print(f"[eeg] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[eeg] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/train.yaml"
    run(fname=fname)
