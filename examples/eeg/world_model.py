"""EEG World Model — 3-geometry JEPA for temporal prediction.

Three parallel prediction heads, each in a different geometric space:

  Euclidean (ambient)
    Embed: encoder.represent() → global avg-pool [B, D]
    Anti-collapse: SIGReg / BCS (Gaussian assumption in Euclidean space)
    WM loss: predictor_amb(z_past) ≈ z_future  (MSE in projected space)

  Tangent (Log-Euclidean, compressed)
    Embed: cov_features() → temporal_covariance → spd_logm → upper_tri_vec [B, d(d+1)/2]
    Anti-collapse: PEIRA (distribution-free, no Gaussian assumption on tangent vectors)
    WM loss: predictor_tan(z_past) ≈ z_future  (MSE on upper-tri log-SPD)

  Riemannian (Log-Euclidean, full matrix)
    Embed: cov_features() → temporal_covariance → log-SPD full d×d flatten [B, d²]
    Anti-collapse: VICReg on projected log-SPD features
    WM loss: predictor decodes to full log-SPD, MSE vs target log-SPD
             (Log-Euclidean metric — valid Riemannian metric on SPD, numerically stable)
    Distinct from tangent head: uses full d×d matrix (includes cross-terms) not just upper-tri.

Data: TUEVDataset mode="wm_ssl" returns (x_past, x_future) consecutive 1-second windows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.losses import VICRegLoss, BCS
from examples.eeg.geometry import (
    temporal_covariance, spd_logm, upper_tri_vec, collapse_metrics,
)
from examples.eeg.main import Projector
from examples.eeg.peira import PEIRALoss


class Predictor(nn.Module):
    """MLP: past latent → predicted future latent (same dim)."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        h = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, h), nn.BatchNorm1d(h), nn.GELU(),
            nn.Linear(h, dim),
        )

    def forward(self, x):
        return self.net(x)


class EEGWorldModel(nn.Module):
    """Three-geometry World Model for EEG sequences.

    Args:
        encoder: EEGEncoder1D instance (shared across all heads)
        cfg: model OmegaConf node with encoder + ssl sub-nodes
    """

    def __init__(self, encoder, cfg):
        super().__init__()
        s = cfg.ssl
        self.encoder = encoder

        d = encoder.out_dim                        # 256
        d_cov = encoder.d_cov                      # 8  (for TUEV: 8 < T'=13)
        d_tan = d_cov * (d_cov + 1) // 2          # 36
        d_spd = d_cov * d_cov                      # 64 (log-SPD flattened, full d×d)
        proj_spec = s.get("proj", "512-512-256")
        self.d_cov = d_cov
        self.wm_weight = s.get("wm_weight", 1.0)
        self.ssl_weight = s.get("ssl_weight", 1.0)

        # ── Euclidean (ambient) head ─────────────────────────────────────────
        self.proj_amb = Projector(d, proj_spec)
        p = self.proj_amb.out_dim                  # 256
        self.pred_amb = Predictor(p)
        self.reg_amb = BCS(
            num_slices=s.get("num_slices", 256), lmbd=s.get("lmbd", 10.0))

        # ── Tangent (Log-Euclidean) head ─────────────────────────────────────
        self.proj_tan = Projector(d_tan, proj_spec)
        self.pred_tan = Predictor(p)
        self.reg_tan = PEIRALoss(dim=p, lam=s.get("lam", 0.1))

        # ── Riemannian (Log-Euclidean, full d×d) head ───────────────────────
        # Uses full log-SPD matrix [B, d²] — richer than upper-tri tangent [B, d(d+1)/2].
        # WM loss = MSE on log-SPD (= Log-Euclidean metric, numerically stable).
        self.proj_riem = Projector(d_spd, proj_spec)
        self.pred_riem = Predictor(p)
        self.riem_decoder = nn.Linear(p, d_spd)   # latent → log-SPD [B, d²]
        self.reg_riem = VICRegLoss(
            std_coeff=s.get("std_coeff", 1.0), cov_coeff=s.get("cov_coeff", 1.0))

    def _embed_all(self, x):
        """Single forward pass → (z_amb, z_tan, z_riem, spd) for one window."""
        feat = self.encoder.feature_map(x)          # [B, D, T']
        B = feat.shape[0]

        # Ambient
        z_amb = self.proj_amb(feat.mean(dim=-1))    # [B, p]

        # Tangent + Riemannian share the same cov pipeline
        cov_feat = self.encoder.cov_proj(feat)      # [B, d_cov, T']
        spd = temporal_covariance(cov_feat)         # [B, d_cov, d_cov]
        log_spd = spd_logm(spd)                     # [B, d_cov, d_cov]

        tan_vec = upper_tri_vec(log_spd)            # [B, d_tan]
        z_tan = self.proj_tan(tan_vec)              # [B, p]

        flat_log = log_spd.reshape(B, self.d_cov * self.d_cov)  # [B, d²]
        z_riem = self.proj_riem(flat_log)           # [B, p]

        return z_amb, z_tan, z_riem, spd

    def compute_loss(self, batch):
        x_past, x_future = batch

        z_past_amb, z_past_tan, z_past_riem, spd_past = self._embed_all(x_past)
        z_fut_amb, z_fut_tan, z_fut_riem, spd_fut = self._embed_all(x_future)

        # ── Euclidean ────────────────────────────────────────────────────────
        pred_fut_amb = self.pred_amb(z_past_amb)
        loss_wm_amb = F.mse_loss(pred_fut_amb, z_fut_amb.detach())
        out_amb = self.reg_amb(z_past_amb, z_fut_amb)
        loss_ssl_amb = out_amb["loss"]

        # ── Tangent ──────────────────────────────────────────────────────────
        pred_fut_tan = self.pred_tan(z_past_tan)
        loss_wm_tan = F.mse_loss(pred_fut_tan, z_fut_tan.detach())
        out_tan = self.reg_tan(z_past_tan, z_fut_tan)
        loss_ssl_tan = out_tan["loss"]

        # ── Riemannian (Log-Euclidean) ────────────────────────────────────────
        # Predict future log-SPD directly; MSE in log-space = Log-Euclidean metric.
        pred_z_riem = self.pred_riem(z_past_riem)              # [B, p]
        log_pred_flat = self.riem_decoder(pred_z_riem)         # [B, d²]

        # Target: full log-SPD of future window (stop-grad)
        B, d = x_past.shape[0], self.d_cov
        log_fut_flat = spd_logm(spd_fut).reshape(B, d * d).detach()

        loss_wm_riem = F.mse_loss(log_pred_flat, log_fut_flat)
        out_riem = self.reg_riem(z_past_riem, z_fut_riem)
        loss_ssl_riem = out_riem["loss"]

        # ── Combined ─────────────────────────────────────────────────────────
        loss = (
            self.ssl_weight * (loss_ssl_amb + loss_ssl_tan + loss_ssl_riem)
            + self.wm_weight * (loss_wm_amb + loss_wm_tan + loss_wm_riem)
        )

        def _f(v):
            return round(v.item(), 5) if torch.is_tensor(v) else round(float(v), 5)

        logs = {
            "wm_amb":  _f(loss_wm_amb),
            "wm_tan":  _f(loss_wm_tan),
            "wm_riem": _f(loss_wm_riem),
            "ssl_amb": _f(loss_ssl_amb),
            "ssl_tan": _f(loss_ssl_tan),
            "ssl_riem": _f(loss_ssl_riem),
            **{f"tan_{k}": _f(v) for k, v in out_tan.items() if k != "loss"},
        }
        logs.update(collapse_metrics(z_past_amb.detach()))
        return loss, logs
