"""PEIRA anti-collapse regulariser (SC-PEIRA).

Arbel, Terver & Ponce, "PEIRA: Learning Predictive Encoders through Inter-View
Regressor Alignment", arXiv:2605.17671 (2026). Algorithm verified from the paper
(Eqs. 5/6/15, Fig. 1).

Why PEIRA here: it maximises the trace of the optimal *ridge regressor* between
the two views (a non-linear-CCA objective) with an L2 scale-control term, and
makes representation collapse provably unstable (when >= 2 canonical
correlations exceed lambda). Crucially it assumes NO distribution on the
embeddings — unlike SIGReg, which drives them toward an isotropic Gaussian. That
makes PEIRA the principled anti-collapse term for a (non-Gaussian) SPD tangent
space: it does not fight the manifold's natural distribution.

Stochastic estimator: the k x k cross-/within-view second-moment matrices
Sigma, N are tracked by EMA; the ridge regressor P = Sigma (N + lambda I)^-1 and
Q = (N + lambda I)^-1 are treated as constants (stop-grad), and an auxiliary
loss whose gradient matches the true objective is back-propagated through the
encoder only.

Interface matches eb_jepa.losses (VICRegLoss / BCS): ``forward(z1, z2) -> dict``
with a ``"loss"`` key. ``z1, z2`` are the projected view embeddings ``[B, k]``
(for the tangent arm these are projections of the SPD tangent vectors, so both
views already share a base point — PEIRA's equal-dim / shared-space requirement).

Caveat (from the paper): collapse-instability needs r_max >= 2, i.e. the two
views must share at least two predictable modes above lambda. If runs collapse,
the suspect is view construction, not the regulariser.
"""
import torch
import torch.nn as nn


class PEIRALoss(nn.Module):
    def __init__(self, dim, lam=0.1, eta_init=0.5, eta_min=0.01, eta_anneal=0.999):
        super().__init__()
        self.lam = lam
        self.eta = eta_init
        self.eta_min = eta_min
        self.eta_anneal = eta_anneal
        # persistent EMA second-moment estimates (k x k)
        self.register_buffer("Sigma", torch.zeros(dim, dim))
        self.register_buffer("N", torch.eye(dim))

    @torch.no_grad()
    def _update_stats(self, x, y):
        B = x.shape[0]
        sig = (x.T @ y + y.T @ x) / B
        nb = (x.T @ x + y.T @ y) / B
        self.Sigma.mul_(1 - self.eta).add_(self.eta * sig)
        self.N.mul_(1 - self.eta).add_(self.eta * nb)
        self.eta = max(self.eta_min, self.eta * self.eta_anneal)

    def forward(self, z1, z2):
        self._update_stats(z1.detach(), z2.detach())
        k = self.N.shape[0]
        eye = torch.eye(k, device=z1.device, dtype=z1.dtype)
        Ninv = torch.linalg.solve(self.N + self.lam * eye, eye).detach()
        P = (self.Sigma @ Ninv).detach()          # ridge regressor, stop-grad
        Q = Ninv                                   # already detached
        rx = z1 @ P.T - z2                          # grad flows via z1, z2 only
        ry = z2 @ P.T - z1
        term = (z1 * (rx @ Q.T)).sum(dim=1) + (z2 * (ry @ Q.T)).sum(dim=1)
        scale = z1.pow(2).sum(dim=1) + z2.pow(2).sum(dim=1)
        loss = 0.5 * term.mean() + 0.5 * self.lam * scale.mean()
        return {"loss": loss,
                "tr_P": torch.diagonal(P).sum(),    # predictability; should rise
                "peira_scale": 0.5 * self.lam * scale.mean()}
