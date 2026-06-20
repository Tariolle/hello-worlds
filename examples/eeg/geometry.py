"""SPD-manifold helpers + collapse diagnostics for the EEG-JEPA track.

The geometry-aware regulariser lives in the tangent space of the per-window
temporal feature covariance (an SPD matrix). We use the Log-Euclidean metric:
the tangent vector is ``vec(logm(C))`` at the identity base point — cheap, a
global diffeomorphism, and (with the sqrt(2) off-diagonal weighting below) its
Euclidean inner product equals the Frobenius inner product of ``logm(C)``. Both
views share this fixed base point, which is exactly the "common base point +
metric-aware vectorisation" PEIRA/SIGReg-in-tangent require.
"""
import torch


def temporal_covariance(feat, eps=1e-4):
    """[B, d, T'] feature map -> [B, d, d] SPD covariance over time (+ eps*I).

    The SPD floor is an ABSOLUTE additive eps*I (not a relative jitter), so it is
    not scale-equivariant: for low-energy windows whose true eigenvalues are
    << eps, the tangent vector is dominated by the floor. Acceptable as a
    regulariser, but note it also manufactures clusters of identical floor-valued
    eigenvalues under collapse — feeding the eigh-backward degeneracy that spd_logm
    guards against below. Use eps*trace(cov)/d for a scale-equivariant floor.
    """
    feat = feat - feat.mean(dim=-1, keepdim=True)
    d, t = feat.shape[1], feat.shape[-1]
    cov = feat @ feat.transpose(-1, -2) / (t - 1)
    eye = torch.eye(d, device=feat.device, dtype=feat.dtype)
    return cov + eps * eye


def spd_logm(C, eps=1e-5):
    """Differentiable symmetric matrix logarithm via eigendecomposition.

    Runs the eigendecomposition (and therefore its backward) in float64 for
    numerical stability: torch.linalg.eigh's backward carries 1/(lambda_i -
    lambda_j) terms that blow up to NaN when eigenvalues (near-)coincide — exactly
    the degeneracy a collapsing cov_proj produces, which is when the anti-collapse
    regulariser most needs a finite gradient. float64 is cheap at d_cov=32 and
    widens the usable eigen-gap by ~10x. The training loop additionally skips any
    optimizer step on a non-finite loss as a second line of defence.

    The clamp_min(eps) here is a redundant safety floor in the shipped path
    (temporal_covariance already adds 1e-4*I > eps=1e-5, so it never fires); it
    only matters if spd_logm is called on an un-floored matrix. It does NOT
    stabilise the backward — float64 + the loss-level isfinite guard do.
    """
    in_dtype = C.dtype
    C = 0.5 * (C + C.transpose(-1, -2))
    C64 = C.to(torch.float64)
    evals, evecs = torch.linalg.eigh(C64)
    log_evals = torch.log(evals.clamp_min(eps))
    out = evecs @ torch.diag_embed(log_evals) @ evecs.transpose(-1, -2)
    return out.to(in_dtype)


def upper_tri_vec(S):
    """[B, d, d] symmetric -> [B, d(d+1)/2]; off-diagonals scaled by sqrt(2) so
    the Euclidean norm of the vector equals the Frobenius norm of S."""
    d = S.shape[-1]
    idx = torch.triu_indices(d, d, device=S.device)
    w = torch.ones(idx.shape[1], device=S.device, dtype=S.dtype)
    w[idx[0] != idx[1]] = 2.0 ** 0.5
    return S[:, idx[0], idx[1]] * w


def tangent_features(feat, eps_cov=1e-4, eps_log=1e-5):
    """[B, d, T'] feature map -> [B, d(d+1)/2] Log-Euclidean tangent vector."""
    return upper_tri_vec(spd_logm(temporal_covariance(feat, eps_cov), eps_log))


@torch.no_grad()
def collapse_metrics(Z):
    """Anti-collapse diagnostics on a batch of embeddings Z: [N, D].

    Returns the effective rank (entropy of the singular-value distribution),
    the mean per-dimension std, and the off-diagonal covariance norm. A
    collapsing run shows eff_rank -> 1 and feat_std -> 0.
    """
    Z = Z - Z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Z)
    p = s / (s.sum() + 1e-12)
    eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum())
    std = Z.std(dim=0).mean()
    D = Z.shape[1]
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    off_norm = off.pow(2).sum().sqrt() / (D * (D - 1)) ** 0.5
    return {"eff_rank": round(eff_rank.item(), 3),
            "feat_std": round(std.item(), 4),
            "offdiag_cov": round(off_norm.item(), 5)}
