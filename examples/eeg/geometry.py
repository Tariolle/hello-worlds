"""SPD-manifold helpers and collapse diagnostics for the EEG-JEPA track.

The geometry-aware regularizer acts on the Log-Euclidean tangent of each
window's temporal feature covariance. Both views use the identity base point.
"""
import torch


def temporal_covariance(feat, eps=1e-4):
    """Map [B, d, T] feature maps to regularized [B, d, d] covariances."""
    feat = feat - feat.mean(dim=-1, keepdim=True)
    d, t = feat.shape[1], feat.shape[-1]
    cov = feat @ feat.transpose(-1, -2) / (t - 1)
    eye = torch.eye(d, device=feat.device, dtype=feat.dtype)
    return cov + eps * eye


def spd_logm(C, eps=1e-5):
    """Compute a differentiable symmetric matrix logarithm in float64."""
    input_dtype = C.dtype
    C = 0.5 * (C + C.transpose(-1, -2))
    covariance64 = C.to(torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance64)
    log_eigenvalues = torch.log(eigenvalues.clamp_min(eps))
    result = eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-1, -2)
    return result.to(input_dtype)


def upper_tri_vec(S):
    """Vectorize a symmetric matrix with Frobenius-norm-preserving weights."""
    d = S.shape[-1]
    indices = torch.triu_indices(d, d, device=S.device)
    weights = torch.ones(indices.shape[1], device=S.device, dtype=S.dtype)
    weights[indices[0] != indices[1]] = 2.0 ** 0.5
    return S[:, indices[0], indices[1]] * weights


def tangent_features(feat, eps_cov=1e-4, eps_log=1e-5):
    """Map [B, d, T] features to Log-Euclidean tangent vectors."""
    return upper_tri_vec(spd_logm(temporal_covariance(feat, eps_cov), eps_log))


@torch.no_grad()
def collapse_metrics(Z):
    """Return effective rank, mean standard deviation, and covariance norm."""
    centered = Z - Z.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / (singular_values.sum() + 1e-12)
    effective_rank = torch.exp(-(probabilities * (probabilities + 1e-12).log()).sum())
    feature_std = centered.std(dim=0).mean()
    dimension = centered.shape[1]
    covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    off_diagonal_norm = off_diagonal.pow(2).sum().sqrt() / (dimension * (dimension - 1)) ** 0.5
    return {
        "eff_rank": round(effective_rank.item(), 3),
        "feat_std": round(feature_std.item(), 4),
        "offdiag_cov": round(off_diagonal_norm.item(), 5),
    }
