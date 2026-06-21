"""Fast contract tests for the retained SIGReg/PEIRA/SPD EEG pipeline."""

import torch

from eb_jepa.losses import BCS
from examples.eeg.encoder import EEGEncoder1D
from examples.eeg.geometry import tangent_features
from examples.eeg.peira import PEIRALoss


def test_core_encoder_exposes_representation_and_spd_tangent_features():
    encoder = EEGEncoder1D(n_channels=19, widths=(16, 24), d_model=32, d_cov=8)
    x = torch.randn(4, 19, 128)

    assert encoder.represent(x).shape == (4, 32)
    tangent = tangent_features(encoder.cov_features(x))
    assert tangent.shape == (4, 36)
    assert torch.isfinite(tangent).all()


def test_sigreg_loss_is_finite_and_differentiable():
    z1 = torch.randn(12, 16, requires_grad=True)
    z2 = torch.randn(12, 16, requires_grad=True)
    out = BCS(num_slices=12, lmbd=1.0)(z1, z2)

    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert z1.grad is not None and z2.grad is not None


def test_peira_loss_is_finite_and_differentiable():
    z1 = torch.randn(12, 16, requires_grad=True)
    z2 = torch.randn(12, 16, requires_grad=True)
    out = PEIRALoss(dim=16, lam=0.1)(z1, z2)

    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert z1.grad is not None and z2.grad is not None
