"""Shape-contract + gradient + SPD-rank tests for the Fourier STFT encoder."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from examples.eeg.fourier_encoder import FourierEEGEncoder1D
from examples.eeg.geometry import tangent_features


def _enc(**kw):
    return FourierEEGEncoder1D(n_channels=19, n_fft=128, hop=32, d_model=256,
                               d_cov=32, d_hidden=64, n_temporal_layers=2, **kw)


def test_represent_and_feature_map_shapes():
    enc = _enc().eval()
    x = torch.randn(4, 19, 2000)
    fm = enc.feature_map(x)
    assert fm.shape[0] == 4 and fm.shape[1] == enc.out_dim == 256
    tprime = fm.shape[-1]
    assert tprime == 63, f"expected 63 STFT frames for T=2000, got {tprime}"
    assert enc.represent(x).shape == (4, 256)
    assert enc.cov_features(x).shape == (4, enc.d_cov, tprime)


def test_temporal_axis_supports_full_rank_covariance():
    # d_cov must be < T' so geometry.temporal_covariance is full rank (tangent arm)
    enc = _enc().eval()
    x = torch.randn(2, 19, 2000)
    assert enc.cov_features(x).shape[-1] > enc.d_cov
    tan = tangent_features(enc.cov_features(x))
    assert tan.shape == (2, enc.d_cov * (enc.d_cov + 1) // 2)
    assert torch.isfinite(tan).all()


def test_backward_reaches_all_learnable_params():
    enc = _enc()
    x = torch.randn(3, 19, 2000)
    enc.represent(x).pow(2).mean().backward()
    missing = [n for n, p in enc.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not missing, f"no gradient reached: {missing}"


def test_window_moves_with_module_and_is_not_a_parameter():
    enc = _enc()
    assert "window" in dict(enc.named_buffers())
    assert "window" not in dict(enc.named_parameters())


def test_variable_window_length():
    # Encoder must not hardcode T=2000; STFT frames scale with input length.
    enc = _enc().eval()
    for t in (1000, 2000, 3000):
        fm = enc.feature_map(torch.randn(2, 19, t))
        assert fm.shape[-1] == t // enc.hop + 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("FOURIER_ENCODER_TESTS_OK")
