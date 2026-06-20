"""Forward/backward + masking tests for the TCP-Graph-JEPA model."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from eb_jepa.graph_jepa.masking import MaskConfig, make_mask
from eb_jepa.graph_jepa.model import ModelConfig, build_model


def _model(T=20, F=5, **kw):
    cfg = ModelConfig(feature_dim=F, hidden_dim=32, n_temporal_layers=1,
                      n_graph_layers=2, n_heads=4, max_time=T, dropout=0.0, **kw)
    return build_model(cfg), cfg


def test_forward_shapes_and_loss():
    B, C, T, F = 4, 22, 20, 5
    model, _ = _model(T=T, F=F)
    x = torch.randn(B, C, T, F)
    cm = torch.ones(B, C, dtype=torch.bool)
    mask = make_mask(B, C, T, MaskConfig(mask_ratio=0.3), channel_mask=cm)
    out = model(x, mask, channel_mask=cm)
    assert out["pred"].shape == (B, C, T, model.cfg.hidden_dim)
    assert out["err"].shape == (B, C, T)
    assert torch.isfinite(out["loss"]) and out["loss"].item() >= 0


def test_backward_only_updates_context():
    B, C, T, F = 2, 22, 16, 5
    model, _ = _model(T=T, F=F, target_mode="ema")
    x = torch.randn(B, C, T, F)
    cm = torch.ones(B, C, dtype=torch.bool)
    mask = make_mask(B, C, T, MaskConfig(mask_ratio=0.4), channel_mask=cm)
    out = model(x, mask, channel_mask=cm)
    out["loss"].backward()
    ctx_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.context_encoder.parameters())
    tgt_grad = any(p.grad is not None for p in model.target_encoder.parameters())
    assert ctx_grad, "context encoder received no gradient"
    assert not tgt_grad, "target encoder must be stop-gradient"


def test_ema_update_changes_target():
    model, _ = _model()
    before = [p.clone() for p in model.target_encoder.parameters()]
    # perturb context params, then EMA
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(torch.randn_like(p))
    model.update_target(0.9)
    changed = any(not torch.allclose(b, a) for b, a in
                  zip(before, model.target_encoder.parameters()))
    assert changed, "EMA update did not move the target encoder"


def test_channel_mask_excludes_unavailable():
    B, C, T, F = 3, 22, 16, 5
    model, _ = _model(T=T, F=F)
    x = torch.randn(B, C, T, F)
    cm = torch.ones(B, C, dtype=torch.bool)
    cm[:, [8, 13]] = False                       # mimic TUAB missing A1-T3 / T4-A2
    mask = make_mask(B, C, T, MaskConfig(mask_ratio=0.5), channel_mask=cm)
    assert not mask[:, [8, 13]].any(), "unavailable channels must never be masked"
    out = model(x, mask, channel_mask=cm)
    assert not out["valid"][:, [8, 13]].any()


def test_shared_target_mode_runs():
    model, _ = _model(target_mode="shared")
    assert model.target_encoder is None
    B, C, T, F = 2, 22, 16, 5
    x = torch.randn(B, C, T, F)
    cm = torch.ones(B, C, dtype=torch.bool)
    mask = make_mask(B, C, T, MaskConfig(mask_ratio=0.3), channel_mask=cm)
    out = model(x, mask, channel_mask=cm)
    out["loss"].backward()                       # must not error
    assert torch.isfinite(out["loss"])


def test_mask_generator_device_and_contralateral_guard():
    # CPU generator + explicit device must not raise (device/generator footgun)
    g = torch.Generator().manual_seed(0)
    cm = torch.ones(2, 22, dtype=torch.bool)
    m = make_mask(2, 22, 20, MaskConfig(), channel_mask=cm, generator=g, device="cpu")
    assert m.shape == (2, 22, 20) and m.device.type == "cpu"
    # contralateral mode must fail loudly on non-TCP (numeric) channel names
    raised = False
    try:
        make_mask(2, 10, 20, MaskConfig(mode="contralateral"),
                  channel_mask=torch.ones(2, 10, dtype=torch.bool), generator=g)
    except ValueError:
        raised = True
    assert raised, "contralateral mode should raise on non-TCP channel names"


def test_mask_modes():
    B, C, T = 4, 22, 20
    cm = torch.ones(B, C, dtype=torch.bool)
    for mode in ("random", "channel", "contralateral", "mixed"):
        m = make_mask(B, C, T, MaskConfig(mode=mode, mask_ratio=0.25,
                                          span_frac=0.5, channel_frac=0.3),
                      channel_mask=cm)
        assert m.shape == (B, C, T)
        assert m.any(), f"mode {mode} produced an empty mask"
        # never mask everything available (min_keep visible)
        for b in range(B):
            assert int(m[b].sum()) < C * T


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("GRAPH_JEPA_FORWARD_TESTS_OK")
