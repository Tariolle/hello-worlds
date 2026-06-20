import torch

from src.models.tcp_graph_jepa import TCPGraphJEPA, TCPGraphJEPAConfig


def _small_model():
    return TCPGraphJEPA(
        TCPGraphJEPAConfig(
            channels=22,
            time_steps=12,
            feature_dim=5,
            hidden_dim=16,
            temporal_layers=1,
            temporal_heads=4,
            graph_layers=1,
            dropout=0.0,
            mask_ratio=0.25,
        )
    )


def test_graph_jepa_forward_accepts_expected_shape():
    model = _small_model()
    x = torch.randn(2, 22, 12, 5)
    mask = model.make_mask(2, 12, x.device, mode="random", mask_ratio=0.25)
    out = model(x, mask=mask)
    assert out["pred"].shape == (2, 22, 12, 16)
    assert out["target"].shape == (2, 22, 12, 16)
    assert out["mask"].shape == (2, 22, 12)
    loss, logs = model.compute_loss(x, mask=mask)
    assert loss.ndim == 0
    assert logs["mask_ratio"] > 0


def test_tiny_fixed_batch_loss_can_decrease():
    torch.manual_seed(0)
    model = _small_model()
    x = torch.randn(4, 22, 12, 5)
    mask = torch.zeros(4, 22, 12, dtype=torch.bool)
    mask[:, :, 3:6] = True
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(6):
        opt.zero_grad(set_to_none=True)
        loss, _logs = model.compute_loss(x, mask=mask)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] <= losses[0]

