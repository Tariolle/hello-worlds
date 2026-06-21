"""Synthetic end-to-end smoke test for TCP-Graph-JEPA (no EEG data needed).

Generates random "normal" feature tensors ``[N,22,70,5]``, injects a high-
amplitude structured anomaly into one channel-time block of a held-out set,
trains the JEPA for a few steps on the normal windows, and confirms the anomaly
scoring assigns a *larger* latent error inside the injected region than the
window average. Checks pipeline correctness only — not model performance.

Run:  python -m archive.graph_jepa.v2.scripts.smoke_graph_jepa
"""
import numpy as np
import torch

from archive.graph_jepa.v2.core.masking import MaskConfig, make_mask
from archive.graph_jepa.v2.core.model import ModelConfig, build_model
from archive.graph_jepa.v2.core.scoring import ScoringConfig, window_error_maps


def run(steps: int = 120, seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, T, F = 22, 70, 5
    N = 96

    # normal windows ~ N(0,1) with mild spatial smoothness across channels
    base = torch.randn(N, C, T, F)
    normal = 0.7 * base + 0.3 * base.roll(1, dims=1)

    cfg = ModelConfig(feature_dim=F, hidden_dim=64, n_temporal_layers=1,
                      n_graph_layers=2, n_heads=4, max_time=T, dropout=0.0,
                      target_mode="ema", ema_decay=0.99)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)
    cm = torch.ones(C, dtype=torch.bool)
    mcfg = MaskConfig(mode="random", mask_ratio=0.25)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    bs = 32
    for step in range(steps):
        idx = torch.randint(0, N, (bs,))
        x = normal[idx].to(device)
        cmb = cm.unsqueeze(0).expand(bs, -1).to(device)
        mask = make_mask(bs, C, T, mcfg, channel_mask=cmb, generator=gen).to(device)
        out = model(x, mask, channel_mask=cmb)
        opt.zero_grad(set_to_none=True); out["loss"].backward(); opt.step()
        model.update_target()
        if verbose and step % 30 == 0:
            print(f"  step {step:3d} loss={out['loss'].detach().item():.4f}", flush=True)

    # build an anomalous window: inject a strong structured block
    c0, c1, t0, t1 = 5, 8, 30, 45
    anom = (0.7 * torch.randn(8, C, T, F) + 0.0)
    bump = torch.linspace(3.0, 6.0, t1 - t0).view(1, 1, -1, 1)
    anom[:, c0:c1, t0:t1, :] += bump            # high-amplitude structured anomaly
    cmb = cm.unsqueeze(0).expand(anom.shape[0], -1)

    scfg = ScoringConfig(n_masks=12, mask_ratio=0.4, seed=seed)
    heat = window_error_maps(model, anom.to(device), cmb, scfg, device=device)  # [B,C,T]
    heat = heat.mean(0).cpu().numpy()           # [C,T]

    region = heat[c0:c1, t0:t1].mean()
    overall = heat.mean()
    ratio = region / (overall + 1e-9)
    if verbose:
        print(f"[smoke] injected-region err={region:.4f}  overall err={overall:.4f}  "
              f"ratio={ratio:.2f}", flush=True)
    assert region > overall, (
        f"anomaly region error ({region:.4f}) not above average ({overall:.4f})")
    if verbose:
        print("GRAPH_JEPA_SMOKE_DONE", flush=True)
    return ratio


if __name__ == "__main__":
    run()
