"""Channel x time masking for TCP-Graph-JEPA.

A mask is a ``BoolTensor[B, C, T]`` where ``True`` marks positions to PREDICT
(hidden from the context encoder, supervised against the target encoder). Three
modes, all configurable and all respecting per-sample channel availability
(missing channels — e.g. TUAB has no A1/A2, so A1-T3 / T4-A2 are absent — are
never masked and never used as targets):

  * ``random``        — random channel-time tokens at ``mask_ratio``;
  * ``channel``       — whole channels masked over short time spans;
  * ``contralateral`` — mask one hemisphere (over a span) and predict it from the
    other side; the homologous edges are exactly what makes this solvable.

A run can mix modes via ``mode="mixed"`` (uniformly samples one mode per batch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .tcp_graph import LEFT_CHANNELS, RIGHT_CHANNELS, TCP_CHANNELS


@dataclass
class MaskConfig:
    mode: str = "random"            # random | channel | contralateral | mixed
    mask_ratio: float = 0.25        # for random mode (fraction of tokens)
    span_frac: float = 0.5          # for channel/contralateral: fraction of T in the span
    channel_frac: float = 0.3       # for channel mode: fraction of channels masked
    min_keep: int = 1               # always keep >=1 available token visible per sample


def _avail_mask(B, C, T, device, channel_mask):
    """Expand a per-channel availability mask to [B, C, T] (True = usable)."""
    if channel_mask is None:
        return torch.ones(B, C, T, dtype=torch.bool, device=device)
    cm = channel_mask.to(device=device, dtype=torch.bool)
    if cm.dim() == 1:
        cm = cm.unsqueeze(0).expand(B, C)
    return cm.unsqueeze(-1).expand(B, C, T)


def _channels_for_side(side: str, channels: List[str]) -> torch.Tensor:
    names = LEFT_CHANNELS if side == "left" else RIGHT_CHANNELS
    idx = [i for i, c in enumerate(channels) if c in names]
    return torch.tensor(idx, dtype=torch.long)


def make_mask(
    B: int,
    C: int,
    T: int,
    cfg: MaskConfig,
    channel_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    device=None,
    channels: Optional[List[str]] = None,
) -> torch.Tensor:
    """Return a ``BoolTensor[B, C, T]`` (True = predict). See module docstring."""
    target_device = (torch.device(device) if device is not None
                     else (channel_mask.device if channel_mask is not None
                           else torch.device("cpu")))
    # All randomness must live on the generator's device: a CPU generator with a
    # CUDA target otherwise raises a device-mismatch RuntimeError. Build the mask
    # on this 'work' device and move it to the target device on return.
    device = generator.device if generator is not None else target_device
    channels = channels or (TCP_CHANNELS if C == len(TCP_CHANNELS) else
                            [str(i) for i in range(C)])
    avail = _avail_mask(B, C, T, device, channel_mask)  # [B,C,T] bool

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device)

    mode = cfg.mode
    if mode == "mixed":
        modes = ["random", "channel", "contralateral"]
        pick = int(torch.randint(len(modes), (1,), generator=generator,
                                 device=device).item())
        mode = modes[pick]

    mask = torch.zeros(B, C, T, dtype=torch.bool, device=device)

    if mode == "random":
        draw = rand(B, C, T) < cfg.mask_ratio
        mask = draw & avail

    elif mode == "channel":
        n_mask = max(1, int(round(cfg.channel_frac * C)))
        span = max(1, int(round(cfg.span_frac * T)))
        for b in range(B):
            avail_ch = torch.nonzero(avail[b, :, 0], as_tuple=False).flatten()
            if avail_ch.numel() == 0:
                continue
            k = min(n_mask, avail_ch.numel())
            perm = torch.randperm(avail_ch.numel(), generator=generator,
                                  device=device)[:k]
            chosen = avail_ch[perm]
            start = 0 if T <= span else int(torch.randint(
                0, T - span + 1, (1,), generator=generator, device=device).item())
            mask[b, chosen, start:start + span] = True
        mask = mask & avail

    elif mode == "contralateral":
        if (_channels_for_side("left", channels).numel() == 0 and
                _channels_for_side("right", channels).numel() == 0):
            raise ValueError(
                "contralateral masking requires TCP channel names; pass channels=")
        span = max(1, int(round(cfg.span_frac * T)))
        for b in range(B):
            side = "left" if rand(1).item() < 0.5 else "right"
            idx = _channels_for_side(side, channels).to(device)
            if idx.numel() == 0:
                continue
            start = 0 if T <= span else int(torch.randint(
                0, T - span + 1, (1,), generator=generator, device=device).item())
            mask[b, idx, start:start + span] = True
        mask = mask & avail
    else:
        raise ValueError(f"unknown mask mode: {cfg.mode!r}")

    # Safety: never mask *all* available tokens of a sample (keep >= min_keep
    # visible) so the context encoder always has something to condition on.
    for b in range(B):
        av = avail[b]
        n_av = int(av.sum())
        if n_av == 0:
            continue
        if int(mask[b].sum()) >= n_av - cfg.min_keep + 1:
            # unmask a few random available tokens
            flat_av = torch.nonzero(av.flatten(), as_tuple=False).flatten()
            keep = flat_av[torch.randperm(flat_av.numel(), generator=generator,
                                          device=device)[:cfg.min_keep]]
            m = mask[b].flatten()
            m[keep] = False
            mask[b] = m.view(C, T)
        if int(mask[b].sum()) == 0:  # ensure at least one prediction target
            flat_av = torch.nonzero(av.flatten(), as_tuple=False).flatten()
            pick = flat_av[torch.randint(flat_av.numel(), (1,),
                                         generator=generator, device=device)]
            m = mask[b].flatten()
            m[pick] = True
            mask[b] = m.view(C, T)
    return mask.to(target_device)
