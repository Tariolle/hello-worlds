"""Tiny YAML config loader + dataclass binding for TCP-Graph-JEPA.

Deliberately depends only on ``pyyaml`` (not ``omegaconf``) so the training /
eval / smoke entrypoints run unchanged on a laptop CPU and on the cluster venv.
Supports dotlist overrides like ``model.hidden_dim=64`` for CLI tweaking.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _coerce(s: str):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    return s


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` strings in place (creating nested keys as needed)."""
    for ov in overrides or []:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg


def bind(dc_type, d: Dict[str, Any]):
    """Build a dataclass of ``dc_type`` from dict ``d`` (ignoring unknown keys)."""
    fields = {f.name for f in dataclasses.fields(dc_type)}
    kw = {k: v for k, v in (d or {}).items() if k in fields}
    return dc_type(**kw)
