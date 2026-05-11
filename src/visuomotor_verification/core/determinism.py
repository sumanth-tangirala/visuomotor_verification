"""Determinism layer.

This is the ONLY module in the codebase that touches global RNGs and cuDNN
flags. All other code receives a `RunConfig` (or a component-specific seed)
and uses local RNG instances.

See docs/superpowers/specs/2026-05-11-foundations-design.md §5 for the design.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class RunMode(str, Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    MIXED = "mixed"


@dataclass(frozen=True)
class Seeds:
    master: int | None = None
    sim: int | None = None
    policy: int | None = None
    torch: int | None = None
    numpy: int | None = None
    python: int | None = None
    dataloader: int | None = None
    cuda_strict: bool = False


@dataclass(frozen=True)
class RunConfig:
    mode: RunMode
    seeds: Seeds
    allow_dirty: bool = False


def derive_seed(master: int, component: str) -> int:
    """Deterministic seed for a component, derived from the master seed.

    Uses blake2b (not Python's `hash()`, which is not stable across processes).
    Returns a value in [0, 2**32) so it fits all common RNG seed types.
    """
    digest = hashlib.blake2b(
        f"{master}:{component}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF
