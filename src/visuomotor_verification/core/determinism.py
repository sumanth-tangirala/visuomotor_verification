"""Determinism layer.

This is the ONLY module in the codebase that touches global RNGs and cuDNN
flags. All other code receives a `RunConfig` (or a component-specific seed)
and uses local RNG instances.

See docs/superpowers/specs/2026-05-11-foundations-design.md §5 for the design.
"""
from __future__ import annotations

import hashlib
import random
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from visuomotor_verification.core import git_info


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


class DirtyTreeError(RuntimeError):
    """Raised when a DETERMINISTIC run is attempted in a dirty git tree."""


def derive_seed(master: int, component: str) -> int:
    """Deterministic seed for a component, derived from the master seed.

    Uses blake2b (not Python's `hash()`, which is not stable across processes).
    Returns a value in [0, 2**32) so it fits all common RNG seed types.
    """
    digest = hashlib.blake2b(
        f"{master}:{component}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


_COMPONENTS = ("sim", "policy", "torch", "numpy", "python", "dataloader")


def resolve_seeds(cfg: RunConfig) -> Seeds:
    """Return a Seeds object with every field that *will be used* filled in.

    - DETERMINISTIC: all unset fields are derived from master. `master` is required.
    - MIXED: only explicitly-set fields are honored; unset ones stay None.
    - STOCHASTIC: every field is forced to None; warn if any was explicitly set.
    """
    if cfg.mode is RunMode.DETERMINISTIC:
        if cfg.seeds.master is None:
            raise ValueError(
                "RunMode.DETERMINISTIC requires seeds.master to be set"
            )
        derived = {
            comp: (
                getattr(cfg.seeds, comp)
                if getattr(cfg.seeds, comp) is not None
                else derive_seed(cfg.seeds.master, comp)
            )
            for comp in _COMPONENTS
        }
        return Seeds(master=cfg.seeds.master, cuda_strict=cfg.seeds.cuda_strict, **derived)

    if cfg.mode is RunMode.MIXED:
        return cfg.seeds  # already has only-explicit semantics

    # STOCHASTIC
    explicit = [
        c for c in (("master",) + _COMPONENTS) if getattr(cfg.seeds, c) is not None
    ]
    if explicit:
        warnings.warn(
            "STOCHASTIC mode ignores explicitly-set seeds: "
            f"{explicit}. This is almost certainly a config bug.",
            stacklevel=2,
        )
    return Seeds(cuda_strict=cfg.seeds.cuda_strict)


def seed_all(cfg: RunConfig, repo_root: Path | None) -> Seeds:
    """Seed every global RNG source according to `cfg`. Returns the resolved Seeds.

    The git-cleanliness gate (see §5.7 of the foundations spec) is enforced
    here when `repo_root` is not None. Pass `repo_root=None` only in tests
    that exercise pure seeding behavior outside of a repo context.

    This is the ONLY function that should touch torch.manual_seed, np.random.seed,
    random.seed, or cuDNN flags.
    """
    if repo_root is not None:
        info = git_info.collect(repo_root)
        if info["dirty"]:
            if cfg.mode is RunMode.DETERMINISTIC and not cfg.allow_dirty:
                raise DirtyTreeError(
                    "RunMode.DETERMINISTIC refuses to run in a dirty tree. "
                    "Commit or stash your changes, or set run.allow_dirty=true "
                    "explicitly (the run will record `git_diff` in metadata.json "
                    "regardless)."
                )
            warnings.warn(
                f"Working tree is dirty (mode={cfg.mode.value}); "
                "the run will proceed but git_sha will not uniquely identify "
                "the code that ran. The diff is captured in metadata.json.",
                UserWarning,
                stacklevel=2,
            )

    resolved = resolve_seeds(cfg)

    if resolved.python is not None:
        random.seed(resolved.python)
    if resolved.numpy is not None:
        np.random.seed(resolved.numpy)
    if resolved.torch is not None:
        torch.manual_seed(resolved.torch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved.torch)

    # cuDNN flags
    strict_cuda = cfg.mode is RunMode.DETERMINISTIC or (
        cfg.mode is RunMode.MIXED and resolved.cuda_strict
    )
    if strict_cuda:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return resolved
