from __future__ import annotations

import pytest

from visuomotor_verification.core.determinism import (
    RunConfig,
    RunMode,
    Seeds,
    derive_seed,
)


def test_runmode_values() -> None:
    assert RunMode("deterministic") is RunMode.DETERMINISTIC
    assert RunMode("stochastic") is RunMode.STOCHASTIC
    assert RunMode("mixed") is RunMode.MIXED


def test_seeds_default_all_none() -> None:
    s = Seeds()
    assert s.master is None
    assert s.sim is None
    assert s.policy is None
    assert s.torch is None
    assert s.numpy is None
    assert s.python is None
    assert s.dataloader is None
    assert s.cuda_strict is False


def test_runconfig_default_allow_dirty_false() -> None:
    cfg = RunConfig(mode=RunMode.STOCHASTIC, seeds=Seeds())
    assert cfg.allow_dirty is False


def test_derive_seed_deterministic_per_component() -> None:
    # Same master + same component => same seed
    assert derive_seed(42, "sim") == derive_seed(42, "sim")
    # Same master + different components => different seeds
    assert derive_seed(42, "sim") != derive_seed(42, "policy")
    # Different masters + same component => different seeds
    assert derive_seed(42, "sim") != derive_seed(43, "sim")


def test_derive_seed_in_uint32_range() -> None:
    for master in [0, 1, 42, 2**31, 2**32 - 1]:
        for comp in ["sim", "policy", "torch", "numpy", "python", "dataloader"]:
            seed = derive_seed(master, comp)
            assert 0 <= seed <= 2**32 - 1


def test_derive_seed_known_value() -> None:
    # Pin to a specific value so we catch accidental hash changes.
    # If this fails after a deliberate change, update the expected value AND
    # be aware that all previously-deterministic runs will now produce different
    # data with the same master seed.
    assert derive_seed(42, "sim") == 2792115813
