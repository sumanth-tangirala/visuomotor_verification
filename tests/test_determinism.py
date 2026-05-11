from __future__ import annotations

import random
import subprocess

import numpy as np
import pytest
import torch

from visuomotor_verification.core.determinism import (
    DirtyTreeError,
    RunConfig,
    RunMode,
    Seeds,
    derive_seed,
    resolve_seeds,
    seed_all,
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


def test_resolve_seeds_deterministic_fills_unset() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=42))
    resolved = resolve_seeds(cfg)
    assert resolved.master == 42
    # Every component seed should be filled deterministically.
    for comp in ["sim", "policy", "torch", "numpy", "python", "dataloader"]:
        v = getattr(resolved, comp)
        assert v is not None, f"{comp} should be derived in DETERMINISTIC mode"
        assert v == derive_seed(42, comp)


def test_resolve_seeds_deterministic_respects_explicit_override() -> None:
    cfg = RunConfig(
        mode=RunMode.DETERMINISTIC,
        seeds=Seeds(master=42, sim=999),
    )
    resolved = resolve_seeds(cfg)
    assert resolved.sim == 999
    assert resolved.policy == derive_seed(42, "policy")


def test_resolve_seeds_stochastic_clears_all() -> None:
    cfg = RunConfig(
        mode=RunMode.STOCHASTIC,
        seeds=Seeds(master=42, sim=999),  # explicit values should be ignored
    )
    with pytest.warns(UserWarning, match="STOCHASTIC mode ignores"):
        resolved = resolve_seeds(cfg)
    assert resolved.master is None
    assert resolved.sim is None
    assert resolved.policy is None


def test_resolve_seeds_mixed_only_uses_explicit() -> None:
    cfg = RunConfig(
        mode=RunMode.MIXED,
        seeds=Seeds(sim=7),
    )
    resolved = resolve_seeds(cfg)
    assert resolved.sim == 7
    assert resolved.policy is None  # not set => not seeded
    assert resolved.torch is None


def test_resolve_seeds_deterministic_requires_master() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds())
    with pytest.raises(ValueError, match="master"):
        resolve_seeds(cfg)


def test_seed_all_reproducible_python_random() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=42))
    # Skip the git-cleanliness gate for this test by using a non-repo cwd.
    seed_all(cfg, repo_root=None)
    a = [random.random() for _ in range(5)]
    seed_all(cfg, repo_root=None)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_all_reproducible_numpy() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=42))
    seed_all(cfg, repo_root=None)
    a = np.random.rand(5).tolist()
    seed_all(cfg, repo_root=None)
    b = np.random.rand(5).tolist()
    assert a == b


def test_seed_all_reproducible_torch() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=42))
    seed_all(cfg, repo_root=None)
    a = torch.randn(5).tolist()
    seed_all(cfg, repo_root=None)
    b = torch.randn(5).tolist()
    assert a == b


def test_seed_all_idempotent_under_same_config() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=7))
    seed_all(cfg, repo_root=None)
    seed_all(cfg, repo_root=None)  # second call must not raise


def test_seed_all_stochastic_runs_without_master() -> None:
    cfg = RunConfig(mode=RunMode.STOCHASTIC, seeds=Seeds())
    seed_all(cfg, repo_root=None)  # must not raise


def test_seed_all_mixed_reproducible_for_explicit_seeds() -> None:
    cfg = RunConfig(mode=RunMode.MIXED, seeds=Seeds(torch=99))
    seed_all(cfg, repo_root=None)
    a = torch.randn(5).tolist()
    seed_all(cfg, repo_root=None)
    b = torch.randn(5).tolist()
    assert a == b


def test_seed_all_returns_resolved_seeds() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=42))
    resolved = seed_all(cfg, repo_root=None)
    assert resolved.master == 42
    assert resolved.torch == derive_seed(42, "torch")
    assert resolved.sim == derive_seed(42, "sim")


def test_resolve_seeds_stochastic_does_not_warn_on_cuda_strict_only() -> None:
    cfg = RunConfig(mode=RunMode.STOCHASTIC, seeds=Seeds(cuda_strict=True))
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")  # turn any warning into an exception
        resolved = resolve_seeds(cfg)
    assert resolved.cuda_strict is True


def _make_git_repo(path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)
    (path / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "a.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_dirty_tree_blocks_deterministic(tmp_path) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=1))
    with pytest.raises(DirtyTreeError):
        seed_all(cfg, repo_root=tmp_path)


def test_dirty_tree_allowed_with_allow_dirty(tmp_path) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")
    cfg = RunConfig(
        mode=RunMode.DETERMINISTIC,
        seeds=Seeds(master=1),
        allow_dirty=True,
    )
    # allow_dirty bypasses the raise but still warns (run is not SHA-reproducible).
    with pytest.warns(UserWarning, match="dirty"):
        seed_all(cfg, repo_root=tmp_path)


def test_dirty_tree_warns_in_stochastic(tmp_path) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")
    cfg = RunConfig(mode=RunMode.STOCHASTIC, seeds=Seeds())
    with pytest.warns(UserWarning, match="dirty"):
        seed_all(cfg, repo_root=tmp_path)


def test_dirty_tree_warns_in_mixed(tmp_path) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")
    cfg = RunConfig(mode=RunMode.MIXED, seeds=Seeds())
    with pytest.warns(UserWarning, match="dirty"):
        seed_all(cfg, repo_root=tmp_path)


def test_clean_tree_passes_deterministic(tmp_path) -> None:
    _make_git_repo(tmp_path)
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=1))
    seed_all(cfg, repo_root=tmp_path)  # must not raise
