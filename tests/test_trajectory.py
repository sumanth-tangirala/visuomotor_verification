from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visuomotor_verification.core.trajectory import Trajectory


def _make_trajectory() -> Trajectory:
    return Trajectory(
        observations=np.arange(30, dtype=np.float32).reshape(10, 3),
        actions=np.arange(20, dtype=np.float32).reshape(10, 2),
        rewards=np.linspace(0.0, 1.0, 10, dtype=np.float32),
        terminated=np.array([False] * 9 + [True]),
        truncated=np.zeros(10, dtype=bool),
        success=True,
        run_metadata={"mode": "deterministic", "master": 42, "git_sha": "abc"},
    )


def test_trajectory_basic_fields() -> None:
    t = _make_trajectory()
    assert t.observations.shape == (10, 3)
    assert t.actions.shape == (10, 2)
    assert t.success is True
    assert t.run_metadata["master"] == 42


def test_trajectory_npz_roundtrip(tmp_path: Path) -> None:
    t = _make_trajectory()
    p = tmp_path / "ep_0.npz"
    t.save_npz(p)
    loaded = Trajectory.load_npz(p)
    np.testing.assert_array_equal(loaded.observations, t.observations)
    np.testing.assert_array_equal(loaded.actions, t.actions)
    np.testing.assert_array_equal(loaded.rewards, t.rewards)
    np.testing.assert_array_equal(loaded.terminated, t.terminated)
    np.testing.assert_array_equal(loaded.truncated, t.truncated)
    assert loaded.success is True
    assert loaded.run_metadata == t.run_metadata


def test_trajectory_length() -> None:
    t = _make_trajectory()
    assert len(t) == 10


def test_trajectory_length_consistency_enforced() -> None:
    with pytest.raises(ValueError, match="length"):
        Trajectory(
            observations=np.zeros((10, 3)),
            actions=np.zeros((9, 2)),  # wrong length
            rewards=np.zeros(10),
            terminated=np.zeros(10, dtype=bool),
            truncated=np.zeros(10, dtype=bool),
            success=False,
            run_metadata={},
        )


def test_trajectory_npz_roundtrip_success_false(tmp_path: Path) -> None:
    t = _make_trajectory()
    t.success = False
    p = tmp_path / "ep_fail.npz"
    t.save_npz(p)
    loaded = Trajectory.load_npz(p)
    assert loaded.success is False


def test_trajectory_save_npz_normalizes_extension(tmp_path: Path) -> None:
    t = _make_trajectory()
    p = tmp_path / "ep_no_ext"
    t.save_npz(p)
    # File should be saved with .npz appended.
    assert (tmp_path / "ep_no_ext.npz").exists()
    loaded = Trajectory.load_npz(tmp_path / "ep_no_ext.npz")
    assert loaded.success is True
