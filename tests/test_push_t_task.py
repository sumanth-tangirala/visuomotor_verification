"""Tests for PushTTask concrete impl."""
from __future__ import annotations

from pathlib import Path

from visuomotor_verification.task.base import Task
from visuomotor_verification.task.push_t import PushTTask


def _make_task(tmp_path: Path) -> PushTTask:
    return PushTTask(
        name="push_t",
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="rgb",
        sim_backend="physx_cuda",
        max_episode_steps=150,
        horizon=150,
        demo_path=str(tmp_path / "fake.h5"),
    )


def test_push_t_task_is_task_subclass(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert isinstance(t, Task)


def test_push_t_task_horizon(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert t.horizon == 150


def test_push_t_task_is_success_true(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert t.is_success(obs=None, info={"success": True}) is True


def test_push_t_task_is_success_false(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert t.is_success(obs=None, info={"success": False}) is False


def test_push_t_task_is_success_missing_key(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert t.is_success(obs=None, info={}) is False


def test_push_t_task_build_env_is_noop(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    # build_env should not raise; it's a no-op for ManiSkill.
    assert t.build_env(sim=None) is None


def test_push_t_task_is_success_accepts_tensor_like(tmp_path: Path) -> None:
    """If info['success'] is a tensor-like object (.item() returns scalar),
    is_success must coerce it before bool()."""
    t = _make_task(tmp_path)

    class FakeTensor:
        def __init__(self, v: bool) -> None:
            self._v = v
        def item(self) -> bool:
            return self._v

    assert t.is_success(obs=None, info={"success": FakeTensor(True)}) is True
    assert t.is_success(obs=None, info={"success": FakeTensor(False)}) is False


def test_push_t_task_is_success_handles_none_info(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert t.is_success(obs=None, info=None) is False
