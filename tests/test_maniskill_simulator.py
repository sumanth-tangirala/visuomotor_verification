"""Tests for ManiSkillSimulator concrete impl."""
from __future__ import annotations

import numpy as np
import pytest

from visuomotor_verification.simulator.base import Simulator
from visuomotor_verification.simulator.maniskill import ManiSkillSimulator


@pytest.mark.slow
def test_maniskill_simulator_is_simulator_subclass() -> None:
    sim = ManiSkillSimulator(
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="state",
        sim_backend="physx_cpu",
        max_episode_steps=50,
    )
    try:
        assert isinstance(sim, Simulator)
    finally:
        sim.close()


@pytest.mark.slow
def test_maniskill_simulator_reset_and_step() -> None:
    sim = ManiSkillSimulator(
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="state",
        sim_backend="physx_cpu",
        max_episode_steps=50,
    )
    try:
        obs = sim.reset(seed=0)
        assert obs is not None
        # Sample a zero action; ManiSkill action spaces are (D,) Box[-1, 1].
        spec = sim.action_spec
        action_dim = spec["shape"][0]
        action = np.zeros(action_dim, dtype=np.float32)
        result = sim.step(action)
        assert result.obs is not None
        assert isinstance(result.reward, float)
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)
        assert isinstance(result.info, dict)
    finally:
        sim.close()


@pytest.mark.slow
def test_maniskill_simulator_observation_and_action_spec_shapes() -> None:
    sim = ManiSkillSimulator(
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="state",
        sim_backend="physx_cpu",
        max_episode_steps=50,
    )
    try:
        a_spec = sim.action_spec
        o_spec = sim.observation_spec
        assert "shape" in a_spec and "low" in a_spec and "high" in a_spec
        assert "shape" in o_spec or "spaces" in o_spec

        # observation_spec.shape must match the unbatched obs from reset() —
        # i.e. it must not carry a leading batch dimension.
        obs = sim.reset(seed=0)
        assert isinstance(obs, np.ndarray), f"expected ndarray in state mode, got {type(obs)}"
        assert tuple(obs.shape) == o_spec["shape"], (
            f"obs shape {obs.shape} != spec shape {o_spec['shape']}; "
            "leading batch dim leaked through observation_spec"
        )

        # action_spec.low must be writeable but disconnected from the Box's buffer.
        a_spec["low"][0] = -999.0  # mutate the returned copy
        # Re-read action_spec and confirm the env's Box is unaffected.
        fresh = sim.action_spec
        assert fresh["low"][0] != -999.0, "action_spec.low is a view, not a copy"
    finally:
        sim.close()


@pytest.mark.slow
def test_maniskill_simulator_render_returns_ndarray() -> None:
    sim = ManiSkillSimulator(
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="state",
        sim_backend="physx_cpu",
        max_episode_steps=50,
    )
    try:
        sim.reset(seed=0)
        frame = sim.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim >= 2
    finally:
        sim.close()
