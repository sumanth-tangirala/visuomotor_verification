"""Smoke test: ManiSkill is installed and a push-T env can reset.

Marked with @pytest.mark.slow so it's easy to skip on CI if needed.
"""
from __future__ import annotations

import pytest


def test_maniskill_importable() -> None:
    import mani_skill  # noqa: F401


@pytest.mark.slow
def test_push_t_env_resets() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  -- registers envs

    push_t_ids = [k for k in gym.envs.registry.keys() if "PushT" in k]
    if not push_t_ids:
        pytest.skip(
            "No PushT env registered. Available push-* envs: "
            + str([k for k in gym.envs.registry.keys() if "push" in k.lower()])
        )
    env = gym.make(push_t_ids[0])
    try:
        obs, info = env.reset(seed=0)
        assert obs is not None
    finally:
        env.close()
