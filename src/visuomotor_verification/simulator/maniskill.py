"""Concrete Simulator backed by a single ManiSkill gym env.

Training-time evaluation uses the vendored make_eval_envs (parallel CPU/GPU
vector envs). This class is for single-env use cases like trajectory
collection in subsequent PRs.
"""
from __future__ import annotations

import gymnasium as gym
import mani_skill.envs  # noqa: F401 -- registers ManiSkill envs
import numpy as np

from visuomotor_verification.core.types import (
    Action,
    ActionSpec,
    Observation,
    ObsSpec,
    StepResult,
)
from visuomotor_verification.simulator.base import Simulator


class ManiSkillSimulator(Simulator):
    """Single-env wrapper over `gym.make` for ManiSkill."""

    def __init__(
        self,
        *,
        env_id: str,
        control_mode: str,
        obs_mode: str,
        sim_backend: str,
        max_episode_steps: int,
    ) -> None:
        self._env = gym.make(
            env_id,
            control_mode=control_mode,
            obs_mode=obs_mode,
            sim_backend=sim_backend,
            max_episode_steps=max_episode_steps,
            reward_mode="sparse",
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
        )

    def reset(self, *, seed: int | None = None) -> Observation:
        obs, _info = self._env.reset(seed=seed)
        return self._to_numpy(obs)

    @staticmethod
    def _to_numpy(value: object) -> np.ndarray:
        """Convert a ManiSkill observation (torch.Tensor or ndarray) to a numpy array.

        ManiSkill returns batched torch tensors even for a single env.  We move
        to CPU, convert to numpy, and squeeze the leading batch dimension so
        the result matches ``single_observation_space.shape``.
        """
        import torch  # local import; torch is a ManiSkill dependency

        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        # Drop a leading batch dim of size 1 that ManiSkill adds for vector envs.
        if arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def step(self, action: Action) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return StepResult(
            obs=self._to_numpy(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        raw = self._env.render()
        return self._to_numpy(raw)

    def close(self) -> None:
        self._env.close()

    @property
    def observation_spec(self) -> ObsSpec:
        # ManiSkill's gym.make returns an env whose observation_space is the
        # batched (vector-env-style) space. Use single_observation_space for
        # the per-env spec that matches what reset()/step() actually return.
        env = self._env.unwrapped
        space = getattr(env, "single_observation_space", None) or self._env.observation_space
        if hasattr(space, "shape") and space.shape is not None:
            return {"shape": tuple(space.shape), "dtype": str(space.dtype)}
        # Dict observation space (rgb/depth modes): expose member spaces.
        return {"spaces": {k: dict(shape=tuple(v.shape), dtype=str(v.dtype))
                           for k, v in space.spaces.items()}}

    @property
    def action_spec(self) -> ActionSpec:
        # Use single_action_space if present (same reasoning as observation_spec).
        env = self._env.unwrapped
        space = getattr(env, "single_action_space", None) or self._env.action_space
        return {
            "shape": tuple(space.shape),
            "low": np.asarray(space.low).copy(),
            "high": np.asarray(space.high).copy(),
            "dtype": str(space.dtype),
        }
