"""PushT task. Judges success by reading info['success'] from the env step."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from visuomotor_verification.simulator.base import Simulator
from visuomotor_verification.task.base import Task


class PushTTask(Task):
    """ManiSkill PushT-v1 task wrapper.

    `build_env` is a no-op: ManiSkill's `gym.make` already wires the task
    into the env when constructed. Kept for ABC compliance and as a typed
    seam for future multi-simulator support.
    """

    def __init__(
        self,
        *,
        name: str,
        env_id: str,
        control_mode: str,
        obs_mode: str,
        sim_backend: str,
        max_episode_steps: int,
        horizon: int,
        demo_path: str,
    ) -> None:
        self.name = name
        self.env_id = env_id
        self.control_mode = control_mode
        self.obs_mode = obs_mode
        self.sim_backend = sim_backend
        self.max_episode_steps = max_episode_steps
        self._horizon = horizon
        self.demo_path = Path(demo_path)

    def build_env(self, sim: Simulator | None) -> None:
        return None

    def is_success(self, obs: Any, info: dict[str, Any] | None) -> bool:
        if not info:
            return False
        val = info.get("success", False)
        # ManiSkill may return success as a 0-d or (1,) torch tensor on GPU; coerce.
        item = getattr(val, "item", None)
        if callable(item):
            val = item()
        return bool(val)

    @property
    def horizon(self) -> int:
        return self._horizon
