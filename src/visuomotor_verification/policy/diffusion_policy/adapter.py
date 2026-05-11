"""Adapter wrapping the vendored DP baseline behind the Policy ABC.

NOT YET IMPLEMENTED. See foundations design spec §9 (initial deliverable scope).
The implementation lands in the next PR, which will:
  1. Wrap the vendored trainer in a class implementing `Policy.act` and `Policy.load`.
  2. Route the diffusion sampler's noise through `seeds.policy`.
  3. Drive training from our Hydra config.
"""
from __future__ import annotations

from pathlib import Path

from visuomotor_verification.core.types import Action, Observation
from visuomotor_verification.policy.base import Policy


class DiffusionPolicy(Policy):
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "DiffusionPolicy adapter is not yet implemented. "
            "Wiring lands in the next PR; see UPSTREAM.md and the foundations "
            "design spec §9."
        )

    def reset(self, *, seed: int | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def act(self, obs_history: list[Observation]) -> Action:  # pragma: no cover
        raise NotImplementedError

    def load(self, ckpt_path: Path) -> None:  # pragma: no cover
        raise NotImplementedError
