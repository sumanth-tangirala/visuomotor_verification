"""Policy ABC. Concrete impls live in sibling modules (e.g. diffusion_policy/adapter.py)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from visuomotor_verification.core.types import Action, Observation


class Policy(ABC):
    """Policy interface.

    `act` takes an observation history (not a single observation) because some
    policies (e.g. diffusion policy) consume an obs window. Single-obs policies
    just ignore all but the last entry.
    """

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> None: ...

    @abstractmethod
    def act(self, obs_history: list[Observation]) -> Action: ...

    @abstractmethod
    def load(self, ckpt_path: Path) -> None: ...
