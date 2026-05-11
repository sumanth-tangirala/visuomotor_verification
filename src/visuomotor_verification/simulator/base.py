"""Simulator ABC. Concrete impls live in sibling modules (e.g. maniskill.py)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from visuomotor_verification.core.types import (
    Action,
    ActionSpec,
    Observation,
    ObsSpec,
    StepResult,
)


class Simulator(ABC):
    """Simulator interface. Wraps an underlying physics engine + gym env.

    Concrete impls are seed-aware: pass `seed` to `reset` to fix the next
    episode's init. The simulator's internal RNG state is otherwise managed
    by the simulator itself, not by global RNG.
    """

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> Observation: ...

    @abstractmethod
    def step(self, action: Action) -> StepResult: ...

    @abstractmethod
    def render(self, mode: str = "rgb_array") -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def observation_spec(self) -> ObsSpec: ...

    @property
    @abstractmethod
    def action_spec(self) -> ActionSpec: ...
