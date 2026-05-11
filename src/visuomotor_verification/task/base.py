"""Task ABC. A Task configures a Simulator and defines the success criterion."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from visuomotor_verification.core.types import Observation
from visuomotor_verification.simulator.base import Simulator


class Task(ABC):
    """Task interface.

    A task knows how to configure a simulator and judge success. Kept
    separate from Simulator so the same task can run in different simulators
    without rewriting task semantics.
    """

    @abstractmethod
    def build_env(self, sim: Simulator) -> None: ...

    @abstractmethod
    def is_success(self, obs: Observation, info: dict[str, Any]) -> bool: ...

    @property
    @abstractmethod
    def horizon(self) -> int: ...
