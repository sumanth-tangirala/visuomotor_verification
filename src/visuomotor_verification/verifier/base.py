"""Verifier ABC -- PROVISIONAL.

This will be revised when the first concrete verifier exists. The current
shape is intentionally minimal so the rest of the package can reference it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from visuomotor_verification.core.trajectory import Trajectory
from visuomotor_verification.core.types import VerifierOutput


class Verifier(ABC):
    """Judges trajectory quality. Provisional — see module docstring."""

    @abstractmethod
    def fit(self, trajectories: Iterable[Trajectory]) -> None: ...

    @abstractmethod
    def predict(self, trajectory: Trajectory) -> VerifierOutput: ...
