from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

Action = np.ndarray
Observation = Any  # narrowed by concrete Simulator impls (dict for state, ndarray for image)
ObsSpec = dict[str, Any]
ActionSpec = dict[str, Any]


class StepResult(NamedTuple):
    obs: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class VerifierOutput(NamedTuple):
    score: float           # probability of success or scalar judgment
    label: bool | None     # binary prediction if applicable, else None
