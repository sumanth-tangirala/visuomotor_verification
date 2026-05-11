from __future__ import annotations

import pytest

from visuomotor_verification.policy.base import Policy
from visuomotor_verification.simulator.base import Simulator
from visuomotor_verification.task.base import Task
from visuomotor_verification.verifier.base import Verifier


@pytest.mark.parametrize("klass", [Simulator, Task, Policy, Verifier])
def test_abc_cannot_be_instantiated_directly(klass) -> None:
    with pytest.raises(TypeError):
        klass()  # type: ignore[abstract]


def test_concrete_simulator_subclass_works() -> None:
    class Dummy(Simulator):
        def reset(self, *, seed=None):
            return None
        def step(self, action):
            from visuomotor_verification.core.types import StepResult
            return StepResult(None, 0.0, False, False, {})
        def render(self, mode="rgb_array"):
            import numpy as np
            return np.zeros((1, 1, 3), dtype=np.uint8)
        def close(self):
            return None
        @property
        def observation_spec(self):
            return {}
        @property
        def action_spec(self):
            return {}

    s = Dummy()
    s.close()
