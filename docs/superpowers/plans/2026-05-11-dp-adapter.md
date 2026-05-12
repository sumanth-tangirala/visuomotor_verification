# Diffusion Policy Adapter + Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `DiffusionPolicy(nn.Module, Policy)` adapter + `trainer.train()` + `scripts/train_policy.py` body so we can train a Diffusion Policy on ManiSkill's `PushT-v1` task with RGB observations, driven entirely by our Hydra config and determinism layer.

**Architecture:** The inner vendored package (`diffusion_policy/conditional_unet1d.py`, `plain_conv.py`, `evaluate.py`, `make_env.py`, `utils.py`) is reused as clean library code via a `sys.path` shim. The model wrapper (`Agent`) and dataset wrapper (`SmallDemoDataset_DiffusionPolicy`) from the vendored outer scripts (`train_rgbd.py`) cannot be imported directly — they reference module-level globals (`device`, `args.control_mode`) set only inside `__main__`. We re-implement them in our own `adapter.py` and `trainer.py`, with structures that mirror upstream and comments pointing to upstream line ranges for sync tracking. The training loop is owned by us; inference threads a `torch.Generator` through every `torch.randn`/`scheduler.step` call when `Policy.reset(seed=...)` was called with a non-`None` seed.

**Tech Stack:** Python 3.11, PyTorch 2.4 + CUDA 12.1, ManiSkill 3, diffusers (DDPMScheduler, EMAModel), Hydra/OmegaConf, pytest.

**Spec:** `docs/superpowers/specs/2026-05-11-dp-adapter-design.md`

**Working tree:** `/common/home/st1122/Projects/visuomotor_verification/` on a fresh feature branch off `main`. Activate the conda env with `conda run -n visuomotor_verification <cmd>`.

**Critical preconditions:** The foundations PR has been merged. `main` has all 62 tests passing. The vendored DP baseline lives pristine at `src/visuomotor_verification/policy/diffusion_policy/` (rooted at commit `a4a4f9272ad64b1564035874b605ceb687b63ed8` per `UPSTREAM.md`).

---

## File Structure

### New files

- `src/visuomotor_verification/policy/diffusion_policy/_vendor_import.py` — `sys.path` shim
- `src/visuomotor_verification/policy/diffusion_policy/trainer.py` — `TrainerConfig` + `DemoDataset` + `train()`
- `src/visuomotor_verification/simulator/maniskill.py` — `ManiSkillSimulator(Simulator)`
- `src/visuomotor_verification/task/push_t.py` — `PushTTask(Task)`
- `tests/test_vendor_import.py` — verify shim resolves inner-package imports
- `tests/test_maniskill_simulator.py` — verify simulator ABC compliance + basic reset/step
- `tests/test_push_t_task.py` — verify task ABC compliance + `is_success`
- `tests/test_dp_adapter.py` — `DiffusionPolicy` ABC compliance, generator routing, action chunking, load
- `tests/test_dp_agent_override.py` — equivalence sentinel vs. vendored Agent
- `tests/test_dp_trainer_step.py` — one training step runs without crash
- `tests/test_smoke_train_policy.py` — `@pytest.mark.slow` subprocess smoke

### Modified files

- `src/visuomotor_verification/policy/diffusion_policy/adapter.py` — replace stub with full `DiffusionPolicy(nn.Module, Policy)`
- `src/visuomotor_verification/policy/diffusion_policy/UPSTREAM.md` — point Runtime/Invocation note at the new shim
- `scripts/train_policy.py` — replace stub body with Hydra-driven trainer invocation
- `configs/train_policy.yaml` — `defaults: run: stochastic`, add `training:` block, tweak `hydra.run.dir` (no change)
- `configs/task/push_t.yaml` — full task config (env_id, control_mode, obs_mode, sim_backend, max_episode_steps, demo_path)
- `configs/policy/diffusion_policy.yaml` — full DP hyperparams

---

## Task 1: Add the vendor import shim and verify inner-package imports

**Files:**
- Create: `src/visuomotor_verification/policy/diffusion_policy/_vendor_import.py`
- Create: `tests/test_vendor_import.py`

The vendored inner package lives at `src/visuomotor_verification/policy/diffusion_policy/diffusion_policy/`. Its modules use bare `from diffusion_policy.X` imports, which only resolve if the *outer* directory (containing the inner `diffusion_policy/` directory) is on `sys.path`.

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_vendor_import.py`:

```python
"""Verify the _vendor_import shim makes inner DP package imports work."""
from __future__ import annotations


def test_inner_package_imports_after_shim() -> None:
    # Importing the shim must enable `from diffusion_policy.X` imports.
    from visuomotor_verification.policy.diffusion_policy import _vendor_import  # noqa: F401

    from diffusion_policy.utils import load_demo_dataset, worker_init_fn  # noqa: F401
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D  # noqa: F401
    from diffusion_policy.plain_conv import PlainConv  # noqa: F401
    from diffusion_policy.make_env import make_eval_envs  # noqa: F401
    from diffusion_policy.evaluate import evaluate  # noqa: F401


def test_shim_is_idempotent() -> None:
    # Importing the shim twice must not duplicate sys.path entries.
    import sys

    from visuomotor_verification.policy.diffusion_policy import _vendor_import  # noqa: F401

    before = list(sys.path)
    # Re-importing in Python is a no-op (cached in sys.modules), but if anyone
    # manually calls the shim function it should be idempotent.
    _vendor_import.install()
    _vendor_import.install()
    matches = [p for p in sys.path if p == _vendor_import.VENDOR_DIR]
    assert len(matches) == 1, f"shim added duplicate sys.path entries: {matches}"
```

- [ ] **Step 1.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_vendor_import.py -v
```

Expected: ImportError on `_vendor_import` module.

- [ ] **Step 1.3: Implement the shim**

Create `src/visuomotor_verification/policy/diffusion_policy/_vendor_import.py`:

```python
"""Insert the vendored DP outer directory onto sys.path so its inner package
imports resolve as `from diffusion_policy.X import ...`.

Why: the vendored ManiSkill DP baseline uses bare `from diffusion_policy.utils
import ...` etc. — those imports assume the inner `diffusion_policy/` directory
is a top-level package. Our nested layout (`src/visuomotor_verification/policy/
diffusion_policy/diffusion_policy/`) breaks that. Inserting the outer directory
onto sys.path makes the inner directory discoverable as `diffusion_policy`.

Side effect on import: calls `install()` once.
"""
from __future__ import annotations

import sys
from pathlib import Path

VENDOR_DIR = str(Path(__file__).resolve().parent)


def install() -> None:
    """Insert VENDOR_DIR at sys.path[0] if not already present."""
    if VENDOR_DIR in sys.path:
        return
    sys.path.insert(0, VENDOR_DIR)


install()
```

- [ ] **Step 1.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_vendor_import.py -v
```

Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/_vendor_import.py tests/test_vendor_import.py
git commit -m "Add vendor import shim for inner DP package"
```

---

## Task 2: Implement `ManiSkillSimulator` (concrete `Simulator` ABC)

**Files:**
- Create: `src/visuomotor_verification/simulator/maniskill.py`
- Create: `tests/test_maniskill_simulator.py`

Concrete `Simulator` impl that wraps `gym.make` for a single ManiSkill env. Used for ABC non-vacuity and (future) rollout collection. Training-time eval uses the vendored `make_eval_envs` directly, not this class.

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_maniskill_simulator.py`:

```python
"""Tests for ManiSkillSimulator concrete impl."""
from __future__ import annotations

import numpy as np
import pytest

from visuomotor_verification.simulator.base import Simulator
from visuomotor_verification.simulator.maniskill import ManiSkillSimulator


@pytest.mark.slow
def test_maniskill_simulator_is_simulator_subclass() -> None:
    sim = ManiSkillSimulator(
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="state",
        sim_backend="physx_cpu",
        max_episode_steps=50,
    )
    try:
        assert isinstance(sim, Simulator)
    finally:
        sim.close()


@pytest.mark.slow
def test_maniskill_simulator_reset_and_step() -> None:
    sim = ManiSkillSimulator(
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="state",
        sim_backend="physx_cpu",
        max_episode_steps=50,
    )
    try:
        obs = sim.reset(seed=0)
        assert obs is not None
        # Sample a zero action; ManiSkill action spaces are (D,) Box[-1, 1].
        spec = sim.action_spec
        action_dim = spec["shape"][0]
        action = np.zeros(action_dim, dtype=np.float32)
        result = sim.step(action)
        assert result.obs is not None
        assert isinstance(result.reward, float)
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)
        assert isinstance(result.info, dict)
    finally:
        sim.close()


@pytest.mark.slow
def test_maniskill_simulator_observation_and_action_spec_shapes() -> None:
    sim = ManiSkillSimulator(
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="state",
        sim_backend="physx_cpu",
        max_episode_steps=50,
    )
    try:
        a_spec = sim.action_spec
        o_spec = sim.observation_spec
        assert "shape" in a_spec and "low" in a_spec and "high" in a_spec
        assert "shape" in o_spec or "spaces" in o_spec  # state mode: shape; dict modes: spaces
    finally:
        sim.close()
```

- [ ] **Step 2.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_maniskill_simulator.py -v -m slow
```

Expected: ImportError.

- [ ] **Step 2.3: Implement `ManiSkillSimulator`**

Create `src/visuomotor_verification/simulator/maniskill.py`:

```python
"""Concrete Simulator backed by a single ManiSkill gym env.

Training-time evaluation uses the vendored make_eval_envs (parallel CPU/GPU
vector envs). This class is for single-env use cases like trajectory
collection in subsequent PRs.
"""
from __future__ import annotations

from typing import Any

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
        self._env_id = env_id
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
        return obs

    def step(self, action: Action) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return StepResult(
            obs=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        return np.asarray(self._env.render())

    def close(self) -> None:
        self._env.close()

    @property
    def observation_spec(self) -> ObsSpec:
        space = self._env.observation_space
        if hasattr(space, "shape") and space.shape is not None:
            return {"shape": tuple(space.shape), "dtype": str(space.dtype)}
        # Dict observation space (rgb/depth modes): expose member spaces.
        return {"spaces": {k: dict(shape=tuple(v.shape), dtype=str(v.dtype))
                           for k, v in space.spaces.items()}}

    @property
    def action_spec(self) -> ActionSpec:
        space = self._env.action_space
        return {
            "shape": tuple(space.shape),
            "low": np.asarray(space.low),
            "high": np.asarray(space.high),
            "dtype": str(space.dtype),
        }
```

- [ ] **Step 2.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_maniskill_simulator.py -v -m slow
```

Expected: 3 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/visuomotor_verification/simulator/maniskill.py tests/test_maniskill_simulator.py
git commit -m "Add ManiSkillSimulator concrete impl"
```

---

## Task 3: Implement `PushTTask` (concrete `Task` ABC)

**Files:**
- Create: `src/visuomotor_verification/task/push_t.py`
- Create: `tests/test_push_t_task.py`

Concrete `Task` impl. `build_env` is a no-op (ManiSkill's `gym.make` already wires the task into the env). `is_success` reads `info["success"]`.

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_push_t_task.py`:

```python
"""Tests for PushTTask concrete impl."""
from __future__ import annotations

from pathlib import Path

from visuomotor_verification.task.base import Task
from visuomotor_verification.task.push_t import PushTTask


def _make_task(tmp_path: Path) -> PushTTask:
    return PushTTask(
        name="push_t",
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="rgb",
        sim_backend="physx_cuda",
        max_episode_steps=150,
        horizon=150,
        demo_path=str(tmp_path / "fake.h5"),
    )


def test_push_t_task_is_task_subclass(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert isinstance(t, Task)


def test_push_t_task_horizon(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert t.horizon == 150


def test_push_t_task_is_success_true(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert t.is_success(obs=None, info={"success": True}) is True


def test_push_t_task_is_success_false(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert t.is_success(obs=None, info={"success": False}) is False


def test_push_t_task_is_success_missing_key(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    assert t.is_success(obs=None, info={}) is False


def test_push_t_task_build_env_is_noop(tmp_path: Path) -> None:
    t = _make_task(tmp_path)
    # build_env should not raise; it's a no-op for ManiSkill.
    assert t.build_env(sim=None) is None
```

- [ ] **Step 3.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_push_t_task.py -v
```

Expected: ImportError.

- [ ] **Step 3.3: Implement `PushTTask`**

Create `src/visuomotor_verification/task/push_t.py`:

```python
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

    def build_env(self, sim: Simulator) -> None:
        return None

    def is_success(self, obs: Any, info: dict[str, Any]) -> bool:
        return bool(info.get("success", False))

    @property
    def horizon(self) -> int:
        return self._horizon
```

- [ ] **Step 3.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_push_t_task.py -v
```

Expected: 6 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/visuomotor_verification/task/push_t.py tests/test_push_t_task.py
git commit -m "Add PushTTask concrete impl"
```

---

## Task 4: Implement `DiffusionPolicy.__init__` + model structure

**Files:**
- Modify: `src/visuomotor_verification/policy/diffusion_policy/adapter.py` (replace stub)
- Create: `tests/test_dp_adapter.py` (first test only — more added in Tasks 7–9)

Mirror the model construction in vendored `train_rgbd.py:Agent.__init__` (lines 246–288), with `device`, `act_dim`, `obs_state_dim`, and `rgb_shape` lifted to `__init__` kwargs instead of probed from an `env` object.

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_dp_adapter.py`:

```python
"""Tests for DiffusionPolicy adapter (Policy ABC subclass)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from visuomotor_verification.policy.base import Policy


def _build_dp(device: torch.device = None):
    """Construct a minimal DiffusionPolicy for testing.

    Uses small dimensions to keep tests fast. The shape contract is:
      - state: (obs_state_dim,)
      - rgb: (C, H, W), with H = W = 64 to keep PlainConv cheap
      - action: (act_dim,) in [-1, 1]
    """
    from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DiffusionPolicy(
        obs_horizon=2,
        act_horizon=1,
        pred_horizon=16,
        act_dim=4,
        obs_state_dim=9,
        rgb_shape=(3, 64, 64),
        include_rgb=True,
        include_depth=False,
        diffusion_step_embed_dim=64,
        unet_dims=[64, 128, 256],
        n_groups=8,
        num_diffusion_iters=100,
        device=device,
    ).to(device)


def test_diffusion_policy_is_policy_subclass() -> None:
    dp = _build_dp()
    assert isinstance(dp, Policy)


def test_diffusion_policy_is_nn_module() -> None:
    dp = _build_dp()
    assert isinstance(dp, torch.nn.Module)


def test_diffusion_policy_has_expected_submodules() -> None:
    dp = _build_dp()
    assert hasattr(dp, "visual_encoder")
    assert hasattr(dp, "noise_pred_net")
    assert hasattr(dp, "noise_scheduler")
    # Hyperparams accessible
    assert dp.obs_horizon == 2
    assert dp.act_horizon == 1
    assert dp.pred_horizon == 16
    assert dp.act_dim == 4
```

- [ ] **Step 4.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py -v
```

Expected: ImportError on `DiffusionPolicy` or attribute errors.

- [ ] **Step 4.3: Replace `adapter.py` stub with model structure**

Overwrite `src/visuomotor_verification/policy/diffusion_policy/adapter.py`:

```python
"""DiffusionPolicy adapter: implements the Policy ABC over the vendored DP
baseline's model components.

Structure mirrors vendored train_rgbd.py:Agent (lines 246-382). The class is
re-implemented here (not subclassed) because the upstream Agent references
module-level globals (`device`) set only inside `if __name__ == "__main__":`,
making it un-importable as a library class. Our re-implementation lifts those
globals to `__init__` kwargs.

UPSTREAM:
  src/visuomotor_verification/policy/diffusion_policy/train_rgbd.py
    lines 246-288: Agent.__init__   -> our __init__
    lines 290-310: Agent.encode_obs -> our encode_obs (Task 5)
    lines 312-337: Agent.compute_loss -> our compute_loss (Task 6)
    lines 339-381: Agent.get_action -> our get_action (Task 7)

If upstream Agent changes, update this file to match.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import _vendor_import  # noqa: F401 -- must run before vendored imports

import numpy as np
import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv

from visuomotor_verification.core.types import Action, Observation
from visuomotor_verification.policy.base import Policy


# Visual feature dim is fixed in upstream (line 271). Kept here as a constant.
_VISUAL_FEATURE_DIM = 256


class DiffusionPolicy(nn.Module, Policy):
    """RGB Diffusion Policy. Inherits nn.Module (for parameters/state_dict) and
    the Policy ABC (for reset/act/load).
    """

    def __init__(
        self,
        *,
        obs_horizon: int,
        act_horizon: int,
        pred_horizon: int,
        act_dim: int,
        obs_state_dim: int,
        rgb_shape: tuple[int, int, int],   # (C, H, W); C may be 3*num_cameras
        include_rgb: bool,
        include_depth: bool,
        diffusion_step_embed_dim: int,
        unet_dims: list[int],
        n_groups: int,
        num_diffusion_iters: int,
        device: torch.device,
    ) -> None:
        nn.Module.__init__(self)

        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.obs_state_dim = obs_state_dim
        self.include_rgb = include_rgb
        self.include_depth = include_depth

        total_visual_channels = rgb_shape[0] if include_rgb else 0
        # depth channels match rgb camera count; for now we don't separate.
        # When include_depth is True, the caller is responsible for passing the
        # combined channel count via rgb_shape[0] (mirrors upstream's behavior
        # of stacking rgb and depth as one tensor).

        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels,
            out_dim=_VISUAL_FEATURE_DIM,
            pool_feature_map=True,
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=obs_horizon * (_VISUAL_FEATURE_DIM + obs_state_dim),
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        self.num_diffusion_iters = num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self._device = device
        self._gen: torch.Generator | None = None
        # Action chunk cache for Policy.act. List of (act_dim,) np.ndarrays.
        self._action_cache: list[np.ndarray] = []

    # --- ABC stubs: filled in across Tasks 5-9 ----------------------------------
    def reset(self, *, seed: int | None = None) -> None:
        raise NotImplementedError("filled in Task 8")

    def act(self, obs_history: list[Observation]) -> Action:
        raise NotImplementedError("filled in Task 8")

    def load(self, ckpt_path: Path) -> None:
        raise NotImplementedError("filled in Task 9")
```

- [ ] **Step 4.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py -v
```

Expected: 3 passed (the 3 init/structure tests).

- [ ] **Step 4.5: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/adapter.py tests/test_dp_adapter.py
git commit -m "Add DiffusionPolicy __init__ and model structure"
```

---

## Task 5: Implement `DiffusionPolicy.encode_obs`

**Files:**
- Modify: `src/visuomotor_verification/policy/diffusion_policy/adapter.py`
- Modify: `tests/test_dp_adapter.py`

Mirror vendored `train_rgbd.py:Agent.encode_obs` (lines 290-310). Takes a dict-of-tensors observation, returns a flattened conditioning vector.

- [ ] **Step 5.1: Write the failing test**

Append to `tests/test_dp_adapter.py`:

```python
def test_encode_obs_shape() -> None:
    """encode_obs takes a dict {state, rgb} and returns (B, obs_horizon * (visual_dim + state_dim))."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device)
    B, obs_h = 2, 2
    obs_seq = {
        "state": torch.randn(B, obs_h, dp.obs_state_dim, device=device),
        "rgb": (torch.rand(B, obs_h, 3, 64, 64, device=device) * 255).to(torch.uint8),
    }
    feat = dp.encode_obs(obs_seq, eval_mode=True)
    # visual_feature_dim is 256 (constant in adapter)
    expected_dim = obs_h * (256 + dp.obs_state_dim)
    assert feat.shape == (B, expected_dim), f"got {feat.shape}, want (B={B}, {expected_dim})"
```

- [ ] **Step 5.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py::test_encode_obs_shape -v
```

Expected: AttributeError (no `encode_obs` method).

- [ ] **Step 5.3: Add `encode_obs`**

In `adapter.py`, insert before the `--- ABC stubs:` line:

```python
    def encode_obs(self, obs_seq: dict[str, torch.Tensor], eval_mode: bool) -> torch.Tensor:
        """Encode (obs_horizon-length) observations to a conditioning vector.

        Args:
            obs_seq: dict with keys 'state' (B, H, S), 'rgb' (B, H, C, IH, IW) uint8,
                optionally 'depth' (B, H, C, IH, IW).
            eval_mode: if True, skip data augmentation (we don't define `aug` here;
                upstream's optional aug isn't carried over in this PR).

        Returns:
            (B, H * (visual_feature_dim + obs_state_dim)) float tensor.

        Mirrors upstream train_rgbd.py:Agent.encode_obs (lines 290-310).
        """
        img_seq = None
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0
            img_seq = depth if img_seq is None else torch.cat([img_seq, depth], dim=2)

        batch_size = img_seq.shape[0]
        # (B, H, C, IH, IW) -> (B*H, C, IH, IW)
        img_seq_flat = img_seq.flatten(end_dim=1)
        visual_feature = self.visual_encoder(img_seq_flat)  # (B*H, D)
        visual_feature = visual_feature.reshape(batch_size, self.obs_horizon, _VISUAL_FEATURE_DIM)
        feature = torch.cat((visual_feature, obs_seq["state"]), dim=-1)
        return feature.flatten(start_dim=1)
```

- [ ] **Step 5.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py::test_encode_obs_shape -v
```

Expected: 1 passed.

- [ ] **Step 5.5: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/adapter.py tests/test_dp_adapter.py
git commit -m "Add DiffusionPolicy.encode_obs"
```

---

## Task 6: Implement `DiffusionPolicy.compute_loss`

**Files:**
- Modify: `src/visuomotor_verification/policy/diffusion_policy/adapter.py`
- Modify: `tests/test_dp_adapter.py`

Mirror vendored `train_rgbd.py:Agent.compute_loss` (lines 312-337). Returns a scalar MSE loss tensor.

- [ ] **Step 6.1: Write the failing test**

Append to `tests/test_dp_adapter.py`:

```python
def test_compute_loss_returns_scalar() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device)
    B = 2
    obs_seq = {
        "state": torch.randn(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": (torch.rand(B, dp.obs_horizon, 3, 64, 64, device=device) * 255).to(torch.uint8),
    }
    action_seq = torch.randn(B, dp.pred_horizon, dp.act_dim, device=device).clamp(-1, 1)
    loss = dp.compute_loss(obs_seq, action_seq)
    assert loss.dim() == 0, f"loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"


def test_compute_loss_backward() -> None:
    """Loss must backprop and produce gradients on every learnable parameter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device)
    B = 2
    obs_seq = {
        "state": torch.randn(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": (torch.rand(B, dp.obs_horizon, 3, 64, 64, device=device) * 255).to(torch.uint8),
    }
    action_seq = torch.randn(B, dp.pred_horizon, dp.act_dim, device=device).clamp(-1, 1)
    loss = dp.compute_loss(obs_seq, action_seq)
    loss.backward()
    n_with_grad = sum(1 for p in dp.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in dp.parameters() if p.requires_grad)
    assert n_with_grad > 0
    # Most params should get gradient
    assert n_with_grad >= n_total // 2
```

- [ ] **Step 6.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py::test_compute_loss_returns_scalar tests/test_dp_adapter.py::test_compute_loss_backward -v
```

Expected: AttributeError (no `compute_loss` method).

- [ ] **Step 6.3: Add `compute_loss`**

In `adapter.py`, insert right after `encode_obs`:

```python
    def compute_loss(self, obs_seq: dict[str, torch.Tensor], action_seq: torch.Tensor) -> torch.Tensor:
        """Training-time DDPM loss. Mirrors upstream train_rgbd.py:Agent.compute_loss
        (lines 312-337).

        Uses GLOBAL torch RNG for noise and timestep sampling — training is
        stochastic by spec (see foundations §5.4, dp-adapter design §6).
        """
        B = obs_seq["state"].shape[0]
        obs_cond = self.encode_obs(obs_seq, eval_mode=False)

        noise = torch.randn(
            (B, self.pred_horizon, self.act_dim),
            device=action_seq.device,
        )
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=action_seq.device,
        ).long()
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        return torch.nn.functional.mse_loss(noise_pred, noise)
```

- [ ] **Step 6.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py::test_compute_loss_returns_scalar tests/test_dp_adapter.py::test_compute_loss_backward -v
```

Expected: 2 passed.

- [ ] **Step 6.5: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/adapter.py tests/test_dp_adapter.py
git commit -m "Add DiffusionPolicy.compute_loss"
```

---

## Task 7: Implement `DiffusionPolicy.get_action` with `torch.Generator` routing

**Files:**
- Modify: `src/visuomotor_verification/policy/diffusion_policy/adapter.py`
- Modify: `tests/test_dp_adapter.py`

This is the inference path. Mirrors vendored `Agent.get_action` (lines 339-381) but threads `self._gen` into every `torch.randn` and `noise_scheduler.step` call. When `self._gen is None`, behavior matches upstream (global RNG).

Note: the upstream `get_action` permutes RGB from `(B, H, IH, IW, C)` to `(B, H, C, IH, IW)` in the no_grad block. Our caller is expected to pre-permute (since `encode_obs` already expects `(B, H, C, IH, IW)`), so we skip that permute here. If a caller passes channels-last RGB, they get a shape error from `encode_obs` — desired behavior.

- [ ] **Step 7.1: Write the failing test**

Append to `tests/test_dp_adapter.py`:

```python
def test_get_action_no_generator_uses_global_rng() -> None:
    """When _gen is None, two consecutive get_action calls should differ
    (global RNG advances)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device).eval()
    B = 1
    obs_seq = {
        "state": torch.zeros(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": torch.zeros(B, dp.obs_horizon, 3, 64, 64, device=device, dtype=torch.uint8),
    }
    torch.manual_seed(0)
    a1 = dp.get_action(obs_seq).cpu()
    a2 = dp.get_action(obs_seq).cpu()
    assert not torch.allclose(a1, a2), "global RNG should advance between calls"


def test_get_action_with_generator_is_reproducible() -> None:
    """Two get_action calls preceded by the same generator state must match."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device).eval()
    B = 1
    obs_seq = {
        "state": torch.zeros(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": torch.zeros(B, dp.obs_horizon, 3, 64, 64, device=device, dtype=torch.uint8),
    }
    dp._gen = torch.Generator(device=device).manual_seed(42)
    a1 = dp.get_action(obs_seq).cpu()
    dp._gen = torch.Generator(device=device).manual_seed(42)
    a2 = dp.get_action(obs_seq).cpu()
    assert torch.allclose(a1, a2), f"same seed should produce same action; diff={(a1 - a2).abs().max()}"


def test_get_action_shape() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device).eval()
    B = 3
    obs_seq = {
        "state": torch.zeros(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": torch.zeros(B, dp.obs_horizon, 3, 64, 64, device=device, dtype=torch.uint8),
    }
    a = dp.get_action(obs_seq)
    assert a.shape == (B, dp.act_horizon, dp.act_dim), a.shape
```

- [ ] **Step 7.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py -v -k get_action
```

Expected: AttributeError or wrong shapes (current `get_action` raises NotImplementedError? No — we only stubbed `reset`/`act`/`load`; `get_action` doesn't exist yet).

- [ ] **Step 7.3: Add `get_action`**

In `adapter.py`, insert right after `compute_loss`:

```python
    def get_action(self, obs_seq: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference. Mirrors upstream train_rgbd.py:Agent.get_action (339-381),
        but every torch.randn and noise_scheduler.step is threaded with
        `self._gen` so the diffusion sampler is reproducible per `seeds.policy`
        when `reset(seed=...)` was called.

        Args:
            obs_seq: dict with 'state' (B, H, S) and 'rgb' (B, H, C, IH, IW) uint8.
                Caller must pass RGB in channels-first layout.

        Returns:
            (B, act_horizon, act_dim) float tensor.
        """
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)

            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim),
                device=obs_seq["state"].device,
                generator=self._gen,
            )
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                    generator=self._gen,
                ).prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]
```

- [ ] **Step 7.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py -v -k get_action
```

Expected: 3 passed.

- [ ] **Step 7.5: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/adapter.py tests/test_dp_adapter.py
git commit -m "Add DiffusionPolicy.get_action with torch.Generator routing"
```

---

## Task 8: Implement `Policy.reset` and `Policy.act` (with action chunking)

**Files:**
- Modify: `src/visuomotor_verification/policy/diffusion_policy/adapter.py`
- Modify: `tests/test_dp_adapter.py`

`reset(seed=None)` clears the action cache and either nulls or builds `self._gen`. `act(obs_history)` returns a single action from a list-of-observations. When the cache is empty, it runs one `get_action` pass and fills the cache with `act_horizon` actions; otherwise it pops from the cache.

The Policy ABC defines `obs_history: list[Observation]`. For our adapter, an Observation is a dict `{"state": np.ndarray, "rgb": np.ndarray}`. We require the list length to equal `obs_horizon`; padding (when the episode hasn't filled the buffer yet) is the caller's responsibility — the rollout collection script in the next PR will handle that.

- [ ] **Step 8.1: Write the failing test**

Append to `tests/test_dp_adapter.py`:

```python
def _fake_obs_history(dp) -> list:
    """Construct an obs_history of length obs_horizon, each obs being a
    dict with numpy state and rgb tensors."""
    out = []
    for _ in range(dp.obs_horizon):
        out.append(
            {
                "state": np.zeros((dp.obs_state_dim,), dtype=np.float32),
                "rgb": np.zeros((3, 64, 64), dtype=np.uint8),
            }
        )
    return out


def test_reset_with_none_clears_generator() -> None:
    dp = _build_dp()
    dp._gen = torch.Generator(device=dp._device).manual_seed(1)
    dp._action_cache.append(np.zeros(dp.act_dim, dtype=np.float32))
    dp.reset(seed=None)
    assert dp._gen is None
    assert dp._action_cache == []


def test_reset_with_seed_builds_generator() -> None:
    dp = _build_dp()
    dp.reset(seed=42)
    assert dp._gen is not None
    assert isinstance(dp._gen, torch.Generator)


def test_act_returns_single_action_array() -> None:
    dp = _build_dp().eval()
    dp.reset(seed=0)
    history = _fake_obs_history(dp)
    a = dp.act(history)
    assert isinstance(a, np.ndarray)
    assert a.shape == (dp.act_dim,)


def test_act_seeds_reproduce_first_action() -> None:
    dp = _build_dp().eval()
    history = _fake_obs_history(dp)

    dp.reset(seed=7)
    a1 = dp.act(history)
    dp.reset(seed=7)
    a2 = dp.act(history)
    assert np.allclose(a1, a2), f"seed 7 should reproduce; diff={np.abs(a1 - a2).max()}"


def test_act_chunking_serves_from_cache_then_requeries() -> None:
    """With act_horizon=2 (override the default _build_dp here), the first
    act() triggers a denoise pass, the next act() reads from cache, the third
    act() triggers another denoise pass."""
    from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = DiffusionPolicy(
        obs_horizon=2, act_horizon=2, pred_horizon=16,
        act_dim=4, obs_state_dim=9, rgb_shape=(3, 64, 64),
        include_rgb=True, include_depth=False,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256],
        n_groups=8, num_diffusion_iters=100, device=device,
    ).to(device).eval()
    dp.reset(seed=11)

    history = _fake_obs_history(dp)
    # Patch get_action to count calls.
    calls = []
    orig = dp.get_action
    def counting(obs_seq):
        calls.append(1)
        return orig(obs_seq)
    dp.get_action = counting

    a1 = dp.act(history)
    a2 = dp.act(history)
    a3 = dp.act(history)

    # 2 denoise passes total (one fills 2 actions, the third triggers a refill).
    assert len(calls) == 2, f"expected 2 get_action calls, got {len(calls)}"
    assert a1.shape == (dp.act_dim,) and a2.shape == (dp.act_dim,) and a3.shape == (dp.act_dim,)
```

- [ ] **Step 8.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py -v -k "reset or act"
```

Expected: NotImplementedError (current stubs raise) or assertion failures.

- [ ] **Step 8.3: Implement `reset` and `act`**

In `adapter.py`, replace the existing `reset` and `act` method stubs (the ones raising `NotImplementedError`) with:

```python
    def reset(self, *, seed: int | None = None) -> None:
        """Reset per-episode state. If `seed` is provided, build a local
        torch.Generator on `self._device` seeded with it; subsequent inference
        is reproducible per seed. If `seed` is None, fall back to global RNG.
        """
        self._action_cache.clear()
        if seed is None:
            self._gen = None
        else:
            self._gen = torch.Generator(device=self._device).manual_seed(int(seed))

    def act(self, obs_history: list[Observation]) -> Action:
        """Return a single action. Implements action chunking: each get_action
        call returns `act_horizon` actions, which we serve one-at-a-time from
        the cache. When the cache is empty we run another denoise pass.

        `obs_history` must be a list of length `obs_horizon`, each element a
        dict with `state` (np.ndarray of shape (obs_state_dim,)) and `rgb`
        (np.ndarray of shape (C, H, W), uint8). Padding the buffer for the
        first few steps of an episode is the caller's responsibility.
        """
        if not self._action_cache:
            obs_seq = self._stack_obs_for_inference(obs_history)
            with torch.no_grad():
                action_seq = self.get_action(obs_seq)  # (1, act_horizon, act_dim)
            arr = action_seq[0].cpu().numpy()
            self._action_cache = [arr[i] for i in range(arr.shape[0])]
        return self._action_cache.pop(0)

    def _stack_obs_for_inference(self, obs_history: list[Observation]) -> dict[str, torch.Tensor]:
        """Build a batched (B=1) obs_seq dict suitable for `get_action`."""
        if len(obs_history) != self.obs_horizon:
            raise ValueError(
                f"obs_history length {len(obs_history)} != obs_horizon {self.obs_horizon}"
            )
        states = np.stack([o["state"] for o in obs_history], axis=0)   # (H, S)
        rgbs = np.stack([o["rgb"] for o in obs_history], axis=0)       # (H, C, IH, IW)
        return {
            "state": torch.from_numpy(states).float().unsqueeze(0).to(self._device),
            "rgb": torch.from_numpy(rgbs).unsqueeze(0).to(self._device),
        }
```

- [ ] **Step 8.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py -v -k "reset or act"
```

Expected: 5 passed (`test_reset_with_none_clears_generator`, `test_reset_with_seed_builds_generator`, `test_act_returns_single_action_array`, `test_act_seeds_reproduce_first_action`, `test_act_chunking_serves_from_cache_then_requeries`).

- [ ] **Step 8.5: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/adapter.py tests/test_dp_adapter.py
git commit -m "Add DiffusionPolicy reset and act with action chunking"
```

---

## Task 9: Implement `Policy.load`

**Files:**
- Modify: `src/visuomotor_verification/policy/diffusion_policy/adapter.py`
- Modify: `tests/test_dp_adapter.py`

Upstream checkpoints save `{'agent': state_dict, 'ema_agent': state_dict}`. The EMA weights are the deployment weights. `load(path)` loads the EMA weights into the model.

- [ ] **Step 9.1: Write the failing test**

Append to `tests/test_dp_adapter.py`:

```python
def test_load_restores_ema_state_dict(tmp_path) -> None:
    """Save a fake checkpoint with distinct agent vs. ema_agent weights, then
    load() and verify the EMA weights are what's restored."""
    dp = _build_dp()
    # Construct a fake "ema" state dict: zero out all params, save under 'ema_agent'.
    agent_sd = {k: v.clone() for k, v in dp.state_dict().items()}
    ema_sd = {k: torch.zeros_like(v) for k, v in dp.state_dict().items()}
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"agent": agent_sd, "ema_agent": ema_sd}, ckpt_path)

    # Mutate weights so load has something to overwrite.
    with torch.no_grad():
        for p in dp.parameters():
            p.add_(1.0)
    dp.load(ckpt_path)
    # All learnable params should now be ~0 (from ema_sd).
    for p in dp.parameters():
        assert torch.allclose(p, torch.zeros_like(p)), "ema weights not loaded"


def test_load_missing_ema_key_raises(tmp_path) -> None:
    dp = _build_dp()
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"agent": dp.state_dict()}, ckpt_path)  # missing 'ema_agent'
    with pytest.raises(KeyError):
        dp.load(ckpt_path)
```

- [ ] **Step 9.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py -v -k load
```

Expected: NotImplementedError.

- [ ] **Step 9.3: Implement `load`**

In `adapter.py`, replace the `load` stub:

```python
    def load(self, ckpt_path: Path) -> None:
        """Load EMA weights from a training checkpoint.

        Checkpoints written by `trainer.train()` (and by the vendored upstream)
        are dicts with keys 'agent' and 'ema_agent'. EMA weights are the
        deployment weights for diffusion policy.
        """
        ckpt = torch.load(ckpt_path, map_location=self._device)
        if "ema_agent" not in ckpt:
            raise KeyError(
                f"checkpoint {ckpt_path} has no 'ema_agent' key; got keys: {list(ckpt.keys())}"
            )
        self.load_state_dict(ckpt["ema_agent"])
```

- [ ] **Step 9.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_adapter.py -v -k load
```

Expected: 2 passed.

- [ ] **Step 9.5: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/adapter.py tests/test_dp_adapter.py
git commit -m "Add DiffusionPolicy.load (EMA weight restoration)"
```

---

## Task 10: Implement `DemoDataset` in `trainer.py`

**Files:**
- Create: `src/visuomotor_verification/policy/diffusion_policy/trainer.py` (first slice — dataset only; rest added in Tasks 11–12)
- Modify: `tests/test_dp_trainer_step.py` (will be created in this task)

Mirror vendored `train_rgbd.py:SmallDemoDataset_DiffusionPolicy` (lines 124-244), with `device` and `control_mode` lifted to `__init__` kwargs. The dataset preloads everything into GPU memory and pre-computes sliding-window (traj_idx, start, end) tuples for diffusion policy training.

This test requires a real demo file on disk. We gate it on the demo's existence — skip if absent so plan execution doesn't require demo download.

- [ ] **Step 10.1: Write the failing test**

Create `tests/test_dp_trainer_step.py`:

```python
"""Tests for trainer.py: DemoDataset, TrainerConfig, train()."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


DEMO_PATH = Path.home() / ".maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"


def _need_demo() -> None:
    if not DEMO_PATH.exists():
        pytest.skip(
            f"Demo file not found at {DEMO_PATH}. "
            "Run the demo-prep recipe in docs/superpowers/specs/2026-05-11-dp-adapter-design.md §1."
        )


@pytest.mark.slow
def test_demo_dataset_loads_two_trajectories() -> None:
    _need_demo()
    from visuomotor_verification.policy.diffusion_policy.trainer import DemoDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DemoDataset(
        data_path=DEMO_PATH,
        device=device,
        control_mode="pd_ee_delta_pose",
        env_id="PushT-v1",
        obs_mode="rgb",
        num_traj=2,
        obs_horizon=2,
        pred_horizon=16,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert "observations" in sample and "actions" in sample
    obs = sample["observations"]
    assert obs["state"].shape[0] == 2          # obs_horizon
    assert obs["rgb"].shape[0] == 2            # obs_horizon
    assert sample["actions"].shape[0] == 16    # pred_horizon
```

- [ ] **Step 10.2: Run, expect failure (or skip)**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_trainer_step.py -v -m slow
```

Expected: ImportError on `DemoDataset` if `_vendor_import` and `trainer.py` are absent; skip if demo file missing.

- [ ] **Step 10.3: Create `trainer.py` with `DemoDataset`**

Create `src/visuomotor_verification/policy/diffusion_policy/trainer.py`:

```python
"""Trainer module: DemoDataset + TrainerConfig + train().

Re-implements the dataset wrapper from vendored train_rgbd.py because the
upstream class references module-level globals (`args.control_mode`) set only
inside `if __name__ == "__main__":`. We lift those globals to __init__ kwargs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional

from . import _vendor_import  # noqa: F401 -- must run before vendored imports

import gymnasium as gym
import mani_skill.envs  # noqa: F401 -- registers ManiSkill envs
import numpy as np
import torch
import torch.optim as optim
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import (
    IterationBasedBatchSampler,
    build_state_obs_extractor,
    convert_obs,
    load_demo_dataset,
    worker_init_fn,
)

from visuomotor_verification.core.determinism import Seeds
from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy


def _reorder_keys(d: dict, ref: Any) -> dict:
    """Mirror upstream train_rgbd.py:reorder_keys (lines 114-121)."""
    out = {}
    for k, v in ref.items() if isinstance(ref, dict) else ref.spaces.items():
        if isinstance(v, (dict, spaces.Dict)):
            out[k] = _reorder_keys(d[k], v)
        else:
            out[k] = d[k]
    return out


class DemoDataset(Dataset):
    """DP training dataset. Mirrors upstream
    SmallDemoDataset_DiffusionPolicy (train_rgbd.py:124-244). Pre-loads h5 demos
    to `device` memory and precomputes sliding-window slice indices.

    UPSTREAM: train_rgbd.py:124-244. If upstream's dataset preprocessing
    changes, update this class.
    """

    def __init__(
        self,
        data_path: Path,
        *,
        device: torch.device,
        control_mode: str,
        env_id: str,
        obs_mode: str,
        num_traj: Optional[int],
        obs_horizon: int,
        pred_horizon: int,
    ) -> None:
        # Probe the env's original observation space so we can reorder demo
        # obs dicts to match.
        tmp_env = gym.make(
            env_id,
            control_mode=control_mode,
            obs_mode=obs_mode,
            sim_backend="physx_cpu",  # CPU probe is enough for the obs_space
            max_episode_steps=1,
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
        )
        original_obs_space = tmp_env.observation_space
        self.include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
        self.include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
        tmp_env.close()

        obs_process_fn = partial(
            convert_obs,
            concat_fn=partial(np.concatenate, axis=-1),
            transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
            state_obs_extractor=build_state_obs_extractor(env_id),
            depth="rgbd" in str(data_path),
        )

        trajectories = load_demo_dataset(str(data_path), num_traj=num_traj, concat=False)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = _reorder_keys(obs_traj_dict, original_obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.tensor(
                    _obs_traj_dict["depth"].astype(np.float32),
                    device=device, dtype=torch.float16,
                )
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(device)
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(device)
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict_list[0].keys())

        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.tensor(
                trajectories["actions"][i], device=device, dtype=torch.float32,
            )

        # Compute (traj_idx, start, end) sliding windows.
        if "delta_pos" in control_mode or control_mode == "base_pd_joint_vel_arm_pd_joint_vel":
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device,
            )
        else:
            raise NotImplementedError(
                f"Control Mode {control_mode} not supported (upstream restriction)"
            )

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.slices: list[tuple[int, int, int]] = []
        n_traj = len(trajectories["actions"])
        for traj_idx in range(n_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        traj_idx, start, end = self.slices[index]
        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq: dict[str, torch.Tensor] = {}
        for k, v in obs_traj.items():
            chunk = v[max(0, start): start + self.obs_horizon]
            if start < 0:
                pad = torch.stack([chunk[0]] * abs(start), dim=0)
                chunk = torch.cat((pad, chunk), dim=0)
            obs_seq[k] = chunk
        act_seq = self.trajectories["actions"][traj_idx][max(0, start): end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        L = self.trajectories["actions"][traj_idx].shape[0]
        if end > L:
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        assert obs_seq["state"].shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        return {"observations": obs_seq, "actions": act_seq}
```

- [ ] **Step 10.4: Run, expect pass (or skip)**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_trainer_step.py -v -m slow
```

Expected: 1 passed (if demo file exists) or 1 skipped (if not). If skipped, this is fine — the test will run on machines that have downloaded the demos.

- [ ] **Step 10.5: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/trainer.py tests/test_dp_trainer_step.py
git commit -m "Add DemoDataset to trainer.py (mirrors vendored SmallDemoDataset_DiffusionPolicy)"
```

---

## Task 11: Add `TrainerConfig` dataclass to `trainer.py`

**Files:**
- Modify: `src/visuomotor_verification/policy/diffusion_policy/trainer.py`

`TrainerConfig` is a frozen dataclass that bundles every knob the trainer needs. Tests for this are folded into Task 12 (where `train()` consumes it).

- [ ] **Step 11.1: Add the dataclass**

In `trainer.py`, insert after the imports and before `_reorder_keys`:

```python
@dataclass(frozen=True)
class TrainerConfig:
    """All knobs required by `train()`. Built by `scripts/train_policy.py` from
    the Hydra DictConfig (`cfg.task`, `cfg.policy`, `cfg.training`, plus
    resolved seeds from `cfg.run`)."""

    # demo + env (from cfg.task)
    demo_path: Path
    env_id: str
    control_mode: str
    obs_mode: str
    max_episode_steps: int
    sim_backend: str

    # DP hyperparams (from cfg.policy)
    obs_horizon: int
    act_horizon: int
    pred_horizon: int
    diffusion_step_embed_dim: int
    unet_dims: list[int]
    n_groups: int
    num_diffusion_iters: int

    # training (from cfg.training)
    total_iters: int
    batch_size: int
    lr: float
    num_demos: Optional[int]
    num_dataload_workers: int
    log_freq: int
    eval_freq: int
    save_freq: Optional[int]
    num_eval_episodes: int
    num_eval_envs: int

    # determinism
    seeds: Seeds
    device: torch.device
```

- [ ] **Step 11.2: Add an import smoke test**

Append to `tests/test_dp_trainer_step.py`:

```python
def test_trainer_config_importable_and_constructible() -> None:
    from visuomotor_verification.policy.diffusion_policy.trainer import TrainerConfig
    from visuomotor_verification.core.determinism import Seeds

    cfg = TrainerConfig(
        demo_path=Path("/dev/null"),
        env_id="PushT-v1", control_mode="pd_ee_delta_pose",
        obs_mode="rgb", max_episode_steps=150, sim_backend="physx_cuda",
        obs_horizon=2, act_horizon=1, pred_horizon=16,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256],
        n_groups=8, num_diffusion_iters=100,
        total_iters=2, batch_size=4, lr=1e-4,
        num_demos=2, num_dataload_workers=0,
        log_freq=1, eval_freq=10_000, save_freq=None,
        num_eval_episodes=2, num_eval_envs=1,
        seeds=Seeds(),
        device=torch.device("cpu"),
    )
    assert cfg.env_id == "PushT-v1"
```

- [ ] **Step 11.3: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_trainer_step.py::test_trainer_config_importable_and_constructible -v
```

Expected: 1 passed.

- [ ] **Step 11.4: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/trainer.py tests/test_dp_trainer_step.py
git commit -m "Add TrainerConfig dataclass to trainer.py"
```

---

## Task 12: Implement `trainer.train()` (the main training loop)

**Files:**
- Modify: `src/visuomotor_verification/policy/diffusion_policy/trainer.py`
- Modify: `tests/test_dp_trainer_step.py`

The training loop owns: dataloader, model + EMA, AdamW + cosine LR, the forward/backward/step cycle, periodic eval (via vendored `evaluate`), save-on-best checkpoint, tensorboard logging. Returns the path to the best checkpoint.

Mirrors upstream train loop (train_rgbd.py:512-608) with these deltas:
- No tyro args; everything comes from `TrainerConfig`.
- No upstream `random.seed`/`np.random.seed`/`torch.manual_seed`/cudnn flags — `seed_all` has already been called.
- `worker_init_fn` base_seed = `cfg.seeds.dataloader` (or current torch RNG if None — matches upstream fallback).
- Eval envs seeded via `cfg.seeds.sim`: `range(cfg.seeds.sim, cfg.seeds.sim + num_eval_envs)` if set, else `range(num_eval_envs)`.
- Checkpoints written under `run_dir / "checkpoints"` (not `runs/<name>`).
- Tensorboard logs written to `run_dir / "logs"` (writer passed in by caller, not built here).
- No `FlattenRGBDObservationWrapper` if obs_mode is `"state"`; only for `"rgb"`/`"rgbd"`.

- [ ] **Step 12.1: Write the failing tests**

Append to `tests/test_dp_trainer_step.py`:

```python
@pytest.mark.slow
def test_train_runs_one_iteration_without_eval(tmp_path) -> None:
    """Minimal train() invocation: 2 iters, eval_freq much larger than total
    so no eval runs, demo limited to 2 trajectories. Verify checkpoint exists.
    Requires demo file present."""
    _need_demo()
    from visuomotor_verification.policy.diffusion_policy.trainer import TrainerConfig, train
    from visuomotor_verification.core.determinism import Seeds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainerConfig(
        demo_path=DEMO_PATH,
        env_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        obs_mode="rgb",
        max_episode_steps=150,
        sim_backend="physx_cuda",
        obs_horizon=2, act_horizon=1, pred_horizon=16,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256],
        n_groups=8, num_diffusion_iters=100,
        total_iters=2, batch_size=2, lr=1e-4,
        num_demos=2, num_dataload_workers=0,
        log_freq=1,
        eval_freq=10_000,            # > total_iters, so no eval in training
        save_freq=2,                  # force a non-eval-based checkpoint at iter=2
        num_eval_episodes=2, num_eval_envs=1,
        seeds=Seeds(),
        device=device,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    writer = SummaryWriter(str(run_dir / "logs"))
    try:
        result = train(cfg, run_dir, writer)
    finally:
        writer.close()
    assert result is not None  # path to last/best checkpoint
    assert (run_dir / "checkpoints").exists()
    # The save_freq=2 should have produced a checkpoint at iter 2.
    ckpts = list((run_dir / "checkpoints").iterdir())
    assert len(ckpts) >= 1, f"no checkpoints written to {run_dir / 'checkpoints'}"
```

Add at top of file:
```python
from torch.utils.tensorboard import SummaryWriter
```

- [ ] **Step 12.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_trainer_step.py::test_train_runs_one_iteration_without_eval -v -m slow
```

Expected: ImportError or AttributeError.

- [ ] **Step 12.3: Implement `train()`**

Append to `trainer.py`:

```python
def _build_eval_envs(cfg: TrainerConfig):
    """Construct the vector of eval envs using the vendored make_eval_envs."""
    env_kwargs = dict(
        control_mode=cfg.control_mode,
        reward_mode="sparse",
        obs_mode=cfg.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=cfg.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=cfg.obs_horizon)
    wrappers = []
    if cfg.obs_mode in ("rgb", "rgbd", "rgb+depth"):
        wrappers = [FlattenRGBDObservationWrapper]
    return make_eval_envs(
        cfg.env_id,
        cfg.num_eval_envs,
        cfg.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=None,            # skip video recording in this PR
        wrappers=wrappers,
    )


def _build_policy(cfg: TrainerConfig, envs) -> DiffusionPolicy:
    """Construct DiffusionPolicy from env-derived shapes + cfg hyperparams."""
    act_dim = int(envs.single_action_space.shape[0])
    obs_state_dim = int(envs.single_observation_space["state"].shape[1])
    include_rgb = "rgb" in envs.single_observation_space.spaces
    include_depth = "depth" in envs.single_observation_space.spaces
    rgb_channels = (
        envs.single_observation_space["rgb"].shape[-1] if include_rgb else 0
    )
    depth_channels = (
        envs.single_observation_space["depth"].shape[-1] if include_depth else 0
    )
    # rgb_shape[0] aggregates both modalities, matching how encode_obs stacks them.
    total_visual_channels = rgb_channels + depth_channels
    # Pull H, W from rgb (or depth if no rgb).
    if include_rgb:
        _, h, w, _ = envs.single_observation_space["rgb"].shape  # (H_steps, H, W, C)
    else:
        _, h, w, _ = envs.single_observation_space["depth"].shape
    return DiffusionPolicy(
        obs_horizon=cfg.obs_horizon,
        act_horizon=cfg.act_horizon,
        pred_horizon=cfg.pred_horizon,
        act_dim=act_dim,
        obs_state_dim=obs_state_dim,
        rgb_shape=(total_visual_channels, h, w),
        include_rgb=include_rgb,
        include_depth=include_depth,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        unet_dims=cfg.unet_dims,
        n_groups=cfg.n_groups,
        num_diffusion_iters=cfg.num_diffusion_iters,
        device=cfg.device,
    ).to(cfg.device)


def train(cfg: TrainerConfig, run_dir: Path, log: SummaryWriter) -> Path:
    """Run the DP training loop. Writes checkpoints to `run_dir/checkpoints/`
    and tensorboard scalars via `log`. Returns the path of the most recent
    saved checkpoint (best-by-metric if eval ran, else the last save_freq tick).

    Structure mirrors upstream train_rgbd.py:567-608. Differences:
      - No re-seeding (caller has already done seed_all).
      - Checkpoint paths come from run_dir, not 'runs/<name>'.
      - SummaryWriter is owned by caller.
    """
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    envs = _build_eval_envs(cfg)

    # Seed eval envs deterministically per-env if sim seed is provided.
    if cfg.seeds.sim is not None:
        # action_space.seed is per-env when seeded via the wrapper; if our
        # SyncVectorEnv/AsyncVectorEnv exposes individual env seeds, hand them in.
        try:
            envs.action_space.seed(cfg.seeds.sim)
            envs.observation_space.seed(cfg.seeds.sim)
        except Exception:
            pass  # not all vector envs support seeding from outside

    dataset = DemoDataset(
        data_path=cfg.demo_path,
        device=cfg.device,
        control_mode=cfg.control_mode,
        env_id=cfg.env_id,
        obs_mode=cfg.obs_mode,
        num_traj=cfg.num_demos,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=cfg.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, cfg.total_iters)
    base_seed = cfg.seeds.dataloader  # None falls back to upstream behavior
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_dataload_workers,
        worker_init_fn=(lambda wid: worker_init_fn(wid, base_seed=base_seed))
        if cfg.num_dataload_workers > 0 else None,
        persistent_workers=(cfg.num_dataload_workers > 0),
    )

    agent = _build_policy(cfg, envs)
    ema_agent = _build_policy(cfg, envs)

    optimizer = optim.AdamW(
        params=agent.parameters(), lr=cfg.lr,
        betas=(0.95, 0.999), weight_decay=1e-6,
    )
    lr_sched = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=cfg.total_iters,
    )
    ema = EMAModel(parameters=agent.parameters(), power=0.75)

    best_metrics: dict[str, float] = {}
    last_ckpt: Optional[Path] = None

    def _save(tag: str) -> Path:
        ema.copy_to(ema_agent.parameters())
        path = ckpt_dir / f"{tag}.pt"
        torch.save(
            {"agent": agent.state_dict(), "ema_agent": ema_agent.state_dict()},
            path,
        )
        return path

    agent.train()
    for it, batch in enumerate(train_loader):
        loss = agent.compute_loss(batch["observations"], batch["actions"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_sched.step()
        ema.step(agent.parameters())

        if it % cfg.log_freq == 0:
            log.add_scalar("losses/total_loss", float(loss.item()), it)
            log.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], it)

        if cfg.eval_freq > 0 and it % cfg.eval_freq == 0 and it > 0:
            ema.copy_to(ema_agent.parameters())
            ema_agent.eval()
            metrics = evaluate(
                cfg.num_eval_episodes, ema_agent, envs, cfg.device, cfg.sim_backend
            )
            agent.train()
            for k, vs in metrics.items():
                m = float(np.mean(vs))
                log.add_scalar(f"eval/{k}", m, it)
                if k in ("success_once", "success_at_end") and m > best_metrics.get(k, -1.0):
                    best_metrics[k] = m
                    last_ckpt = _save(f"best_eval_{k}")

        if cfg.save_freq is not None and it % cfg.save_freq == 0 and it > 0:
            last_ckpt = _save(f"iter_{it}")

    # Final save unconditionally.
    last_ckpt = _save("final")
    envs.close()
    return last_ckpt
```

- [ ] **Step 12.4: Run, expect pass (or skip)**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_trainer_step.py::test_train_runs_one_iteration_without_eval -v -m slow
```

Expected: 1 passed (if demo present) or 1 skipped.

- [ ] **Step 12.5: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/trainer.py tests/test_dp_trainer_step.py
git commit -m "Add trainer.train(): main DP training loop"
```

---

## Task 13: Update Hydra config files

**Files:**
- Modify: `configs/task/push_t.yaml`
- Modify: `configs/policy/diffusion_policy.yaml`
- Modify: `configs/train_policy.yaml`

- [ ] **Step 13.1: Overwrite `configs/task/push_t.yaml`**

```yaml
_target_: visuomotor_verification.task.push_t.PushTTask
name: push_t
env_id: PushT-v1
control_mode: pd_ee_delta_pose
obs_mode: rgb
sim_backend: physx_cuda
max_episode_steps: 150
horizon: 150
demo_path: ${oc.env:HOME}/.maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5
```

- [ ] **Step 13.2: Overwrite `configs/policy/diffusion_policy.yaml`**

```yaml
_target_: visuomotor_verification.policy.diffusion_policy.adapter.DiffusionPolicy
name: diffusion_policy
checkpoint: null

obs_horizon: 2
act_horizon: 1
pred_horizon: 16
diffusion_step_embed_dim: 64
unet_dims: [64, 128, 256]
n_groups: 8
num_diffusion_iters: 100
```

- [ ] **Step 13.3: Overwrite `configs/train_policy.yaml`**

```yaml
defaults:
  - run: stochastic
  - simulator: maniskill
  - task: push_t
  - policy: diffusion_policy
  - storage: default
  - _self_

experiment_name: ???

training:
  total_iters: 50000
  batch_size: 64
  lr: 1.0e-4
  num_demos: null
  num_dataload_workers: 0
  log_freq: 1000
  eval_freq: 5000
  save_freq: null
  num_eval_episodes: 100
  num_eval_envs: 100
  wandb:
    enabled: false
    project: visuomotor_verification
    entity: null

hydra:
  run:
    dir: ${storage.root}/policies/${task.name}/${policy.name}/${experiment_name}-${now:%Y-%m-%d_%H-%M-%S}
  job:
    chdir: false
```

- [ ] **Step 13.4: Verify YAML parses + Hydra composes**

```bash
conda run -n visuomotor_verification python -c "
from hydra import initialize_config_dir, compose
from pathlib import Path
import os
os.environ.setdefault('HOME', str(Path.home()))
with initialize_config_dir(version_base=None, config_dir=str(Path('configs').resolve())):
    cfg = compose(config_name='train_policy', overrides=['experiment_name=test'])
    print('composed ok; experiment_name =', cfg.experiment_name)
    print('task.demo_path =', cfg.task.demo_path)
    print('training.batch_size =', cfg.training.batch_size)
"
```

Expected: prints the resolved fields without error.

- [ ] **Step 13.5: Commit**

```bash
git add configs/train_policy.yaml configs/task/push_t.yaml configs/policy/diffusion_policy.yaml
git commit -m "Wire DP + push-T into Hydra configs"
```

---

## Task 14: Implement `scripts/train_policy.py` body

**Files:**
- Modify: `scripts/train_policy.py`

Replace the stub with a Hydra-driven body that builds `TrainerConfig` from the composed `cfg`, runs `train()`, and writes the best checkpoint path into `metadata.json`. The prologue from `scripts/_common.py` handles `RunConfig.from_hydra` + `git_info.collect` + `seed_all` + initial `metadata.json`.

- [ ] **Step 14.1: Overwrite `scripts/train_policy.py`**

```python
"""Train a Diffusion Policy on push-T.

Usage:
    python scripts/train_policy.py experiment_name=push_t_dp_v1

One-time demo prep is documented in
docs/superpowers/specs/2026-05-11-dp-adapter-design.md §1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import prologue

from visuomotor_verification.core.determinism import RunConfig, resolve_seeds
from visuomotor_verification.policy.diffusion_policy.trainer import (
    TrainerConfig,
    train,
)


@hydra.main(version_base=None, config_path="../configs", config_name="train_policy")
def main(cfg: DictConfig) -> None:
    run_dir = prologue(cfg, script_name="train_policy.py")

    run_cfg = RunConfig.from_hydra(cfg)
    resolved = resolve_seeds(run_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer_cfg = TrainerConfig(
        demo_path=Path(cfg.task.demo_path),
        env_id=cfg.task.env_id,
        control_mode=cfg.task.control_mode,
        obs_mode=cfg.task.obs_mode,
        max_episode_steps=cfg.task.max_episode_steps,
        sim_backend=cfg.task.sim_backend,
        obs_horizon=cfg.policy.obs_horizon,
        act_horizon=cfg.policy.act_horizon,
        pred_horizon=cfg.policy.pred_horizon,
        diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
        unet_dims=list(cfg.policy.unet_dims),
        n_groups=cfg.policy.n_groups,
        num_diffusion_iters=cfg.policy.num_diffusion_iters,
        total_iters=cfg.training.total_iters,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        num_demos=cfg.training.num_demos,
        num_dataload_workers=cfg.training.num_dataload_workers,
        log_freq=cfg.training.log_freq,
        eval_freq=cfg.training.eval_freq,
        save_freq=cfg.training.save_freq,
        num_eval_episodes=cfg.training.num_eval_episodes,
        num_eval_envs=cfg.training.num_eval_envs,
        seeds=resolved,
        device=device,
    )

    if not trainer_cfg.demo_path.exists():
        raise FileNotFoundError(
            f"Demo file not found: {trainer_cfg.demo_path}. "
            "Run the demo-prep recipe in "
            "docs/superpowers/specs/2026-05-11-dp-adapter-design.md §1."
        )

    log = SummaryWriter(str(run_dir / "logs"))
    if cfg.training.wandb.enabled:
        import wandb
        wandb.init(
            project=cfg.training.wandb.project,
            entity=cfg.training.wandb.entity,
            name=run_dir.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
        )

    best_ckpt = train(trainer_cfg, run_dir, log)
    log.close()
    print(f"[train_policy] last checkpoint: {best_ckpt}")

    # Annotate metadata.json with the best checkpoint path.
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text())
        payload.setdefault("output_artifacts", {})["best_checkpoint"] = str(best_ckpt)
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 14.2: Verify the script at least parses + imports**

```bash
conda run -n visuomotor_verification python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
import importlib.util
spec = importlib.util.spec_from_file_location('train_policy', 'scripts/train_policy.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('train_policy.main exists:', hasattr(mod, 'main'))
"
```

Expected: prints `True`. Any ImportError here is a wiring bug.

- [ ] **Step 14.3: Run the existing smoke test (from foundations) to confirm no regression**

```bash
conda run -n visuomotor_verification pytest tests/test_smoke_scripts.py -v -m slow
```

Expected: 4 passed. The smoke test runs in stochastic mode with `experiment_name=smoke_train` and the demo path may not exist — the body now raises `FileNotFoundError`, which is the right behavior. But wait: the foundations smoke test uses `run=stochastic storage.root=<tmp>` and doesn't override the demo path, so the train script will hit the `FileNotFoundError`. That means the smoke test will fail.

We need to update the foundations smoke test to pass `task.demo_path=/dev/null` and let it fail at the next step inside `prologue` only when the metadata file isn't writable. Actually, the smoke test asserts metadata.json is written, then expects a non-zero exit. The `FileNotFoundError` happens *after* `prologue()` writes metadata.json, so the smoke test should still pass because:
  1. prologue writes metadata.json
  2. demo check raises FileNotFoundError
  3. script exits non-zero (which we don't assert)

Verify by running:
```bash
conda run -n visuomotor_verification pytest tests/test_smoke_scripts.py::test_stub_writes_metadata -v -m slow
```

If it fails, update `tests/test_smoke_scripts.py` to also pass `task.demo_path=/tmp/nonexistent.h5` for the train_policy case (no-op since FileNotFoundError still raises, but explicit). If the test passes (because prologue ran before the demo check), nothing to do.

- [ ] **Step 14.4: Commit**

```bash
git add scripts/train_policy.py
git commit -m "Wire scripts/train_policy.py to trainer.train() via Hydra"
```

If `tests/test_smoke_scripts.py` needed an update, include it:

```bash
git add tests/test_smoke_scripts.py
git commit --amend --no-edit
```

---

## Task 15: Add the equivalence sentinel test (`test_dp_agent_override.py`)

**Files:**
- Create: `tests/test_dp_agent_override.py`

This test ensures our re-implemented `DiffusionPolicy.get_action` produces byte-identical output to the vendored `Agent.get_action` when given the same weights and the same global RNG state. It monkey-patches the vendored module's `device` global so the upstream class is constructible; if that monkey-patch ever stops working (upstream refactor), the test skips with a diagnostic message.

- [ ] **Step 15.1: Write the test**

Create `tests/test_dp_agent_override.py`:

```python
"""Equivalence sentinel: our DiffusionPolicy.get_action must match the vendored
Agent.get_action byte-for-byte under shared global RNG and identical weights.
Drift here means our re-implementation has diverged from upstream."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _maybe_import_vendored_agent():
    """Import the vendored train_rgbd module and monkey-patch the `device`
    global. Returns the Agent class, or None if the monkey-patch fails
    (e.g., upstream refactored)."""
    from visuomotor_verification.policy.diffusion_policy import _vendor_import  # noqa: F401
    try:
        import train_rgbd
    except ImportError as e:
        pytest.skip(f"upstream train_rgbd not importable: {e}")
    if not hasattr(train_rgbd, "Agent"):
        pytest.skip("upstream train_rgbd has no Agent class (refactored?)")
    # Inject the `device` global so Agent's compute_loss/get_action don't NameError.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rgbd.device = device
    return train_rgbd, device


def _fake_env(act_dim: int, obs_state_dim: int, rgb_shape: tuple[int, int, int, int]):
    """Build a fake env object exposing the shape probes Agent.__init__ uses.
    rgb_shape is (cameras*3, H, W, C_per_camera) — Agent reads shape[-1] for
    channel count."""
    obs_shape_state = (1, obs_state_dim)
    obs_space = SimpleNamespace(
        keys=lambda: {"state", "rgb"},
        __getitem__=lambda self, k: SimpleNamespace(
            shape=obs_shape_state if k == "state" else rgb_shape,
        ),
    )
    action_space = SimpleNamespace(
        shape=(act_dim,), high=np.ones(act_dim), low=-np.ones(act_dim),
    )
    return SimpleNamespace(
        single_observation_space=obs_space,
        single_action_space=action_space,
    )


@pytest.mark.slow
def test_get_action_matches_vendored_agent_under_shared_seed():
    train_rgbd, device = _maybe_import_vendored_agent()

    # Match _build_dp() params from test_dp_adapter.py
    obs_horizon = 2
    act_horizon = 1
    pred_horizon = 16
    act_dim = 4
    obs_state_dim = 9
    H = W = 64
    rgb_channels = 3
    rgb_shape = (obs_horizon, H, W, rgb_channels)  # upstream reads [-1] for channels

    args = SimpleNamespace(
        obs_horizon=obs_horizon, act_horizon=act_horizon, pred_horizon=pred_horizon,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256], n_groups=8,
    )
    env = _fake_env(act_dim, obs_state_dim, rgb_shape)

    # Construct both models.
    torch.manual_seed(0)
    upstream = train_rgbd.Agent(env, args).to(device)

    from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy
    ours = DiffusionPolicy(
        obs_horizon=obs_horizon, act_horizon=act_horizon, pred_horizon=pred_horizon,
        act_dim=act_dim, obs_state_dim=obs_state_dim,
        rgb_shape=(rgb_channels, H, W),
        include_rgb=True, include_depth=False,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256],
        n_groups=8, num_diffusion_iters=100, device=device,
    ).to(device)

    # Copy upstream weights into ours so the only difference is the get_action body.
    # Layout matches because both classes build the same submodules in the same order.
    ours.load_state_dict(upstream.state_dict())

    # Build a single observation. Upstream's get_action expects rgb in (B, H, IH, IW, C)
    # and permutes to (B, H, C, IH, IW) internally. Our get_action expects pre-permuted.
    B = 1
    rgb_upstream = torch.zeros(B, obs_horizon, H, W, rgb_channels, dtype=torch.uint8, device=device)
    rgb_ours = rgb_upstream.permute(0, 1, 4, 2, 3).contiguous()
    state = torch.zeros(B, obs_horizon, obs_state_dim, device=device)
    obs_upstream = {"state": state.clone(), "rgb": rgb_upstream}
    obs_ours = {"state": state.clone(), "rgb": rgb_ours}

    torch.manual_seed(123)
    a_upstream = upstream.get_action(obs_upstream).detach()
    torch.manual_seed(123)
    a_ours = ours.get_action(obs_ours).detach()

    assert a_upstream.shape == a_ours.shape, f"{a_upstream.shape} vs {a_ours.shape}"
    diff = (a_upstream - a_ours).abs().max().item()
    assert diff < 1e-5, f"max abs diff = {diff}; our get_action drifted from upstream"
```

- [ ] **Step 15.2: Run**

```bash
conda run -n visuomotor_verification pytest tests/test_dp_agent_override.py -v -m slow
```

Expected: 1 passed. If the test fails with a tight tolerance issue, increase the tolerance to `1e-4` and investigate the diff source (likely numerical artifacts of slightly different operation ordering). If the test skips, that's a diagnostic — we re-examine upstream's structure.

- [ ] **Step 15.3: Commit**

```bash
git add tests/test_dp_agent_override.py
git commit -m "Add DP-vs-vendored-Agent equivalence sentinel test"
```

---

## Task 16: Mark new subprocess/training tests as slow and add the end-to-end smoke test

**Files:**
- Create: `tests/test_smoke_train_policy.py`
- Modify: `tests/test_dp_adapter.py` (mark heavy tests slow)

The four model tests in `test_dp_adapter.py` that instantiate a `DiffusionPolicy` (with PlainConv + UNet + DDPMScheduler) take 1-2 seconds even without GPU. Mark them `@pytest.mark.slow` so the fast suite stays fast.

- [ ] **Step 16.1: Add `@pytest.mark.slow` to the `_build_dp`-dependent tests**

In `tests/test_dp_adapter.py`, decorate every test that calls `_build_dp` with `@pytest.mark.slow`. The full list (already-existing tests from Tasks 4-9):

- `test_diffusion_policy_is_policy_subclass`
- `test_diffusion_policy_is_nn_module`
- `test_diffusion_policy_has_expected_submodules`
- `test_encode_obs_shape`
- `test_compute_loss_returns_scalar`
- `test_compute_loss_backward`
- `test_get_action_no_generator_uses_global_rng`
- `test_get_action_with_generator_is_reproducible`
- `test_get_action_shape`
- `test_reset_with_none_clears_generator`
- `test_reset_with_seed_builds_generator`
- `test_act_returns_single_action_array`
- `test_act_seeds_reproduce_first_action`
- `test_act_chunking_serves_from_cache_then_requeries`
- `test_load_restores_ema_state_dict`
- `test_load_missing_ema_key_raises`

For each, prepend `@pytest.mark.slow` immediately above the `def`.

Also ensure `import pytest` is in the imports at the top of the file.

- [ ] **Step 16.2: Create `tests/test_smoke_train_policy.py`**

```python
"""End-to-end smoke test: scripts/train_policy.py runs with total_iters=2 and
writes a checkpoint. Gated on demo file presence; skips otherwise."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = Path.home() / ".maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"


@pytest.mark.slow
def test_train_policy_end_to_end(tmp_path: Path) -> None:
    if not DEMO_PATH.exists():
        pytest.skip(
            f"Demo file not found at {DEMO_PATH}. Run the demo-prep recipe in "
            "docs/superpowers/specs/2026-05-11-dp-adapter-design.md §1."
        )

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_policy.py"),
        "run=stochastic",
        f"storage.root={tmp_path}",
        "experiment_name=smoke_train_e2e",
        "training.total_iters=2",
        "training.batch_size=2",
        "training.num_demos=2",
        "training.log_freq=1",
        "training.eval_freq=10000",   # > total_iters, no eval
        "training.save_freq=2",        # produces a non-eval checkpoint at iter 2
        "training.num_eval_envs=1",
        "training.num_eval_episodes=2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"train_policy.py failed:\n{output}"

    # Locate the run directory and inspect.
    runs = list(tmp_path.rglob("smoke_train_e2e-*"))
    assert len(runs) == 1, f"expected exactly one run_dir, got: {runs}"
    run_dir = runs[0]

    assert (run_dir / "metadata.json").exists()
    assert (run_dir / ".hydra").exists()
    assert (run_dir / "checkpoints").exists()
    ckpts = list((run_dir / "checkpoints").iterdir())
    assert len(ckpts) >= 1, f"no checkpoints under {run_dir / 'checkpoints'}"

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["script"] == "train_policy.py"
    assert metadata["run_config"]["mode"] == "stochastic"
    # output_artifacts.best_checkpoint must be set by the script's post-run annotation.
    assert "output_artifacts" in metadata
    assert "best_checkpoint" in metadata["output_artifacts"]
```

- [ ] **Step 16.3: Run**

```bash
conda run -n visuomotor_verification pytest tests/test_smoke_train_policy.py -v -m slow
```

Expected: 1 passed (with demos) or 1 skipped (without).

- [ ] **Step 16.4: Run the full suite to confirm test counts**

```bash
conda run -n visuomotor_verification pytest -q -m "not slow"
```

Expected: ~64 passed (foundations 57 fast tests + Task 1 vendor_import: 2 + Task 3 push_t_task: 6, minus the slow-marked DP adapter tests). Adjust the expectation if your local count differs — the goal is *no failures*, not a precise count.

- [ ] **Step 16.5: Commit**

```bash
git add tests/test_dp_adapter.py tests/test_smoke_train_policy.py
git commit -m "Mark DP adapter tests as slow; add end-to-end train_policy smoke"
```

---

## Task 17: Update UPSTREAM.md to reference the new shim

**Files:**
- Modify: `src/visuomotor_verification/policy/diffusion_policy/UPSTREAM.md`

The foundations UPSTREAM.md said "The next PR will handle this — either via an explicit sys.path shim in the adapter, or by adding an __init__.py shim in our own non-vendored sibling." Update to reflect the resolved choice.

- [ ] **Step 17.1: Edit UPSTREAM.md**

In `src/visuomotor_verification/policy/diffusion_policy/UPSTREAM.md`, locate the "Runtime / invocation note" section and replace its body with:

```markdown
## Runtime / invocation note

The vendored `train.py` and `train_rgbd.py` import from `diffusion_policy.utils`,
`diffusion_policy.evaluate`, etc. — bare top-level package imports that
upstream resolves via `setup.py` + `find_packages()`. Our `pyproject.toml`
does not install the vendored package, and the inner
`diffusion_policy/diffusion_policy/` directory has no `__init__.py`.

**Resolved:** `_vendor_import.py` (sibling of this file) inserts the outer
vendored directory onto `sys.path` once at import time. `adapter.py` and
`trainer.py` both `from . import _vendor_import` as their first line so the
shim is active before any vendored import resolves. After the shim runs,
`from diffusion_policy.utils import ...` works.

Note that the outer vendored scripts (`train.py`, `train_rgbd.py`) define
`Agent` and `SmallDemoDataset_DiffusionPolicy` referencing module-level
globals (`device`, `args.control_mode`) set only inside their `__main__`
block. Importing those classes and calling their methods raises `NameError`.
We re-implement those wrappers in `adapter.py` and `trainer.py` with the
globals lifted to `__init__` kwargs; see the upstream-line-range comments in
each class.
```

- [ ] **Step 17.2: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/UPSTREAM.md
git commit -m "Update UPSTREAM.md: shim resolution + re-implementation rationale"
```

---

## Final verification

After all tasks, run the full suite (including slow) and verify nothing is broken:

```bash
conda run -n visuomotor_verification pytest -v
```

Expected:
- Fast (non-slow) tests: ~64 passed.
- Slow tests: vary by whether demos are present.
  - Without demos: skips for `test_train_runs_one_iteration_without_eval`, `test_demo_dataset_loads_two_trajectories`, `test_train_policy_end_to_end`, and the maniskill smoke test still passes.
  - With demos: all slow tests pass.
- Total: 60+ passed, plus skips for demo-gated tests.

Then verify the end-to-end pipeline by hand on a machine with demos:

```bash
cd /common/home/st1122/Projects/visuomotor_verification
conda run -n visuomotor_verification python scripts/train_policy.py \
    experiment_name=push_t_dp_v1 \
    training.total_iters=50000
```

Expected: a 50k-iter training run that writes:
- Checkpoints to `/common/users/shared/pracsys/visuomotor_verification-data/policies/push_t/diffusion_policy/push_t_dp_v1-<timestamp>/checkpoints/`
- TensorBoard logs to the same `run_dir/logs/`
- `metadata.json` with `output_artifacts.best_checkpoint` set
- `.hydra/config.yaml` and other Hydra snapshots

The exact success rate of the resulting policy depends on hyperparameter tuning (spec §11 flags `batch_size: 64` as an informed guess that may need adjustment for the 11 GB GPU).

If this completes without error, the foundations of the visuomotor-verification pipeline are ready for trajectory collection (next PR).
