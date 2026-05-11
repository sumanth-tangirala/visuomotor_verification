# Diffusion Policy on Push-T — Adapter & Training Design

**Date:** 2026-05-11
**Status:** Draft, awaiting user review
**Scope:** Wire the vendored ManiSkill diffusion-policy baseline through our `Policy` ABC + Hydra configs + determinism plumbing. Produce a trained DP checkpoint for push-T (RGB observations). Trajectory collection and verifier work are deferred to subsequent PRs.

**Depends on:** Foundations PR (`docs/superpowers/specs/2026-05-11-foundations-design.md`).

---

## 1. Project context

This PR is the first end-to-end use of the foundations scaffolding. It:

1. Implements `DiffusionPolicy` — a concrete `Policy` ABC subclass that wraps the vendored ManiSkill DP baseline.
2. Implements `scripts/train_policy.py` — Hydra-driven training entry point that produces a checkpoint on push-T.
3. Adds concrete `ManiSkillSimulator(Simulator)` and `PushTTask(Task)` so the four-axis ABCs are no longer all stubs.

The downstream goal is to roll out the trained policy and collect labeled trajectories for verifier training, but that pipeline phase is **out of scope here**.

### One-time setup (before any training run)

```bash
# 1. Download RL-generated state demos for push-T
python -m mani_skill.utils.download_demo PushT-v1 -o ~/.maniskill/demos

# 2. Replay state demos to produce RGB trajectories
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_ee_delta_pose.physx_cuda.h5 \
    --use-env-states --obs-mode rgb --target-control-mode pd_ee_delta_pose \
    --save-traj --num-procs 4
```

This produces `~/.maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5`. Demos are downloaded and replayed once per machine; `scripts/train_policy.py` asserts the file exists and points back to this section if not. `scripts/train_policy.py`'s module docstring references this spec section so the recipe is discoverable from the script.

---

## 2. Design decisions

Five upfront decisions, with the chosen option recorded:

| # | Decision | Chosen | Rationale |
|---|---|---|---|
| 1 | PR scope | Adapter + train only (no rollout collection, no verifier) | Smaller scope; the rollout script and verifier are clean follow-ups once a checkpoint exists |
| 2 | Observation modality | RGB (visuomotor) | Project name commits to visuomotor policies; start the visual track now even though baseline hyperparams aren't tuned for push-T RGB |
| 3 | Training-loop ownership | Own the loop, reuse the *inner* vendored library (`ConditionalUnet1D`, `evaluate`, `make_env`, `utils`); re-implement the model/dataset wrappers that live in vendored scripts | Hydra-driven training + our determinism plumbing + our checkpoint paths require owning the loop. The vendored outer scripts (`train.py`/`train_rgbd.py`) define `Agent` and `SmallDemoDataset_DiffusionPolicy` referencing module-level globals (`device`, `args.control_mode`) only set inside `__main__` — they cannot be imported and used as-is. We re-implement these wrappers in our code; the inner package is clean library code and is imported normally |
| 4 | Determinism in the diffusion sampler | Inference via `torch.Generator` override (config-gated); training via global RNG; training defaults to STOCHASTIC mode | Per-component reproducibility matters at rollout time (verifier consumes it); training byte-determinism on CUDA is mostly aspirational |
| 5 | Class shape | Single `DiffusionPolicy(nn.Module, Policy)` with multiple inheritance | One user-facing class. ABCMeta + nn.Module's metaclass compose cleanly. The model/scheduler internals mirror the vendored `Agent` structure but with `device` as an `__init__` kwarg instead of a module global |

---

## 3. Repository additions

```
src/visuomotor_verification/policy/diffusion_policy/
├── adapter.py                 # was a stub; now fully implemented
├── trainer.py                 # NEW — TrainerConfig + train() function
└── _vendor_import.py          # NEW — sys.path shim for vendored `from diffusion_policy.X` imports

src/visuomotor_verification/simulator/
└── maniskill.py               # NEW — ManiSkillSimulator(Simulator)

src/visuomotor_verification/task/
└── push_t.py                  # NEW — PushTTask(Task)

scripts/
└── train_policy.py            # was a stub; now Hydra-driven, calls trainer.train()

configs/
├── train_policy.yaml          # updated (defaults to run: stochastic; training: block)
├── policy/
│   └── diffusion_policy.yaml  # updated (DP hyperparams)
└── task/
    └── push_t.yaml            # updated (env_id, control_mode, obs_mode, demo_path)

tests/
├── test_dp_agent_override.py  # NEW
├── test_dp_adapter.py         # NEW
├── test_dp_trainer_step.py    # NEW
└── test_smoke_train_policy.py # NEW, @pytest.mark.slow

src/visuomotor_verification/policy/diffusion_policy/
└── UPSTREAM.md                # "Runtime / invocation note" section updated to reference _vendor_import.py
```

### Vendor import shim

The inner vendored package (`diffusion_policy/diffusion_policy/`) is reusable library code: `ConditionalUnet1D`, `PlainConv`, `evaluate`, `make_eval_envs`, `IterationBasedBatchSampler`, `worker_init_fn`, `load_demo_dataset`. These have no problematic globals. But their import path is `diffusion_policy.utils`, `diffusion_policy.evaluate`, etc. — bare top-level `diffusion_policy` package — which our nested layout breaks.

`_vendor_import.py` does `sys.path.insert(0, <outer_vendored_dir>)` once at import time so the inner `diffusion_policy/` package is discoverable. `trainer.py` and `adapter.py` both `from . import _vendor_import` as their first line so the shim is in place before any vendored import resolves.

**We do NOT import from the outer vendored scripts** (`train.py`, `train_rgbd.py`). Those define `Agent` and `SmallDemoDataset_DiffusionPolicy` but reference module-level globals (`device`, `args.control_mode`) set only inside `if __name__ == "__main__":`. Importing them and instantiating their classes raises `NameError` at the first method call. Instead, we re-implement those two wrappers in our code (`adapter.py` and `trainer.py`) with the relevant globals lifted to `__init__` kwargs. The class bodies otherwise mirror the upstream structure, with a comment at each class pointing to its upstream source file + line range for sync-tracking purposes.

The `UPSTREAM.md` "Runtime / invocation note" added in the foundations PR is updated to point at this shim as the chosen resolution.

---

## 4. Adapter architecture (`adapter.py`)

```python
from . import _vendor_import  # noqa: F401 — must precede vendored imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Inner package only — no outer-script imports
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv  # RGB encoder

from visuomotor_verification.policy.base import Policy


class DiffusionPolicy(nn.Module, Policy):
    """RGB diffusion policy. Structure mirrors `Agent` in vendored
    train_rgbd.py:246-382, with the module-level `device` global lifted to
    `__init__(device=...)`. Implements the Policy ABC interface and overrides
    `get_action` to thread a torch.Generator through the diffusion sampler so
    `seeds.policy` can independently control inference noise.

    UPSTREAM: src/visuomotor_verification/policy/diffusion_policy/train_rgbd.py
    Lines 246-382 (class Agent). If upstream Agent changes, update this class
    to match.
    """

    def __init__(
        self,
        *,
        obs_horizon: int,
        act_horizon: int,
        pred_horizon: int,
        act_dim: int,
        obs_state_dim: int,
        rgb_shape: tuple[int, int, int],  # (C, H, W)
        diffusion_step_embed_dim: int,
        unet_dims: list[int],
        n_groups: int,
        num_diffusion_iters: int,
        device: torch.device,
    ):
        nn.Module.__init__(self)
        # build encoder + UNet + scheduler (same structure as upstream Agent.__init__)
        # ...
        self._device = device
        self._gen: torch.Generator | None = None
        # Action chunk cache: list of remaining single-step actions to return.
        self._action_cache: list[np.ndarray] = []

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the policy's per-episode state.

        If `seed` is provided, build a local torch.Generator seeded with it; all
        subsequent `get_action`/`act` calls thread this generator through every
        sampling op. If `seed` is None, fall back to global RNG (matches the
        vendored Agent's behavior).
        """
        self._action_cache.clear()
        if seed is None:
            self._gen = None
        else:
            self._gen = torch.Generator(device=self._device).manual_seed(seed)

    def compute_loss(self, obs_seq: dict, action_seq: torch.Tensor) -> torch.Tensor:
        """Training-time loss. Mirrors upstream Agent.compute_loss.

        `torch.randn` and `torch.randint` use global RNG (training is stochastic
        by spec §6).
        """
        # ... structure as upstream lines 309-340 ...

    def get_action(self, obs_seq: dict) -> torch.Tensor:
        """Inference-time action sampling. Mirrors upstream Agent.get_action,
        but every `torch.randn` and `noise_scheduler.step` receives
        `generator=self._gen` so inference is reproducible per `seeds.policy`
        when `reset(seed=...)` was called.
        """
        # ... structure as upstream lines 342-382 ...

    def act(self, obs_history: list[Observation]) -> Action:
        """Single-step action interface. Implements action chunking:
        - If cache is empty: run `get_action` (one denoise pass), populate cache
          with the predicted `act_horizon` actions.
        - Pop and return the next action from the cache.
        """
        if not self._action_cache:
            obs_seq = self._stack_obs(obs_history)         # (1, obs_horizon, *obs_dims)
            action_seq = self.get_action(obs_seq)          # (1, act_horizon, act_dim)
            self._action_cache = list(action_seq[0].cpu().numpy())
        return self._action_cache.pop(0)

    def load(self, ckpt_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self._device)
        # Load EMA weights (deployment weights); the checkpoint dict has 'agent' and 'ema_agent'.
        self.load_state_dict(ckpt["ema_agent"])
```

**Action chunking semantics.** DP predicts `pred_horizon` actions, of which `act_horizon` are executable (the rest is overlap with past predictions to stabilize denoising). Each `act()` call returns one action. The first call after `reset()` triggers a denoise pass that fills the cache with `act_horizon` actions. Subsequent calls pop from the cache. When the cache is empty, the next `act()` triggers another denoise pass.

**`_stack_obs(obs_history)`** turns a `list[Observation]` (length up to `obs_horizon`) into the tensor shape expected by `get_action`: `(1, obs_horizon, *obs_dims)`. For push-T RGB this is `(1, 2, C, H, W)` plus a `state` channel.

---

## 5. Trainer module (`trainer.py`)

### Types and entry point

```python
from . import _vendor_import  # noqa: F401

# Inner-package imports only (clean library code).
from diffusion_policy.utils import (
    IterationBasedBatchSampler, worker_init_fn, load_demo_dataset,
)
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.evaluate import evaluate

from visuomotor_verification.core.determinism import Seeds
from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy


class DemoDataset(torch.utils.data.Dataset):
    """Dataset for DP training. Structure mirrors
    `SmallDemoDataset_DiffusionPolicy` in vendored train_rgbd.py:124-244, with
    `device` and `control_mode` lifted to __init__ params (upstream pulls them
    from module globals set in __main__).

    UPSTREAM: train_rgbd.py:124-244. If upstream's dataset changes, update.
    """
    def __init__(
        self,
        data_path: Path,
        *,
        device: torch.device,
        control_mode: str,
        num_traj: int | None,
        obs_horizon: int,
        pred_horizon: int,
    ):
        # ... structure as upstream ...


@dataclass
class TrainerConfig:
    # demo + env (from cfg.task + cfg.training)
    demo_path: Path
    env_id: str
    control_mode: str
    obs_mode: str
    max_episode_steps: int
    num_eval_envs: int
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
    num_demos: int | None
    num_dataload_workers: int
    log_freq: int
    eval_freq: int
    save_freq: int | None
    num_eval_episodes: int
    # determinism (from resolve_seeds(run_cfg))
    seeds: Seeds
    device: torch.device


def train(cfg: TrainerConfig, run_dir: Path, log: SummaryWriter) -> Path:
    """Run the DP training loop. Writes checkpoints under `run_dir/checkpoints/`
    and tensorboard logs under `run_dir/logs/`. Returns the best-checkpoint path.

    Phases:
      1. Build dataset + dataloader (vendored load_demo_dataset, IterationBasedBatchSampler).
      2. Build eval envs (vendored make_eval_envs; seed = cfg.seeds.sim + env_idx).
      3. Construct DiffusionPolicy + EMA copy.
      4. AdamW(lr, betas=(0.95, 0.999), weight_decay=1e-6) + cosine LR (500 warmup).
      5. Loop: forward `compute_loss`, backward, optimizer.step, lr_scheduler.step, EMA.step.
         Every `eval_freq` iterations: evaluate(), save on best success_at_end.
         Every `log_freq` iterations: tensorboard scalars.
      6. Final eval + save.
    """
```

### `scripts/train_policy.py`

Replaces the stub from the foundations PR. The body is:

```python
import hydra
from omegaconf import DictConfig
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# (sys.path shim for _common, as in foundations)
from _common import prologue
from visuomotor_verification.core.determinism import resolve_seeds, RunConfig
from visuomotor_verification.policy.diffusion_policy.trainer import (
    TrainerConfig, train,
)


@hydra.main(version_base=None, config_path="../configs", config_name="train_policy")
def main(cfg: DictConfig) -> None:
    """Train a Diffusion Policy. See spec §1 for one-time demo-prep recipe."""
    run_dir = prologue(cfg, script_name="train_policy.py")

    # Build TrainerConfig from Hydra cfg
    run_cfg = RunConfig.from_hydra(cfg)
    resolved = resolve_seeds(run_cfg)
    trainer_cfg = TrainerConfig(
        demo_path=Path(cfg.task.demo_path),
        env_id=cfg.task.env_id,
        # ... all fields from cfg.task / cfg.policy / cfg.training ...
        seeds=resolved,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Demo file check
    if not trainer_cfg.demo_path.exists():
        raise FileNotFoundError(
            f"Demo file not found: {trainer_cfg.demo_path}. "
            "Run the demo-prep recipe in "
            "docs/superpowers/specs/2026-05-11-dp-adapter-design.md §1."
        )

    log = SummaryWriter(str(run_dir / "logs"))
    if cfg.training.wandb.enabled:
        import wandb
        wandb.init(project=cfg.training.wandb.project,
                   entity=cfg.training.wandb.entity,
                   name=run_dir.name,
                   config=OmegaConf.to_container(cfg, resolve=True),
                   sync_tensorboard=True)

    best_ckpt = train(trainer_cfg, run_dir, log)
    log.close()
    print(f"[train_policy] best checkpoint: {best_ckpt}")
```

---

## 6. Determinism plumbing

| Surface | Source seed | Mechanism |
|---|---|---|
| Global Python RNG | `seeds.python` | `random.seed(...)` via `seed_all` |
| Global numpy RNG | `seeds.numpy` | `np.random.seed(...)` via `seed_all` |
| Global torch RNG | `seeds.torch` | `torch.manual_seed(...)` + `torch.cuda.manual_seed_all(...)` via `seed_all` |
| cuDNN flags | `cfg.mode` | `seed_all` (off in STOCHASTIC, the training default) |
| Dataloader workers | `seeds.dataloader` | `worker_init_fn(worker_id, base_seed=resolved.dataloader)` |
| Eval env init | `seeds.sim` | `env.action_space.seed(resolved.sim + env_idx)` and same for `observation_space`; per-env seed offsets |
| Diffusion noise (training) | `seeds.torch` (via global) | Vendored `compute_loss` calls `torch.randn`/`torch.randint` with no `generator=`; flows through global |
| Diffusion noise (inference) | `seeds.policy` if set, else global | Our `get_action` override: `torch.randn(..., generator=self._gen)` and `scheduler.step(generator=self._gen)`; `self._gen` is set by `reset(seed=...)` |

**Training default.** `configs/train_policy.yaml` uses `run: stochastic`. cuDNN benchmark is on. Training is not byte-deterministic across runs — that's an explicit project choice (spec §5.4 of foundations acknowledges the CUDA determinism cost). Same-seed runs are reproducible *up to* CUDA kernel non-determinism, which is the standard tradeoff for visuomotor training.

**Inference paths.** When `seeds.policy` is unset (or `seed=None` passed to `reset`), `get_action` uses the global RNG (matches upstream behavior). When `seeds.policy` is set, the local `torch.Generator` overrides — this is the path the rollout-collection PR will use to produce reproducible trajectories per `(policy_seed, sim_seed, master)` triple.

---

## 7. Hydra configuration

### `configs/task/push_t.yaml`

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

### `configs/policy/diffusion_policy.yaml`

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

### `configs/train_policy.yaml`

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
  batch_size: 64           # informed guess for RGB on 11GB; tune in follow-up if OOM
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

---

## 8. `ManiSkillSimulator` and `PushTTask`

Concrete impls of the foundations ABCs. Small wrappers; their purpose is to make the four-axis abstraction non-vacuous and to give later phases (rollout, eval) a typed seam.

### `src/visuomotor_verification/simulator/maniskill.py`

```python
class ManiSkillSimulator(Simulator):
    """Thin wrapper around gym.make for a single ManiSkill env. The training
    pipeline uses `make_eval_envs` (vendored) for parallel eval; this class is
    for single-env use cases like rollout collection (next PR)."""

    def __init__(self, *, env_id: str, control_mode: str, obs_mode: str,
                 sim_backend: str, max_episode_steps: int):
        ...

    def reset(self, *, seed: int | None = None) -> Observation: ...
    def step(self, action: Action) -> StepResult: ...
    def render(self, mode: str = "rgb_array") -> np.ndarray: ...
    def close(self) -> None: ...
    @property
    def observation_spec(self) -> ObsSpec: ...
    @property
    def action_spec(self) -> ActionSpec: ...
```

### `src/visuomotor_verification/task/push_t.py`

```python
class PushTTask(Task):
    """Push-T task: judge success by reading info['success'] from the env."""

    def __init__(self, *, name: str, env_id: str, control_mode: str,
                 obs_mode: str, sim_backend: str, max_episode_steps: int,
                 horizon: int, demo_path: str):
        # Stash for build_env / accessors
        self._horizon = horizon
        self.demo_path = Path(demo_path)
        # ...

    def build_env(self, sim: Simulator) -> None:
        """No-op for ManiSkill — the env is already constructed with the right
        task by `gym.make(env_id, ...)`. Kept for ABC compliance."""
        return None

    def is_success(self, obs, info) -> bool:
        return bool(info.get("success", False))

    @property
    def horizon(self) -> int:
        return self._horizon
```

---

## 9. Initial deliverable scope

**In:**
1. `_vendor_import.py` sys.path shim.
2. `adapter.py` fully implemented: `DiffusionPolicy(Agent, Policy)` with `get_action` override, `reset(seed)`, `act` with action chunking, `load`.
3. `trainer.py` fully implemented: `TrainerConfig` + `train()` covering dataloader, EMA, cosine LR, eval, save-on-best.
4. `simulator/maniskill.py` and `task/push_t.py` as concrete ABC subclasses.
5. `scripts/train_policy.py` body, Hydra-driven, with the demo-file existence check.
6. Hydra config updates per §7.
7. UPSTREAM.md update referencing `_vendor_import.py`.
8. Four tests:
   - `test_dp_agent_override.py` — upstream-equivalence under shared global seed.
   - `test_dp_adapter.py` — Policy ABC compliance + `reset(seed)` determinism via Generator.
   - `test_dp_trainer_step.py` — one training step runs without crashing (mock or skip eval).
   - `test_smoke_train_policy.py` — `@pytest.mark.slow`, subprocess invocation with `total_iters=2`.
9. A demonstration that the full pipeline produces a non-trivial checkpoint on push-T (the user runs the actual long training; this is verification, not a deliverable file).

**Out (subsequent PRs):**
- `scripts/collect_trajectories.py` body — rolls out the trained policy under `seeds.policy` and saves `Trajectory` objects under `datasets/<task>/rollouts/<run_id>/`.
- Concrete `Verifier` subclass + training script body.
- Push-T RGB hyperparameter tuning beyond the initial guesses (batch_size in particular).

---

## 10. Testing posture

### `test_dp_agent_override.py`

Pins our `DiffusionPolicy.get_action` against the structural equivalent in vendored `train_rgbd.py:Agent.get_action`. Because the upstream `Agent` can't be cleanly imported (script globals), the equivalence test compares against a re-instantiated copy: we monkey-patch the upstream module's `device` global, instantiate upstream `Agent`, and compare outputs under shared global seed.

```python
def test_dp_get_action_matches_upstream_under_shared_global_seed():
    """When `gen=None` (no local generator), our get_action must produce
    byte-identical output to the upstream Agent.get_action given the same
    global RNG state and same weights."""
    import train_rgbd  # vendored outer script
    train_rgbd.device = torch.device("cuda")  # monkey-patch the global

    # Construct minimal env stand-in for upstream Agent.__init__ shape probes
    fake_env = build_fake_env_with_specs(...)
    fake_args = build_fake_args(...)
    torch.manual_seed(0)
    upstream = train_rgbd.Agent(fake_env, fake_args).to(device)

    dp = DiffusionPolicy(... same kwargs ..., device=device)
    dp.load_state_dict(upstream.state_dict())

    obs_seq = build_fake_obs(...)
    torch.manual_seed(123)
    a_upstream = upstream.get_action(obs_seq)
    torch.manual_seed(123)
    a_dp = dp.get_action(obs_seq)
    assert torch.allclose(a_upstream, a_dp)
```

This test runs only if the monkey-patch import succeeds (skips gracefully if upstream's structure changes). It's a regression sentinel, not a load-bearing assertion.

### `test_dp_adapter.py`

```python
def test_reset_with_seed_uses_generator():
    """Two reset(seed=42); act() calls produce identical actions; reset(seed=43)
    produces a different action."""

def test_reset_with_none_uses_global_rng():
    """reset(seed=None) -> act() uses global RNG (regression sanity)."""

def test_action_chunking_serves_from_cache():
    """First act() after reset triggers get_action; next (act_horizon-1) act()
    calls do not trigger get_action again; act_horizon+1th call re-queries."""

def test_load_restores_ema_weights():
    """load(ckpt_path) loads ema_agent weights, not agent weights."""
```

### `test_dp_trainer_step.py`

```python
def test_one_training_step_runs_without_crash(tmp_path):
    """TrainerConfig with total_iters=2, num_demos=2; train() runs without crash
    and writes a checkpoint."""
```

Construction tolerates absence of eval envs by setting `num_eval_envs=0` or by skipping eval entirely with `eval_freq > total_iters`.

### `test_smoke_train_policy.py`

```python
@pytest.mark.slow
def test_train_policy_smoke(tmp_path):
    """Subprocess scripts/train_policy.py with total_iters=2, eval_freq=3,
    num_demos=2, storage.root=tmp_path. Assert run_dir exists with
    checkpoints/, metadata.json, and .hydra/."""
```

The test is gated on demo file presence — if `~/.maniskill/demos/...` doesn't exist, the test skips with a message pointing to the prep recipe (§1).

---

## 11. Open questions / known unknowns

- **`batch_size: 64` for RGB on 11GB.** Informed guess. The smoke test will reveal OOM if too aggressive. If OOM, halve and retry; if generous headroom, bump up to 128 in a follow-up. This is not a blocker for landing the PR — the goal here is correct plumbing, not tuned performance.
- **Hyperparameter tuning for push-T RGB.** Beyond batch_size, the recipe uses state-task defaults (lr, total_iters, obs_horizon, etc.). A subsequent PR may sweep these once we see baseline success rate.
- **Demo replay determinism.** `replay_trajectory` is itself a stochastic process (env reset, controller behavior). Two replays with the same seed should produce identical RGB demos, but this isn't asserted by ManiSkill. We trust the resulting `.h5` as the canonical artifact. If reproducibility of *demos* matters later, we'll add a replay-determinism check.
- **WandB usage.** Off by default. Project name `visuomotor_verification`; entity TBD with first long training run.
- **Action chunking with `act_horizon=1`.** For push-T with `act_horizon=1`, the cache pattern collapses: every `act()` call triggers a fresh `get_action`. The code path is exercised but the optimization (cache reuse) is moot for this task. Other tasks with `act_horizon > 1` benefit from the cache.
