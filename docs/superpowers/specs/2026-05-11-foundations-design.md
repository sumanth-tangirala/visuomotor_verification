# Visuomotor Verification — Foundations Design

**Date:** 2026-05-11
**Status:** Draft, awaiting user review
**Scope:** Initial repository scaffolding, abstractions, determinism mechanism, conda environment, and storage conventions. Sets up the system to train a diffusion policy on ManiSkill push-T (next PR) and eventually train verifiers on its rollouts.

---

## 1. Project context

The goal of this project is research on **visuomotor verification**: training a verifier that, given a trajectory produced by a visuomotor policy, predicts whether the policy will succeed or fail at the task. The pipeline is:

1. Train a visuomotor policy (initially diffusion policy on ManiSkill push-T).
2. Roll out the trained policy to collect trajectories labeled with success/failure.
3. Train a verifier on these trajectories.
4. Evaluate the verifier on held-out rollouts.

The project will move across **simulators** (starting with ManiSkill), **tasks** (starting with push-T), **policy types** (starting with diffusion policy), and **verifier architectures** (TBD). The repository must support easy swapping along each of these four axes.

### Special concern: determinism

Verifiers are highly sensitive to the determinism of the data they are trained on and the inference paths that produce trajectories at test time. The repository must support **full deterministic control** of the data collection and inference pipeline — and equally must support a **fully stochastic mode** when desired. Determinism is a first-class design concern, not an afterthought.

### Storage

All disk-space-heavy artifacts (datasets, policy checkpoints, verifier experiments, Hydra outputs) are written under the shared lab data directory:

```
/common/users/shared/pracsys/visuomotor_verification-data/
```

The repository itself contains only code, configs, docs, and tests.

---

## 2. Design decisions

Four upfront decisions, with the chosen option recorded for future reference:

| # | Decision | Chosen | Notes |
|---|---|---|---|
| 1 | Determinism granularity | Layered: global mode flag **plus** per-component seed overrides | Common case is one flag; advanced experiments can ablate per-component |
| 2 | Abstraction scope on day 1 | Full ABCs for `Simulator`, `Task`, `Policy`, `Verifier` from the start, with Hydra `_target_` instantiation as the registry | User explicitly chose maximum abstraction up front |
| 3 | ManiSkill DP baseline integration | Install ManiSkill as a library; **vendor** only the diffusion-policy baseline code | Sim updates flow in from upstream; DP code is owned so we can wire it into our determinism layer and ABC |
| 4 | Pipeline composition | Per-phase entry points (`scripts/train_policy.py`, `collect_trajectories.py`, `train_verifier.py`, `evaluate_verifier.py`) — no monolithic runner | Each phase has its own determinism story; iterate per phase |

The `Verifier` ABC is **provisional**: its concrete shape will be revised once the first concrete verifier impl exists. The day-1 ABC is a placeholder so the rest of the package can reference it without changes.

---

## 3. Repository layout

```
visuomotor_verification/
├── pyproject.toml                  # PEP 621 + setuptools; deps + console scripts
├── environment.yml                 # conda env spec
├── README.md
├── CLAUDE.md                       # vision, conventions, paths, determinism principle
├── configs/                        # Hydra config tree (see §6)
├── docs/
│   └── superpowers/specs/          # design docs
├── scripts/                        # Hydra entry points (one per phase)
├── src/visuomotor_verification/
│   ├── __init__.py
│   ├── core/
│   │   ├── determinism.py          # RunMode, Seeds, RunConfig, seed_all
│   │   ├── trajectory.py           # Trajectory dataclass + I/O
│   │   └── storage.py              # run_id minting + path resolution
│   ├── simulator/
│   │   ├── base.py                 # Simulator ABC
│   │   └── maniskill.py            # ManiSkillSimulator
│   ├── task/
│   │   ├── base.py                 # Task ABC
│   │   └── push_t.py               # PushTTask
│   ├── policy/
│   │   ├── base.py                 # Policy ABC
│   │   └── diffusion_policy/       # vendored from ManiSkill examples
│   │       ├── UPSTREAM.md         # records source commit hash + diff plan
│   │       ├── model.py
│   │       ├── trainer.py
│   │       └── adapter.py          # DiffusionPolicy(Policy) wrapper
│   └── verifier/
│       └── base.py                 # Verifier ABC (provisional)
└── tests/
    ├── test_determinism.py         # core seeding invariants
    └── test_storage.py             # run_id + path conventions
```

The `src/` layout makes the package installable in editable mode (`pip install -e .`) and keeps imports unambiguous.

---

## 4. Abstract base classes

The four ABCs are the contracts the pipeline talks to. They live under `src/visuomotor_verification/<axis>/base.py`. Signatures only — bodies are filled by concrete subclasses.

```python
# simulator/base.py
class Simulator(ABC):
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

# task/base.py
class Task(ABC):
    @abstractmethod
    def build_env(self, sim: Simulator) -> None: ...
    @abstractmethod
    def is_success(self, obs: Observation, info: dict) -> bool: ...
    @property
    @abstractmethod
    def horizon(self) -> int: ...

# policy/base.py
class Policy(ABC):
    @abstractmethod
    def reset(self, *, seed: int | None = None) -> None: ...
    @abstractmethod
    def act(self, obs_history: list[Observation]) -> Action: ...
    @abstractmethod
    def load(self, ckpt_path: Path) -> None: ...

# verifier/base.py  (PROVISIONAL — will be revised when first concrete impl lands)
class Verifier(ABC):
    @abstractmethod
    def fit(self, trajectories: Iterable[Trajectory]) -> None: ...
    @abstractmethod
    def predict(self, trajectory: Trajectory) -> VerifierOutput: ...
```

Intentional design choices:

- **`Task` is separate from `Simulator`** so we can run the same task in a different simulator without rewriting task semantics. `Task.build_env(sim)` is the seam.
- **`Policy.act` takes an observation history**, not a single observation, because diffusion policy consumes an obs window. Policies that only need the latest obs ignore the rest.
- **`Trajectory`** is a first-class dataclass in `core/`, not in any of the four axes, because it is the data interface between trajectory collection and verifier training.

---

## 5. Determinism mechanism

### 5.1 Core types (`core/determinism.py`)

```python
class RunMode(Enum):
    DETERMINISTIC = "deterministic"     # all RNG sources fixed from master seed
    STOCHASTIC    = "stochastic"        # all RNG sources free
    MIXED         = "mixed"             # per-component overrides; unset = stochastic

@dataclass(frozen=True)
class Seeds:
    master: int | None = None           # required if mode != STOCHASTIC
    sim:    int | None = None           # explicit override
    policy: int | None = None
    torch:  int | None = None
    numpy:  int | None = None
    python: int | None = None
    dataloader: int | None = None
    cuda_strict: bool = False           # see §5.4

@dataclass(frozen=True)
class RunConfig:
    mode:        RunMode
    seeds:       Seeds
    allow_dirty: bool = False           # see §5.7
```

### 5.2 Seed derivation

In `DETERMINISTIC` mode, all unset per-component seeds are derived from `master` via a fixed hash:

```python
component_seed = int.from_bytes(
    hashlib.blake2b(f"{master}:{component_name}".encode(), digest_size=8).digest(),
    "big",
) & 0xFFFFFFFF
```

This guarantees: same `master` → same per-component seeds, and each component sees a different RNG stream.

- **`MIXED`** mode: only explicitly-set component seeds are honored; unset ones run from the OS RNG. Useful for "fix env init, leave policy sampling free."
- **`STOCHASTIC`** mode: all per-component seeds are ignored. If any were explicitly set, `seed_all` emits a warning (almost certainly a config bug).

### 5.3 Enforcement at the boundary

Every phase entry point (`scripts/*.py`) does, exactly once:

```python
run_cfg = RunConfig.from_hydra(cfg)
seed_all(run_cfg)
# instantiate sim, policy, etc. via Hydra `_target_`, passing run_cfg through
```

`seed_all` is the **only** function that touches global RNGs and cuDNN flags. It seeds:

- `random.seed(seeds.python)`
- `np.random.seed(seeds.numpy)`
- `torch.manual_seed(seeds.torch)` (CPU + CUDA)
- cuDNN flags per §5.4

Nothing else in the codebase calls `torch.manual_seed`, `np.random.seed`, or `random.seed` directly — this is a lint rule we enforce by code review and (later) by a test that greps for these calls outside `seed_all`.

Components that have their own RNG (the diffusion sampler, the sim's reset seed) receive their component seed via constructor / method argument and create a **local** `torch.Generator` or `np.random.Generator`. They do not read from globals. This makes a component's stochastic behavior auditable from its inputs alone.

### 5.4 CUDA cost knob

Full determinism on CUDA requires:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These have a meaningful speed cost. Behavior:

- `RunMode.DETERMINISTIC` → all three flags on.
- `RunMode.MIXED` → flags off by default; `seeds.cuda_strict=True` turns them on.
- `RunMode.STOCHASTIC` → all three flags off.

This tradeoff is documented in `CLAUDE.md` so it is never surprising.

### 5.5 Trajectory metadata

Every saved `Trajectory` records the `RunConfig` that produced it: `mode`, `master`, and **all component seeds actually used** (not just requested). This makes "was this dataset collected deterministically?" a question with a real answer.

### 5.6 Testing posture

`tests/test_determinism.py` asserts:

- Same `(mode=DETERMINISTIC, master=k)` → byte-identical trajectories across two runs of a sim+policy combination.
- Seed derivation is stable: `derive_seed(master, component)` produces the same value across Python versions / platforms (we pin to `blake2b` for this reason — Python's `hash()` is not stable).
- `seed_all` is idempotent under the same `RunConfig`.
- `RunMode.DETERMINISTIC` with a dirty tree raises `DirtyTreeError`; setting `allow_dirty=True` bypasses (see §5.7).

This suite is a CI guardrail; flakiness here means a determinism leak.

### 5.7 Git cleanliness gate

A dirty working tree means `git_sha` in `metadata.json` points to a different code state than what actually ran, breaking the reproducibility claim that `RunMode.DETERMINISTIC` is supposed to make. Policy:

- **Always capture**, regardless of mode: `git_dirty: bool`, and when dirty, the full `git diff HEAD` (staged + unstaged) into `metadata.json` under `git_diff`. This makes dirty runs reproducible *in principle*: replay the SHA, then apply the diff. Untracked file paths are listed under `git_untracked` but their contents are not snapshotted — users who need that should commit.
- **`RunMode.DETERMINISTIC` + dirty tree** → `seed_all` raises `DirtyTreeError` unless `RunConfig.allow_dirty=true`. The error message points the user to commit or stash. Default is `false` because the cost of an unreproducible "deterministic" run is much higher than the cost of being asked to commit.
- **`RunMode.MIXED` / `STOCHASTIC` + dirty tree** → warning logged, run proceeds. These modes do not claim reproducibility, so blocking would be friction without benefit.
- The `allow_dirty` knob lives on `RunConfig`, not on `Seeds`, since it is about reproducibility hygiene, not RNG sources.

---

## 6. Hydra configuration

### 6.1 Config tree

```
configs/
├── train_policy.yaml              # top-level for scripts/train_policy.py
├── collect_trajectories.yaml
├── train_verifier.yaml
├── evaluate_verifier.yaml
├── run/
│   ├── deterministic.yaml         # mode: deterministic, master: <int>
│   ├── stochastic.yaml
│   └── mixed.yaml
├── simulator/
│   └── maniskill.yaml             # _target_: ...ManiSkillSimulator + kwargs
├── task/
│   └── push_t.yaml                # _target_: ...PushTTask + kwargs
├── policy/
│   └── diffusion_policy.yaml      # _target_: ...DiffusionPolicy + hyperparams
├── verifier/
│   └── placeholder.yaml
└── storage/
    └── default.yaml               # roots + run_dir interpolation
```

Each top-level script's config uses Hydra `defaults` to compose: `run` + `simulator` + `task` + `policy` (or `verifier`) + `storage`. Hydra `_target_` does the registry job — no custom registry needed.

### 6.2 Mandatory top-level fields

Every top-level config requires:

- `experiment_name`: human-readable slug (e.g. `push_t_dp_v1`). Used in the run_id. Hydra will error if missing.

### 6.3 Hydra output directory

`hydra.run.dir` is interpolated to point at the run's artifact directory inside the shared data dir (see §7). This means Hydra's `.hydra/{config,overrides,hydra}.yaml` files land directly in the run dir alongside `metadata.json`, `checkpoints/`, `trajectories/`, etc. — a single directory holds everything about a run.

---

## 7. Storage layout

Root: `/common/users/shared/pracsys/visuomotor_verification-data/` (already exists with `datasets/`, `policies/`, `experiments/` subdirectories pre-created).

### 7.1 Layout

```
datasets/
  <task>/
    <source>/                          # demos | rollouts
      <run_id>/
        trajectories/                  # .npz or .hdf5, one per episode
        metadata.json                  # RunConfig + git SHA + script + cmdline
        .hydra/                        # Hydra-generated config snapshots

policies/
  <task>/<policy>/
    <run_id>/
      checkpoints/                     # epoch-N.pt, best.pt
      logs/                            # tensorboard event files
      metadata.json
      .hydra/

experiments/
  verifier/<task>/<verifier>/
    <run_id>/
      checkpoints/
      eval/                            # evaluation outputs
      logs/
      metadata.json
      .hydra/
```

### 7.2 Run ID format

```
<run_id> = "<experiment_name>-<YYYY-MM-DD>_<HH-MM-SS>"
```

Example: `push_t_dp_v1-2026-05-11_17-23-08`.

`experiment_name` is supplied via Hydra; timestamp is generated at process start via `${now:%Y-%m-%d_%H-%M-%S}` interpolation. Re-running with the same `experiment_name` produces a different `run_id` thanks to the timestamp; the name groups related runs by intent rather than by directory hierarchy.

### 7.3 `metadata.json`

Every run directory has a `metadata.json` with at minimum:

- `run_id`
- `experiment_name`
- `script` (entry-point name)
- `cmdline` (full argv)
- `git_sha` and `git_dirty` (bool)
- `git_diff`: full `git diff HEAD` output (staged + unstaged) when `git_dirty` is true; absent otherwise
- `git_untracked`: list of untracked file paths when `git_dirty` is true; absent otherwise
- `timestamp` (ISO 8601)
- `run_config`: full `RunConfig` (mode + seeds actually used)
- `resolved_config`: the resolved Hydra config (also available in `.hydra/config.yaml`, but mirrored here for ergonomics)
- `input_artifacts`: dict of upstream artifact paths this run consumed (e.g. policy checkpoint path for trajectory collection)

### 7.4 Inter-phase artifact paths

Paths between phases are passed via Hydra overrides — **no implicit "latest" magic**. For example:

```bash
python scripts/collect_trajectories.py \
  experiment_name=push_t_dp_rollouts_v1 \
  policy=diffusion_policy \
  policy.checkpoint=/common/users/.../policies/push_t/diffusion_policy/push_t_dp_v1-2026-05-11_17-23-08/checkpoints/best.pt \
  run=deterministic \
  run.seeds.master=42
```

This is explicit on purpose: verifier reproducibility depends on knowing *exactly* which policy checkpoint produced which rollouts.

---

## 8. Per-phase entry points

| Script | Reads | Writes |
|---|---|---|
| `scripts/train_policy.py` | `datasets/<task>/demos/<run_id>` | `policies/<task>/<policy>/<run_id>` |
| `scripts/collect_trajectories.py` | `policies/<task>/<policy>/<run_id>` (checkpoint) | `datasets/<task>/rollouts/<run_id>` |
| `scripts/train_verifier.py` | `datasets/<task>/rollouts/<run_id>` | `experiments/verifier/<task>/<verifier>/<run_id>` |
| `scripts/evaluate_verifier.py` | verifier checkpoint + held-out rollouts | `experiments/verifier/.../<run_id>/eval/` |

Each is a Hydra app (`@hydra.main(...)`). Each calls `seed_all(RunConfig.from_hydra(cfg))` exactly once before instantiating any sim/policy/verifier.

---

## 9. Initial deliverable scope (this PR)

**In:**

1. Repo scaffolding: `pyproject.toml`, `environment.yml`, `src/` package layout, `configs/` skeleton, `scripts/` stubs, `tests/` skeleton.
2. The four ABCs (signatures + brief docstrings).
3. `core/determinism.py` fully implemented + unit tests for seed derivation and idempotency.
4. `core/trajectory.py` (dataclass + npz I/O) skeleton.
5. `core/storage.py` (run_id minting + path resolution from `storage` config) skeleton.
6. `CLAUDE.md`: project vision, paths, conventions, determinism principle, storage rules.
7. Conda env created on disk: `visuomotor_verification` with Python 3.11, PyTorch 2.4 + CUDA 12.1 build (driver is 12.2), ManiSkill 3 latest from PyPI, plus Hydra/OmegaConf/numpy/h5py/tensorboard/wandb/diffusers/pytest.
8. Smoke verification: `import mani_skill; gym.make("PushT-v1").reset()` works.
9. Vendor the DP baseline from ManiSkill's `examples/baselines/diffusion_policy/` into `src/visuomotor_verification/policy/diffusion_policy/` with `UPSTREAM.md` recording the source commit hash. **No refactor yet** — pristine copy with a sibling `adapter.py` stub that says "TODO: wire to Policy ABC" but is not yet implemented.

**Out (next PRs):**

- Wiring the DP baseline through `Policy` ABC + Hydra config (its own PR).
- First end-to-end DP training run on push-T.
- Trajectory collection script implementation.
- Concrete `Verifier` subclass (its design will revise §4's provisional ABC).

---

## 10. Conda environment

`environment.yml`:

```yaml
name: visuomotor_verification
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pytorch=2.4.*
  - pytorch-cuda=12.1
  - torchvision
  - pip:
      - mani_skill
      - hydra-core
      - omegaconf
      - numpy
      - h5py
      - tensorboard
      - wandb
      - tqdm
      - diffusers
      - einops
      - pytest
```

Driver is CUDA 12.2; PyTorch 2.4 with the CUDA 12.1 build is forward-compatible. Additional pip deps the vendored DP baseline pulls in will be folded into this file when the baseline is vendored (read its `requirements.txt`).

---

## 11. Open questions / known unknowns

- **`Verifier` ABC will be revised.** When we know what the first concrete verifier looks like, we will rewrite §4's verifier signature. The current shape is intentionally minimal.
- **Trajectory storage format.** Starting with per-episode `.npz`; may switch to a single HDF5 file per run later if loading becomes a bottleneck. The `core/trajectory.py` I/O layer hides this.
- **wandb usage policy.** Included as a dep, off by default unless `logging.wandb.enabled=true` in config. Project / entity defaults TBD with first training run.
- **`Trajectory` schema.** Will be finalized when the first DP rollout is collected. Expected fields: `observations`, `actions`, `rewards`, `terminated`, `truncated`, `success`, `info`, plus `run_config` for provenance. Subject to revision based on what the first verifier actually needs.

---

## 12. CLAUDE.md content

`CLAUDE.md` at repo root will record (for future Claude Code sessions and for human readers):

- One-paragraph project vision (verbatim from §1).
- Storage root: `/common/users/shared/pracsys/visuomotor_verification-data/`.
- Determinism principle: "Every phase entry point calls `seed_all(RunConfig.from_hydra(cfg))` exactly once. Nothing else in the codebase touches global RNGs or cuDNN flags."
- Git cleanliness gate: `RunMode.DETERMINISTIC` refuses to run with a dirty tree unless `allow_dirty=true` is set explicitly. Do not paper over this with a blanket override — commit or stash instead.
- The four abstraction axes and where their ABCs live.
- Run ID format and `metadata.json` invariant.
- Per-phase entry-point script names and what each one reads/writes.
- Conda env name: `visuomotor_verification`.
- The fact that `Verifier` ABC is provisional.
