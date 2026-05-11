# Visuomotor Verification — Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the `visuomotor_verification` repository with a conda env, OOP abstractions for `Simulator`/`Task`/`Policy`/`Verifier`, a determinism layer (`RunMode`/seeds/git-cleanliness gate), Hydra configs, per-phase entry-point stubs, and a pristine vendoring of ManiSkill's diffusion-policy baseline. Sets up the system to train DP on push-T in a subsequent PR — no end-to-end training in this plan.

**Architecture:** Python `src/` package layout. Conda env handles Python+PyTorch+CUDA; `pyproject.toml` handles pip deps. Four ABCs (`Simulator`, `Task`, `Policy`, `Verifier`) live under their own subpackages. `core/determinism.py` is the only module that touches global RNGs and cuDNN flags. Hydra `_target_` instantiation is the registry; `hydra.run.dir` interpolates into the shared data dir so each run's `.hydra/` snapshot lives alongside its checkpoints. All disk-heavy artifacts go to `/common/users/shared/pracsys/visuomotor_verification-data/`.

**Tech Stack:** Python 3.11, PyTorch 2.4 + CUDA 12.1 build (driver is 12.2), ManiSkill 3, Hydra/OmegaConf, pytest. Conda for environment management.

**Spec:** `docs/superpowers/specs/2026-05-11-foundations-design.md`

**Repository root for all paths below:** `/common/home/st1122/Projects/visuomotor_verification/`

---

## File Structure

### Top level

- `pyproject.toml` — package metadata + pip deps (everything except python/pytorch/cuda)
- `environment.yml` — conda env spec (python + pytorch + cuda)
- `README.md` — one-paragraph description, env setup, links to spec/plan
- `CLAUDE.md` — guidance for future Claude Code sessions

### Source (`src/visuomotor_verification/`)

- `__init__.py`
- `core/__init__.py`
- `core/types.py` — `Action`/`Observation`/`StepResult`/`VerifierOutput` aliases
- `core/git_info.py` — git SHA/dirty/diff/untracked helpers
- `core/determinism.py` — `RunMode`/`Seeds`/`RunConfig`/`derive_seed`/`seed_all`/`DirtyTreeError`
- `core/storage.py` — `StorageConfig`/`mint_run_id`/`run_dir`/`write_metadata`
- `core/trajectory.py` — `Trajectory` dataclass + `save_npz`/`load_npz`
- `simulator/__init__.py`, `simulator/base.py` — `Simulator` ABC
- `task/__init__.py`, `task/base.py` — `Task` ABC
- `policy/__init__.py`, `policy/base.py` — `Policy` ABC
- `policy/diffusion_policy/UPSTREAM.md` — source URL + upstream commit hash
- `policy/diffusion_policy/<vendored files from ManiSkill examples>`
- `policy/diffusion_policy/adapter.py` — TODO stub for `Policy` ABC adapter
- `verifier/__init__.py`, `verifier/base.py` — `Verifier` ABC (provisional)

### Configs (`configs/`)

- `train_policy.yaml`, `collect_trajectories.yaml`, `train_verifier.yaml`, `evaluate_verifier.yaml`
- `run/deterministic.yaml`, `run/stochastic.yaml`, `run/mixed.yaml`
- `simulator/maniskill.yaml`
- `task/push_t.yaml`
- `policy/diffusion_policy.yaml`
- `verifier/placeholder.yaml`
- `storage/default.yaml`

### Scripts (`scripts/`, Hydra entry-point stubs)

- `train_policy.py`, `collect_trajectories.py`, `train_verifier.py`, `evaluate_verifier.py`

### Tests (`tests/`)

- `__init__.py`
- `test_git_info.py`
- `test_determinism.py`
- `test_storage.py`
- `test_trajectory.py`
- `test_smoke_maniskill.py`
- `test_smoke_scripts.py`

---

## Task 1: Create conda env, package metadata, and editable install

**Files:**
- Create: `environment.yml`
- Create: `pyproject.toml`
- Create: `src/visuomotor_verification/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1.1: Write `environment.yml`**

Create file `/common/home/st1122/Projects/visuomotor_verification/environment.yml`:

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
```

- [ ] **Step 1.2: Write `pyproject.toml`**

Create file `/common/home/st1122/Projects/visuomotor_verification/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "visuomotor_verification"
version = "0.0.1"
description = "Research on visuomotor verification: training verifiers that predict policy success/failure from trajectories."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "mani_skill",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "numpy",
    "h5py",
    "tensorboard",
    "wandb",
    "tqdm",
    "diffusers",
    "einops",
]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-xdist"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 1.3: Create empty package `__init__` files**

Create `src/visuomotor_verification/__init__.py` with single line:

```python
__version__ = "0.0.1"
```

Create empty `tests/__init__.py` (zero bytes).

- [ ] **Step 1.4: Create the conda environment** (~5 minutes)

Run:
```bash
cd /common/home/st1122/Projects/visuomotor_verification
conda env create -f environment.yml
```

Expected: env `visuomotor_verification` created. If the env already exists, abort and ask the user — do **not** recreate without permission.

- [ ] **Step 1.5: Install package in editable mode + dev deps**

Run:
```bash
conda run -n visuomotor_verification pip install -e ".[dev]"
```

Expected: installs all pip deps including ManiSkill. ManiSkill may pull in `sapien`, `gymnasium`, etc. — this is expected.

- [ ] **Step 1.6: Verify imports work**

Run:
```bash
conda run -n visuomotor_verification python -c "import visuomotor_verification; import mani_skill; import hydra; import torch; print(torch.cuda.is_available())"
```

Expected: `True` (or `False` if no GPU is visible — that is OK, but record the actual output). No `ImportError`.

- [ ] **Step 1.7: Commit**

```bash
git add environment.yml pyproject.toml src/visuomotor_verification/__init__.py tests/__init__.py
git commit -m "Add conda env + package metadata + editable install"
```

---

## Task 2: Implement `core/types.py`

Plain type aliases. Used by the ABCs in later tasks. No tests beyond importability.

**Files:**
- Create: `src/visuomotor_verification/core/__init__.py`
- Create: `src/visuomotor_verification/core/types.py`

- [ ] **Step 2.1: Create `core/__init__.py`** (zero bytes)

- [ ] **Step 2.2: Implement `core/types.py`**

```python
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
```

- [ ] **Step 2.3: Verify import**

```bash
conda run -n visuomotor_verification python -c "from visuomotor_verification.core.types import StepResult, VerifierOutput; print('ok')"
```

Expected output: `ok`.

- [ ] **Step 2.4: Commit**

```bash
git add src/visuomotor_verification/core/__init__.py src/visuomotor_verification/core/types.py
git commit -m "Add core/types.py with Action/Observation/StepResult/VerifierOutput"
```

---

## Task 3: Implement `core/git_info.py` (TDD)

Provides `git_sha()`, `git_dirty()`, `git_diff()`, `git_untracked()` for `metadata.json` and the cleanliness gate.

**Files:**
- Create: `src/visuomotor_verification/core/git_info.py`
- Create: `tests/test_git_info.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_git_info.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from visuomotor_verification.core import git_info


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)
    (path / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "a.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_clean_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    info = git_info.collect(tmp_path)
    assert info["dirty"] is False
    assert len(info["sha"]) == 40
    assert "diff" not in info
    assert "untracked" not in info


def test_dirty_modified_file(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("hello world\n")
    info = git_info.collect(tmp_path)
    assert info["dirty"] is True
    assert "hello world" in info["diff"]
    assert info.get("untracked", []) == []


def test_dirty_untracked_file(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "scratch.py").write_text("print('hi')\n")
    info = git_info.collect(tmp_path)
    assert info["dirty"] is True
    assert "scratch.py" in info["untracked"]


def test_staged_change_is_dirty(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("staged change\n")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    info = git_info.collect(tmp_path)
    assert info["dirty"] is True
    assert "staged change" in info["diff"]


def test_not_a_repo(tmp_path: Path) -> None:
    with pytest.raises(git_info.NotARepoError):
        git_info.collect(tmp_path)
```

- [ ] **Step 3.2: Run the test, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_git_info.py -v
```

Expected: `ImportError: cannot import name 'git_info'` or similar — module does not yet exist.

- [ ] **Step 3.3: Implement `core/git_info.py`**

```python
"""Git working-tree inspection. Used by metadata.json and the determinism gate.

This module is read-only with respect to git: it never modifies the repo.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class NotARepoError(RuntimeError):
    """Raised when the given path is not inside a git repository."""


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # `git rev-parse` returns 128 when not in a repo.
        if "not a git repository" in result.stderr.lower():
            raise NotARepoError(f"{cwd} is not inside a git repository")
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def collect(repo: Path) -> dict[str, Any]:
    """Collect git working-tree info as a JSON-serializable dict.

    Keys:
      - sha:        always present, 40-char HEAD SHA
      - dirty:      always present, bool
      - diff:       present iff dirty; full `git diff HEAD` output (staged + unstaged)
      - untracked:  present iff dirty; list of untracked file paths
    """
    repo = Path(repo)
    # `rev-parse HEAD` validates we're in a repo and gives the SHA.
    sha = _git(["rev-parse", "HEAD"], cwd=repo).strip()

    # `status --porcelain` is the canonical "is anything dirty" check.
    status = _git(["status", "--porcelain"], cwd=repo)
    dirty = bool(status.strip())

    info: dict[str, Any] = {"sha": sha, "dirty": dirty}
    if dirty:
        info["diff"] = _git(["diff", "HEAD"], cwd=repo)
        untracked_raw = _git(
            ["ls-files", "--others", "--exclude-standard"], cwd=repo
        )
        info["untracked"] = [
            line for line in untracked_raw.splitlines() if line
        ]
    return info
```

- [ ] **Step 3.4: Run the test, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_git_info.py -v
```

Expected: 5 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/visuomotor_verification/core/git_info.py tests/test_git_info.py
git commit -m "Add core/git_info with SHA/dirty/diff/untracked collection"
```

---

## Task 4: Implement `core/determinism.py` — dataclasses + `derive_seed` (TDD)

Locks in `RunMode`, `Seeds`, `RunConfig`, `derive_seed`. No global side effects yet.

**Files:**
- Create: `src/visuomotor_verification/core/determinism.py` (partial)
- Create: `tests/test_determinism.py` (partial)

- [ ] **Step 4.1: Write failing tests**

Create `tests/test_determinism.py`:

```python
from __future__ import annotations

import pytest

from visuomotor_verification.core.determinism import (
    RunConfig,
    RunMode,
    Seeds,
    derive_seed,
)


def test_runmode_values() -> None:
    assert RunMode("deterministic") is RunMode.DETERMINISTIC
    assert RunMode("stochastic") is RunMode.STOCHASTIC
    assert RunMode("mixed") is RunMode.MIXED


def test_seeds_default_all_none() -> None:
    s = Seeds()
    assert s.master is None
    assert s.sim is None
    assert s.policy is None
    assert s.torch is None
    assert s.numpy is None
    assert s.python is None
    assert s.dataloader is None
    assert s.cuda_strict is False


def test_runconfig_default_allow_dirty_false() -> None:
    cfg = RunConfig(mode=RunMode.STOCHASTIC, seeds=Seeds())
    assert cfg.allow_dirty is False


def test_derive_seed_deterministic_per_component() -> None:
    # Same master + same component => same seed
    assert derive_seed(42, "sim") == derive_seed(42, "sim")
    # Same master + different components => different seeds
    assert derive_seed(42, "sim") != derive_seed(42, "policy")
    # Different masters + same component => different seeds
    assert derive_seed(42, "sim") != derive_seed(43, "sim")


def test_derive_seed_in_uint32_range() -> None:
    for master in [0, 1, 42, 2**31, 2**32 - 1]:
        for comp in ["sim", "policy", "torch", "numpy", "python", "dataloader"]:
            seed = derive_seed(master, comp)
            assert 0 <= seed <= 2**32 - 1


def test_derive_seed_known_value() -> None:
    # Pin to a specific value so we catch accidental hash changes.
    # If this fails after a deliberate change, update the expected value AND
    # be aware that all previously-deterministic runs will now produce different
    # data with the same master seed.
    import hashlib

    master, comp = 42, "sim"
    expected = (
        int.from_bytes(
            hashlib.blake2b(f"{master}:{comp}".encode(), digest_size=8).digest(),
            "big",
        )
        & 0xFFFFFFFF
    )
    assert derive_seed(master, comp) == expected
```

- [ ] **Step 4.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_determinism.py -v
```

Expected: `ImportError` — module does not exist.

- [ ] **Step 4.3: Implement `core/determinism.py` (partial)**

Create `src/visuomotor_verification/core/determinism.py`:

```python
"""Determinism layer.

This is the ONLY module in the codebase that touches global RNGs and cuDNN
flags. All other code receives a `RunConfig` (or a component-specific seed)
and uses local RNG instances.

See docs/superpowers/specs/2026-05-11-foundations-design.md §5 for the design.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class RunMode(str, Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    MIXED = "mixed"


@dataclass(frozen=True)
class Seeds:
    master: int | None = None
    sim: int | None = None
    policy: int | None = None
    torch: int | None = None
    numpy: int | None = None
    python: int | None = None
    dataloader: int | None = None
    cuda_strict: bool = False


@dataclass(frozen=True)
class RunConfig:
    mode: RunMode
    seeds: Seeds
    allow_dirty: bool = False


def derive_seed(master: int, component: str) -> int:
    """Deterministic seed for a component, derived from the master seed.

    Uses blake2b (not Python's `hash()`, which is not stable across processes).
    Returns a value in [0, 2**32) so it fits all common RNG seed types.
    """
    digest = hashlib.blake2b(
        f"{master}:{component}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF
```

- [ ] **Step 4.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_determinism.py -v
```

Expected: 6 passed.

- [ ] **Step 4.5: Commit**

```bash
git add src/visuomotor_verification/core/determinism.py tests/test_determinism.py
git commit -m "Add RunMode/Seeds/RunConfig/derive_seed"
```

---

## Task 5: Implement `seed_all` (TDD)

The boundary-enforcing function that seeds python/numpy/torch/cuDNN once per phase.

**Files:**
- Modify: `src/visuomotor_verification/core/determinism.py`
- Modify: `tests/test_determinism.py`

- [ ] **Step 5.1: Add failing tests**

Append to `tests/test_determinism.py`:

```python
import random

import numpy as np
import torch

from visuomotor_verification.core.determinism import resolve_seeds, seed_all


def test_resolve_seeds_deterministic_fills_unset() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=42))
    resolved = resolve_seeds(cfg)
    assert resolved.master == 42
    # Every component seed should be filled deterministically.
    for comp in ["sim", "policy", "torch", "numpy", "python", "dataloader"]:
        v = getattr(resolved, comp)
        assert v is not None, f"{comp} should be derived in DETERMINISTIC mode"
        assert v == derive_seed(42, comp)


def test_resolve_seeds_deterministic_respects_explicit_override() -> None:
    cfg = RunConfig(
        mode=RunMode.DETERMINISTIC,
        seeds=Seeds(master=42, sim=999),
    )
    resolved = resolve_seeds(cfg)
    assert resolved.sim == 999
    assert resolved.policy == derive_seed(42, "policy")


def test_resolve_seeds_stochastic_clears_all() -> None:
    cfg = RunConfig(
        mode=RunMode.STOCHASTIC,
        seeds=Seeds(master=42, sim=999),  # explicit values should be ignored
    )
    with pytest.warns(UserWarning, match="STOCHASTIC mode ignores"):
        resolved = resolve_seeds(cfg)
    assert resolved.master is None
    assert resolved.sim is None
    assert resolved.policy is None


def test_resolve_seeds_mixed_only_uses_explicit() -> None:
    cfg = RunConfig(
        mode=RunMode.MIXED,
        seeds=Seeds(sim=7),
    )
    resolved = resolve_seeds(cfg)
    assert resolved.sim == 7
    assert resolved.policy is None  # not set => not seeded
    assert resolved.torch is None


def test_resolve_seeds_deterministic_requires_master() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds())
    with pytest.raises(ValueError, match="master"):
        resolve_seeds(cfg)


def test_seed_all_reproducible_python_random() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=42))
    # Skip the git-cleanliness gate for this test by using a non-repo cwd.
    seed_all(cfg, repo_root=None)
    a = [random.random() for _ in range(5)]
    seed_all(cfg, repo_root=None)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_all_reproducible_numpy() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=42))
    seed_all(cfg, repo_root=None)
    a = np.random.rand(5).tolist()
    seed_all(cfg, repo_root=None)
    b = np.random.rand(5).tolist()
    assert a == b


def test_seed_all_reproducible_torch() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=42))
    seed_all(cfg, repo_root=None)
    a = torch.randn(5).tolist()
    seed_all(cfg, repo_root=None)
    b = torch.randn(5).tolist()
    assert a == b


def test_seed_all_idempotent_under_same_config() -> None:
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=7))
    seed_all(cfg, repo_root=None)
    seed_all(cfg, repo_root=None)  # second call must not raise


def test_seed_all_stochastic_runs_without_master() -> None:
    cfg = RunConfig(mode=RunMode.STOCHASTIC, seeds=Seeds())
    seed_all(cfg, repo_root=None)  # must not raise
```

- [ ] **Step 5.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_determinism.py -v
```

Expected: failures for `resolve_seeds` and `seed_all` not defined.

- [ ] **Step 5.3: Add `resolve_seeds` and `seed_all` to `core/determinism.py`**

Append to `src/visuomotor_verification/core/determinism.py`:

```python
import random
import warnings
from pathlib import Path

import numpy as np
import torch


_COMPONENTS = ("sim", "policy", "torch", "numpy", "python", "dataloader")


def resolve_seeds(cfg: RunConfig) -> Seeds:
    """Return a Seeds object with every field that *will be used* filled in.

    - DETERMINISTIC: all unset fields are derived from master. `master` is required.
    - MIXED: only explicitly-set fields are honored; unset ones stay None.
    - STOCHASTIC: every field is forced to None; warn if any was explicitly set.
    """
    if cfg.mode is RunMode.DETERMINISTIC:
        if cfg.seeds.master is None:
            raise ValueError(
                "RunMode.DETERMINISTIC requires seeds.master to be set"
            )
        derived = {
            comp: (
                getattr(cfg.seeds, comp)
                if getattr(cfg.seeds, comp) is not None
                else derive_seed(cfg.seeds.master, comp)
            )
            for comp in _COMPONENTS
        }
        return Seeds(master=cfg.seeds.master, cuda_strict=cfg.seeds.cuda_strict, **derived)

    if cfg.mode is RunMode.MIXED:
        return cfg.seeds  # already has only-explicit semantics

    # STOCHASTIC
    explicit = [
        c for c in (("master",) + _COMPONENTS) if getattr(cfg.seeds, c) is not None
    ]
    if explicit:
        warnings.warn(
            "STOCHASTIC mode ignores explicitly-set seeds: "
            f"{explicit}. This is almost certainly a config bug.",
            stacklevel=2,
        )
    return Seeds(cuda_strict=cfg.seeds.cuda_strict)


def seed_all(cfg: RunConfig, repo_root: Path | None) -> Seeds:
    """Seed every global RNG source according to `cfg`. Returns the resolved Seeds.

    The git-cleanliness gate (see §5.7 of the foundations spec) is enforced
    here when `repo_root` is not None. Pass `repo_root=None` only in tests
    that exercise pure seeding behavior outside of a repo context.

    This is the ONLY function that should touch torch.manual_seed, np.random.seed,
    random.seed, or cuDNN flags.
    """
    # Note: git gate is added in a later task.
    resolved = resolve_seeds(cfg)

    if resolved.python is not None:
        random.seed(resolved.python)
    if resolved.numpy is not None:
        np.random.seed(resolved.numpy)
    if resolved.torch is not None:
        torch.manual_seed(resolved.torch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved.torch)

    # cuDNN flags
    if cfg.mode is RunMode.DETERMINISTIC or resolved.cuda_strict:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return resolved
```

- [ ] **Step 5.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_determinism.py -v
```

Expected: all tests pass. There may be a `UserWarning` about `torch.use_deterministic_algorithms` — that is fine, the function is being exercised correctly.

- [ ] **Step 5.5: Commit**

```bash
git add src/visuomotor_verification/core/determinism.py tests/test_determinism.py
git commit -m "Add resolve_seeds and seed_all"
```

---

## Task 6: Implement git cleanliness gate in `seed_all` (TDD)

Adds `DirtyTreeError` and the §5.7 mode-gated check.

**Files:**
- Modify: `src/visuomotor_verification/core/determinism.py`
- Modify: `tests/test_determinism.py`

- [ ] **Step 6.1: Write failing tests**

Append to `tests/test_determinism.py`:

```python
import subprocess

from visuomotor_verification.core.determinism import DirtyTreeError


def _make_git_repo(path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)
    (path / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "a.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_dirty_tree_blocks_deterministic(tmp_path) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=1))
    with pytest.raises(DirtyTreeError):
        seed_all(cfg, repo_root=tmp_path)


def test_dirty_tree_allowed_with_allow_dirty(tmp_path) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")
    cfg = RunConfig(
        mode=RunMode.DETERMINISTIC,
        seeds=Seeds(master=1),
        allow_dirty=True,
    )
    seed_all(cfg, repo_root=tmp_path)  # must not raise


def test_dirty_tree_warns_in_stochastic(tmp_path) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")
    cfg = RunConfig(mode=RunMode.STOCHASTIC, seeds=Seeds())
    with pytest.warns(UserWarning, match="dirty"):
        seed_all(cfg, repo_root=tmp_path)


def test_dirty_tree_warns_in_mixed(tmp_path) -> None:
    _make_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")
    cfg = RunConfig(mode=RunMode.MIXED, seeds=Seeds())
    with pytest.warns(UserWarning, match="dirty"):
        seed_all(cfg, repo_root=tmp_path)


def test_clean_tree_passes_deterministic(tmp_path) -> None:
    _make_git_repo(tmp_path)
    cfg = RunConfig(mode=RunMode.DETERMINISTIC, seeds=Seeds(master=1))
    seed_all(cfg, repo_root=tmp_path)  # must not raise
```

- [ ] **Step 6.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_determinism.py -v -k "dirty or clean_tree"
```

Expected: ImportError on `DirtyTreeError` and/or behavior mismatch.

- [ ] **Step 6.3: Wire the gate into `seed_all`**

Replace the body of `seed_all` in `src/visuomotor_verification/core/determinism.py` with the version below (note the new `DirtyTreeError`, import of `git_info`, and the early gate-check):

```python
class DirtyTreeError(RuntimeError):
    """Raised when a DETERMINISTIC run is attempted in a dirty git tree."""


def seed_all(cfg: RunConfig, repo_root: Path | None) -> Seeds:
    """Seed every global RNG source according to `cfg`. Returns the resolved Seeds.

    The git-cleanliness gate (see §5.7 of the foundations spec) is enforced
    here when `repo_root` is not None. Pass `repo_root=None` only in tests
    that exercise pure seeding behavior outside of a repo context.

    This is the ONLY function that should touch torch.manual_seed, np.random.seed,
    random.seed, or cuDNN flags.
    """
    from visuomotor_verification.core import git_info

    if repo_root is not None:
        info = git_info.collect(repo_root)
        if info["dirty"]:
            if cfg.mode is RunMode.DETERMINISTIC and not cfg.allow_dirty:
                raise DirtyTreeError(
                    "RunMode.DETERMINISTIC refuses to run in a dirty tree. "
                    "Commit or stash your changes, or set run.allow_dirty=true "
                    "explicitly (the run will record `git_diff` in metadata.json "
                    "regardless)."
                )
            warnings.warn(
                f"Working tree is dirty (mode={cfg.mode.value}); "
                "the run will proceed but git_sha will not uniquely identify "
                "the code that ran. The diff is captured in metadata.json.",
                stacklevel=2,
            )

    resolved = resolve_seeds(cfg)

    if resolved.python is not None:
        random.seed(resolved.python)
    if resolved.numpy is not None:
        np.random.seed(resolved.numpy)
    if resolved.torch is not None:
        torch.manual_seed(resolved.torch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved.torch)

    if cfg.mode is RunMode.DETERMINISTIC or resolved.cuda_strict:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return resolved
```

- [ ] **Step 6.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_determinism.py -v
```

Expected: all tests pass (including the 5 new ones).

- [ ] **Step 6.5: Commit**

```bash
git add src/visuomotor_verification/core/determinism.py tests/test_determinism.py
git commit -m "Add DirtyTreeError and git cleanliness gate in seed_all"
```

---

## Task 7: Implement `RunConfig.from_hydra` (TDD)

Lets entry-point scripts do `seed_all(RunConfig.from_hydra(cfg), repo_root=...)`.

**Files:**
- Modify: `src/visuomotor_verification/core/determinism.py`
- Modify: `tests/test_determinism.py`

- [ ] **Step 7.1: Write failing test**

Append to `tests/test_determinism.py`:

```python
from omegaconf import OmegaConf


def test_from_hydra_deterministic() -> None:
    cfg = OmegaConf.create(
        {
            "run": {
                "mode": "deterministic",
                "allow_dirty": False,
                "seeds": {"master": 42, "sim": 999},
            }
        }
    )
    rc = RunConfig.from_hydra(cfg)
    assert rc.mode is RunMode.DETERMINISTIC
    assert rc.allow_dirty is False
    assert rc.seeds.master == 42
    assert rc.seeds.sim == 999
    assert rc.seeds.policy is None  # not in config => None


def test_from_hydra_stochastic_minimal() -> None:
    cfg = OmegaConf.create({"run": {"mode": "stochastic", "seeds": {}}})
    rc = RunConfig.from_hydra(cfg)
    assert rc.mode is RunMode.STOCHASTIC
    assert rc.seeds.master is None


def test_from_hydra_mixed_with_partial_seeds() -> None:
    cfg = OmegaConf.create(
        {"run": {"mode": "mixed", "seeds": {"sim": 7, "cuda_strict": True}}}
    )
    rc = RunConfig.from_hydra(cfg)
    assert rc.mode is RunMode.MIXED
    assert rc.seeds.sim == 7
    assert rc.seeds.cuda_strict is True
```

- [ ] **Step 7.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_determinism.py -v -k "from_hydra"
```

- [ ] **Step 7.3: Add `from_hydra` classmethod**

In `src/visuomotor_verification/core/determinism.py`, replace the `@dataclass(frozen=True) class RunConfig:` block with:

```python
@dataclass(frozen=True)
class RunConfig:
    mode: RunMode
    seeds: Seeds
    allow_dirty: bool = False

    @classmethod
    def from_hydra(cls, cfg) -> "RunConfig":
        """Build a RunConfig from a Hydra/OmegaConf DictConfig.

        Expects `cfg.run.mode`, `cfg.run.seeds` (with any subset of fields),
        and optional `cfg.run.allow_dirty`.
        """
        run = cfg.run
        seeds_cfg = run.get("seeds", {}) or {}
        seeds = Seeds(
            master=seeds_cfg.get("master"),
            sim=seeds_cfg.get("sim"),
            policy=seeds_cfg.get("policy"),
            torch=seeds_cfg.get("torch"),
            numpy=seeds_cfg.get("numpy"),
            python=seeds_cfg.get("python"),
            dataloader=seeds_cfg.get("dataloader"),
            cuda_strict=bool(seeds_cfg.get("cuda_strict", False)),
        )
        return cls(
            mode=RunMode(run.mode),
            seeds=seeds,
            allow_dirty=bool(run.get("allow_dirty", False)),
        )
```

- [ ] **Step 7.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_determinism.py -v
```

- [ ] **Step 7.5: Commit**

```bash
git add src/visuomotor_verification/core/determinism.py tests/test_determinism.py
git commit -m "Add RunConfig.from_hydra"
```

---

## Task 8: Implement `core/storage.py` (TDD)

`mint_run_id` + `run_dir` resolution + `write_metadata`.

**Files:**
- Create: `src/visuomotor_verification/core/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 8.1: Write failing tests**

Create `tests/test_storage.py`:

```python
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from visuomotor_verification.core.storage import (
    StorageConfig,
    mint_run_id,
    policy_run_dir,
    rollout_run_dir,
    verifier_run_dir,
    write_metadata,
)


def test_mint_run_id_format() -> None:
    rid = mint_run_id("push_t_dp_v1")
    assert re.fullmatch(
        r"push_t_dp_v1-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", rid
    ), rid


def test_mint_run_id_two_calls_differ() -> None:
    import time

    a = mint_run_id("exp")
    time.sleep(1.1)  # second-precision timestamp; must differ in adjacent seconds
    b = mint_run_id("exp")
    assert a != b


def test_mint_run_id_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="experiment_name"):
        mint_run_id("")


def test_mint_run_id_rejects_path_unsafe() -> None:
    # Reject characters that would break path parsing (slashes, etc.).
    with pytest.raises(ValueError):
        mint_run_id("bad/name")
    with pytest.raises(ValueError):
        mint_run_id("bad name")


def test_policy_run_dir(tmp_path: Path) -> None:
    cfg = StorageConfig(root=tmp_path)
    d = policy_run_dir(cfg, task="push_t", policy="diffusion_policy", run_id="exp-2026-01-01_00-00-00")
    assert d == tmp_path / "policies" / "push_t" / "diffusion_policy" / "exp-2026-01-01_00-00-00"


def test_rollout_run_dir(tmp_path: Path) -> None:
    cfg = StorageConfig(root=tmp_path)
    d = rollout_run_dir(cfg, task="push_t", run_id="rid")
    assert d == tmp_path / "datasets" / "push_t" / "rollouts" / "rid"


def test_verifier_run_dir(tmp_path: Path) -> None:
    cfg = StorageConfig(root=tmp_path)
    d = verifier_run_dir(cfg, task="push_t", verifier="v1", run_id="rid")
    assert d == tmp_path / "experiments" / "verifier" / "push_t" / "v1" / "rid"


def test_write_metadata_roundtrip(tmp_path: Path) -> None:
    payload = {"run_id": "exp-2026-01-01_00-00-00", "git_sha": "abc"}
    target = tmp_path / "metadata.json"
    write_metadata(target, payload)
    assert target.exists()
    loaded = json.loads(target.read_text())
    assert loaded == payload


def test_write_metadata_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "metadata.json"
    write_metadata(target, {"ok": True})
    assert target.exists()
```

- [ ] **Step 8.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_storage.py -v
```

- [ ] **Step 8.3: Implement `core/storage.py`**

```python
"""Storage layout helpers.

Resolves paths under the shared data root and mints run IDs of the form
`<experiment_name>-<YYYY-MM-DD>_<HH-MM-SS>` (see §7 of the foundations spec).
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
DEFAULT_ROOT = Path("/common/users/shared/pracsys/visuomotor_verification-data")


@dataclass(frozen=True)
class StorageConfig:
    root: Path = DEFAULT_ROOT


def mint_run_id(experiment_name: str, *, now: _dt.datetime | None = None) -> str:
    """Build `<experiment_name>-<YYYY-MM-DD>_<HH-MM-SS>`.

    `experiment_name` must be a non-empty path-safe slug (alnum, `_`, `-`, `.`).
    """
    if not experiment_name:
        raise ValueError("experiment_name must be non-empty")
    if not _NAME_RE.match(experiment_name):
        raise ValueError(
            f"experiment_name {experiment_name!r} must match {_NAME_RE.pattern}"
        )
    when = now or _dt.datetime.now()
    return f"{experiment_name}-{when.strftime('%Y-%m-%d_%H-%M-%S')}"


def policy_run_dir(cfg: StorageConfig, *, task: str, policy: str, run_id: str) -> Path:
    return cfg.root / "policies" / task / policy / run_id


def demo_dataset_dir(cfg: StorageConfig, *, task: str, run_id: str) -> Path:
    return cfg.root / "datasets" / task / "demos" / run_id


def rollout_run_dir(cfg: StorageConfig, *, task: str, run_id: str) -> Path:
    return cfg.root / "datasets" / task / "rollouts" / run_id


def verifier_run_dir(
    cfg: StorageConfig, *, task: str, verifier: str, run_id: str
) -> Path:
    return cfg.root / "experiments" / "verifier" / task / verifier / run_id


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Write a metadata.json (or any JSON) file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
```

- [ ] **Step 8.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_storage.py -v
```

Expected: 9 passed.

- [ ] **Step 8.5: Commit**

```bash
git add src/visuomotor_verification/core/storage.py tests/test_storage.py
git commit -m "Add core/storage with run_id minting and path resolution"
```

---

## Task 9: Implement `core/trajectory.py` (TDD)

Provides the `Trajectory` dataclass + npz save/load with metadata roundtrip.

**Files:**
- Create: `src/visuomotor_verification/core/trajectory.py`
- Create: `tests/test_trajectory.py`

- [ ] **Step 9.1: Write failing tests**

Create `tests/test_trajectory.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from visuomotor_verification.core.trajectory import Trajectory


def _make_trajectory() -> Trajectory:
    return Trajectory(
        observations=np.arange(30, dtype=np.float32).reshape(10, 3),
        actions=np.arange(20, dtype=np.float32).reshape(10, 2),
        rewards=np.linspace(0.0, 1.0, 10, dtype=np.float32),
        terminated=np.array([False] * 9 + [True]),
        truncated=np.zeros(10, dtype=bool),
        success=True,
        run_metadata={"mode": "deterministic", "master": 42, "git_sha": "abc"},
    )


def test_trajectory_basic_fields() -> None:
    t = _make_trajectory()
    assert t.observations.shape == (10, 3)
    assert t.actions.shape == (10, 2)
    assert t.success is True
    assert t.run_metadata["master"] == 42


def test_trajectory_npz_roundtrip(tmp_path: Path) -> None:
    t = _make_trajectory()
    p = tmp_path / "ep_0.npz"
    t.save_npz(p)
    loaded = Trajectory.load_npz(p)
    np.testing.assert_array_equal(loaded.observations, t.observations)
    np.testing.assert_array_equal(loaded.actions, t.actions)
    np.testing.assert_array_equal(loaded.rewards, t.rewards)
    np.testing.assert_array_equal(loaded.terminated, t.terminated)
    np.testing.assert_array_equal(loaded.truncated, t.truncated)
    assert loaded.success is True
    assert loaded.run_metadata == t.run_metadata


def test_trajectory_length() -> None:
    t = _make_trajectory()
    assert len(t) == 10


def test_trajectory_length_consistency_enforced() -> None:
    import pytest

    with pytest.raises(ValueError, match="length"):
        Trajectory(
            observations=np.zeros((10, 3)),
            actions=np.zeros((9, 2)),  # wrong length
            rewards=np.zeros(10),
            terminated=np.zeros(10, dtype=bool),
            truncated=np.zeros(10, dtype=bool),
            success=False,
            run_metadata={},
        )
```

- [ ] **Step 9.2: Run, expect failure**

```bash
conda run -n visuomotor_verification pytest tests/test_trajectory.py -v
```

- [ ] **Step 9.3: Implement `core/trajectory.py`**

```python
"""Trajectory dataclass + npz I/O.

A `Trajectory` is the unit of data exchanged between policy rollouts and
verifier training. Schema is intentionally minimal; will be revised when the
first concrete verifier is implemented (see spec §11).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Trajectory:
    observations: np.ndarray   # shape (T+1, *obs_dims) or (T, *obs_dims) -- impl decides
    actions: np.ndarray        # shape (T, *act_dims)
    rewards: np.ndarray        # shape (T,)
    terminated: np.ndarray     # shape (T,), bool
    truncated: np.ndarray      # shape (T,), bool
    success: bool
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = len(self.actions)
        for name in ("rewards", "terminated", "truncated"):
            arr = getattr(self, name)
            if len(arr) != T:
                raise ValueError(
                    f"length mismatch: actions has {T} but {name} has {len(arr)}"
                )

    def __len__(self) -> int:
        return len(self.actions)

    def save_npz(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # run_metadata is JSON-serialized into a 0-d string array for portability.
        np.savez(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminated=self.terminated,
            truncated=self.truncated,
            success=np.array(self.success),
            run_metadata=np.array(json.dumps(self.run_metadata)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "Trajectory":
        data = np.load(path, allow_pickle=False)
        return cls(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            terminated=data["terminated"],
            truncated=data["truncated"],
            success=bool(data["success"]),
            run_metadata=json.loads(str(data["run_metadata"])),
        )
```

- [ ] **Step 9.4: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_trajectory.py -v
```

- [ ] **Step 9.5: Commit**

```bash
git add src/visuomotor_verification/core/trajectory.py tests/test_trajectory.py
git commit -m "Add Trajectory dataclass and npz I/O"
```

---

## Task 10: Define the four ABCs

Skeleton signatures only. ABC instantiation must error without all `@abstractmethod`s; tests assert this.

**Files:**
- Create: `src/visuomotor_verification/simulator/__init__.py`
- Create: `src/visuomotor_verification/simulator/base.py`
- Create: `src/visuomotor_verification/task/__init__.py`
- Create: `src/visuomotor_verification/task/base.py`
- Create: `src/visuomotor_verification/policy/__init__.py`
- Create: `src/visuomotor_verification/policy/base.py`
- Create: `src/visuomotor_verification/verifier/__init__.py`
- Create: `src/visuomotor_verification/verifier/base.py`
- Create: `tests/test_abcs.py`

- [ ] **Step 10.1: Create empty `__init__.py`s**

Create zero-byte files:
- `src/visuomotor_verification/simulator/__init__.py`
- `src/visuomotor_verification/task/__init__.py`
- `src/visuomotor_verification/policy/__init__.py`
- `src/visuomotor_verification/verifier/__init__.py`

- [ ] **Step 10.2: Write failing test for ABCs**

Create `tests/test_abcs.py`:

```python
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
```

- [ ] **Step 10.3: Run, expect failure** (modules don't exist yet)

```bash
conda run -n visuomotor_verification pytest tests/test_abcs.py -v
```

- [ ] **Step 10.4: Implement `simulator/base.py`**

```python
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
```

- [ ] **Step 10.5: Implement `task/base.py`**

```python
"""Task ABC. A Task configures a Simulator and defines the success criterion."""
from __future__ import annotations

from abc import ABC, abstractmethod

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
    def is_success(self, obs: Observation, info: dict) -> bool: ...

    @property
    @abstractmethod
    def horizon(self) -> int: ...
```

- [ ] **Step 10.6: Implement `policy/base.py`**

```python
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
```

- [ ] **Step 10.7: Implement `verifier/base.py`**

```python
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
    @abstractmethod
    def fit(self, trajectories: Iterable[Trajectory]) -> None: ...

    @abstractmethod
    def predict(self, trajectory: Trajectory) -> VerifierOutput: ...
```

- [ ] **Step 10.8: Run, expect pass**

```bash
conda run -n visuomotor_verification pytest tests/test_abcs.py -v
```

Expected: 5 passed.

- [ ] **Step 10.9: Commit**

```bash
git add src/visuomotor_verification/simulator src/visuomotor_verification/task src/visuomotor_verification/policy/__init__.py src/visuomotor_verification/policy/base.py src/visuomotor_verification/verifier tests/test_abcs.py
git commit -m "Add Simulator/Task/Policy/Verifier ABCs"
```

---

## Task 11: Create Hydra configs skeleton

Each config below is a stub: enough to instantiate via `_target_` when the concrete classes land, and enough to compose for the script smoke tests.

**Files:**
- Create: 12 YAML files under `configs/` (listed below)

- [ ] **Step 11.1: Create `configs/storage/default.yaml`**

```yaml
root: /common/users/shared/pracsys/visuomotor_verification-data
```

- [ ] **Step 11.2: Create `configs/run/deterministic.yaml`**

```yaml
mode: deterministic
allow_dirty: false
seeds:
  master: 42
  # All other seeds derive from master. Override here to fix a single component
  # (e.g. `sim: 999`) while leaving the rest derived.
```

- [ ] **Step 11.3: Create `configs/run/stochastic.yaml`**

```yaml
mode: stochastic
allow_dirty: true
seeds: {}
```

- [ ] **Step 11.4: Create `configs/run/mixed.yaml`**

```yaml
mode: mixed
allow_dirty: false
seeds:
  # Set only the components you want to fix. Unset ones run from the OS RNG.
  # Example:
  # sim: 7
```

- [ ] **Step 11.5: Create `configs/simulator/maniskill.yaml`**

```yaml
_target_: visuomotor_verification.simulator.maniskill.ManiSkillSimulator
# Concrete impl is added in a later PR; this stub records the target path.
name: maniskill
env_id: PushT-v1
control_mode: pd_ee_delta_pose
obs_mode: state
```

- [ ] **Step 11.6: Create `configs/task/push_t.yaml`**

```yaml
_target_: visuomotor_verification.task.push_t.PushTTask
name: push_t
horizon: 200
```

- [ ] **Step 11.7: Create `configs/policy/diffusion_policy.yaml`**

```yaml
_target_: visuomotor_verification.policy.diffusion_policy.adapter.DiffusionPolicy
name: diffusion_policy
# Hyperparams are filled in when the adapter is implemented (next PR).
checkpoint: null
```

- [ ] **Step 11.8: Create `configs/verifier/placeholder.yaml`**

```yaml
_target_: null  # filled in when the first concrete verifier exists
name: placeholder
```

- [ ] **Step 11.9: Create `configs/train_policy.yaml`**

```yaml
defaults:
  - run: deterministic
  - simulator: maniskill
  - task: push_t
  - policy: diffusion_policy
  - storage: default
  - _self_

experiment_name: ???   # must be provided on the command line

hydra:
  run:
    dir: ${storage.root}/policies/${task.name}/${policy.name}/${experiment_name}-${now:%Y-%m-%d_%H-%M-%S}
  job:
    chdir: false
```

- [ ] **Step 11.10: Create `configs/collect_trajectories.yaml`**

```yaml
defaults:
  - run: deterministic
  - simulator: maniskill
  - task: push_t
  - policy: diffusion_policy
  - storage: default
  - _self_

experiment_name: ???
num_episodes: 100

hydra:
  run:
    dir: ${storage.root}/datasets/${task.name}/rollouts/${experiment_name}-${now:%Y-%m-%d_%H-%M-%S}
  job:
    chdir: false
```

- [ ] **Step 11.11: Create `configs/train_verifier.yaml`**

```yaml
defaults:
  - run: deterministic
  - task: push_t
  - verifier: placeholder
  - storage: default
  - _self_

experiment_name: ???
rollouts_run: ???    # path to the rollouts run_dir feeding this verifier

hydra:
  run:
    dir: ${storage.root}/experiments/verifier/${task.name}/${verifier.name}/${experiment_name}-${now:%Y-%m-%d_%H-%M-%S}
  job:
    chdir: false
```

- [ ] **Step 11.12: Create `configs/evaluate_verifier.yaml`**

```yaml
defaults:
  - run: stochastic
  - task: push_t
  - verifier: placeholder
  - storage: default
  - _self_

experiment_name: ???
verifier_checkpoint: ???
rollouts_run: ???

hydra:
  run:
    dir: ${storage.root}/experiments/verifier/${task.name}/${verifier.name}/${experiment_name}-${now:%Y-%m-%d_%H-%M-%S}/eval
  job:
    chdir: false
```

- [ ] **Step 11.13: Commit**

```bash
git add configs/
git commit -m "Add Hydra config skeleton for all four phases"
```

---

## Task 12: Create per-phase entry-point stubs

Each script: loads Hydra config, calls `seed_all` with the repo root, writes a `metadata.json` into `hydra.run.dir`, then raises `NotImplementedError`. Enough to verify Hydra composition works end-to-end.

**Files:**
- Create: `scripts/_common.py` — shared metadata writer
- Create: `scripts/train_policy.py`
- Create: `scripts/collect_trajectories.py`
- Create: `scripts/train_verifier.py`
- Create: `scripts/evaluate_verifier.py`
- Create: `tests/test_smoke_scripts.py`

- [ ] **Step 12.1: Create `scripts/_common.py`**

```python
"""Shared boilerplate for Hydra entry-point scripts.

Every phase entry point does the same prologue:
  1. Build RunConfig from cfg.
  2. seed_all() with the repo root for the cleanliness gate.
  3. Write metadata.json into hydra.run.dir.
After that, each script does its phase-specific work.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from visuomotor_verification.core import git_info
from visuomotor_verification.core.determinism import RunConfig, seed_all
from visuomotor_verification.core.storage import write_metadata


REPO_ROOT = Path(__file__).resolve().parent.parent


def prologue(cfg: DictConfig, *, script_name: str) -> Path:
    """Standard prologue. Returns the resolved run directory (hydra.run.dir)."""
    run_cfg = RunConfig.from_hydra(cfg)
    resolved_seeds = seed_all(run_cfg, repo_root=REPO_ROOT)

    run_dir = Path(HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    info = git_info.collect(REPO_ROOT)
    metadata: dict[str, Any] = {
        "run_id": run_dir.name,
        "experiment_name": cfg.experiment_name,
        "script": script_name,
        "cmdline": sys.argv,
        "git_sha": info["sha"],
        "git_dirty": info["dirty"],
        "timestamp": _dt.datetime.now().isoformat(),
        "run_config": {
            "mode": run_cfg.mode.value,
            "allow_dirty": run_cfg.allow_dirty,
            "seeds": {
                "master": resolved_seeds.master,
                "sim": resolved_seeds.sim,
                "policy": resolved_seeds.policy,
                "torch": resolved_seeds.torch,
                "numpy": resolved_seeds.numpy,
                "python": resolved_seeds.python,
                "dataloader": resolved_seeds.dataloader,
                "cuda_strict": resolved_seeds.cuda_strict,
            },
        },
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    if info["dirty"]:
        metadata["git_diff"] = info["diff"]
        metadata["git_untracked"] = info["untracked"]

    write_metadata(run_dir / "metadata.json", metadata)
    return run_dir
```

- [ ] **Step 12.2: Create `scripts/train_policy.py`**

```python
"""Train a policy. Stub: prologue runs, body raises NotImplementedError.

Usage:
  python scripts/train_policy.py experiment_name=push_t_dp_v1
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import prologue


@hydra.main(version_base=None, config_path="../configs", config_name="train_policy")
def main(cfg: DictConfig) -> None:
    run_dir = prologue(cfg, script_name="train_policy.py")
    print(f"[train_policy] run_dir={run_dir}")
    raise NotImplementedError(
        "train_policy is a stub. Real DP-baseline integration is the next PR."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.3: Create `scripts/collect_trajectories.py`**

```python
"""Collect trajectories from a trained policy. Stub.

Usage:
  python scripts/collect_trajectories.py \\
      experiment_name=push_t_rollouts_v1 \\
      policy.checkpoint=/path/to/best.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import prologue


@hydra.main(
    version_base=None, config_path="../configs", config_name="collect_trajectories"
)
def main(cfg: DictConfig) -> None:
    run_dir = prologue(cfg, script_name="collect_trajectories.py")
    print(f"[collect_trajectories] run_dir={run_dir}")
    raise NotImplementedError(
        "collect_trajectories is a stub. Implementation in a subsequent PR."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.4: Create `scripts/train_verifier.py`**

```python
"""Train a verifier. Stub."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import prologue


@hydra.main(version_base=None, config_path="../configs", config_name="train_verifier")
def main(cfg: DictConfig) -> None:
    run_dir = prologue(cfg, script_name="train_verifier.py")
    print(f"[train_verifier] run_dir={run_dir}")
    raise NotImplementedError(
        "train_verifier is a stub. Implementation deferred until first verifier exists."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.5: Create `scripts/evaluate_verifier.py`**

```python
"""Evaluate a verifier on held-out rollouts. Stub."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import prologue


@hydra.main(
    version_base=None, config_path="../configs", config_name="evaluate_verifier"
)
def main(cfg: DictConfig) -> None:
    run_dir = prologue(cfg, script_name="evaluate_verifier.py")
    print(f"[evaluate_verifier] run_dir={run_dir}")
    raise NotImplementedError(
        "evaluate_verifier is a stub. Implementation deferred."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.6: Create `tests/test_smoke_scripts.py`**

```python
"""Smoke tests: each script's prologue runs end-to-end and writes metadata.json.

These tests use stochastic mode + a tmp_path storage root so they don't
touch the shared data dir and don't need a clean git tree.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_script(script: str, tmp_path: Path, extra: list[str]) -> tuple[Path, int, str]:
    """Run a stub script in stochastic mode with tmp_path as storage root."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / script),
        "run=stochastic",
        f"storage.root={tmp_path}",
        *extra,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return tmp_path, result.returncode, result.stdout + result.stderr


@pytest.mark.parametrize(
    "script,extra",
    [
        ("train_policy.py", ["experiment_name=smoke_train"]),
        (
            "collect_trajectories.py",
            ["experiment_name=smoke_collect", "policy.checkpoint=/dev/null"],
        ),
        (
            "train_verifier.py",
            ["experiment_name=smoke_verifier", "rollouts_run=/tmp/none"],
        ),
        (
            "evaluate_verifier.py",
            [
                "experiment_name=smoke_eval",
                "verifier_checkpoint=/tmp/none",
                "rollouts_run=/tmp/none",
            ],
        ),
    ],
)
def test_stub_writes_metadata(script: str, extra: list[str], tmp_path: Path) -> None:
    root, rc, out = _run_script(script, tmp_path, extra)
    # The stub raises NotImplementedError, which Hydra surfaces as a non-zero
    # exit; we just need the prologue to have run, which we verify via metadata.
    metadata_files = list(root.rglob("metadata.json"))
    assert metadata_files, f"no metadata.json under {root}; output was:\n{out}"
    payload = json.loads(metadata_files[0].read_text())
    assert payload["script"] == script
    assert "experiment_name" in payload
    assert payload["run_config"]["mode"] == "stochastic"
```

- [ ] **Step 12.7: Run the smoke tests**

```bash
conda run -n visuomotor_verification pytest tests/test_smoke_scripts.py -v
```

Expected: 4 passed. If any script raises before `prologue()` completes (e.g. a config interpolation error), the test will fail with the diagnostic output captured — fix accordingly.

- [ ] **Step 12.8: Commit**

```bash
git add scripts/ tests/test_smoke_scripts.py
git commit -m "Add per-phase entry-point stubs with shared prologue + smoke tests"
```

---

## Task 13: Vendor ManiSkill diffusion-policy baseline

Pristine copy from upstream, with `UPSTREAM.md` recording the source URL and commit hash. No refactor in this PR — only an `adapter.py` TODO stub for the `Policy`-ABC wiring (deferred to next PR).

**Files:**
- Create: `src/visuomotor_verification/policy/diffusion_policy/` (vendored files)
- Create: `src/visuomotor_verification/policy/diffusion_policy/UPSTREAM.md`
- Create: `src/visuomotor_verification/policy/diffusion_policy/__init__.py`
- Create: `src/visuomotor_verification/policy/diffusion_policy/adapter.py`

- [ ] **Step 13.1: Clone upstream ManiSkill to a temp location**

```bash
mkdir -p /tmp/maniskill-upstream
git clone --depth 1 https://github.com/haosulab/ManiSkill.git /tmp/maniskill-upstream
cd /tmp/maniskill-upstream
git log -1 --format='%H' > /tmp/maniskill-sha
cat /tmp/maniskill-sha
```

Expected: a 40-char SHA prints. Save it; you'll paste it into `UPSTREAM.md` in step 13.4.

- [ ] **Step 13.2: Verify the DP baseline path exists upstream**

```bash
ls /tmp/maniskill-upstream/examples/baselines/diffusion_policy/
```

Expected: file listing. If the directory is named differently (e.g. `diffusion-policy`), use the actual path and note this in `UPSTREAM.md`. If no DP baseline exists in `examples/baselines/`, run `find /tmp/maniskill-upstream -type d -iname '*diffusion*' -not -path '*.git*'` and report back — do not proceed to copy without confirming the source location.

- [ ] **Step 13.3: Copy the vendored files into our repo**

```bash
SRC=/tmp/maniskill-upstream/examples/baselines/diffusion_policy
DST=/common/home/st1122/Projects/visuomotor_verification/src/visuomotor_verification/policy/diffusion_policy
mkdir -p "$DST"
cp -r "$SRC"/. "$DST"/
ls "$DST"
```

Expected: vendored files in place. Do not delete or modify any of them in this PR.

- [ ] **Step 13.4: Write `UPSTREAM.md`**

Create `src/visuomotor_verification/policy/diffusion_policy/UPSTREAM.md`:

```markdown
# Vendored from ManiSkill

This directory is a **pristine vendored copy** of ManiSkill's diffusion-policy
baseline. Do not modify these files inline; if changes are needed, do them in
`adapter.py` or a sibling module so the diff against upstream stays auditable.

- **Source:** https://github.com/haosulab/ManiSkill
- **Path in source:** `examples/baselines/diffusion_policy/`
- **Commit hash:** <PASTE THE SHA FROM STEP 13.1 HERE>
- **Vendored on:** 2026-05-11

## Re-syncing with upstream

To compare against newer upstream:

```bash
git clone --depth 1 https://github.com/haosulab/ManiSkill.git /tmp/ms-new
diff -r /tmp/ms-new/examples/baselines/diffusion_policy/ \
        src/visuomotor_verification/policy/diffusion_policy/ \
        | grep -v UPSTREAM.md | grep -v adapter.py | grep -v __init__.py
```

## Adapter status

`adapter.py` will wrap this baseline behind our `Policy` ABC. As of this PR,
`adapter.py` is a TODO stub — not yet implemented. Implementation is deferred
to the next PR (see foundations design spec §9).
```

**Important:** paste the actual SHA you captured in step 13.1 into the `Commit hash:` field. Do not leave the placeholder.

- [ ] **Step 13.5: Create `__init__.py` and `adapter.py` stub**

`src/visuomotor_verification/policy/diffusion_policy/__init__.py`:

```python
# Vendored diffusion-policy baseline from ManiSkill. See UPSTREAM.md.
```

`src/visuomotor_verification/policy/diffusion_policy/adapter.py`:

```python
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
```

- [ ] **Step 13.6: Verify the package still imports**

```bash
conda run -n visuomotor_verification python -c "from visuomotor_verification.policy.diffusion_policy import adapter; print('ok')"
```

Expected: `ok`. (Importing `adapter` is fine; instantiating `DiffusionPolicy()` would raise.)

- [ ] **Step 13.7: Check that no test was inadvertently broken**

```bash
conda run -n visuomotor_verification pytest -q
```

Expected: all previous tests pass; nothing new is exercised.

- [ ] **Step 13.8: Commit**

```bash
git add src/visuomotor_verification/policy/diffusion_policy/
git commit -m "Vendor ManiSkill diffusion-policy baseline (pristine) + adapter stub"
```

---

## Task 14: ManiSkill smoke test

Verifies that ManiSkill is importable and a push-T env can be created and reset.

**Files:**
- Create: `tests/test_smoke_maniskill.py`

- [ ] **Step 14.1: Write the smoke test**

```python
"""Smoke test: ManiSkill is installed and a push-T env can reset.

Marked with @pytest.mark.slow so it's easy to skip on CI if needed.
"""
from __future__ import annotations

import pytest


def test_maniskill_importable() -> None:
    import mani_skill  # noqa: F401


@pytest.mark.slow
def test_push_t_env_resets() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  -- registers envs

    push_t_ids = [k for k in gym.envs.registry.keys() if "PushT" in k]
    if not push_t_ids:
        pytest.skip(
            "No PushT env registered. Available push-* envs: "
            + str([k for k in gym.envs.registry.keys() if "push" in k.lower()])
        )
    env = gym.make(push_t_ids[0])
    obs, info = env.reset(seed=0)
    assert obs is not None
    env.close()
```

- [ ] **Step 14.2: Register the `slow` marker in `pyproject.toml`**

Add to the existing `[tool.pytest.ini_options]` section of `pyproject.toml` (after `addopts`):

```toml
markers = ["slow: marks tests that take >1s (deselect with -m 'not slow')"]
```

The full updated section should look like:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = ["slow: marks tests that take >1s (deselect with -m 'not slow')"]
```

- [ ] **Step 14.3: Run the fast portion** (should already pass)

```bash
conda run -n visuomotor_verification pytest tests/test_smoke_maniskill.py::test_maniskill_importable -v
```

Expected: 1 passed.

- [ ] **Step 14.4: Run the slow portion**

```bash
conda run -n visuomotor_verification pytest tests/test_smoke_maniskill.py::test_push_t_env_resets -v -m slow --no-header
```

Expected: 1 passed (or skipped with a descriptive message if `PushT-v1` isn't the right env ID). If skipped, look at the message — it lists available `push*` envs — and update `configs/simulator/maniskill.yaml`'s `env_id` to whatever the actual push-T env ID is in this ManiSkill version. Re-run after editing the config; the test should still pass (the test doesn't read the config, only registry).

- [ ] **Step 14.5: Commit**

```bash
git add tests/test_smoke_maniskill.py pyproject.toml
git commit -m "Add ManiSkill smoke test"
```

If you updated `configs/simulator/maniskill.yaml` in step 14.4, include it in this commit (or a separate one — your call).

---

## Task 15: Write `CLAUDE.md` and `README.md`

Last task. Records the project's load-bearing facts so future sessions don't relearn them.

**Files:**
- Create: `CLAUDE.md`
- Create: `README.md`

- [ ] **Step 15.1: Write `CLAUDE.md`**

Create `/common/home/st1122/Projects/visuomotor_verification/CLAUDE.md`:

```markdown
# Visuomotor Verification — Project Guide

## Vision

Research on **visuomotor verification**: training a verifier that, given a
trajectory produced by a visuomotor policy, predicts whether the policy will
succeed or fail at the task. Pipeline: train policy → roll out → label
success/failure → train verifier → evaluate.

The project moves across **simulators**, **tasks**, **policy types**, and
**verifier architectures**. Code is structured so each axis is swappable via
the four ABCs in `src/visuomotor_verification/{simulator,task,policy,verifier}/base.py`,
instantiated via Hydra `_target_`.

## Paths

- **Shared data root:** `/common/users/shared/pracsys/visuomotor_verification-data/`
  - All datasets, checkpoints, verifier experiments, and Hydra outputs live here.
  - The repo itself contains only code, configs, docs, and tests.
- **Subdirectories** (already created):
  - `datasets/<task>/{demos,rollouts}/<run_id>/`
  - `policies/<task>/<policy>/<run_id>/`
  - `experiments/verifier/<task>/<verifier>/<run_id>/`

## Determinism principle

Every phase entry-point script (`scripts/*.py`) calls
`seed_all(RunConfig.from_hydra(cfg), repo_root=REPO_ROOT)` exactly once at
the top. **Nothing else in the codebase touches global RNGs or cuDNN flags.**
Components that have their own RNG receive a component seed via constructor
argument and create a local `torch.Generator` / `np.random.Generator`.

`RunMode` is one of `deterministic` / `mixed` / `stochastic`. In
`deterministic` mode, all unset per-component seeds derive from
`seeds.master` via blake2b — so a single master seed reproduces a whole run.

## Git cleanliness gate

`RunMode.DETERMINISTIC` refuses to run with a dirty tree unless
`run.allow_dirty=true` is set explicitly. Do **not** paper over this with a
blanket override — commit or stash instead. The diff is always recorded in
`metadata.json` regardless of mode.

## Run IDs and metadata

- **Run ID format:** `<experiment_name>-<YYYY-MM-DD>_<HH-MM-SS>`
- **`experiment_name`** is a mandatory top-level Hydra field for every script.
- **`metadata.json`** lives in every run directory (alongside `.hydra/`) and
  records: run_id, experiment_name, script, cmdline, git_sha, git_dirty,
  git_diff (if dirty), git_untracked (if dirty), timestamp, run_config,
  resolved_config.

## Per-phase entry points

| Script | Reads | Writes |
|---|---|---|
| `scripts/train_policy.py` | `datasets/<task>/demos/<run_id>` | `policies/<task>/<policy>/<run_id>` |
| `scripts/collect_trajectories.py` | policy checkpoint | `datasets/<task>/rollouts/<run_id>` |
| `scripts/train_verifier.py` | rollouts | `experiments/verifier/<task>/<verifier>/<run_id>` |
| `scripts/evaluate_verifier.py` | verifier ckpt + held-out rollouts | eval subdir |

Inter-phase artifact paths are passed via Hydra overrides; **no implicit "latest" magic**.

## Conda env

```bash
conda env create -f environment.yml
conda activate visuomotor_verification
pip install -e ".[dev]"
```

Env name: `visuomotor_verification`. Python 3.11, PyTorch 2.4 with CUDA 12.1
build (driver is 12.2).

## Status notes

- The `Verifier` ABC at `src/visuomotor_verification/verifier/base.py` is
  **provisional** — its shape will be revised when the first concrete
  verifier exists.
- The diffusion-policy baseline under
  `src/visuomotor_verification/policy/diffusion_policy/` is a **pristine
  vendored copy** from upstream ManiSkill. See `UPSTREAM.md` in that
  directory. The `adapter.py` wiring it to the `Policy` ABC is a TODO stub.

## Spec & plan

- Foundations design spec: `docs/superpowers/specs/2026-05-11-foundations-design.md`
- Foundations implementation plan: `docs/superpowers/plans/2026-05-11-foundations.md`
```

- [ ] **Step 15.2: Write `README.md`**

Create `/common/home/st1122/Projects/visuomotor_verification/README.md`:

```markdown
# visuomotor_verification

Research on visuomotor verification — training a verifier that predicts
whether a visuomotor policy will succeed or fail at a task, from its
trajectories. See `CLAUDE.md` for the project guide, or
`docs/superpowers/specs/2026-05-11-foundations-design.md` for the full
foundations design.

## Quick setup

```bash
conda env create -f environment.yml
conda activate visuomotor_verification
pip install -e ".[dev]"
pytest -m "not slow"
```

Disk-heavy artifacts (datasets, checkpoints, experiments) live under
`/common/users/shared/pracsys/visuomotor_verification-data/`.

## Status

Foundations PR: repo scaffolding, ABCs for simulator/task/policy/verifier,
determinism layer with git-cleanliness gate, Hydra configs, per-phase
entry-point stubs, vendored ManiSkill diffusion-policy baseline. End-to-end
training and first verifier are subsequent PRs.
```

- [ ] **Step 15.3: Verify the full test suite still passes**

```bash
conda run -n visuomotor_verification pytest -q -m "not slow"
```

Expected: all tests pass.

- [ ] **Step 15.4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Add CLAUDE.md and README.md"
```

---

## Final verification

After all tasks above, run the full suite (including slow tests) once and
inspect the test count:

```bash
conda run -n visuomotor_verification pytest -v
```

Expected counts (approximate):
- `test_git_info.py`: 5
- `test_determinism.py`: ~20 (varies as tests were added across tasks 4–7)
- `test_storage.py`: 9
- `test_trajectory.py`: 4
- `test_abcs.py`: 5
- `test_smoke_scripts.py`: 4
- `test_smoke_maniskill.py`: 2 (1 if PushT is skipped)

Total ~47–48 tests.

Then verify the foundations are end-to-end-usable with a real Hydra run:

```bash
cd /common/home/st1122/Projects/visuomotor_verification
conda run -n visuomotor_verification python scripts/train_policy.py \
    experiment_name=foundations_smoke \
    run=stochastic
```

Expected:
- The script raises `NotImplementedError` (this is correct — the stub did its job).
- A run directory exists at
  `/common/users/shared/pracsys/visuomotor_verification-data/policies/push_t/diffusion_policy/foundations_smoke-<timestamp>/`
- That directory contains `metadata.json` and a `.hydra/` subdir.

Inspect:

```bash
ls /common/users/shared/pracsys/visuomotor_verification-data/policies/push_t/diffusion_policy/foundations_smoke-*/
cat /common/users/shared/pracsys/visuomotor_verification-data/policies/push_t/diffusion_policy/foundations_smoke-*/metadata.json | head -30
```

You should see `run_id`, `experiment_name`, `git_sha`, `run_config`, `resolved_config`.

If everything checks out, the foundations PR is ready for review.

## Deferred from spec §5.6

The spec calls for a `test_determinism.py` assertion that the same
`(mode=DETERMINISTIC, master=k)` produces byte-identical trajectories across
two runs of a real sim+policy combination. That test cannot be written in
this PR because there is no concrete `Simulator` or `Policy` implementation
yet. It is deferred to the PR that wires the DP baseline through the
`Policy` ABC; track it there. The current `test_determinism.py` covers the
underlying invariants (seed derivation, per-source RNG reproducibility,
idempotency) but not the end-to-end trajectory-equality claim.
