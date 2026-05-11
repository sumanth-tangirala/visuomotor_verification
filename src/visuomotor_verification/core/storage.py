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

    Note: production scripts get their run_id from Hydra's `${now:...}`
    interpolation in `hydra.run.dir` (see `configs/*.yaml`), not from this
    function. `mint_run_id` is for callers that need to build paths
    programmatically outside a Hydra app (tests, ad-hoc helpers).
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


def demo_run_dir(cfg: StorageConfig, *, task: str, run_id: str) -> Path:
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
