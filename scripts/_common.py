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
    info = git_info.collect(REPO_ROOT)
    resolved_seeds = seed_all(run_cfg, repo_root=REPO_ROOT, git_info_cache=info)

    run_dir = Path(HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "run_id": run_dir.name,
        "experiment_name": cfg.experiment_name,
        "script": script_name,
        "cmdline": sys.argv,
        "git_sha": info["sha"],
        "git_dirty": info["dirty"],
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
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
