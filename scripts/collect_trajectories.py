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
