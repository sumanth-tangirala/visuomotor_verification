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
