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
