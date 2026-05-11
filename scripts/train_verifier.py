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
