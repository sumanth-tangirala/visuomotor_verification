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
        "training.eval_freq=0",        # disable eval for speed
        "training.save_freq=1",        # checkpoint at iter 1
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
    assert (run_dir / "logs").exists()
    ckpts = list((run_dir / "checkpoints").iterdir())
    assert len(ckpts) >= 1, f"no checkpoints under {run_dir / 'checkpoints'}"

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["script"] == "train_policy.py"
    assert metadata["run_config"]["mode"] == "stochastic"
    # output_artifacts.last_checkpoint must be set by the script's post-run annotation.
    assert "output_artifacts" in metadata
    assert "last_checkpoint" in metadata["output_artifacts"]
