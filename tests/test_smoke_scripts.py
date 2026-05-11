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


@pytest.mark.slow
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
    assert "input_artifacts" in payload
    assert payload["input_artifacts"] == {}
