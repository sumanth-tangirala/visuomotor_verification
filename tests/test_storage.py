from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from visuomotor_verification.core.storage import (
    StorageConfig,
    demo_run_dir,
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
    import datetime as _dt

    t0 = _dt.datetime(2026, 1, 1, 0, 0, 0)
    t1 = _dt.datetime(2026, 1, 1, 0, 0, 1)
    assert mint_run_id("exp", now=t0) != mint_run_id("exp", now=t1)


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


def test_demo_run_dir(tmp_path: Path) -> None:
    cfg = StorageConfig(root=tmp_path)
    d = demo_run_dir(cfg, task="push_t", run_id="rid")
    assert d == tmp_path / "datasets" / "push_t" / "demos" / "rid"
