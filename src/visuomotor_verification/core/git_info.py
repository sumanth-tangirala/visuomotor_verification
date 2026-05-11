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
