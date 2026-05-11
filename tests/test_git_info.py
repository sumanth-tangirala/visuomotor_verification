from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from visuomotor_verification.core import git_info


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)
    (path / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "a.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_clean_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    info = git_info.collect(tmp_path)
    assert info["dirty"] is False
    assert len(info["sha"]) == 40
    assert "diff" not in info
    assert "untracked" not in info


def test_dirty_modified_file(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("hello world\n")
    info = git_info.collect(tmp_path)
    assert info["dirty"] is True
    assert "hello world" in info["diff"]
    assert "untracked" in info
    assert info["untracked"] == []


def test_dirty_untracked_file(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "scratch.py").write_text("print('hi')\n")
    info = git_info.collect(tmp_path)
    assert info["dirty"] is True
    assert "scratch.py" in info["untracked"]


def test_staged_change_is_dirty(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("staged change\n")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    info = git_info.collect(tmp_path)
    assert info["dirty"] is True
    assert "staged change" in info["diff"]


def test_not_a_repo(tmp_path: Path) -> None:
    with pytest.raises(git_info.NotARepoError):
        git_info.collect(tmp_path)
