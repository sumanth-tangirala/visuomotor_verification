"""Trajectory dataclass + npz I/O.

A `Trajectory` is the unit of data exchanged between policy rollouts and
verifier training. Schema is intentionally minimal; will be revised when the
first concrete verifier is implemented (see spec §11).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Trajectory:
    observations: np.ndarray   # shape (T+1, *obs_dims) or (T, *obs_dims) -- impl decides
    actions: np.ndarray        # shape (T, *act_dims)
    rewards: np.ndarray        # shape (T,)
    terminated: np.ndarray     # shape (T,), bool
    truncated: np.ndarray      # shape (T,), bool
    success: bool
    # TODO: narrow to a typed RunConfig once the schema is finalized (spec §11).
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = len(self.actions)
        for name in ("rewards", "terminated", "truncated"):
            arr = getattr(self, name)
            if len(arr) != T:
                raise ValueError(
                    f"length mismatch: actions has {T} but {name} has {len(arr)}"
                )

    def __len__(self) -> int:
        return len(self.actions)

    def save_npz(self, path: Path) -> None:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        # run_metadata is JSON-serialized into a 0-d string array for portability.
        np.savez(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminated=self.terminated,
            truncated=self.truncated,
            success=np.array(self.success),
            run_metadata=np.array(json.dumps(self.run_metadata)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "Trajectory":
        with np.load(path, allow_pickle=False) as data:
            # Materialize all arrays inside the `with` so they survive after close.
            return cls(
                observations=np.array(data["observations"]),
                actions=np.array(data["actions"]),
                rewards=np.array(data["rewards"]),
                terminated=np.array(data["terminated"]),
                truncated=np.array(data["truncated"]),
                success=bool(data["success"]),
                run_metadata=json.loads(str(data["run_metadata"])),
            )
