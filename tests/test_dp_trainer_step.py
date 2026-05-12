"""Tests for trainer.py: DemoDataset, TrainerConfig, train()."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


DEMO_PATH = Path.home() / ".maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"


def _need_demo() -> None:
    if not DEMO_PATH.exists():
        pytest.skip(
            f"Demo file not found at {DEMO_PATH}. "
            "Run the demo-prep recipe in docs/superpowers/specs/2026-05-11-dp-adapter-design.md §1."
        )


@pytest.mark.slow
def test_demo_dataset_loads_two_trajectories() -> None:
    _need_demo()
    from visuomotor_verification.policy.diffusion_policy.trainer import DemoDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DemoDataset(
        data_path=DEMO_PATH,
        device=device,
        control_mode="pd_ee_delta_pose",
        env_id="PushT-v1",
        obs_mode="rgb",
        num_traj=2,
        obs_horizon=2,
        pred_horizon=16,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert "observations" in sample and "actions" in sample
    obs = sample["observations"]
    assert obs["state"].shape[0] == 2          # obs_horizon
    assert obs["rgb"].shape[0] == 2            # obs_horizon
    assert sample["actions"].shape[0] == 16    # pred_horizon


def test_trainer_config_importable_and_constructible() -> None:
    from visuomotor_verification.policy.diffusion_policy.trainer import TrainerConfig
    from visuomotor_verification.core.determinism import Seeds

    cfg = TrainerConfig(
        demo_path=Path("/dev/null"),
        env_id="PushT-v1", control_mode="pd_ee_delta_pose",
        obs_mode="rgb", max_episode_steps=150, sim_backend="physx_cuda",
        obs_horizon=2, act_horizon=1, pred_horizon=16,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256],
        n_groups=8, num_diffusion_iters=100,
        total_iters=2, batch_size=4, lr=1e-4,
        num_demos=2, num_dataload_workers=0,
        log_freq=1, eval_freq=10_000, save_freq=None,
        num_eval_episodes=2, num_eval_envs=1,
        seeds=Seeds(),
        device=torch.device("cpu"),
    )
    assert cfg.env_id == "PushT-v1"
