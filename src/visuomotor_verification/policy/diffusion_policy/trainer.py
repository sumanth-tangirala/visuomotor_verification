"""Trainer module: DemoDataset + TrainerConfig + train().

Re-implements the dataset wrapper from vendored train_rgbd.py because the
upstream class references module-level globals (`args.control_mode`) set only
inside `if __name__ == "__main__":`. We lift those globals to __init__ kwargs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional

from . import _vendor_import  # noqa: F401 -- must run before vendored imports

import gymnasium as gym
import mani_skill.envs  # noqa: F401 -- registers ManiSkill envs
import numpy as np
import torch
import torch.optim as optim
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import (
    IterationBasedBatchSampler,
    build_state_obs_extractor,
    convert_obs,
    load_demo_dataset,
    worker_init_fn,
)

from visuomotor_verification.core.determinism import Seeds
from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy


def _reorder_keys(d: dict, ref: Any) -> dict:
    """Mirror upstream train_rgbd.py:reorder_keys (lines 114-121)."""
    out = {}
    for k, v in ref.items() if isinstance(ref, dict) else ref.spaces.items():
        if isinstance(v, (dict, spaces.Dict)):
            out[k] = _reorder_keys(d[k], v)
        else:
            out[k] = d[k]
    return out


class DemoDataset(Dataset):
    """DP training dataset. Mirrors upstream
    SmallDemoDataset_DiffusionPolicy (train_rgbd.py:124-244). Pre-loads h5 demos
    to `device` memory and precomputes sliding-window slice indices.

    UPSTREAM: train_rgbd.py:124-244. If upstream's dataset preprocessing
    changes, update this class.
    """

    def __init__(
        self,
        data_path: Path,
        *,
        device: torch.device,
        control_mode: str,
        env_id: str,
        obs_mode: str,
        num_traj: Optional[int],
        obs_horizon: int,
        pred_horizon: int,
    ) -> None:
        # Probe the env's original observation space so we can reorder demo
        # obs dicts to match.
        tmp_env = gym.make(
            env_id,
            control_mode=control_mode,
            obs_mode=obs_mode,
            sim_backend="physx_cpu",  # CPU probe is enough for the obs_space
            max_episode_steps=1,
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
        )
        original_obs_space = tmp_env.observation_space
        self.include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
        self.include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
        tmp_env.close()

        obs_process_fn = partial(
            convert_obs,
            concat_fn=partial(np.concatenate, axis=-1),
            transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
            state_obs_extractor=build_state_obs_extractor(env_id),
            depth="rgbd" in str(data_path),
        )

        trajectories = load_demo_dataset(str(data_path), num_traj=num_traj, concat=False)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = _reorder_keys(obs_traj_dict, original_obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.tensor(
                    _obs_traj_dict["depth"].astype(np.float32),
                    device=device, dtype=torch.float16,
                )
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(device)
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(device)
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict_list[0].keys())

        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.tensor(
                trajectories["actions"][i], device=device, dtype=torch.float32,
            )

        # Compute (traj_idx, start, end) sliding windows.
        if "delta_pos" in control_mode or control_mode == "base_pd_joint_vel_arm_pd_joint_vel":
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device,
            )
        else:
            raise NotImplementedError(
                f"Control Mode {control_mode} not supported (upstream restriction)"
            )

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.slices: list[tuple[int, int, int]] = []
        n_traj = len(trajectories["actions"])
        for traj_idx in range(n_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        traj_idx, start, end = self.slices[index]
        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq: dict[str, torch.Tensor] = {}
        for k, v in obs_traj.items():
            chunk = v[max(0, start): start + self.obs_horizon]
            if start < 0:
                pad = torch.stack([chunk[0]] * abs(start), dim=0)
                chunk = torch.cat((pad, chunk), dim=0)
            obs_seq[k] = chunk
        act_seq = self.trajectories["actions"][traj_idx][max(0, start): end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        L = self.trajectories["actions"][traj_idx].shape[0]
        if end > L:
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        assert obs_seq["state"].shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        return {"observations": obs_seq, "actions": act_seq}
