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

# EMAModel is imported lazily inside train() to avoid a diffusers/torch
# version incompatibility (flash-attn custom-op schema error) that occurs when
# diffusers.training_utils is imported at module level on this environment.

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


@dataclass(frozen=True)
class TrainerConfig:
    """All knobs required by `train()`. Built by `scripts/train_policy.py` from
    the Hydra DictConfig (`cfg.task`, `cfg.policy`, `cfg.training`, plus
    resolved seeds from `cfg.run`)."""

    # demo + env (from cfg.task)
    demo_path: Path
    env_id: str
    control_mode: str
    obs_mode: str
    max_episode_steps: int
    sim_backend: str

    # DP hyperparams (from cfg.policy)
    obs_horizon: int
    act_horizon: int
    pred_horizon: int
    diffusion_step_embed_dim: int
    unet_dims: list[int]
    n_groups: int
    num_diffusion_iters: int

    # training (from cfg.training)
    total_iters: int
    batch_size: int
    lr: float
    num_demos: Optional[int]
    num_dataload_workers: int
    log_freq: int
    eval_freq: int
    save_freq: Optional[int]
    num_eval_episodes: int
    num_eval_envs: int

    # determinism
    seeds: Seeds
    device: torch.device


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


def _build_eval_envs(cfg: TrainerConfig):
    """Construct the vector of eval envs using the vendored make_eval_envs."""
    env_kwargs = dict(
        control_mode=cfg.control_mode,
        reward_mode="sparse",
        obs_mode=cfg.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=cfg.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=cfg.obs_horizon)
    wrappers = []
    if cfg.obs_mode in ("rgb", "rgbd", "rgb+depth"):
        wrappers = [FlattenRGBDObservationWrapper]
    return make_eval_envs(
        cfg.env_id,
        cfg.num_eval_envs,
        cfg.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=None,            # skip video recording in this PR
        wrappers=wrappers,
    )


def _build_policy(cfg: TrainerConfig, envs) -> DiffusionPolicy:
    """Construct DiffusionPolicy from env-derived shapes + cfg hyperparams."""
    act_dim = int(envs.single_action_space.shape[0])
    obs_state_dim = int(envs.single_observation_space["state"].shape[1])
    include_rgb = "rgb" in envs.single_observation_space.spaces
    include_depth = "depth" in envs.single_observation_space.spaces
    rgb_channels = (
        envs.single_observation_space["rgb"].shape[-1] if include_rgb else 0
    )
    depth_channels = (
        envs.single_observation_space["depth"].shape[-1] if include_depth else 0
    )
    total_visual_channels = rgb_channels + depth_channels
    if include_rgb:
        _, h, w, _ = envs.single_observation_space["rgb"].shape
    else:
        _, h, w, _ = envs.single_observation_space["depth"].shape
    return DiffusionPolicy(
        obs_horizon=cfg.obs_horizon,
        act_horizon=cfg.act_horizon,
        pred_horizon=cfg.pred_horizon,
        act_dim=act_dim,
        obs_state_dim=obs_state_dim,
        rgb_shape=(total_visual_channels, h, w),
        include_rgb=include_rgb,
        include_depth=include_depth,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        unet_dims=cfg.unet_dims,
        n_groups=cfg.n_groups,
        num_diffusion_iters=cfg.num_diffusion_iters,
        device=cfg.device,
    ).to(cfg.device)


def train(cfg: TrainerConfig, run_dir: Path, log: SummaryWriter) -> Path:
    """Run the DP training loop. Writes checkpoints to `run_dir/checkpoints/`
    and tensorboard scalars via `log`. Returns the path of the most recent
    saved checkpoint (best-by-metric if eval ran, else the last save_freq tick).

    Structure mirrors upstream train_rgbd.py:567-608. Differences:
      - No re-seeding (caller has already done seed_all).
      - Checkpoint paths come from run_dir, not 'runs/<name>'.
      - SummaryWriter is owned by caller.
    """
    # Local import: EMAModel's parent module triggers an env-dependent schema
    # error if imported at module level on this stack.
    from diffusers.training_utils import EMAModel

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    envs = _build_eval_envs(cfg)

    # Seed eval envs deterministically per-env if sim seed is provided.
    if cfg.seeds.sim is not None:
        try:
            envs.action_space.seed(cfg.seeds.sim)
            envs.observation_space.seed(cfg.seeds.sim)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Could not seed eval envs (sim_seed={cfg.seeds.sim}): {e}. "
                "Eval rollouts will not be deterministic across runs.",
                stacklevel=2,
            )

    dataset = DemoDataset(
        data_path=cfg.demo_path,
        device=cfg.device,
        control_mode=cfg.control_mode,
        env_id=cfg.env_id,
        obs_mode=cfg.obs_mode,
        num_traj=cfg.num_demos,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=cfg.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, cfg.total_iters)
    base_seed = cfg.seeds.dataloader
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_dataload_workers,
        worker_init_fn=(lambda wid: worker_init_fn(wid, base_seed=base_seed))
        if cfg.num_dataload_workers > 0 else None,
        persistent_workers=(cfg.num_dataload_workers > 0),
    )

    agent = _build_policy(cfg, envs)
    ema_agent = _build_policy(cfg, envs)

    optimizer = optim.AdamW(
        params=agent.parameters(), lr=cfg.lr,
        betas=(0.95, 0.999), weight_decay=1e-6,
    )
    lr_sched = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=cfg.total_iters,
    )
    ema = EMAModel(parameters=agent.parameters(), power=0.75)

    best_metrics: dict[str, float] = {}
    last_ckpt: Optional[Path] = None

    def _save(tag: str) -> Path:
        ema.copy_to(ema_agent.parameters())
        path = ckpt_dir / f"{tag}.pt"
        torch.save(
            {"agent": agent.state_dict(), "ema_agent": ema_agent.state_dict()},
            path,
        )
        return path

    def _eval_and_save_best(it: int) -> None:
        """Eval ema_agent and update best-* checkpoints."""
        nonlocal last_ckpt
        ema.copy_to(ema_agent.parameters())
        ema_agent.eval()
        metrics = evaluate(
            cfg.num_eval_episodes, ema_agent, envs, cfg.device, cfg.sim_backend
        )
        for k, vs in metrics.items():
            m = float(np.mean(vs))
            log.add_scalar(f"eval/{k}", m, it)
            if k in ("success_once", "success_at_end") and m > best_metrics.get(k, -1.0):
                best_metrics[k] = m
                last_ckpt = _save(f"best_eval_{k}")

    agent.train()
    for it, batch in enumerate(train_loader):
        loss = agent.compute_loss(batch["observations"], batch["actions"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_sched.step()
        ema.step(agent.parameters())

        if it % cfg.log_freq == 0:
            log.add_scalar("losses/total_loss", float(loss.item()), it)
            log.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], it)

        if cfg.eval_freq > 0 and it % cfg.eval_freq == 0 and it > 0:
            _eval_and_save_best(it)
            agent.train()  # restore training mode after eval

        if cfg.save_freq is not None and it % cfg.save_freq == 0 and it > 0:
            last_ckpt = _save(f"iter_{it}")

    # Final eval (matches upstream train_rgbd.py:607) — captures the best-eval
    # checkpoint at the final step if total_iters is not a multiple of eval_freq.
    if cfg.eval_freq > 0:
        _eval_and_save_best(cfg.total_iters)

    # Final save unconditionally — always provides a fallback checkpoint.
    last_ckpt = _save("final")
    envs.close()
    return last_ckpt
