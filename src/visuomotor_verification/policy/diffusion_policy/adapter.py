"""DiffusionPolicy adapter: implements the Policy ABC over the vendored DP
baseline's model components.

Structure mirrors vendored train_rgbd.py:Agent (lines 246-382). The class is
re-implemented here (not subclassed) because the upstream Agent references
module-level globals (`device`) set only inside `if __name__ == "__main__":`,
making it un-importable as a library class. Our re-implementation lifts those
globals to `__init__` kwargs.

UPSTREAM:
  src/visuomotor_verification/policy/diffusion_policy/train_rgbd.py
    lines 246-288: Agent.__init__   -> our __init__
    lines 290-310: Agent.encode_obs -> our encode_obs (Task 5)
    lines 312-337: Agent.compute_loss -> our compute_loss (Task 6)
    lines 339-381: Agent.get_action -> our get_action (Task 7)

If upstream Agent changes, update this file to match.
"""
from __future__ import annotations

from pathlib import Path

from . import _vendor_import  # noqa: F401 -- must run before vendored imports

import numpy as np
import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv

from visuomotor_verification.core.types import Action, Observation
from visuomotor_verification.policy.base import Policy


# Visual feature dim is fixed in upstream (line 271). Kept here as a constant.
_VISUAL_FEATURE_DIM = 256


class DiffusionPolicy(nn.Module, Policy):
    """RGB Diffusion Policy. Inherits nn.Module (for parameters/state_dict) and
    the Policy ABC (for reset/act/load).
    """

    def __init__(
        self,
        *,
        obs_horizon: int,
        act_horizon: int,
        pred_horizon: int,
        act_dim: int,
        obs_state_dim: int,
        rgb_shape: tuple[int, int, int],   # (C, H, W); C may be 3*num_cameras
        include_rgb: bool,
        include_depth: bool,
        diffusion_step_embed_dim: int,
        unet_dims: list[int],
        n_groups: int,
        num_diffusion_iters: int,
        device: torch.device,
    ) -> None:
        nn.Module.__init__(self)

        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.obs_state_dim = obs_state_dim
        self.include_rgb = include_rgb
        self.include_depth = include_depth

        total_visual_channels = rgb_shape[0] if include_rgb else 0
        # When include_depth is True, the caller is responsible for passing the
        # combined channel count via rgb_shape[0] (mirrors upstream's behavior
        # of stacking rgb and depth as one tensor).

        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels,
            out_dim=_VISUAL_FEATURE_DIM,
            pool_feature_map=True,
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=obs_horizon * (_VISUAL_FEATURE_DIM + obs_state_dim),
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        self.num_diffusion_iters = num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # `_device` must match the device the model parameters end up on (i.e.,
        # whatever the caller passes to `.to()`). It is consumed in `reset()`
        # (Task 8) to construct a torch.Generator; a mismatch with the actual
        # parameter device produces a runtime error when the generator is used.
        # By convention: construct with `DiffusionPolicy(..., device=d).to(d)`.
        self._device = device
        self._gen: torch.Generator | None = None
        # Action chunk cache for Policy.act. List of (act_dim,) np.ndarrays.
        self._action_cache: list[np.ndarray] = []

    def encode_obs(self, obs_seq: dict[str, torch.Tensor], eval_mode: bool) -> torch.Tensor:
        """Encode (obs_horizon-length) observations to a conditioning vector.

        Args:
            obs_seq: dict with keys 'state' (B, H, S), 'rgb' (B, H, C, IH, IW) uint8,
                optionally 'depth' (B, H, C, IH, IW).
            eval_mode: if True, skip data augmentation (we don't define `aug` here;
                upstream's optional aug isn't carried over in this PR).

        Returns:
            (B, H * (visual_feature_dim + obs_state_dim)) float tensor.

        Mirrors upstream train_rgbd.py:Agent.encode_obs (lines 290-310).
        """
        img_seq = None
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0
            img_seq = depth if img_seq is None else torch.cat([img_seq, depth], dim=2)

        if img_seq is None:
            raise ValueError(
                "encode_obs requires at least one of include_rgb or include_depth to be True"
            )
        batch_size = img_seq.shape[0]
        # (B, H, C, IH, IW) -> (B*H, C, IH, IW)
        img_seq_flat = img_seq.flatten(end_dim=1)
        visual_feature = self.visual_encoder(img_seq_flat)  # (B*H, D)
        visual_feature = visual_feature.reshape(batch_size, self.obs_horizon, _VISUAL_FEATURE_DIM)
        feature = torch.cat((visual_feature, obs_seq["state"]), dim=-1)
        return feature.flatten(start_dim=1)

    def compute_loss(self, obs_seq: dict[str, torch.Tensor], action_seq: torch.Tensor) -> torch.Tensor:
        """Training-time DDPM loss. Mirrors upstream train_rgbd.py:Agent.compute_loss
        (lines 312-337).

        Uses GLOBAL torch RNG for noise and timestep sampling — training is
        stochastic by spec (see foundations §5.4, dp-adapter design §6).
        """
        B = obs_seq["state"].shape[0]
        obs_cond = self.encode_obs(obs_seq, eval_mode=False)

        noise = torch.randn(
            (B, self.pred_horizon, self.act_dim),
            device=action_seq.device,
        )
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=action_seq.device,
        ).long()
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        return torch.nn.functional.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference. Mirrors upstream train_rgbd.py:Agent.get_action (339-381)
        exactly, including the in-method channels-last → channels-first permute,
        but with `self._gen` threaded through every torch.randn and
        noise_scheduler.step so the diffusion sampler is reproducible per
        `seeds.policy` when `reset(seed=...)` was called.

        Args:
            obs_seq: dict with 'state' (B, H, S) and 'rgb' (B, H, IH, IW, C) uint8
                in CHANNELS-LAST layout (matches what env steps return).
                Optionally 'depth' in the same layout.

        Returns:
            (B, act_horizon, act_dim) float tensor.
        """
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            # Permute channels-last → channels-first to match what encode_obs +
            # PlainConv expect. Mirrors upstream lines 349-352.
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(obs_seq, eval_mode=True)

            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim),
                device=obs_seq["state"].device,
                generator=self._gen,
            )
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                    generator=self._gen,
                ).prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]

    # --- ABC stubs: filled in across Tasks 5-9 ----------------------------------
    def reset(self, *, seed: int | None = None) -> None:
        """Reset per-episode state. If `seed` is provided, build a local
        torch.Generator on `self._device` seeded with it; subsequent inference
        is reproducible per seed. If `seed` is None, fall back to global RNG.
        """
        self._action_cache.clear()
        if seed is None:
            self._gen = None
        else:
            self._gen = torch.Generator(device=self._device).manual_seed(int(seed))

    def act(self, obs_history: list[Observation]) -> Action:
        """Return a single action. Implements action chunking: each get_action
        call returns `act_horizon` actions, which we serve one-at-a-time from
        the cache. When the cache is empty we run another denoise pass.

        `obs_history` must be a list of length `obs_horizon`, each element a
        dict with `state` (np.ndarray of shape (obs_state_dim,)) and `rgb`
        (np.ndarray of shape (H, W, C), uint8 — channels-last). Padding the
        buffer for the first few steps of an episode is the caller's
        responsibility.
        """
        if not self._action_cache:
            obs_seq = self._stack_obs_for_inference(obs_history)
            with torch.no_grad():
                action_seq = self.get_action(obs_seq)  # (1, act_horizon, act_dim)
            arr = action_seq[0].cpu().numpy()
            self._action_cache = [arr[i] for i in range(arr.shape[0])]
        return self._action_cache.pop(0)

    def _stack_obs_for_inference(self, obs_history: list[Observation]) -> dict[str, torch.Tensor]:
        """Build a batched (B=1) obs_seq dict suitable for `get_action`.

        Each element of `obs_history` is a dict with `state` (np.ndarray of
        shape (obs_state_dim,)) and `rgb` (np.ndarray of shape (H, W, C),
        uint8 — channels-last, matching env step return). `get_action`
        handles the channels-last → channels-first permute internally.
        """
        if len(obs_history) != self.obs_horizon:
            raise ValueError(
                f"obs_history length {len(obs_history)} != obs_horizon {self.obs_horizon}"
            )
        states = np.stack([o["state"] for o in obs_history], axis=0)   # (H, S)
        rgbs = np.stack([o["rgb"] for o in obs_history], axis=0)       # (H, IH, IW, C)
        return {
            "state": torch.from_numpy(states).float().unsqueeze(0).to(self._device),
            "rgb": torch.from_numpy(rgbs).unsqueeze(0).to(self._device),
        }

    def load(self, ckpt_path: Path) -> None:
        """Load EMA weights from a training checkpoint.

        Checkpoints written by `trainer.train()` (and by the vendored upstream)
        are dicts with keys 'agent' and 'ema_agent'. EMA weights are the
        deployment weights for diffusion policy.

        Raises:
            KeyError: if the checkpoint dict has no 'ema_agent' key.
            RuntimeError: from `load_state_dict(strict=True)` if the checkpoint's
                architecture doesn't match the model (missing or unexpected keys,
                or shape mismatches).
        """
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=True)
        if "ema_agent" not in ckpt:
            raise KeyError(
                f"checkpoint {ckpt_path} has no 'ema_agent' key; got keys: {list(ckpt.keys())}"
            )
        self.load_state_dict(ckpt["ema_agent"])
