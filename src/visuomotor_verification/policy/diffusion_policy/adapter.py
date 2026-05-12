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

    # --- ABC stubs: filled in across Tasks 5-9 ----------------------------------
    def reset(self, *, seed: int | None = None) -> None:
        raise NotImplementedError("filled in Task 8")

    def act(self, obs_history: list[Observation]) -> Action:
        raise NotImplementedError("filled in Task 8")

    def load(self, ckpt_path: Path) -> None:
        raise NotImplementedError("filled in Task 9")
