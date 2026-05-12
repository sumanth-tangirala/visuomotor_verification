"""Tests for DiffusionPolicy adapter (Policy ABC subclass)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from visuomotor_verification.policy.base import Policy


def _build_dp(device: torch.device = None):
    """Construct a minimal DiffusionPolicy for testing.

    Uses small dimensions to keep tests fast. The shape contract is:
      - state: (obs_state_dim,)
      - rgb: (C, H, W), with H = W = 64 to keep PlainConv cheap
      - action: (act_dim,) in [-1, 1]
    """
    from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DiffusionPolicy(
        obs_horizon=2,
        act_horizon=1,
        pred_horizon=16,
        act_dim=4,
        obs_state_dim=9,
        rgb_shape=(3, 64, 64),
        include_rgb=True,
        include_depth=False,
        diffusion_step_embed_dim=64,
        unet_dims=[64, 128, 256],
        n_groups=8,
        num_diffusion_iters=100,
        device=device,
    ).to(device)


def test_diffusion_policy_is_policy_subclass() -> None:
    dp = _build_dp()
    assert isinstance(dp, Policy)


def test_diffusion_policy_is_nn_module() -> None:
    dp = _build_dp()
    assert isinstance(dp, torch.nn.Module)


def test_diffusion_policy_has_expected_submodules() -> None:
    dp = _build_dp()
    assert hasattr(dp, "visual_encoder")
    assert hasattr(dp, "noise_pred_net")
    assert hasattr(dp, "noise_scheduler")
    # Hyperparams accessible
    assert dp.obs_horizon == 2
    assert dp.act_horizon == 1
    assert dp.pred_horizon == 16
    assert dp.act_dim == 4
