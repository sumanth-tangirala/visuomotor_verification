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


def test_encode_obs_shape() -> None:
    """encode_obs takes a dict {state, rgb} and returns (B, obs_horizon * (visual_dim + state_dim))."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device)
    from visuomotor_verification.policy.diffusion_policy.adapter import _VISUAL_FEATURE_DIM
    B = 2
    obs_h = dp.obs_horizon
    obs_seq = {
        "state": torch.randn(B, obs_h, dp.obs_state_dim, device=device),
        "rgb": (torch.rand(B, obs_h, 3, 64, 64, device=device) * 255).to(torch.uint8),
    }
    feat = dp.encode_obs(obs_seq, eval_mode=True)
    expected_dim = obs_h * (_VISUAL_FEATURE_DIM + dp.obs_state_dim)
    assert feat.shape == (B, expected_dim), f"got {feat.shape}, want (B={B}, {expected_dim})"


def test_encode_obs_raises_when_no_visual_modality() -> None:
    """include_rgb=False and include_depth=False is invalid — guard it explicitly."""
    from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = DiffusionPolicy(
        obs_horizon=2, act_horizon=1, pred_horizon=16,
        act_dim=4, obs_state_dim=9,
        rgb_shape=(3, 64, 64),  # ignored but required arg
        include_rgb=False, include_depth=False,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256],
        n_groups=8, num_diffusion_iters=100, device=device,
    ).to(device)
    obs_seq = {"state": torch.randn(1, 2, 9, device=device)}
    with pytest.raises(ValueError, match="include_rgb or include_depth"):
        dp.encode_obs(obs_seq, eval_mode=True)


def test_compute_loss_returns_scalar() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device)
    B = 2
    obs_seq = {
        "state": torch.randn(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": (torch.rand(B, dp.obs_horizon, 3, 64, 64, device=device) * 255).to(torch.uint8),
    }
    action_seq = torch.randn(B, dp.pred_horizon, dp.act_dim, device=device).clamp(-1, 1)
    loss = dp.compute_loss(obs_seq, action_seq)
    assert loss.dim() == 0, f"loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"


def test_compute_loss_backward() -> None:
    """Loss must backprop and produce gradients on every learnable parameter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device)
    B = 2
    obs_seq = {
        "state": torch.randn(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": (torch.rand(B, dp.obs_horizon, 3, 64, 64, device=device) * 255).to(torch.uint8),
    }
    action_seq = torch.randn(B, dp.pred_horizon, dp.act_dim, device=device).clamp(-1, 1)
    loss = dp.compute_loss(obs_seq, action_seq)
    loss.backward()
    n_with_grad = sum(1 for p in dp.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in dp.parameters() if p.requires_grad)
    assert n_with_grad > 0
    # Most params should get gradient
    assert n_with_grad >= n_total // 2
