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


@pytest.mark.slow
def test_diffusion_policy_is_policy_subclass() -> None:
    dp = _build_dp()
    assert isinstance(dp, Policy)


@pytest.mark.slow
def test_diffusion_policy_is_nn_module() -> None:
    dp = _build_dp()
    assert isinstance(dp, torch.nn.Module)


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_compute_loss_backward() -> None:
    """Loss must backprop and produce gradients on every learnable parameter
    (both visual_encoder and noise_pred_net).
    """
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

    # Every learnable parameter must have a gradient.
    missing = [n for n, p in dp.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params without grad: {missing[:5]}{'...' if len(missing) > 5 else ''}"

    # And both submodules (visual encoder + UNet) must receive non-zero gradients
    # (regression sentinel against accidental detachment).
    enc_grad = sum(
        p.grad.abs().sum().item()
        for p in dp.visual_encoder.parameters()
        if p.grad is not None
    )
    unet_grad = sum(
        p.grad.abs().sum().item()
        for p in dp.noise_pred_net.parameters()
        if p.grad is not None
    )
    assert enc_grad > 0, f"visual_encoder received zero gradient (sum={enc_grad})"
    assert unet_grad > 0, f"noise_pred_net received zero gradient (sum={unet_grad})"


@pytest.mark.slow
def test_get_action_no_generator_uses_global_rng() -> None:
    """When _gen is None, two consecutive get_action calls should differ
    (global RNG advances)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device).eval()
    B = 1
    obs_seq = {
        "state": torch.zeros(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": torch.zeros(B, dp.obs_horizon, 3, 64, 64, device=device, dtype=torch.uint8),
    }
    torch.manual_seed(0)
    a1 = dp.get_action(obs_seq).cpu()
    a2 = dp.get_action(obs_seq).cpu()
    assert not torch.allclose(a1, a2), "global RNG should advance between calls"


@pytest.mark.slow
def test_get_action_with_generator_is_reproducible() -> None:
    """Two get_action calls preceded by the same generator state must match."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device).eval()
    B = 1
    obs_seq = {
        "state": torch.zeros(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": torch.zeros(B, dp.obs_horizon, 3, 64, 64, device=device, dtype=torch.uint8),
    }
    dp._gen = torch.Generator(device=device).manual_seed(42)
    a1 = dp.get_action(obs_seq).cpu()
    dp._gen = torch.Generator(device=device).manual_seed(42)
    a2 = dp.get_action(obs_seq).cpu()
    assert torch.allclose(a1, a2), f"same seed should produce same action; diff={(a1 - a2).abs().max()}"


@pytest.mark.slow
def test_get_action_shape() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = _build_dp(device).to(device).eval()
    B = 3
    obs_seq = {
        "state": torch.zeros(B, dp.obs_horizon, dp.obs_state_dim, device=device),
        "rgb": torch.zeros(B, dp.obs_horizon, 3, 64, 64, device=device, dtype=torch.uint8),
    }
    a = dp.get_action(obs_seq)
    assert a.shape == (B, dp.act_horizon, dp.act_dim), a.shape


def _fake_obs_history(dp) -> list:
    """Construct an obs_history of length obs_horizon, each obs being a
    dict with numpy state and rgb tensors."""
    out = []
    for _ in range(dp.obs_horizon):
        out.append(
            {
                "state": np.zeros((dp.obs_state_dim,), dtype=np.float32),
                "rgb": np.zeros((3, 64, 64), dtype=np.uint8),
            }
        )
    return out


@pytest.mark.slow
def test_reset_with_none_clears_generator() -> None:
    dp = _build_dp()
    dp._gen = torch.Generator(device=dp._device).manual_seed(1)
    dp._action_cache.append(np.zeros(dp.act_dim, dtype=np.float32))
    dp.reset(seed=None)
    assert dp._gen is None
    assert dp._action_cache == []


@pytest.mark.slow
def test_reset_with_seed_builds_generator() -> None:
    dp = _build_dp()
    dp.reset(seed=42)
    assert dp._gen is not None
    assert isinstance(dp._gen, torch.Generator)


@pytest.mark.slow
def test_act_returns_single_action_array() -> None:
    dp = _build_dp().eval()
    dp.reset(seed=0)
    history = _fake_obs_history(dp)
    a = dp.act(history)
    assert isinstance(a, np.ndarray)
    assert a.shape == (dp.act_dim,)


@pytest.mark.slow
def test_act_seeds_reproduce_first_action() -> None:
    dp = _build_dp().eval()
    history = _fake_obs_history(dp)

    dp.reset(seed=7)
    a1 = dp.act(history)
    dp.reset(seed=7)
    a2 = dp.act(history)
    assert np.allclose(a1, a2), f"seed 7 should reproduce; diff={np.abs(a1 - a2).max()}"


@pytest.mark.slow
def test_act_chunking_serves_from_cache_then_requeries() -> None:
    """With act_horizon=2, first act() triggers a denoise pass (fills 2 actions),
    second act() reads from cache, third act() triggers another denoise pass."""
    from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = DiffusionPolicy(
        obs_horizon=2, act_horizon=2, pred_horizon=16,
        act_dim=4, obs_state_dim=9, rgb_shape=(3, 64, 64),
        include_rgb=True, include_depth=False,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256],
        n_groups=8, num_diffusion_iters=100, device=device,
    ).to(device).eval()
    dp.reset(seed=11)

    history = _fake_obs_history(dp)
    # Patch get_action to count calls.
    calls = []
    orig = dp.get_action
    def counting(obs_seq):
        calls.append(1)
        return orig(obs_seq)
    dp.get_action = counting

    a1 = dp.act(history)
    a2 = dp.act(history)
    a3 = dp.act(history)

    # 2 denoise passes total (one fills 2 actions, the third triggers a refill).
    assert len(calls) == 2, f"expected 2 get_action calls, got {len(calls)}"
    assert a1.shape == (dp.act_dim,) and a2.shape == (dp.act_dim,) and a3.shape == (dp.act_dim,)


@pytest.mark.slow
def test_load_restores_ema_state_dict(tmp_path) -> None:
    """Save a fake checkpoint with distinct agent vs. ema_agent weights, then
    load() and verify the EMA weights are what's restored."""
    dp = _build_dp()
    # Construct a fake "ema" state dict: zero out all params, save under 'ema_agent'.
    agent_sd = {k: v.clone() for k, v in dp.state_dict().items()}
    ema_sd = {k: torch.zeros_like(v) for k, v in dp.state_dict().items()}
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"agent": agent_sd, "ema_agent": ema_sd}, ckpt_path)

    # Mutate weights so load has something to overwrite.
    with torch.no_grad():
        for p in dp.parameters():
            p.add_(1.0)
    dp.load(ckpt_path)
    # All learnable params should now be ~0 (from ema_sd).
    for p in dp.parameters():
        assert torch.allclose(p, torch.zeros_like(p)), "ema weights not loaded"


@pytest.mark.slow
def test_load_missing_ema_key_raises(tmp_path) -> None:
    dp = _build_dp()
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"agent": dp.state_dict()}, ckpt_path)  # missing 'ema_agent'
    with pytest.raises(KeyError, match="ema_agent"):
        dp.load(ckpt_path)
