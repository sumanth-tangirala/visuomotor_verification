"""Equivalence sentinel: our DiffusionPolicy.get_action must match the vendored
Agent.get_action byte-for-byte under shared global RNG and identical weights.
Drift here means our re-implementation has diverged from upstream."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _maybe_import_vendored_agent():
    """Import the vendored train_rgbd module and monkey-patch the `device`
    global. Returns the train_rgbd module + device, or skips if the
    monkey-patch fails (e.g., upstream refactored)."""
    from visuomotor_verification.policy.diffusion_policy import _vendor_import  # noqa: F401
    try:
        import train_rgbd
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"upstream train_rgbd not importable: {e}")
    if not hasattr(train_rgbd, "Agent"):
        pytest.skip("upstream train_rgbd has no Agent class (refactored?)")
    # Inject the `device` global so Agent's compute_loss/get_action don't NameError.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rgbd.device = device
    return train_rgbd, device


def _fake_env(act_dim: int, obs_state_dim: int, rgb_shape: tuple[int, int, int, int]):
    """Build a fake env object exposing the shape probes Agent.__init__ uses.

    Agent reads:
      env.single_observation_space["state"].shape
      env.single_observation_space["rgb"].shape   (or "depth")
      env.single_observation_space.keys()  (for `"rgb" in keys()` check)
      env.single_action_space.shape / high / low

    rgb_shape is (cameras*3, H, W, C_per_camera) — Agent reads shape[-1] for
    channel count.
    """
    class _ObsSpace(dict):
        """dict subclass — supports both `["state"]` indexing and `.keys()`."""
        pass

    obs_space = _ObsSpace(
        {
            "state": SimpleNamespace(shape=(1, obs_state_dim)),
            "rgb": SimpleNamespace(shape=rgb_shape),
        }
    )
    action_space = SimpleNamespace(
        shape=(act_dim,), high=np.ones(act_dim), low=-np.ones(act_dim),
    )
    return SimpleNamespace(
        single_observation_space=obs_space,
        single_action_space=action_space,
    )


@pytest.mark.slow
def test_get_action_matches_vendored_agent_under_shared_seed():
    train_rgbd, device = _maybe_import_vendored_agent()

    # Match _build_dp() params from test_dp_adapter.py
    obs_horizon = 2
    act_horizon = 1
    pred_horizon = 16
    act_dim = 4
    obs_state_dim = 9
    H = W = 64
    rgb_channels = 3
    rgb_shape = (obs_horizon, H, W, rgb_channels)  # upstream reads [-1] for channels

    args = SimpleNamespace(
        obs_horizon=obs_horizon, act_horizon=act_horizon, pred_horizon=pred_horizon,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256], n_groups=8,
    )
    env = _fake_env(act_dim, obs_state_dim, rgb_shape)

    # Construct both models.
    torch.manual_seed(0)
    upstream = train_rgbd.Agent(env, args).to(device)

    from visuomotor_verification.policy.diffusion_policy.adapter import DiffusionPolicy
    ours = DiffusionPolicy(
        obs_horizon=obs_horizon, act_horizon=act_horizon, pred_horizon=pred_horizon,
        act_dim=act_dim, obs_state_dim=obs_state_dim,
        rgb_shape=(rgb_channels, H, W),
        include_rgb=True, include_depth=False,
        diffusion_step_embed_dim=64, unet_dims=[64, 128, 256],
        n_groups=8, num_diffusion_iters=100, device=device,
    ).to(device)

    # Copy upstream weights into ours so the only difference is the get_action body.
    # Layout matches because both classes build the same submodules in the same order.
    ours.load_state_dict(upstream.state_dict())

    # Both upstream and our get_action expect channels-last RGB (env step layout)
    # and permute internally. Pass identical inputs.
    B = 1
    rgb = torch.zeros(B, obs_horizon, H, W, rgb_channels, dtype=torch.uint8, device=device)
    state = torch.zeros(B, obs_horizon, obs_state_dim, device=device)
    obs_upstream = {"state": state.clone(), "rgb": rgb.clone()}
    obs_ours = {"state": state.clone(), "rgb": rgb.clone()}

    torch.manual_seed(123)
    a_upstream = upstream.get_action(obs_upstream).detach()
    torch.manual_seed(123)
    a_ours = ours.get_action(obs_ours).detach()

    assert a_upstream.shape == a_ours.shape, f"{a_upstream.shape} vs {a_ours.shape}"
    diff = (a_upstream - a_ours).abs().max().item()
    assert diff < 1e-5, f"max abs diff = {diff}; our get_action drifted from upstream"
