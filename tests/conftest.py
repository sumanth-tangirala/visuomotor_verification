"""Shared pytest fixtures for visuomotor_verification tests."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_cudnn_flags():
    """Snapshot torch.use_deterministic_algorithms + cuDNN flags around each test.

    `seed_all` in DETERMINISTIC mode mutates these globals. Without this fixture,
    a determinism test running first can leave cuDNN in strict mode, which then
    raises in any downstream test that runs a non-deterministic CUDA kernel
    (e.g. UNet conv backward in DP training tests).
    """
    # Snapshot
    prior_det_algos = torch.are_deterministic_algorithms_enabled()
    prior_cudnn_det = torch.backends.cudnn.deterministic
    prior_cudnn_bench = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        # Restore — use warn_only=False to match the default off state.
        torch.use_deterministic_algorithms(prior_det_algos)
        torch.backends.cudnn.deterministic = prior_cudnn_det
        torch.backends.cudnn.benchmark = prior_cudnn_bench
