"""Verify the _vendor_import shim makes inner DP package imports work."""
from __future__ import annotations


def test_inner_package_imports_after_shim() -> None:
    # Guard: if a previous test already cached diffusion_policy in sys.modules,
    # the imports below would short-circuit and not actually exercise the shim.
    # In that case the test is meaningless; skip it.
    import sys
    import pytest

    if "diffusion_policy" in sys.modules:
        pytest.skip(
            "diffusion_policy already in sys.modules; cannot prove the shim is "
            "what enables the imports"
        )

    # Importing the shim must enable `from diffusion_policy.X` imports.
    from visuomotor_verification.policy.diffusion_policy import _vendor_import  # noqa: F401

    from diffusion_policy.utils import load_demo_dataset, worker_init_fn  # noqa: F401
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D  # noqa: F401
    from diffusion_policy.plain_conv import PlainConv  # noqa: F401
    from diffusion_policy.make_env import make_eval_envs  # noqa: F401
    from diffusion_policy.evaluate import evaluate  # noqa: F401


def test_shim_is_idempotent() -> None:
    # Importing the shim twice must not duplicate sys.path entries.
    import sys

    from visuomotor_verification.policy.diffusion_policy import _vendor_import  # noqa: F401

    before = list(sys.path)
    # Re-importing in Python is a no-op (cached in sys.modules), but if anyone
    # manually calls the shim function it should be idempotent.
    _vendor_import.install()
    _vendor_import.install()
    matches = [p for p in sys.path if p == _vendor_import.VENDOR_DIR]
    assert len(matches) == 1, f"shim added duplicate sys.path entries: {matches}"
    assert sys.path == before, "install() mutated sys.path on idempotent calls"
