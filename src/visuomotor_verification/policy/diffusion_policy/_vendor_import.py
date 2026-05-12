"""Insert the vendored DP outer directory onto sys.path so its inner package
imports resolve as `from diffusion_policy.X import ...`.

Why: the vendored ManiSkill DP baseline uses bare `from diffusion_policy.utils
import ...` etc. — those imports assume the inner `diffusion_policy/` directory
is a top-level package. Our nested layout (`src/visuomotor_verification/policy/
diffusion_policy/diffusion_policy/`) breaks that. Inserting the outer directory
onto sys.path makes the inner directory discoverable as `diffusion_policy`.

Side effect on import: calls `install()` once.
"""
from __future__ import annotations

import sys
from pathlib import Path

VENDOR_DIR = str(Path(__file__).resolve().parent)


def install() -> None:
    """Insert VENDOR_DIR at sys.path[0] if not already present."""
    if VENDOR_DIR in sys.path:
        return
    sys.path.insert(0, VENDOR_DIR)


install()
