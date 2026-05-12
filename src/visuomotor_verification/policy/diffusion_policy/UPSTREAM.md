# Vendored from ManiSkill

This directory is a **pristine vendored copy** of ManiSkill's diffusion-policy
baseline. Do not modify these files inline; if changes are needed, do them in
`adapter.py` or a sibling module so the diff against upstream stays auditable.

- **Source:** https://github.com/haosulab/ManiSkill
- **Path in source:** `examples/baselines/diffusion_policy/`
- **Commit hash:** a4a4f9272ad64b1564035874b605ceb687b63ed8
- **Vendored on:** 2026-05-11

## Re-syncing with upstream

To compare against newer upstream:

```bash
git clone --depth 1 https://github.com/haosulab/ManiSkill.git /tmp/ms-new
diff -r /tmp/ms-new/examples/baselines/diffusion_policy/ \
        src/visuomotor_verification/policy/diffusion_policy/ \
        | grep -v UPSTREAM.md | grep -v adapter.py | grep -v __init__.py
```

## Adapter status

`adapter.py` wraps the vendored DP model components behind our `Policy` ABC.
It re-implements (not subclasses) the upstream `Agent` class because the
upstream class references module-level globals (`device`) set only inside
`if __name__ == "__main__":`. Our re-implementation lifts those globals to
`__init__` kwargs. Upstream line ranges are recorded in the docstring of
each method (`__init__`, `encode_obs`, `compute_loss`, `get_action`).

The diffusion-policy sampler in `get_action` threads a `torch.Generator`
through every `torch.randn` and `noise_scheduler.step` so the inference
RNG is reproducible per `seeds.policy` independently of the global torch
RNG. See the foundations design spec §5.2 and the dp-adapter design §6.

## Runtime / invocation note

The vendored `train.py` and `train_rgbd.py` import from `diffusion_policy.utils`,
`diffusion_policy.evaluate`, etc. — bare top-level package imports that
upstream resolves via `setup.py` + `find_packages()`. Our `pyproject.toml`
does not install the vendored package, and the inner
`diffusion_policy/diffusion_policy/` directory has no `__init__.py`.

**Resolved:** `_vendor_import.py` (sibling of this file) inserts the outer
vendored directory onto `sys.path` once at import time. `adapter.py` and
`trainer.py` both `from . import _vendor_import` as their first line so the
shim is active before any vendored import resolves. After the shim runs,
`from diffusion_policy.utils import ...` works.

Note that the outer vendored scripts (`train.py`, `train_rgbd.py`) define
`Agent` and `SmallDemoDataset_DiffusionPolicy` referencing module-level
globals (`device`, `args.control_mode`) set only inside their `__main__`
block. Importing those classes and calling their methods raises `NameError`.
We re-implement those wrappers in `adapter.py` and `trainer.py` with the
globals lifted to `__init__` kwargs; see the upstream-line-range comments in
each class.

Additionally, on some environments `from diffusers.training_utils import EMAModel`
triggers a flash-attn / torch op-schema cascade at module import. `trainer.py`
defers this import into the `train()` function body to avoid the failure at
module-import time.
