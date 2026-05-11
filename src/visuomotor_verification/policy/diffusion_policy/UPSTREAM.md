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

`adapter.py` will wrap this baseline behind our `Policy` ABC. As of this PR,
`adapter.py` is a TODO stub — not yet implemented. Implementation is deferred
to the next PR (see foundations design spec §9).

## Runtime / invocation note

The vendored `train.py` and `train_rgbd.py` import from `diffusion_policy.utils`,
`diffusion_policy.evaluate`, etc. Upstream resolves these via the vendored
`setup.py` + `find_packages()`. Our `pyproject.toml` does **not** install the
vendored package, and the inner `diffusion_policy/diffusion_policy/` directory
has no `__init__.py`, so the upstream entry-point scripts cannot be invoked
directly from our repo as-is.

The next PR (DP adapter wiring) will handle this — either via an explicit
`sys.path` shim in the adapter, or by adding an `__init__.py` shim in our
own non-vendored sibling. Until then, do not invoke the vendored `train.py`
directly from this repo layout.
