# Visuomotor Verification

Research on visuomotor verification — training a verifier that predicts
whether a visuomotor policy will succeed or fail at a task, from its
trajectories. See `CLAUDE.md` for the project guide, or
`docs/superpowers/specs/2026-05-11-foundations-design.md` for the full
foundations design.

## Quick setup

```bash
conda env create -f environment.yml
conda activate visuomotor_verification
pip install -e ".[dev]"
pytest -m "not slow"
```

Disk-heavy artifacts (datasets, checkpoints, experiments) live under
`/common/users/shared/pracsys/visuomotor_verification-data/`.

## Status

Foundations PR: repo scaffolding, ABCs for simulator/task/policy/verifier,
determinism layer with git-cleanliness gate, Hydra configs, per-phase
entry-point stubs, vendored ManiSkill diffusion-policy baseline. End-to-end
training and first verifier are subsequent PRs.
