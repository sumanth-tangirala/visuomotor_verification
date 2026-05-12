# Visuomotor Verification — Project Guide

## Vision

Research on **visuomotor verification**: training a verifier that, given a
trajectory produced by a visuomotor policy, predicts whether the policy will
succeed or fail at the task. Pipeline: train policy → roll out → label
success/failure → train verifier → evaluate.

The project moves across **simulators**, **tasks**, **policy types**, and
**verifier architectures**. Code is structured so each axis is swappable via
the four ABCs in `src/visuomotor_verification/{simulator,task,policy,verifier}/base.py`,
instantiated via Hydra `_target_`.

## Paths

- **Shared data root:** `/common/users/shared/pracsys/visuomotor_verification-data/`
  - All disk-space-heavy artifacts go here: demonstration datasets, rollout
    trajectories, policy checkpoints, verifier experiments, and per-run Hydra
    outputs (`.hydra/`, `metadata.json`, `checkpoints/`, `logs/`).
  - The repo itself contains only code, configs, docs, and tests. Never write
    large artifacts under the repo or under `~/` — the home filesystem is
    quota-limited.
- **Layout under the shared root:**
  - `datasets/policy_demos/<task>/` — demonstration datasets used to train
    policies (e.g., ManiSkill-downloaded + replayed `.h5` files). Shared across
    runs; not run-id scoped.
  - `datasets/<task>/rollouts/<run_id>/` — rollouts of trained policies,
    consumed by verifier training. Run-id scoped.
  - `policies/<task>/<policy>/<run_id>/` — policy training runs (one run dir
    per training invocation).
  - `experiments/verifier/<task>/<verifier>/<run_id>/` — verifier training and
    evaluation runs.
- **Pre-existing artifacts** that need to land under the shared root (e.g.
  ManiSkill demos downloaded into `~/.maniskill/`) should be moved into
  `datasets/policy_demos/<task>/` after retrieval.

## Determinism principle

Every phase entry-point script (`scripts/*.py`) calls `prologue(cfg, script_name=...)`
from `scripts/_common.py`, which runs the standard sequence:
`RunConfig.from_hydra(cfg)` → `git_info.collect(REPO_ROOT)` → `seed_all(run_cfg, repo_root=REPO_ROOT, git_info_cache=info)` → write `metadata.json`.
**Nothing else in our project-owned code touches global RNGs or cuDNN flags.** The vendored ManiSkill diffusion-policy baseline at `src/visuomotor_verification/policy/diffusion_policy/` does call `random.seed`/`np.random.seed`/`torch.manual_seed`/cuDNN flags inside its own `train.py`/`utils.py`, but the adapter at `policy/diffusion_policy/adapter.py` is not yet implemented — those vendored RNG calls are dead code from our perspective until the adapter PR lands. The adapter PR will route the vendored seeding through our `seed_all`.
Components that have their own RNG receive a component seed via constructor
argument and create a local `torch.Generator` / `np.random.Generator`.

`RunMode` is one of `deterministic` / `mixed` / `stochastic`. In
`deterministic` mode, all unset per-component seeds derive from
`seeds.master` via blake2b — so a single master seed reproduces a whole run.

## Git cleanliness gate

`RunMode.DETERMINISTIC` refuses to run with a dirty tree unless
`run.allow_dirty=true` is set explicitly. Do **not** paper over this with a
blanket override — commit or stash instead. The diff is always recorded in
`metadata.json` regardless of mode.

## Run IDs and metadata

- **Run ID format:** `<experiment_name>-<YYYY-MM-DD>_<HH-MM-SS>`
- **`experiment_name`** is a mandatory top-level Hydra field for every script.
- **`metadata.json`** lives in every run directory (alongside `.hydra/`) and
  records: run_id, experiment_name, script, cmdline, git_sha, git_dirty,
  git_diff (if dirty), git_untracked (if dirty), timestamp, run_config,
  resolved_config.

## Per-phase entry points

| Script | Reads | Writes |
|---|---|---|
| `scripts/train_policy.py` | `datasets/policy_demos/<task>/` | `policies/<task>/<policy>/<run_id>` |
| `scripts/collect_trajectories.py` | policy checkpoint | `datasets/<task>/rollouts/<run_id>` |
| `scripts/train_verifier.py` | rollouts | `experiments/verifier/<task>/<verifier>/<run_id>` |
| `scripts/evaluate_verifier.py` | verifier ckpt + held-out rollouts | eval subdir |

Inter-phase artifact paths are passed via Hydra overrides; **no implicit "latest" magic**.

## Conda env

```bash
conda env create -f environment.yml
conda activate visuomotor_verification
pip install -e ".[dev]"
```

Env name: `visuomotor_verification`. Python 3.11, PyTorch 2.4 with CUDA 12.1
build (driver is 12.2).

## Status notes

- The `Verifier` ABC at `src/visuomotor_verification/verifier/base.py` is
  **provisional** — its shape will be revised when the first concrete
  verifier exists.
- The diffusion-policy baseline under
  `src/visuomotor_verification/policy/diffusion_policy/` is a **pristine
  vendored copy** from upstream ManiSkill. See
  `src/visuomotor_verification/policy/diffusion_policy/UPSTREAM.md`. The `adapter.py` wiring it to the `Policy` ABC is a TODO stub.

## Spec & plan

- Foundations design spec: `docs/superpowers/specs/2026-05-11-foundations-design.md`
- Foundations implementation plan: `docs/superpowers/plans/2026-05-11-foundations.md`
