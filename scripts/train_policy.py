"""Train a Diffusion Policy on push-T.

Usage:
    python scripts/train_policy.py experiment_name=push_t_dp_v1

One-time demo prep is documented in
docs/superpowers/specs/2026-05-11-dp-adapter-design.md §1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import prologue

from visuomotor_verification.core.determinism import RunConfig, resolve_seeds
from visuomotor_verification.policy.diffusion_policy.trainer import (
    TrainerConfig,
    train,
)


@hydra.main(version_base=None, config_path="../configs", config_name="train_policy")
def main(cfg: DictConfig) -> None:
    run_dir = prologue(cfg, script_name="train_policy.py")

    run_cfg = RunConfig.from_hydra(cfg)
    resolved = resolve_seeds(run_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer_cfg = TrainerConfig(
        demo_path=Path(cfg.task.demo_path),
        env_id=cfg.task.env_id,
        control_mode=cfg.task.control_mode,
        obs_mode=cfg.task.obs_mode,
        max_episode_steps=cfg.task.max_episode_steps,
        sim_backend=cfg.task.sim_backend,
        obs_horizon=cfg.policy.obs_horizon,
        act_horizon=cfg.policy.act_horizon,
        pred_horizon=cfg.policy.pred_horizon,
        diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
        unet_dims=list(cfg.policy.unet_dims),
        n_groups=cfg.policy.n_groups,
        num_diffusion_iters=cfg.policy.num_diffusion_iters,
        total_iters=cfg.training.total_iters,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        num_demos=cfg.training.num_demos,
        num_dataload_workers=cfg.training.num_dataload_workers,
        log_freq=cfg.training.log_freq,
        eval_freq=cfg.training.eval_freq,
        save_freq=cfg.training.save_freq,
        num_eval_episodes=cfg.training.num_eval_episodes,
        num_eval_envs=cfg.training.num_eval_envs,
        seeds=resolved,
        device=device,
    )

    if not trainer_cfg.demo_path.exists():
        raise FileNotFoundError(
            f"Demo file not found: {trainer_cfg.demo_path}. "
            "Run the demo-prep recipe in "
            "docs/superpowers/specs/2026-05-11-dp-adapter-design.md §1."
        )

    log = SummaryWriter(str(run_dir / "logs"))
    if cfg.training.wandb.enabled:
        import wandb
        wandb.init(
            project=cfg.training.wandb.project,
            entity=cfg.training.wandb.entity,
            name=run_dir.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
        )

    wandb_active = bool(cfg.training.wandb.enabled)
    try:
        last_ckpt = train(trainer_cfg, run_dir, log)
    finally:
        log.close()
        if wandb_active:
            import wandb
            wandb.finish()
    print(f"[train_policy] last checkpoint: {last_ckpt}")

    # Annotate metadata.json with the final-checkpoint path. The "best_eval_*"
    # checkpoints (if eval ran) live alongside it in `checkpoints/` as separate
    # files; consumers should glob the directory if they want all artifacts.
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text())
        payload.setdefault("output_artifacts", {})["last_checkpoint"] = str(last_ckpt)
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
