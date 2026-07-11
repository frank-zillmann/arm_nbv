"""Train 3D reconstruction policy with PPO."""

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from arm_nbv.robot_policies import (
    CameraPoseExtractor,
    CameraPoseHistoryExtractor,
    ImageExtractor,
    WeightGridExtractor,
    CombinedExtractor,
)
from arm_nbv.utils.env_factory import make_env_fn
from arm_nbv.utils.callbacks import TimingCallback, LoggingEvalCallback, LoggingTrainCallback
from arm_nbv.config import TrainConfig


def train(config: TrainConfig, checkpoint: Optional[str] = None):
    """Train PPO agent."""
    # Setup paths
    run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(str(log_dir / "config.yaml"))

    # Create environments (with timing collection if benchmark enabled)
    # Note: timing only works with DummyVecEnv (n_envs=1), not SubprocVecEnv
    collect_timing = config.benchmark and config.n_envs == 1
    train_env_fns = [
        make_env_fn(config, seed=i, collect_timing=collect_timing)
        for i in range(config.n_envs)
    ]
    train_env = (
        SubprocVecEnv(train_env_fns)  # type: ignore[arg-type]
        if config.n_envs > 1
        else DummyVecEnv(train_env_fns)  # type: ignore[arg-type]
    )
    
    # Eval env: always a single env to limit memory usage
    # Logs images and rewards to log_dir/eval_data/.
    eval_log_dir = log_dir / "eval_data"
    eval_env = DummyVecEnv([make_env_fn(config, seed=42, eval_log_dir=eval_log_dir)])

    # Build extractors config based on configured observations
    extractors_config = []
    if "camera_pose" in config.observations:
        extractors_config.append(
            (CameraPoseExtractor, {"features_dim": 32, "hidden_dims": [64]})
        )
    if "camera_pose_history" in config.observations:
        extractors_config.append(
            (
                CameraPoseHistoryExtractor,
                {
                    "features_dim": 64,
                    "d_model": 64,
                    "n_heads": 4,
                    "n_layers": 1,
                    "max_steps": config.horizon,
                },
            )
        )
    if "mesh_render" in config.observations or "birdview_image" in config.observations:
        image_keys = [k for k in config.observations if k in ("mesh_render", "birdview_image")]
        extractors_config.append(
            (ImageExtractor, {"features_dim": 128, "image_keys": image_keys})
        )
    if "sdf_grid" in config.observations:
        # TODO: Add SdfGridExtractor when implemented
        print(
            "Warning: sdf_grid observation enabled but no SdfGridExtractor exists yet"
        )
    if "weight_grid" in config.observations:
        extractors_config.append((WeightGridExtractor, {"features_dim": 128}))

    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": {
            "features_dim": config.features_dim,
            "extractors_config": extractors_config,
        },
        "net_arch": dict(pi=config.hidden_dims, vf=config.hidden_dims),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PyTorch/SB3] Using device: {device}")

    # Create or load model
    if checkpoint:
        print(f"Resuming from: {checkpoint}")
        model = PPO.load(
            checkpoint, env=train_env, tensorboard_log=str(log_dir), device=device
        )
    else:
        model = PPO(
            "MultiInputPolicy",  # Required for Dict observation space
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=config.lr,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=config.seed,
            device=device,
        )

    # Callbacks
    callback_list = [
        CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=str(log_dir / "checkpoints"),
            name_prefix="model",
        ),
        LoggingEvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best"),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            eval_on_start=not checkpoint,
        ),
        # LoggingTrainCallback(), # enable for debugging / disable to save some memory in TensorBoard
    ]

    # Add timing callback if benchmarking
    if collect_timing:
        callback_list.append(TimingCallback(log_dir))

    callbacks = CallbackList(callback_list)

    # Train
    print(f"Training for {config.total_timesteps} steps. Logs: {log_dir}")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not checkpoint,
    )
    print(f"Training complete.")
    model.save(str(log_dir / "final_model"))

    train_env.close()
    eval_env.close()
    print(f"Environments closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, nargs="+", default=[], help="Path(s) to config YAML"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from checkpoint"
    )
    args = parser.parse_args()

    config = TrainConfig.load(args.config)
    train(config, args.checkpoint)
