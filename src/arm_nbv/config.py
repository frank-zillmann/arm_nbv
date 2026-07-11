"""Training configuration with YAML support."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Union
import yaml


@dataclass
class TrainConfig:
    """Training configuration with defaults."""

    # Reconstruction policy and metric (required - must be specified in config)
    reconstruction_policy: str = field()  # 'open3d' or 'nvblox'
    reconstruction_metric: str = field()  # 'chamfer_distance' or 'voxelwise_tsdf_error'

    # Robot Environment
    horizon: int = 32
    control_freq: int = 4
    camera_height: int = 128
    camera_width: int = 128
    render_height: int = 128
    render_width: int = 128

    # Observations to include (camera_pose, mesh_render, sdf_grid, weight_grid)
    observations: List[str] = field(default_factory=lambda: ["camera_pose"])

    # Reward settings
    sdf_gt_size: int = 32  # Size of the ground truth SDF grid along each dimension
    # Factor by which the SDF box is expanded on each side beyond the object bounds
    bbox_padding: float = 0.05

    reward_scale: float = 1.0
    characteristic_error: float = 1.0 / 32  # expected error decrease per step
    reward_mode: str = "delta"  # "exponential" or "delta"
    action_penalty_scale: float = 0.1

    # Network
    features_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # PPO and training
    total_timesteps: int = 2_000_000
    n_envs: int = 1  # Number of parallel environments, PPO default is 1
    lr: float = 3e-4  # Learning rate
    n_steps: int = 512  # Steps per env before update
    batch_size: int = 128  # Minibatch size for gradient updates, PPO default is 64
    n_epochs: int = 5  # Passes over rollout buffer per update, PPO default is 10
    gamma: float = 0.98  # Discount factor, PPO default is 0.99
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    clip_range: float = 0.2  # PPO clipping parameter
    ent_coef: float = 0.01  # Entropy bonus for exploration, PPO default is 0.0
    seed: int = 0  # Random seed for PPO (not for envs)

    checkpoint_freq: int = 100_000 # every n_envs * checkpoint_freq steps
    eval_freq: int = 100_000 # every n_envs * eval_freq steps
    n_eval_episodes: int = 10

    # Logging
    log_dir: str = "data/logs"

    # Benchmarking - collect timing stats during training (requires n_envs=1)
    benchmark: bool = False

    def save(self, path: str):
        Path(path).write_text(
            yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
        )

    @classmethod
    def load(cls, paths: Union[str, List[str]]) -> "TrainConfig":
        """Load config, merging with defaults (only override specified fields)."""
        if isinstance(paths, str):
            paths = [paths]

        data = {}
        for path in paths:
            data.update(yaml.safe_load(Path(path).read_text()) or {})
        return cls(**data)
