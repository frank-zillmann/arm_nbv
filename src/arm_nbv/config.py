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

    # Observations to include (camera_pose, camera_pose_history, recon_grid, wrist_image)
    observations: List[str] = field(default_factory=lambda: ["camera_pose"])

    # Reward settings
    sdf_gt_size: int = 32  # Size of the ground truth SDF grid along each dimension
    # Factor by which the SDF box is expanded on each side beyond the object bounds
    bbox_padding: float = 0.05

    reward_scale: float = 1.0
    characteristic_error: float = 1.0 / 32  # expected error decrease per step
    reward_mode: str = "delta"  # "exponential" or "delta"
    action_penalty_scale: float = 0.1

    # Point-cloud reconstruction
    point_min_dist: float = 0.005  # dedup: discard a new point within this of an existing one (m)
    depth_max: float = 1.0  # max integrated depth (m)
    coverage_distance: float = 0.01  # GT point counts as covered if a recon point is within this (m)
    coverage_n_samples: int = 50_000  # GT surface points sampled per episode for the coverage reward

    # Network
    features_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    recon_grid_size: int = 32  # Resolution of the derived 3D observation grid

    # RL algorithm and training
    algorithm: str = "sac"  # off-policy SAC (replay buffer, sample efficient)
    total_timesteps: int = 2_000_000
    n_envs: int = 1  # Number of parallel environments (SAC default is 1)
    lr: float = 3e-4  # Learning rate
    gamma: float = 0.98  # Discount factor (fits the short 32-step horizon)
    seed: int = 0  # Random seed for the RL algorithm (not for envs)

    # SAC (off-policy) hyperparameters
    buffer_size: int = 100_000  # Replay buffer size (uint8 grid obs -> ~3GB at 100k)
    learning_starts: int = 1_000  # Steps of random exploration before learning
    batch_size: int = 256  # Minibatch size for gradient updates
    tau: float = 0.005  # Target network soft-update coefficient
    train_freq: int = 1  # Env steps between gradient updates
    gradient_steps: int = 1  # Gradient steps per update
    ent_coef: str = "auto"  # Entropy temperature ("auto" = learned)

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
