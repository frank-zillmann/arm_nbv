"""Weight grid feature extractor for Stable Baselines 3."""

from typing import Dict

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WeightGridExtractor(BaseFeaturesExtractor):
    """Feature extractor for observation weight voxel grids.

    Uses a 3D CNN to process the weight volumes from the reconstruction.
    The weight grid indicates how many observations have contributed to each voxel,
    which helps the policy understand which regions have been well-observed.

    Expects input shape:
        - 'weight_grid': (batch, 1, 32, 32, 32) - observation weights/counts

    Args:
        observation_space: Gym observation space (must contain 'weight_grid')
        features_dim: Output dimension of the feature extractor
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        grid_size: int = 32,
        max_steps: int = 32,
    ):
        super().__init__(observation_space, features_dim)
        self.grid_size = grid_size

        # 3D CNN for weight grid (1 channel)
        # Input: (batch, 1, 32, 32, 32)
        self.cnn3d = nn.Sequential(
            # Layer 1: 32^3 -> 16^3
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            # Layer 2: 16^3 -> 8^3
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # Layer 3: 8^3 -> 4^3
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),  # -> 32x2x2x2 = 256
            nn.Flatten(),
            nn.Linear(32 * 2 * 2 * 2, features_dim),
            nn.ReLU(),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"[WeightGridExtractor] grid_size={self.grid_size}, "
            f"features_dim={features_dim} | {n_params:,} params"
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        weight_grid = observations["weight_grid"]

        # Resize grid if needed (should not happen with hardcoded size)
        if weight_grid.shape[-3:] != (self.grid_size, self.grid_size, self.grid_size):
            weight_grid = nn.functional.interpolate(
                weight_grid,
                size=(self.grid_size, self.grid_size, self.grid_size),
                mode="trilinear",
                align_corners=False,
            )

        # Use ln(1+x) scaling for weights 
        weight_normalized = (
            torch.log1p(weight_grid) / torch.log1p(torch.tensor(32.0))
        )

        return self.cnn3d(weight_normalized)

