"""3D grid feature extractor for Stable Baselines 3."""

from typing import Dict

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Grid3DExtractor(BaseFeaturesExtractor):
    """3D-CNN feature extractor for the reconstruction grid observation.

    Processes ``recon_grid`` — a dense grid derived from the accumulated point
    cloud (a truncated distance field, stored as uint8 in [0, 255]) — giving the
    policy direct access to the current reconstruction in a 3D sense.

    Expects input shape ``(batch, 1, G, G, G)``.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        grid_key: str = "recon_grid",
    ):
        super().__init__(observation_space, features_dim)
        self.grid_key = grid_key
        self.grid_size = observation_space[grid_key].shape[-1]

        # 3D CNN: (batch, 1, G, G, G) -> features
        self.cnn3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),  # -> 32 x 2x2x2 = 256
            nn.Flatten(),
            nn.Linear(32 * 2 * 2 * 2, features_dim),
            nn.ReLU(),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"[Grid3DExtractor] grid_key={grid_key}, grid_size={self.grid_size}, "
            f"features_dim={features_dim} | {n_params:,} params"
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations[self.grid_key].float() / 255.0  # uint8 -> [0, 1]
        return self.cnn3d(grid)
