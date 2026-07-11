"""Camera pose feature extractor for Stable Baselines 3."""

from typing import Dict, List

import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import utils as gym_utils
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CameraPoseExtractor(BaseFeaturesExtractor):
    """Feature extractor for camera_pose only.

    Processes the 7D camera pose (position + quaternion) through an MLP.

    Args:
        observation_space: Gym observation space (must contain 'camera_pose')
        features_dim: Output dimension of the feature extractor
        hidden_dims: List of hidden layer dimensions for the MLP
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        hidden_dims: List[int] = [64],
    ):
        super().__init__(observation_space, features_dim)
        input_dim = gym_utils.flatdim(observation_space["camera_pose"])

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.extend([nn.Linear(prev_dim, features_dim), nn.ReLU()])

        self.net = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"[CameraPoseExtractor] input_dim={input_dim}, hidden_dims={hidden_dims}, "
            f"features_dim={features_dim} | {n_params:,} params"
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(observations["camera_pose"])
