"""Combined feature extractor that chains multiple extractors."""

from typing import Dict, List, Type

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor that combines multiple sub-extractors.

    Each sub-extractor processes its own subset of the observation space,
    and their outputs are concatenated and passed through a final combining layer.

    Args:
        observation_space: Full Gym observation space (Dict)
        features_dim: Final output dimension after combining all features
        extractors_config: List of tuples (ExtractorClass, kwargs_dict) where
            kwargs_dict contains arguments for that extractor (excluding observation_space)
            Each extractor's features_dim will be set automatically based on the config.

    Example:
        extractors_config = [
            (CameraPoseExtractor, {"features_dim": 32, "hidden_dims": [64, 64]}),
            (MeshRenderingExtractor, {"features_dim": 128}),
            (SDFWeightExtractor, {"features_dim": 128, "grid_size": 32}),
        ]
        extractor = CombinedExtractor(obs_space, features_dim=256, extractors_config=extractors_config)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        extractors_config: List[tuple],
    ):
        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleList()
        total_features = 0

        for extractor_class, kwargs in extractors_config:
            extractor = extractor_class(observation_space, **kwargs)
            self.extractors.append(extractor)
            total_features += kwargs["features_dim"]

        # Combining layer: concatenated features -> final features_dim
        self.combine = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.ReLU(),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"[CombinedExtractor] {len(extractors_config)} sub-extractors, "
            f"total_features={total_features} -> features_dim={features_dim} | "
            f"{n_params:,} total params"
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract features from each sub-extractor
        features_list = []
        for extractor in self.extractors:
            features = extractor(observations)
            features_list.append(features)

        # Concatenate all features
        combined = torch.cat(features_list, dim=1)

        # Apply combining layer
        return self.combine(combined)
