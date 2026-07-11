"""Image feature extractor for Stable Baselines 3."""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImageExtractor(BaseFeaturesExtractor):
    """Feature extractor for 2D image observations.

    Concatenates one or more image observations channel-wise and processes
    them through a CNN with BatchNorm.  Each input is resized to 64x64 in
    the forward pass if needed, so camera and render resolutions can differ.

    Supports any combination of single-channel (e.g. ``mesh_render``) and
    multi-channel (e.g. ``birdview_image`` RGB) observations.  The total
    number of input channels is inferred from the observation space.

    Args:
        observation_space: Gym observation space
        features_dim: Output dimension of the feature extractor
        image_keys: Observation keys to concatenate channel-wise.
            Defaults to ``["mesh_render"]`` for backward compatibility.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        image_keys: Optional[List[str]] = None,
    ):
        super().__init__(observation_space, features_dim)
        if image_keys is None:
            image_keys = ["mesh_render"]
        self.image_keys = image_keys

        # Infer total input channels from observation space shapes
        in_channels = sum(
            observation_space[k].shape[0]  # type: ignore[index]
            for k in image_keys
        )

        ch1 = min(8 * in_channels, 16)
        ch2 = min(16 * in_channels, 32)
        ch3 = min(32 * in_channels, 64)

        # CNN: (in_channels, 64, 64) -> features
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=3, stride=2, padding=1),  # -> ch1 x 32x32
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            nn.Conv2d(ch1, ch2, kernel_size=3, stride=2, padding=1),  # -> ch2 x 16x16
            nn.BatchNorm2d(ch2),
            nn.ReLU(),
            nn.Conv2d(ch2, ch3, kernel_size=3, stride=2, padding=1),  # -> 64x8x8
            nn.BatchNorm2d(ch3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # -> 64x2x2 = 256
            nn.Flatten(),
            nn.Linear(ch3 * 2 * 2, features_dim),
            nn.ReLU(),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"[ImageExtractor] image_keys={image_keys}, in_channels={in_channels}, "
            f"features_dim={features_dim} | {n_params:,} params"
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key in self.image_keys:
            img = observations[key]
            if img.shape[-2:] != (64, 64):
                img = nn.functional.interpolate(
                    img, size=(64, 64), mode="bilinear", align_corners=False
                )
            parts.append(img)
        x = torch.cat(parts, dim=1)  # (B, C_total, 64, 64)
        return self.cnn(x)
