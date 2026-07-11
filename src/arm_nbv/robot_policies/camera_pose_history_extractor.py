"""Camera pose history feature extractor using a Transformer encoder."""

from typing import Dict

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CameraPoseHistoryExtractor(BaseFeaturesExtractor):
    """Processes the episode's camera pose trajectory via a small Transformer.

    Reads ``camera_pose_history`` (horizon, 7) from observations.
    Zero-rows are treated as padding (valid poses always have non-zero quaternion).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        max_steps: int = 32,
    ):
        super().__init__(observation_space, features_dim)
        pose_dim = observation_space["camera_pose_history"].shape[-1]  # 7

        self.input_proj = nn.Linear(pose_dim, d_model)
        self.pos_embedding = nn.Embedding(max_steps, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(nn.Linear(d_model, features_dim), nn.ReLU())

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"[CameraPoseHistoryExtractor] d_model={d_model}, n_heads={n_heads}, "
            f"n_layers={n_layers}, dim_ff={d_model * 4}, max_steps={max_steps}, "
            f"features_dim={features_dim} | {n_params:,} params"
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        poses = observations["camera_pose_history"]  # (B, T, 7)
        B, T, _ = poses.shape

        # Infer padding mask from data: all-zero rows are padding
        valid = poses.abs().sum(dim=-1) > 0  # (B, T)

        x = self.input_proj(poses)  # (B, T, d_model)
        x = x + self.pos_embedding(torch.arange(T, device=x.device))
        x = self.transformer(x, src_key_padding_mask=~valid)  # (B, T, d_model)

        # Masked mean-pool over valid timesteps
        v = valid.unsqueeze(-1).float()  # (B, T, 1)
        x = (x * v).sum(dim=1) / v.sum(dim=1).clamp(min=1.0)  # (B, d_model)
        return self.output_proj(x)
