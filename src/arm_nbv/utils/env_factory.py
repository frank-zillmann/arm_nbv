"""Factory functions for creating reconstruction environments."""

from pathlib import Path
from typing import Optional

from stable_baselines3.common.monitor import Monitor

from arm_nbv.reconstruct3d_gym_wrapper import Reconstruct3DGymWrapper
from arm_nbv.config import TrainConfig


def create_reconstruction_policy(policy_name: str):
    """Create a reconstruction policy by name.
    
    Args:
        policy_name: 'open3d' or 'nvblox'
        
    Returns:
        Reconstruction policy instance with default parameters
    """
    if policy_name == "open3d":
        from arm_nbv.reconstruction_policies.open3d_reconstruction_policy import (
            Open3DReconstructionPolicy,
        )
        return Open3DReconstructionPolicy()
        
    elif policy_name == "nvblox":
        from arm_nbv.reconstruction_policies.nvblox_reconstruction_policy import (
            NvbloxReconstructionPolicy,
        )
        return NvbloxReconstructionPolicy()
        
    else:
        raise ValueError(f"Unknown reconstruction policy: {policy_name}")


def create_env(
    config: TrainConfig,
    seed: int = 0,
    collect_timing: bool = False,
    eval_log_dir: Optional[Path] = None,
    wrap_monitor: bool = True,
):
    """Create a Reconstruct3DGymWrapper environment from config.
    
    Args:
        config: Training configuration
        seed: Random seed for environment
        collect_timing: Whether to collect timing statistics
        eval_log_dir: Directory for evaluation logging (enables eval mode)
        wrap_monitor: Whether to wrap with SB3 Monitor
        
    Returns:
        Environment instance (optionally wrapped with Monitor)
    """
    reconstruction_policy = create_reconstruction_policy(config.reconstruction_policy)
    
    env = Reconstruct3DGymWrapper(
        reconstruction_policy=reconstruction_policy,
        reconstruction_metric=config.reconstruction_metric,
        observations=config.observations,
        horizon=config.horizon,
        control_freq=config.control_freq,
        camera_height=config.camera_height,
        camera_width=config.camera_width,
        render_height=config.render_height,
        render_width=config.render_width,
        collect_timing=collect_timing,
        sdf_gt_size=config.sdf_gt_size,
        bbox_padding=config.bbox_padding,
        reward_scale=config.reward_scale,
        characteristic_error=config.characteristic_error,
        reward_mode=config.reward_mode,
        action_penalty_scale=config.action_penalty_scale,
        eval_log_dir=eval_log_dir,
    )
    
    if wrap_monitor:
        env = Monitor(env)
        
    env.reset(seed=seed)
    return env


def make_env_fn(
    config: TrainConfig,
    seed: int,
    collect_timing: bool = False,
    eval_log_dir: Optional[Path] = None,
):
    """Create a callable that returns an environment (for vectorized envs).
    
    Args:
        config: Training configuration
        seed: Random seed for environment
        collect_timing: Whether to collect timing statistics
        eval_log_dir: Directory for evaluation logging
        
    Returns:
        Callable that creates and returns an environment
    """
    def _init():
        return create_env(config, seed, collect_timing, eval_log_dir)
    return _init
