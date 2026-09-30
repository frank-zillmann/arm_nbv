"""Gym-compatible wrapper for 3D reconstruction RL environment."""

import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
from gymnasium import spaces

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)

from arm_nbv.reconstruction_policies.base import BaseReconstructionPolicy
from arm_nbv.utils.render_mesh import render_mesh


@dataclass
class TimingStats:
    """Accumulated timing statistics for steps and resets."""

    enabled: bool = True

    # Step timing
    n_steps: int = 0
    simulation_total: float = 0.0
    obs_integration_total: float = 0.0
    reconstruction_total: float = 0.0
    reward_total: float = 0.0
    obs_creation_total: float = 0.0

    # Reset timing
    n_resets: int = 0
    reset_env_total: float = 0.0
    reset_reconstruction_total: float = 0.0
    reset_mesh_total: float = 0.0
    reset_sdf_total: float = 0.0

    @contextmanager
    def time(self, name: str):
        """Context manager to accumulate timing for a named field."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        setattr(self, name, getattr(self, name) + time.perf_counter() - t0)

    def step(self) -> None:
        """Call at end of step to increment step count."""
        if self.enabled:
            self.n_steps += 1

    def summary(self) -> str:
        if not self.enabled:
            return "Timing stats collection is disabled."

        if self.n_steps == 0:
            return "No timing data collected"

        step_total = (
            self.simulation_total
            + self.obs_integration_total
            + self.reconstruction_total
            + self.reward_total
            + self.obs_creation_total
        )
        avg_step = step_total / self.n_steps * 1000  # ms

        lines = [
            f"Timing over {self.n_steps} steps ({self.n_resets} resets):",
            f"",
            f"  Step breakdown (avg {avg_step:.1f} ms):",
            f"    Simulation:     {self.simulation_total/self.n_steps*1000:6.1f} ms ({100*self.simulation_total/step_total:5.1f}%)",
            f"    Obs integration:{self.obs_integration_total/self.n_steps*1000:6.1f} ms ({100*self.obs_integration_total/step_total:5.1f}%)",
            f"    Reconstruction: {self.reconstruction_total/self.n_steps*1000:6.1f} ms ({100*self.reconstruction_total/step_total:5.1f}%)",
            f"    Reward:         {self.reward_total/self.n_steps*1000:6.1f} ms ({100*self.reward_total/step_total:5.1f}%)",
            f"    Obs creation:   {self.obs_creation_total/self.n_steps*1000:6.1f} ms ({100*self.obs_creation_total/step_total:5.1f}%)",
        ]

        if self.n_resets > 0:
            reset_total = (
                self.reset_env_total
                + self.reset_reconstruction_total
                + self.reset_mesh_total
                + self.reset_sdf_total
            )
            avg_reset = reset_total / self.n_resets * 1000
            lines.extend(
                [
                    f"",
                    f"  Reset breakdown (avg {avg_reset:.1f} ms):",
                    f"    Env reset:      {self.reset_env_total/self.n_resets*1000:6.1f} ms ({100*self.reset_env_total/reset_total:5.1f}%)",
                    f"    Recon reset:    {self.reset_reconstruction_total/self.n_resets*1000:6.1f} ms ({100*self.reset_reconstruction_total/reset_total:5.1f}%)",
                    f"    GT mesh:        {self.reset_mesh_total/self.n_resets*1000:6.1f} ms ({100*self.reset_mesh_total/reset_total:5.1f}%)",
                    f"    GT SDF:         {self.reset_sdf_total/self.n_resets*1000:6.1f} ms ({100*self.reset_sdf_total/reset_total:5.1f}%)",
                ]
            )

        return "\n".join(lines)


class Reconstruct3DGymWrapper(gym.Env):
    """
    Gym-compatible wrapper combining robosuite environment with reconstruction policy.

    The RL agent controls camera trajectories to optimize 3D reconstruction quality.

    Observation (Dict):
        - camera_pose: (7,) position (3) + quaternion wxyz (4)

    Action:
        - Uses OSC_POSE controller (Operational Space Control)
        - 6D: position delta (3) + axis-angle rotation delta (3)
        - Deltas are in robot base frame, scaled from [-1,1] to physical limits
        - Position: ±10cm, Rotation: ±0.5rad per step

    Reward: reconstruction quality (chamfer distance or SDF error)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        reconstruction_policy: BaseReconstructionPolicy,
        reconstruction_metric: str = "chamfer_distance",
        observations: list = [],
        horizon=32,
        control_freq=4,
        camera_height=128,
        camera_width=128,
        render_height=64,
        render_width=64,
        sdf_gt_size=32,
        recon_grid_size=32,
        bbox_padding=0.05,
        reward_mode="exponential",
        reward_scale=1.0,
        characteristic_error=0.01,
        action_penalty_scale=0.0,
        coverage_distance=0.01,
        coverage_n_samples=50_000,
        collect_timing=False,
        eval_log_dir: Optional[Path] = None,
    ):
        self._step_count = -1  # -1 so first reset() increments to episode 1
        self._episode_count = 0
        self.horizon = horizon

        self.collect_timing = collect_timing
        self.camera_resolution = (camera_height, camera_width)
        self.render_resolution = (render_height, render_width)
        self.timing_stats = TimingStats(enabled=collect_timing)
        self.reconstruction_metric = reconstruction_metric
        self.sdf_gt_size = sdf_gt_size
        self.recon_grid_size = recon_grid_size
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.characteristic_error = characteristic_error
        self.action_penalty_scale = action_penalty_scale
        self.coverage_distance = coverage_distance
        self.coverage_n_samples = coverage_n_samples

        # Observations to include in observation space
        self.observations = observations

        # Camera pose history buffer (for camera_pose_history observation)
        self._pose_history = np.zeros((horizon, 7), dtype=np.float32)
        self._pose_history_len = 0

        # Evaluation logging (only active in val mode with log_dir set)
        self.eval_log_dir = Path(eval_log_dir) if eval_log_dir else None
        self.eval_mode = self.eval_log_dir is not None

        # Use default Panda controller (BASIC with OSC_POSE)
        # This gives delta control in base frame: [dx, dy, dz, dax, day, daz, gripper]
        controller_config = load_composite_controller_config(
            controller=None,  # Load default for robot
            robot="Panda",
        )

        # Increase action limits for camera exploration task
        # Default: position ±0.05m, orientation ±0.5rad
        controller_config["body_parts"]["right"]["output_max"] = [
            0.05,
            0.05,
            0.05,
            0.25,
            0.25,
            0.25,
        ]
        controller_config["body_parts"]["right"]["output_min"] = [
            -0.05,
            -0.05,
            -0.05,
            -0.25,
            -0.25,
            -0.25,
        ]

        # print(f"Using controller config: {controller_config}")

        if self.eval_mode:
            self.camera_names = [
                "robot0_eye_in_hand",
                "birdview",
                "frontview",
                # "sideview",
            ] # multiple views for better saved eval visualization
        else:
            self.camera_names = ["robot0_eye_in_hand"] # for performance

        self.robot_env = robosuite.make(
            env_name="Reconstruct3D",
            robots="Panda",
            controller_configs=controller_config,
            horizon=horizon,
            control_freq=control_freq,
            camera_names=self.camera_names,
            camera_heights=camera_height,
            camera_widths=camera_width,
            sdf_size=sdf_gt_size,
            bbox_padding=bbox_padding,
        )

        # Initialize reconstruction policy
        self.reconstruction_policy = reconstruction_policy

        # Define observation space as Dict (only include configured observations)
        obs_space_dict = {}

        if "camera_pose" in self.observations:
            obs_space_dict["camera_pose"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
        if "camera_rotation_matrix" in self.observations:
            obs_space_dict["camera_rotation_matrix"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32
            )
        if "wrist_image" in self.observations:
            obs_space_dict["wrist_image"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, camera_height, camera_width),
                dtype=np.float32,
            )
        if "recon_grid" in self.observations:
            obs_space_dict["recon_grid"] = spaces.Box(
                low=0,
                high=255,
                shape=(1, recon_grid_size, recon_grid_size, recon_grid_size),
                dtype=np.uint8,
            )
        if "sdf_grid" in self.observations:
            obs_space_dict["sdf_grid"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, sdf_gt_size, sdf_gt_size, sdf_gt_size),
                dtype=np.float32,
            )
        if "camera_pose_history" in self.observations:
            obs_space_dict["camera_pose_history"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(horizon, 7),
                dtype=np.float32,
            )

        print(f"Observation space keys: {list(obs_space_dict.keys())}")
        self.observation_space = spaces.Dict(obs_space_dict)

        # Action space: use robot's native action space
        low, high = self.robot_env.action_spec
        self.action_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )
        print(
            f"Action space shape: {self.action_space.shape}, low: {self.action_space.low}, high: {self.action_space.high}"
        )

    def _get_camera_pose(self) -> np.ndarray:
        """Extract camera pose (position + quaternion) from environment."""
        extrinsic = get_camera_extrinsic_matrix(
            self.robot_env.sim, "robot0_eye_in_hand"
        )
        position = extrinsic[:3, 3]
        rotation_matrix = extrinsic[:3, :3]

        # Convert rotation matrix to quaternion (wxyz format)
        from robosuite.utils.transform_utils import mat2quat

        quat = mat2quat(rotation_matrix)

        pose_7d = np.concatenate([position, quat]).astype(np.float32)
        return pose_7d

    def _get_obs(
        self, recon_grid=None, sdf_grid=None, obs_dict=None
    ) -> Dict[str, np.ndarray]:
        """Assemble the observation dict from the configured observation keys.

        Args:
            recon_grid: (1, G, G, G) uint8 reconstruction grid for ``recon_grid``.
            sdf_grid: SDF grid for the (legacy) ``sdf_grid`` observation.
            obs_dict: Raw robosuite observation dict (for ``wrist_image``).
        """
        obs = {}

        if "camera_rotation_matrix" in self.observation_space.spaces:
            obs["camera_rotation_matrix"] = get_camera_extrinsic_matrix(
                self.robot_env.sim, "robot0_eye_in_hand"
            )[:3, :3].astype(np.float32)

        if "camera_pose" in self.observation_space.spaces:
            obs["camera_pose"] = self._get_camera_pose()

        if "camera_pose_history" in self.observation_space.spaces:
            obs["camera_pose_history"] = self._pose_history.copy()

        if "wrist_image" in self.observation_space.spaces:
            wrist_rgb = obs_dict["robot0_eye_in_hand_image"]
            # robosuite returns (H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]
            obs["wrist_image"] = (
                wrist_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        if recon_grid is not None:
            obs["recon_grid"] = recon_grid.astype(np.uint8)

        if sdf_grid is not None:
            obs["sdf_grid"] = sdf_grid.reshape(
                1, self.sdf_gt_size, self.sdf_gt_size, self.sdf_gt_size
            ).astype(np.float32)

        return obs

    def _integrate_and_reconstruct(
        self, obs_dict: dict, action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], float, Dict[str, Any]]:
        """Integrate camera obs, reconstruct, compute reward, and build gym obs.

        Shared pipeline used by both step() and reset().

        Args:
            obs_dict: Raw observation dict from robosuite (must contain camera images).
            action: Robot action (None during reset — disables action penalty).

        Returns:
            (obs, reward, reward_info_dict)
        """
        # Get depth observation and convert to real depth map
        rgb_image = obs_dict.get("robot0_eye_in_hand_image")
        depth_image = obs_dict.get("robot0_eye_in_hand_depth")
        if depth_image is not None:
            if (
                np.any(np.isnan(depth_image))
                or np.any(depth_image < 0.0)
                or np.any(depth_image > 1.0)
            ):
                n_nan = np.sum(np.isnan(depth_image))
                n_invalid = np.sum((depth_image < 0.0) | (depth_image > 1.0))
                print(
                    f"[WARNING] Depth map has {n_nan} NaN values and {n_invalid} out-of-range values at step {self._step_count}"
                )
                depth_image = np.nan_to_num(
                    depth_image, nan=1.0, posinf=1.0, neginf=0.0
                )
                depth_image = np.clip(depth_image, 0.0, 1.0)

            depth_image = get_real_depth_map(self.robot_env.sim, depth_image)

        # Get camera intrinsics and extrinsics
        intrinsic = get_camera_intrinsic_matrix(
            self.robot_env.sim,
            "robot0_eye_in_hand",
            self.robot_env.camera_heights[0],
            self.robot_env.camera_widths[0],
        )
        extrinsic = get_camera_extrinsic_matrix(
            self.robot_env.sim, "robot0_eye_in_hand"
        )

        # Record current camera pose in history buffer
        if "camera_pose_history" in self.observation_space.spaces:
            pose = self._get_camera_pose()
            if self._pose_history_len < self.horizon:
                self._pose_history[self._pose_history_len] = pose
                self._pose_history_len += 1

        # Obs integration
        with self.timing_stats.time("obs_integration_total"):
            self.reconstruction_policy.add_obs(
                camera_intrinsic=intrinsic,
                camera_extrinsic=extrinsic,
                rgb_image=rgb_image,
                depth_image=depth_image,
            )

        # Get required reconstructions
        bbox_kwargs = dict(
            bbox_center=self.robot_env.bbox_center,
            bbox_size=self.robot_env.bbox_size,
        )
        mesh_reconstruction = None
        recon_grid = None
        reward_reconstruction = None
        with self.timing_stats.time("reconstruction_total"):
            # Mesh: needed for the chamfer metric and always for eval visualization
            if self.reconstruction_metric == "chamfer_distance" or self.eval_mode:
                mesh_reconstruction = self.reconstruction_policy.reconstruct(
                    type="mesh", **bbox_kwargs
                )

            # Reconstruction passed to the reward, depending on the metric
            if self.reconstruction_metric == "chamfer_distance":
                reward_reconstruction = mesh_reconstruction
            elif self.reconstruction_metric == "voxelwise_tsdf_error":
                reward_reconstruction = self.reconstruction_policy.reconstruct(
                    type="tsdf", sdf_size=self.sdf_gt_size, **bbox_kwargs
                )
            elif self.reconstruction_metric == "point_cloud_coverage":
                reward_reconstruction = self.reconstruction_policy.reconstruct(
                    type="point_cloud"
                )
            else:
                raise ValueError(
                    f"Unknown reconstruction metric: {self.reconstruction_metric}"
                )

            # 3D grid observation derived from the reconstruction
            if "recon_grid" in self.observation_space.spaces:
                recon_grid = self.reconstruction_policy.reconstruct(
                    type="grid", grid_size=self.recon_grid_size, **bbox_kwargs
                )

        # Compute reward based on reconstruction quality
        with self.timing_stats.time("reward_total"):
            reward, reward_info_dict = self.robot_env.reward(
                action=action,
                reconstruction=reward_reconstruction,
                reconstruction_metric=self.reconstruction_metric,
                truncation_distance=getattr(
                    self.reconstruction_policy, "sdf_trunc", None
                ),
                output_info_dict=True,
                reward_mode=self.reward_mode,
                reward_scale=self.reward_scale,
                characteristic_error=self.characteristic_error,
                action_penalty_scale=self.action_penalty_scale,
                coverage_distance=self.coverage_distance,
                coverage_n_samples=self.coverage_n_samples,
            )

        # Save eval data if in eval mode
        if self.eval_mode:
            self._save_eval_data(
                reward_info_dict=reward_info_dict,
                obs_dict=obs_dict,
                mesh_reconstruction=mesh_reconstruction,
                sdf_reconstruction=None,
            )

        with self.timing_stats.time("obs_creation_total"):
            obs = self._get_obs(
                recon_grid=recon_grid,
                obs_dict=obs_dict,
            )

        return obs, reward, reward_info_dict

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""

        # Step robot environment (physics + rendering)
        with self.timing_stats.time("simulation_total"):
            obs_dict, _, done, info = self.robot_env.step(action)

        self._step_count += 1

        obs, reward, reward_info_dict = self._integrate_and_reconstruct(
            obs_dict, action
        )

        info.update(reward_info_dict)
        self.timing_stats.step()

        return obs, reward, done, self._step_count >= self.robot_env.horizon, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset robot environment (MuJoCo state) — returns initial obs with camera images
        with self.timing_stats.time("reset_env_total"):
            obs_dict = self.robot_env.reset()

        # Reset reconstruction policy (TSDF volume)
        with self.timing_stats.time("reset_reconstruction_total"):
            self.reconstruction_policy.reset()

        # Skip episode increment on spurious resets (no steps since last reset)
        if self._step_count != 0:
            self._episode_count += 1
        self._step_count = 0

        # Reset camera pose history buffer
        self._pose_history[:] = 0.0
        self._pose_history_len = 0

        # Compute ground truth mesh for reward calculation (chamfer distance)
        with self.timing_stats.time("reset_mesh_total"):
            self.robot_env.compute_gt_mesh()

        # Compute SDF ground truth if using TSDF-based metric
        with self.timing_stats.time("reset_sdf_total"):
            if self.reconstruction_metric in ("voxelwise_tsdf_error"):
                self.robot_env.compute_static_env_sdf()

        if self.timing_stats.enabled:
            self.timing_stats.n_resets += 1

        # Integrate initial camera obs and compute baseline error for cached_error
        obs, _, _ = self._integrate_and_reconstruct(obs_dict, action=None)

        return obs, {}

    def get_timing_stats(self) -> Optional[TimingStats]:
        """Get accumulated timing statistics (only if collect_timing=True)."""
        return self.timing_stats

    def render(self) -> Optional[np.ndarray]:
        """Render current camera view."""
        return self.robot_env.render()

    def _save_eval_data(
        self,
        reward_info_dict,
        obs_dict,
        mesh_reconstruction,
        sdf_reconstruction=None,
    ):
        """Save buffered evaluation data at end of episode."""
        episode_dir = self.eval_log_dir / f"episode_{self._episode_count:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save rewards CSV
        if not (episode_dir / "rewards.csv").exists():
            with open(episode_dir / "rewards.csv", "w") as f:
                f.write(",".join(reward_info_dict.keys()) + "\n")
        with open(episode_dir / "rewards.csv", "a") as f:
            f.write(f"{','.join(str(v) for v in reward_info_dict.values())}\n")

        # Save images and mesh renders
        for camera_name, obs in obs_dict.items():
            # RGB image
            if "image" in camera_name:
                plt.imsave(
                    episode_dir / f"step_{self._step_count:03d}_{camera_name}.png",
                    obs,
                )

            # Depth image
            if "depth" in camera_name:
                depth_2d = np.squeeze(obs)
                plt.imsave(
                    episode_dir / f"step_{self._step_count:03d}_{camera_name}.png",
                    depth_2d,
                    cmap="viridis",
                )

        for camera_name in self.camera_names:
            # Camera extrinsics and intrinsics (use render resolution for intrinsic)
            extrinsic = get_camera_extrinsic_matrix(self.robot_env.sim, camera_name)
            intrinsic = get_camera_intrinsic_matrix(
                self.robot_env.sim,
                camera_name,
                self.render_resolution[0],
                self.render_resolution[1],
            )

            rendered_reconstruction = render_mesh(
                mesh_reconstruction[0],
                mesh_reconstruction[1],
                extrinsic,
                intrinsic,
                resolution=self.render_resolution,
                grayscale=False,
            )

            plt.imsave(
                episode_dir
                / f"step_{self._step_count:03d}_{camera_name}_mesh_reconstruction.png",
                rendered_reconstruction,
            )

            if self._step_count == 0:
                rendered_gt = render_mesh(
                    self.robot_env.gt_vertices,
                    self.robot_env.gt_faces,
                    extrinsic,
                    intrinsic,
                    resolution=self.render_resolution,
                    grayscale=False,
                )

                plt.imsave(
                    episode_dir / f"{camera_name}_mesh_gt.png",
                    rendered_gt,
                )

        # Save SDF voxel plots if provided
        if sdf_reconstruction is not None:
            self._plot_sdf_voxels(
                sdf_reconstruction,
                episode_dir / f"step_{self._step_count:03d}_tsdf_reconstruction.png",
                self.reconstruction_policy.sdf_trunc,
            )

        if self._step_count == 0:
            if self.robot_env.sdf_grid is not None:
                self._plot_sdf_voxels(
                    self.robot_env.sdf_grid,
                    episode_dir / f"tsdf_gt.png",
                    self.reconstruction_policy.sdf_trunc,
                )

    def _plot_sdf_voxels(self, sdf_grid, save_path, threshold):
        """Plot observed voxels from SDF grid using scatter plot for speed."""
        # Identify observed voxels
        mask = np.abs(sdf_grid) < threshold

        if not np.any(mask):
            return  # Nothing to plot

        # Get voxel indices where mask is True
        x_idx, y_idx, z_idx = np.where(mask)

        # Flip x-axis to match robosuite's coordinate system
        x_idx = sdf_grid.shape[2] - 1 - x_idx

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Use scatter plot instead of voxels - much faster
        ax.scatter(x_idx, y_idx, z_idx, c="blue", marker="o", s=10, alpha=0.6)

        # Set axis limits to full grid size
        ax.set_xlim(0, sdf_grid.shape[2])
        ax.set_ylim(0, sdf_grid.shape[1])
        ax.set_zlim(0, sdf_grid.shape[0])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.title("Observed SDF Voxels with |SDF| < {:.3f} m".format(threshold))
        plt.savefig(save_path, dpi=100)
        plt.close(fig)

    def close(self):
        """Clean up resources."""
        self.robot_env.close()
