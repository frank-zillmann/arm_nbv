"""Scripted policy that moves TCP along table edges while looking at table center."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.spatial.transform import Rotation


class ScriptedPolicy:
    """
    A hard-coded robot policy that moves the TCP along the edges of a table
    while the camera always points toward the table center.

    Compatible with StableBaselines3 interface (predict method).

    Default table geometry (based on robosuite Reconstruct3D environment):
        - Table surface at z ≈ 0.8
        - Object/BBox center at approximately (0, 0, 0.9)
        - TCP moves at z ≈ 1.1 (about 0.3m above table)
        - Corners at (±0.3, ±0.3) relative to center

    The policy moves along a rectangular path visiting the corners.

    Action space: Normalized [-1, 1] which the OSC_POSE controller scales to:
        - Position: ±0.1m per step
        - Orientation: ±0.5 rad per step
    """

    def __init__(
        self,
        horizon: int,
        table_center: Tuple[float, float, float] = (0.0, 0.0, 0.9),
        tcp_height: float = 1.3,
    ):
        """
        Initialize the table edge policy.

        Args:
            table_center: (x, y, z) coordinates of the point to look at
            tcp_height: Height (z) at which TCP should move
            corner_offset: Distance from center to corners (in x and y)
            position_gain: Gain for position control (scales normalized action)
            orientation_gain: Gain for orientation control (scales normalized action)
        """
        self.table_center = np.array(table_center)
        self.tcp_height = tcp_height
        self.horizon = horizon

        self._step_count = 0

        # Physical limits (for computing normalized actions)
        # OSC_POSE controller scales [-1, 1] to these limits
        self.pos_output_max = 0.05  # meters
        self.rot_output_max = 0.25  # radians

        # Define waypoints (corners at TCP height, centered on table)
        # Moving counter-clockwise when viewed from above
        cx, cy = table_center[0], table_center[1]

        self.cornerpoints = np.array(
            [
                [-0.5, 0.0, tcp_height],
                [-0.2, -0.3, tcp_height],
                [0.0, 0.0, tcp_height],
                [-0.2, +0.3, tcp_height],
            ]
        )

    def _get_current_waypoint(self, step_count: int) -> np.ndarray:
        """Get the current target waypoint via smooth linear interpolation.

        Interpolates along the 4 corners in one complete loop as step_count
        goes from 0 to horizon.
        """
        # Fraction of episode completed [0, 1]
        frac = step_count / self.horizon
        frac = np.clip(frac, 0.0, 1.0 - 1e-8)  # Avoid edge case at exactly 1.0

        n_cornerpoints = len(self.cornerpoints)
        # Map to segment (0-3) and interpolation parameter t within segment
        segment_frac = frac * n_cornerpoints  # Scale to [0, n_cornerpoints)
        segment_idx = int(segment_frac)  # Which segment (0, 1, 2, ...)
        t = segment_frac - segment_idx  # Interpolation param within segment [0, 1)

        # Get start and end corners for this segment
        corner_start = self.cornerpoints[segment_idx]
        corner_end = self.cornerpoints[
            (segment_idx + 1) % n_cornerpoints
        ]  # Wrap around to first corner

        # Linear interpolation
        return corner_start + t * (corner_end - corner_start)

    def reset(self):
        """Reset policy state."""
        self._step_count = 0

    def _compute_camera_rotation(self) -> np.ndarray:

        R_down = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )

        return R_down

    def _compute_look_at_rotation(self, current_pos: np.ndarray) -> np.ndarray:
        """
        Compute axis-angle rotation to make camera look at table center.

        The camera should look from current_pos toward table_center.
        Returns axis-angle representation for the action.
        """
        # Direction from camera to target
        direction = self.table_center - current_pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Camera looks along negative z-axis in camera frame (OpenGL convention)
        # We want to align -z_camera with the direction vector

        # Compute desired rotation matrix
        # z-axis points away from target (since camera looks along -z)
        z_axis = -direction

        # Choose up vector (world z typically, but handle edge cases)
        world_up = np.array([0.0, 0.0, 1.0])

        # x-axis = up × z (right vector)
        x_axis = np.cross(world_up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            # Looking straight up or down, use different up vector
            world_up = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(world_up, z_axis)
            x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / (x_norm + 1e-8)

        # y-axis = z × x (up vector in camera frame)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        # Desired rotation matrix (columns are x, y, z axes)
        R_desired = np.column_stack([x_axis, y_axis, z_axis])

        return R_desired

    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation."""
        eps = 1e-6

        if R.shape != (3, 3):
            raise ValueError("Input must be a 3x3 rotation matrix")
        if (R.T @ R - np.eye(3) > eps).any():
            raise ValueError("Input is not a valid rotation matrix (not orthogonal)")
        if abs(abs(np.linalg.det(R)) - 1.0) > eps:
            raise ValueError(
                f"Input is not a valid rotation matrix (determinant = {np.linalg.det(R)})"
            )

        trace = np.trace(R)
        if trace > (3.0 + eps) or trace < (-1.0 - eps):
            # Numerical issues, return zero rotation and print warning
            print("Warning: Invalid rotation matrix with trace =", trace)
            return np.zeros(3)

        if abs(trace - 3.0) < eps:
            # No rotation
            return np.zeros(3)

        if abs(trace + 1.0) < eps:
            # Handle 180 degree rotation
            # Find axis from eigenvector with eigenvalue 1
            _, V = np.linalg.eig(R)
            axis = np.real(V[:, np.argmax(np.abs(np.diag(R) + 1))])
            theta = np.pi
            return axis * theta

        # Rodrigues formula inverse
        theta = np.arccos((trace - 1.0) / 2.0)

        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
            2 * np.sin(theta)
        )

        return axis * theta

    def predict(
        self,
        observation: Dict[str, np.ndarray],
        state: Optional[Any] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Compute action to move along table edges while looking at center.

        Compatible with StableBaselines3 policy interface.

        Args:
            observation: Dict containing at least 'camera_pose'
            state: RNN hidden state (unused, for API compatibility)
            episode_start: Whether this is an episode start (resets waypoint index)
            deterministic: Whether to be deterministic (always True for scripted)

        Returns:
            action: 6D action [dx, dy, dz, dax, day, daz]
            state: None (no RNN state)
        """
        # Reset on episode start
        if episode_start is not None and (
            episode_start is True
            or (hasattr(episode_start, "__iter__") and any(episode_start))
        ):
            self.reset()

        # Get current position and orientation
        current_pos = observation["camera_pose"][:3]  # (x, y, z)
        current_R = observation["camera_rotation_matrix"]  # (3, 3)

        # Get target waypoint via smooth interpolation
        target_pos = self._get_current_waypoint(self._step_count)
        self._step_count += 1

        pos_error = target_pos - current_pos
        # Normalize to [-1, 1] action space
        pos_action = pos_error / self.pos_output_max
        pos_action = np.clip(pos_action, -1.0, 1.0)

        # Compute desired orientation
        # R_desired = self._compute_look_at_rotation(current_pos)
        R_desired = self._compute_camera_rotation()

        # Compute rotation error
        R_error = current_R.T @ R_desired

        # Convert to axis-angle
        r_error = self._rotation_matrix_to_axis_angle(R_error)
        print("Rotation error (axis-angle):", r_error)
        r_error = Rotation.from_matrix(R_error).as_rotvec()
        print("Rotation error (scipy rotvec):", r_error)

        # Apply gain and normalize to [-1, 1] action space
        rot_action = r_error / self.rot_output_max
        rot_action = np.clip(rot_action, -1.0, 1.0)

        # Combine into action (6D normalized: position + rotation)
        action = np.concatenate([pos_action, rot_action]).astype(np.float32)

        # For testing: np.array([0, 0, 0.5, 0, 0, 0]).astype(np.float64)
        action[3:] = np.zeros((3))  # Zero rotation for testing

        return action, None
