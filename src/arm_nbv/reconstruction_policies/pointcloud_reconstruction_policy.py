"""Point-cloud reconstruction policy.

The reconstruction is a point cloud accumulated by back-projecting each depth
image into world coordinates. A fine-voxel dedup rule keeps memory bounded
(discard a new point if one is already very close), so — unlike nvblox — memory
stays flat across long episodes.

Consumers:
- ``reconstruct("point_cloud")`` → (N, 3) points, used for the coverage reward.
- ``reconstruct("grid", ...)``   → uint8 (1, G, G, G) distance-field grid, the NN
  observation (point clouds aren't NN-friendly; a grid is).
- ``reconstruct("mesh", ...)``   → (vertices, faces) via marching cubes, eval-only.
"""

from typing import Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from arm_nbv.reconstruction_policies.base import BaseReconstructionPolicy


class PointCloudReconstructionPolicy(BaseReconstructionPolicy):
    def __init__(
        self,
        point_min_dist: float = 0.005,
        depth_max: float = 1.0,
        sdf_trunc: float = 0.04,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            point_min_dist: Fine-voxel dedup resolution in meters. A new point is
                discarded if a kept point already occupies its ``point_min_dist``
                voxel (the "discard if one is already very close" rule).
            depth_max: Maximum depth to back-project (meters). Farther pixels (and
                background) are ignored.
            sdf_trunc: Truncation distance (meters) for the derived distance-field
                grid observation and eval mesh level.
            device: torch device for back-projection ("cuda"/"cpu"); auto if None.
        """
        super().__init__(**kwargs)
        self.point_min_dist = point_min_dist
        self.depth_max = depth_max
        self.sdf_trunc = sdf_trunc
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[PointCloud] Using device: {self.device}, point_min_dist={point_min_dist}m")
        self.reset()

    def reset(self, **kwargs):
        """Clear the accumulated point cloud and dedup hash."""
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._seen = set()  # occupied fine-voxel keys (i, j, k)

    def add_obs(
        self,
        camera_intrinsic,
        camera_extrinsic,
        rgb_image=None,
        depth_image=None,
        **kwargs,
    ):
        """Back-project a depth image and merge new points into the cloud."""
        if depth_image is None:
            raise ValueError("depth_image is required for point-cloud reconstruction")

        depth = np.squeeze(np.asarray(depth_image, dtype=np.float32))  # (H, W)
        H, W = depth.shape

        depth_t = torch.as_tensor(depth, device=self.device)
        K = torch.as_tensor(camera_intrinsic, dtype=torch.float32, device=self.device)
        # camera-to-world (already axis-corrected by get_camera_extrinsic_matrix)
        E = torch.as_tensor(camera_extrinsic, dtype=torch.float32, device=self.device)

        # Valid pixels: positive, finite, within range
        valid = (depth_t > 0) & (depth_t < self.depth_max) & torch.isfinite(depth_t)
        vs, us = torch.nonzero(valid, as_tuple=True)  # row (v), col (u)
        if vs.numel() == 0:
            return

        z = depth_t[vs, us]
        u = us.float()
        v = vs.float()
        # Pinhole back-projection to camera coordinates
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (v - K[1, 2]) / K[1, 1] * z
        cam_pts = torch.stack([x, y, z], dim=1)  # (M, 3)
        # Camera -> world
        world_pts = cam_pts @ E[:3, :3].T + E[:3, 3]

        new = world_pts.cpu().numpy().astype(np.float32)
        self._merge_points(new)

    def _merge_points(self, pts: np.ndarray):
        """Add points whose fine-voxel cell is not yet occupied (dedup)."""
        if len(pts) == 0:
            return
        keys = np.floor(pts / self.point_min_dist).astype(np.int64)
        # Dedup within this frame first
        keys_u, idx = np.unique(keys, axis=0, return_index=True)
        mask = np.fromiter(
            (tuple(k) not in self._seen for k in keys_u), dtype=bool, count=len(keys_u)
        )
        if not mask.any():
            return
        for k in keys_u[mask]:
            self._seen.add(tuple(k))
        self._points = np.vstack([self._points, pts[idx[mask]]])

    def reconstruct(self, type="point_cloud", **kwargs):
        if type == "point_cloud":
            return self._points
        elif type == "grid":
            return self._to_grid(**kwargs)
        elif type == "mesh":
            return self._to_mesh(**kwargs)
        else:
            raise ValueError(f"Unknown reconstruction type: {type}")

    def _closeness_field(self, grid_size, bbox_center, bbox_size):
        """Occupancy -> truncated closeness field in [0, 1] (1 at surfaces)."""
        origin = np.asarray(bbox_center) - bbox_size / 2.0
        edge = bbox_size / grid_size
        field = np.zeros((grid_size,) * 3, dtype=np.float32)
        if len(self._points) == 0:
            return field, origin, edge

        idx = np.floor((self._points - origin) / edge).astype(np.int64)
        inb = np.all((idx >= 0) & (idx < grid_size), axis=1)
        idx = idx[inb]
        if len(idx) == 0:
            return field, origin, edge

        occ = np.zeros((grid_size,) * 3, dtype=bool)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        # Distance (in meters) to nearest occupied voxel, then truncated closeness
        dist = distance_transform_edt(~occ).astype(np.float32) * edge
        field = 1.0 - np.clip(dist / self.sdf_trunc, 0.0, 1.0)
        return field, origin, edge

    def _to_grid(self, grid_size, bbox_center, bbox_size, **kwargs):
        """Derive the uint8 (1, G, G, G) distance-field observation grid."""
        field, _, _ = self._closeness_field(grid_size, bbox_center, bbox_size)
        grid = (field * 255.0).astype(np.uint8)
        return grid[np.newaxis]  # (1, G, G, G)

    def _to_mesh(self, grid_size=64, bbox_center=None, bbox_size=None, **kwargs):
        """Marching-cubes mesh for eval visualization (not on the hot path)."""
        from skimage.measure import marching_cubes

        field, origin, edge = self._closeness_field(grid_size, bbox_center, bbox_size)
        if field.max() < 0.5:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
        try:
            verts, faces, _, _ = marching_cubes(field, level=0.5)
        except (ValueError, RuntimeError):
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
        verts = origin + (verts + 0.5) * edge  # voxel index -> world
        return verts.astype(np.float32), faces.astype(np.int32)
