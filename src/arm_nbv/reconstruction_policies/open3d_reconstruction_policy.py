from typing import Optional

import numpy as np
import open3d as o3d

from arm_nbv.reconstruction_policies.base import BaseReconstructionPolicy


def get_default_device() -> str:
    """Get the best available device (CUDA if available, else CPU)."""
    if o3d.core.cuda.is_available():
        return "CUDA:0"
    return "CPU:0"


class Open3DReconstructionPolicy(BaseReconstructionPolicy):
    """
    Generate Truncated Signed Distance Fields from depth observations using Open3D.

    Uses the tensor-based VoxelBlockGrid API which supports depth-only integration
    without dummy color images.

    Note: bbox_min/bbox_max are only used for the `get_sdf_grid()` method to define
    the query region. The VoxelBlockGrid itself dynamically allocates voxel blocks
    based on where depth observations fall.

    Note: The `get_sdf_grid` method approximates SDF by computing signed distances
    to the extracted mesh, which may differ from the internally stored TSDF values.
    """

    def __init__(
        self,
        voxel_size: float = 0.01,
        sdf_trunc: float = 0.04,
        depth_max: float = 1.0,
        device: Optional[str] = None,
    ):
        """
        Initialize TSDF generator.

        Args:
            voxel_size: Size of voxel in meters
            sdf_trunc: Truncation distance for TSDF in meters. Controls how far from
                surfaces we track signed distance. Larger values = more computation but
                less truncation.
            depth_max: Maximum depth to integrate in meters. Depth pixels farther
                than this are ignored (useful for filtering noisy far-field depth).
            device: Device to use for computation ("CPU:0" or "CUDA:0").
                If None, automatically selects CUDA if available, else CPU.
        """
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.depth_max = depth_max

        # Auto-select device if not specified
        device_str = device if device is not None else get_default_device()
        self.device = o3d.core.Device(device_str)
        print(f"[Open3D TSDF] Using device: {device_str}")

        # Block size for VoxelBlockGrid (typically 8 or 16)
        self.block_resolution = 8

        # Initialize the voxel block grid
        self._init_volume()

        super().__init__()

    def _init_volume(self):
        """Initialize or reset the voxel block grid."""
        # VoxelBlockGrid with only TSDF and weight (no color)
        self.volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight"),
            attr_dtypes=(o3d.core.float32, o3d.core.float32),
            attr_channels=((1,), (1,)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=50000,  # Initial block count, grows as needed
            device=self.device,
        )

    def add_obs(
        self,
        camera_intrinsic: np.ndarray,
        camera_extrinsic: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        depth_scale: float = 1.0,
        **kwargs,
    ):
        """
        Add an observation (depth image) to the TSDF volume.

        Args:
            camera_intrinsic: 3x3 camera intrinsic matrix
            camera_extrinsic: 4x4 camera pose (camera to world transform)
            rgb_image: RGB image (H, W, 3) - currently unused, for API compatibility
            depth_image: Depth map (H, W) in meters (if depth_scale=1.0)
            depth_scale: Scale factor to convert depth values to meters.
                Use 1.0 if depth is already in meters, 1000.0 if in millimeters.
            **kwargs: Additional arguments (ignored)
        """
        if depth_image is None:
            raise ValueError("depth_image is required for TSDF integration")
        # Convert to tensor format
        depth_t = o3d.t.geometry.Image(
            o3d.core.Tensor(depth_image.astype(np.float32), device=self.device)
        )

        # Create intrinsic tensor (3x3)
        # Note: Open3D's compute_unique_block_coordinates expects intrinsic/extrinsic on CPU
        intrinsic_t = o3d.core.Tensor(
            camera_intrinsic.astype(np.float64), device=o3d.core.Device("CPU:0")
        )

        # Create extrinsic tensor (4x4) - world to camera
        # Note: Open3D's compute_unique_block_coordinates expects extrinsic on CPU
        extrinsic_t = o3d.core.Tensor(
            np.linalg.inv(camera_extrinsic).astype(np.float64),
            device=o3d.core.Device("CPU:0"),
        )

        # Compute truncation in voxels (trunc_voxel_multiplier)
        trunc_voxel_multiplier = self.sdf_trunc / self.voxel_size

        # Get frustum block coordinates to determine which voxel blocks to update
        # This may fail if no voxels are within the depth range (camera pointing at empty space)
        try:
            frustum_block_coords = self.volume.compute_unique_block_coordinates(
                depth_t, intrinsic_t, extrinsic_t, depth_scale, self.depth_max
            )
        except RuntimeError as e:
            if "No block is touched" in str(e):
                # Camera is not seeing any geometry within depth range - skip this observation
                return
            raise

        # Integrate depth into volume (no color)
        # Note: trunc_voxel_multiplier controls TSDF truncation distance in voxel units
        self.volume.integrate(
            frustum_block_coords,
            depth_t,
            intrinsic_t,
            extrinsic_t,
            depth_scale,
            self.depth_max,
            trunc_voxel_multiplier,
        )

    def reconstruct(self, type="mesh", bbox_center=None, bbox_size=None, **kwargs):
        if type == "mesh":
            return self.extract_mesh(**kwargs)
        elif type == "point_cloud":
            return self.extract_point_cloud(**kwargs)
        else:
            raise ValueError(f"Unknown reconstruction type: {type}")

    def extract_mesh(self, weight_threshold: float = 1.0) -> tuple:
        """
        Extract triangle mesh from the TSDF volume using marching cubes.

        Args:
            weight_threshold: Minimum integration weight for a voxel to be included.
                Lower values include voxels seen fewer times (noisier but faster start).
                Default 1.0 means voxel must be seen at least once.

        Returns:
            tuple: (vertices, faces) as numpy arrays, or (empty, empty) if no surface
        """
        try:
            mesh = self.volume.extract_triangle_mesh(weight_threshold=weight_threshold)
            if mesh.vertex.positions.shape[0] == 0:
                # Empty volume - return empty arrays
                return (
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.int32),
                )
            mesh_legacy = mesh.to_legacy()
            vertices = np.asarray(mesh_legacy.vertices, dtype=np.float32)
            faces = np.asarray(mesh_legacy.triangles, dtype=np.int32)
            return (vertices, faces)
        except RuntimeError:
            # Empty volume - return empty arrays
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
            )

    def extract_point_cloud(
        self, weight_threshold: float = 0.0
    ) -> o3d.geometry.PointCloud:
        """
        Extract point cloud from the TSDF volume.

        Args:
            weight_threshold: Minimum weight for a voxel to be included.

        Returns:
            pcd: Open3D PointCloud
        """
        pcd = self.volume.extract_point_cloud(weight_threshold=weight_threshold)
        return pcd.to_legacy()

    def get_sdf_grid(
        self, grid_resolution: int, weight_threshold: float = 0.0
    ) -> np.ndarray:
        """
        Get SDF values on a regular grid within the bounding box.

        Note: This computes signed distances to the extracted mesh surface,
        which approximates the TSDF but is not truncated.

        Args:
            grid_resolution: Number of points along each axis
            weight_threshold: Minimum weight for mesh extraction (default: 0.0)

        Returns:
            sdf_grid: (grid_resolution, grid_resolution, grid_resolution) array of SDF values
        """
        # Create grid of query points
        x = np.linspace(self.bbox_min[0], self.bbox_max[0], grid_resolution)
        y = np.linspace(self.bbox_min[1], self.bbox_max[1], grid_resolution)
        z = np.linspace(self.bbox_min[2], self.bbox_max[2], grid_resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        query_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Extract mesh and compute signed distances
        vertices, faces = self.extract_mesh(weight_threshold=weight_threshold)

        if len(vertices) == 0:
            # No mesh extracted yet, return large positive values (outside)
            return np.full(
                (grid_resolution, grid_resolution, grid_resolution),
                self.sdf_trunc,
                dtype=np.float32,
            )

        # Create raycasting scene for distance queries
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)

        # Compute signed distances
        query_points_t = o3d.core.Tensor(
            query_points.astype(np.float32), dtype=o3d.core.Dtype.Float32
        )
        signed_distances = scene.compute_signed_distance(query_points_t).numpy()

        # Reshape to grid
        sdf_grid = signed_distances.reshape(
            grid_resolution, grid_resolution, grid_resolution
        )

        return sdf_grid

    def get_native_tsdf_points(
        self, weight_threshold: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get native TSDF values at all integrated voxel positions.

        Returns sparse data (only voxels that have been observed).

        **WARNING**: The returned positions are in VoxelBlockGrid's internal coordinate
        system, NOT world coordinates. The voxel_coordinates_and_flattened_indices()
        method returns coordinates relative to an internal grid origin (appears to be
        near the first integrated observation), not the world origin or bbox_min.

        For correct world-space coordinates, use extract_mesh() or get_sdf_grid().

        Args:
            weight_threshold: Minimum weight for a voxel to be returned.

        Returns:
            positions: (N, 3) array of voxel center positions (VoxelBlockGrid internal coordinates!)
            tsdf_values: (N,) array of TSDF values
            weights: (N,) array of integration weights
        """
        # Extract voxel coordinates and indices
        voxel_coords, voxel_indices = (
            self.volume.voxel_coordinates_and_flattened_indices()
        )

        if voxel_coords.shape[0] == 0:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0)

        # Get attribute tensors
        tsdf_tensor = self.volume.attribute("tsdf")
        weight_tensor = self.volume.attribute("weight")

        # Flatten and convert to numpy
        tsdf_values = tsdf_tensor.reshape(-1).cpu().numpy()
        weight_values = weight_tensor.reshape(-1).cpu().numpy()

        # Use indices to get values
        flat_indices = voxel_indices.cpu().numpy()
        voxel_tsdf = tsdf_values[flat_indices]
        voxel_weights = weight_values[flat_indices]

        # Convert voxel coordinates to positions
        voxel_coords_np = voxel_coords.cpu().numpy()

        # NOTE: VoxelBlockGrid's voxel_coordinates_and_flattened_indices() returns
        # coordinates in an internal sparse hashmap coordinate system. Multiplying
        # by voxel_size gives metric positions, but they're relative to an arbitrary
        # origin, not world coordinates.
        positions = voxel_coords_np * self.voxel_size

        # Filter by weight threshold
        mask = voxel_weights >= weight_threshold

        return positions[mask], voxel_tsdf[mask], voxel_weights[mask]

    def reset(self, **kwargs):
        """Reset the TSDF volume to empty state."""
        self._init_volume()
