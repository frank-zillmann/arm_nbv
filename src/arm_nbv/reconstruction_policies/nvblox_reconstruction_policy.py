import os

# Suppress nvblox C++ logging (GLOG)
# 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
# Must be set before importing nvblox_torch or any library using glog
os.environ["GLOG_minloglevel"] = "1"

import numpy as np
import torch
from arm_nbv.reconstruction_policies.base import BaseReconstructionPolicy
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.mapper_params import MapperParams
from nvblox_torch.sensor import Sensor


class NvbloxReconstructionPolicy(BaseReconstructionPolicy):
    """
    Simple reconstruction policy using nvblox_torch for TSDF and mesh extraction.
    """

    def __init__(self, voxel_size=0.01, sdf_trunc=0.04, depth_max=1.0, **kwargs):
        super().__init__(**kwargs)
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.depth_max = depth_max
        # sdf_trunc is in voxels for nvblox_torch
        self.trunc_voxels = sdf_trunc / voxel_size

        # Cache for SDF query points
        self._cached_query_params = None
        self._cached_query_tensor = None

        # Store mapper params for recreation on reset
        self._mapper_params = self._create_mapper_params()

        self.nvblox_mapper = Mapper(
            voxel_sizes_m=self.voxel_size,
            mapper_parameters=self._mapper_params,
        )

    def _create_mapper_params(self):
        """Create mapper params (reused on every reset)."""
        mapper_params = MapperParams()
        integrator_params = mapper_params.get_projective_integrator_params()
        integrator_params.projective_integrator_truncation_distance_vox = (
            self.trunc_voxels
        )
        integrator_params.projective_integrator_max_integration_distance_m = (
            self.depth_max
        )
        mapper_params.set_projective_integrator_params(integrator_params)
        return mapper_params

    def add_obs(
        self,
        camera_intrinsic,
        camera_extrinsic,
        rgb_image=None,
        depth_image=None,
        **kwargs,
    ):
        # nvblox_torch expects:
        # - depth_frame: torch.Tensor (H, W) on GPU in meters
        # - t_w_c: torch.Tensor (4, 4) on CPU - camera extrinsic (world to camera transform)
        # - intrinsics: torch.Tensor (3, 3) on CPU - camera intrinsic matrix

        # Convert numpy arrays to torch tensors if needed
        # Depth goes on GPU
        if isinstance(depth_image, np.ndarray):
            depth_tensor = torch.from_numpy(depth_image).float().cuda()
        else:
            depth_tensor = depth_image.float().cuda()

        # Squeeze to 2D if depth has a channel dimension (e.g., H, W, 1) -> (H, W)
        depth_tensor = depth_tensor.squeeze()

        # Intrinsics and extrinsics stay on CPU
        if isinstance(camera_intrinsic, np.ndarray):
            K = camera_intrinsic.astype(np.float32)
        else:
            K = camera_intrinsic.float().cpu().numpy()

        if isinstance(camera_extrinsic, np.ndarray):
            extrinsic_tensor = torch.from_numpy(camera_extrinsic).float()
        else:
            extrinsic_tensor = camera_extrinsic.float().cpu()

        # nvblox 0.0.9: add_depth_frame() takes a Sensor object instead of an intrinsics tensor
        h, w = depth_tensor.shape
        sensor = Sensor.from_camera(
            fu=float(K[0, 0]),
            fv=float(K[1, 1]),
            cu=float(K[0, 2]),
            cv=float(K[1, 2]),
            width=w,
            height=h,
        )

        self.nvblox_mapper.add_depth_frame(
            depth_frame=depth_tensor,
            t_w_c=extrinsic_tensor,
            sensor=sensor,
        )

    def reconstruct(
        self,
        type="mesh",
        sdf_size=32,
        bbox_center=None,
        bbox_size=None,
        **kwargs,
    ):
        """
        Reconstruct the scene from integrated observations.

        Args:
            type: "mesh" returns (vertices, faces) tuple as numpy arrays
                  "tsdf" returns a 3D numpy array of SDF values on a regular grid
            sdf_size: Resolution of the SDF grid for type="tsdf_dense" (default 32)
            bbox_center: Center of the bounding box for SDF query (numpy array shape (3,))
            bbox_size: Size of the bounding box for SDF query (scalar, length of longest side)

        Returns:
            For "mesh": tuple of (vertices, faces) as numpy arrays
            For "tsdf_dense": numpy array of shape (sdf_size, sdf_size, sdf_size)
            For "tsdf_sparse": dict with 'positions' (N, 3), 'sdf_values' (N,), 'truncation_distance'
        """
        if type == "mesh":
            self.nvblox_mapper.update_color_mesh()
            color_mesh = self.nvblox_mapper.get_color_mesh()

            vertices = color_mesh.vertices().cpu().numpy()
            faces = color_mesh.triangles().cpu().numpy()

            return (vertices, faces)
        elif type == "tsdf":
            return self._get_dense_tsdf(
                sdf_size, bbox_center, bbox_size, QueryType.TSDF
            )
        else:
            raise ValueError(f"Unknown reconstruction type: {type}")

    def _get_dense_tsdf(
        self,
        sdf_size: float,
        bbox_center: np.ndarray,
        bbox_size: float,
        query_type: QueryType = QueryType.TSDF,
    ):
        """
        Query the TSDF on a regular grid.

        The grid is constructed to match the same coordinate system used by mesh2sdf
        in reconstruct3D.compute_static_env_sdf():
        - The mesh is normalized to [-1, 1] centered at bbox_center with scale bbox_size/2
        - The SDF grid samples this [-1, 1]^3 space uniformly

        Args:
            sdf_size: Resolution of the grid (grid will be sdf_size^3)
            bbox_center: Center of bounding box in world coordinates, shape (3,)
            bbox_size: Size of bounding box (length of longest axis after padding)

        Returns:
            numpy array of shape (sdf_size, sdf_size, sdf_size) with SDF values.
            Unobserved regions are filled with a large positive value (100.0).

            numpy array of shape (sdf_size, sdf_size, sdf_size) with weights.
        """
        if bbox_center is None or bbox_size is None:
            raise ValueError(
                "bbox_center and bbox_size must be provided for type='tsdf'"
            )

        bbox_center = np.asarray(bbox_center)

        query_params = (sdf_size, bbox_center, bbox_size)
        if (
            self._cached_query_params == None
            or not np.isclose(self._cached_query_params[0], query_params[0])
            or not np.allclose(self._cached_query_params[1], query_params[1])
            or not np.isclose(self._cached_query_params[2], query_params[2])
        ):
            # print("Creating new SDF query grid (should be cached and not happen every step)")

            # Create grid in normalized [-1, 1] space (matching mesh2sdf convention)
            # mesh2sdf uses uniform sampling in [-1, 1]^3
            coords_1d = np.linspace(-1, 1, sdf_size)
            xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")

            # Stack into query points (N, 3) in normalized space
            query_points_normalized = np.stack(
                [xx.ravel(), yy.ravel(), zz.ravel()], axis=1
            )

            # Transform to world coordinates and create torch tensor
            query_points_world = query_points_normalized * (bbox_size / 2) + bbox_center
            self._cached_query_tensor = (
                torch.from_numpy(query_points_world).float().cuda()
            )

            # Update param cache
            self._cached_query_params = query_params

        # Query TSDF layer
        # Returns [N, 2] where column 0 is SDF value, column 1 is weight
        tsdf_result = self.nvblox_mapper.query_layer(
            query_type, self._cached_query_tensor
        )

        # Extract SDF values and weights
        sdf_values = tsdf_result[:, 0].cpu().numpy()
        weights = tsdf_result[:, 1].cpu().numpy()

        # print(
        #     f"TSDF stats: weight=0 voxels: {(weights == 0).sum()}, truncated voxels: {(np.abs(sdf_values) >= self.sdf_trunc).sum()}, sdf>99 voxels: {(sdf_values >= 99.0).sum()}"
        # )
        # # TSDF stats: weight=0 voxels: 32507, truncated voxels: 31798, sdf>99 voxels: 31702 => not expected

        # # find values of weight=0 and sdf<99 and print stats about them => all 0 => no clear explanation
        # # SDF values for unobserved voxels (weight=0, sdf<99): mean=0.0, std=0.0, min=0.0, max=0.0
        # unobserved_mask = (weights == 0) & (sdf_values < 99.0)
        # print(
        #     f"SDF values for unobserved voxels (weight=0, sdf<99): mean={sdf_values[unobserved_mask].mean()}, std={sdf_values[unobserved_mask].std()}, min={sdf_values[unobserved_mask].min()}, max={sdf_values[unobserved_mask].max()}"
        # )
        # # find values which are truncated but <99 and print stats about them => all barely below truncation value => just a numerical issue on my side
        # # SDF values for truncated voxels (|sdf|>=0.04, sdf<99): mean=0.03999999538064003, std=3.725290298461914e-09, min=0.03999999910593033, max=0.03999999910593033
        # truncated_mask = (np.abs(sdf_values) >= self.sdf_trunc) & (sdf_values < 99.0)
        # print(
        #     f"SDF values for truncated voxels (|sdf|>={self.sdf_trunc}, sdf<99): mean={sdf_values[truncated_mask].mean()}, std={sdf_values[truncated_mask].std()}, min={sdf_values[truncated_mask].min()}, max={sdf_values[truncated_mask].max()}"
        # )

        # set unobserved and truncated sdf values to 100 manually # TODO: How does nvblox handle this?
        sdf_values[(weights == 0)] = 100.0

        # print(f"Manipulated TSDF stats: weight=0 voxels: {(weights == 0).sum()}, truncated voxels: {(np.abs(sdf_values) >= self.sdf_trunc).sum()}, sdf>99 voxels: {(sdf_values >= 99.0).sum()}")

        # Reshape to 3D grid
        sdf_grid = sdf_values.reshape(sdf_size, sdf_size, sdf_size)
        weights_grid = weights.reshape(sdf_size, sdf_size, sdf_size)

        return sdf_grid, weights_grid

    def reset(self, **kwargs):
        # clear() frees all voxel block data in the C++ mapper (GPU memory).
        self.nvblox_mapper.clear()

        # The following was a workaround for a memory leak in render_mesh_open3d (now removed).
        # Kept here for reference in case a leak reappears.
        # del self.nvblox_mapper
        # gc.collect()
        # torch.cuda.synchronize()
        # self.nvblox_mapper = Mapper(
        #     voxel_sizes_m=self.voxel_size,
        #     mapper_parameters=self._mapper_params,
        # )
