"""
Test TSDF_generator_open3d with Reconstruct3D environment.

Integrates a single depth observation from robot0_eye_in_hand camera
and extracts the reconstructed mesh.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)

from arm_nbv.reconstruction_policies.open3d_reconstruction_policy import Open3DReconstructionPolicy
from arm_nbv.utils.plot_sdf_slices import plot_sdf_slices


def test_tsdf_single_observation(save_dir: str = "./data/test_TSDF_generator_open3d/"):
    """
    Test TSDF integration with a single observation from robot0_eye_in_hand.

    Args:
        save_dir: Directory to save output mesh
    """
    print("=" * 60)
    print("TEST: TSDF Generator with Reconstruct3D Environment")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # Camera settings
    camera_name = "robot0_eye_in_hand"
    camera_height = 128
    camera_width = 128

    # Create environment with default Panda controller (OSC_POSE)
    print("\nCreating Reconstruct3D environment...")
    controller_config = load_composite_controller_config(
        robot="Panda",
    )

    env = suite.make(
        env_name="Reconstruct3D",
        robots="Panda",
        controller_configs=controller_config,
        camera_names=[camera_name],
        camera_heights=camera_height,
        camera_widths=camera_width,
    )

    print("Environment created!")

    try:
        # Reset and get observation
        print("\nResetting environment and getting observation...")
        obs = env.reset()

        # Get depth observation
        depth_key = f"{camera_name}_depth"
        if depth_key not in obs:
            print(f"Available keys: {list(obs.keys())}")
            raise KeyError(f"Depth key '{depth_key}' not found in observations")

        # Get normalized depth and convert to real depth (meters)
        depth_normalized = obs[depth_key][:, :, 0]  # Remove channel dim
        depth_real = get_real_depth_map(env.sim, depth_normalized)

        print(f"Depth shape: {depth_real.shape}")
        print(f"Depth range: [{depth_real.min():.3f}, {depth_real.max():.3f}] meters")

        # Get RGB image if available
        rgb_key = f"{camera_name}_image"
        rgb_image = obs.get(rgb_key, None)

        # Save depth and RGB images for inspection
        print(f"\nSaving input images to {save_dir}...")

        # Save depth as colormap visualization
        depth_vis_path = os.path.join(save_dir, "input_depth.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(depth_real, cmap="viridis")
        plt.colorbar(label="Depth (meters)")
        plt.title(
            f"Depth Image\nRange: [{depth_real.min():.3f}, {depth_real.max():.3f}] m"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(depth_vis_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved depth visualization to: {depth_vis_path}")

        # Save depth as raw numpy array
        depth_npy_path = os.path.join(save_dir, "input_depth.npy")
        np.save(depth_npy_path, depth_real)
        print(f"  Saved depth array to: {depth_npy_path}")

        # Save RGB image if available
        if rgb_image is not None:
            rgb_path = os.path.join(save_dir, "input_rgb.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb_image)
            plt.title("RGB Image")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(rgb_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved RGB image to: {rgb_path}")
        else:
            print(f"  No RGB image available (key '{rgb_key}' not in observations)")

        # Get camera intrinsic and extrinsic matrices
        intrinsic = get_camera_intrinsic_matrix(
            env.sim, camera_name, camera_height, camera_width
        )
        extrinsic = get_camera_extrinsic_matrix(env.sim, camera_name)

        print(f"\nCamera intrinsic matrix:\n{intrinsic}")
        print(f"\nCamera extrinsic (camera to world):\n{extrinsic}")

        # Define workspace bounding box (should cover the table area)
        bbox_min = np.array([-0.5, -0.5, 0.5])  # meters
        bbox_max = np.array([0.5, 0.5, 1.5])  # meters

        print(f"\nWorkspace bounds: min={bbox_min}, max={bbox_max}")

        # Create TSDF generator
        print("\nInitializing TSDF generator...")
        tsdf = Open3DReconstructionPolicy(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            voxel_size=0.02,  # 2cm voxels
            sdf_trunc=0.05,  # 5cm truncation
            # device auto-selected (CUDA if available, else CPU)
        )

        print(f"Device: {tsdf.device}")

        print(f"Volume size: {tsdf.volume_size}")
        print(f"Grid shape: {tsdf.grid_shape}")

        # Integrate depth observation
        print("\nIntegrating depth observation...")
        tsdf.add_obs(
            camera_intrinsic=intrinsic,
            camera_extrinsic=extrinsic,
            depth_image=depth_real,
            depth_max=1.0,  # Ignore depth beyond 1m
            depth_scale=1.0,  # Already in meters
        )
        print("Integration complete!")

        # Debug: Extract point cloud first to verify data
        print("\nExtracting point cloud...")
        pcd = tsdf.extract_point_cloud(weight_threshold=0.0)
        print(f"Point cloud points: {len(pcd.points)}")

        # Extract mesh
        print("\nExtracting mesh...")
        mesh = tsdf.extract_mesh(weight_threshold=0.0)

        print(f"Mesh vertices: {len(mesh.vertices)}")
        print(f"Mesh triangles: {len(mesh.triangles)}")

        if len(mesh.vertices) > 0:
            # Save mesh as OBJ
            mesh_path = os.path.join(save_dir, "tsdf_mesh_single_obs.obj")
            import open3d as o3d

            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"\nMesh saved to: {mesh_path}")
        else:
            print("\nWarning: No mesh extracted (possibly no surface visible)")

        # --- New: test native TSDF grid extraction ---
        print("\nExtracting native TSDF grid from VoxelBlockGrid...")

        # First check what native voxel positions we have
        positions, tsdf_vals, weights = tsdf.get_native_tsdf_points(
            weight_threshold=0.0
        )
        print(f"Native voxel positions: {len(positions)} voxels")
        if len(positions) > 0:
            print(
                f"  Position range: x=[{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]"
            )
            print(
                f"                  y=[{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]"
            )
            print(
                f"                  z=[{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]"
            )
            print(f"  TSDF range: [{tsdf_vals.min():.4f}, {tsdf_vals.max():.4f}]")
            print(f"  Weight range: [{weights.min():.1f}, {weights.max():.1f}]")

            print(f"Example voxel data (first 5):")
            for i in range(min(5, len(positions))):
                print(
                    f"  Pos: ({positions[i,0]:.3f}, {positions[i,1]:.3f}, {positions[i,2]:.3f}), "
                    f"TSDF: {tsdf_vals[i]:.4f}, Weight: {weights[i]:.1f}"
                )

            # Compare with mesh vertices to understand coordinate system
            mesh_vertices = np.asarray(mesh.vertices)
            print(f"\nComparison with mesh vertices:")
            print(f"  Mesh vertex count: {len(mesh_vertices)}")
            print(
                f"  Mesh X range: [{mesh_vertices[:,0].min():.3f}, {mesh_vertices[:,0].max():.3f}]"
            )
            print(
                f"  Mesh Y range: [{mesh_vertices[:,1].min():.3f}, {mesh_vertices[:,1].max():.3f}]"
            )
            print(
                f"  Mesh Z range: [{mesh_vertices[:,2].min():.3f}, {mesh_vertices[:,2].max():.3f}]"
            )
        else:
            print("  No native voxels found above weight threshold.")

        grid_resolution = 16
        t0 = time.time()
        sdf_grid = tsdf.get_sdf_grid(grid_resolution=grid_resolution)
        dt = time.time() - t0

        print(f"SDF grid shape: {sdf_grid.shape}")
        print(
            f"SDF grid stats: min={np.nanmin(sdf_grid):.4f}, max={np.nanmax(sdf_grid):.4f}"
        )
        print(f"Extraction took {dt:.3f} s")

        # Save the grid for inspection
        sdf_npy_path = os.path.join(save_dir, "sdf_grid.npy")
        sdf_plot_path = os.path.join(save_dir, "sdf_slices.png")

        np.save(sdf_npy_path, sdf_grid)
        plot_sdf_slices(sdf_grid, sdf_plot_path)

        print(f"Saved SDF grid to: {sdf_npy_path}")
        print(f"Saved SDF slices plot to: {sdf_plot_path}")

        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        env.close()


if __name__ == "__main__":
    test_tsdf_single_observation()
