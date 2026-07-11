"""
Test function to generate mesh from SDF and render it for comparison.
"""

import numpy as np
import os
from skimage import measure
import trimesh
import matplotlib.pyplot as plt

from arm_nbv.utils.render_mesh import render_mesh


def test_render_sdf_mesh(
    env,
    sdf_grid=None,
    bbox_center=None,
    bbox_size=None,
    path_to_save="./data/test_reconstruct3D_env/sdf_mesh/",
):
    """
    Extract mesh from SDF using marching cubes and render it.

    Args:
        env: Reconstruct3D environment instance
        sdf_grid: SDF grid array (N, N, N)
        bbox_center: Center of the bounding box used for normalization
        bbox_size: Size of the bounding box used for normalization
        path_to_save: Directory to save renderings
    """

    if sdf_grid is None:
        sdf_grid = env.sdf_grid
    if bbox_center is None:
        bbox_center = env.bbox_center
    if bbox_size is None:
        bbox_size = env.bbox_size

    print("\n" + "=" * 60)
    print("TEST 3: Rendering Mesh from SDF")
    print("=" * 60)

    os.makedirs(path_to_save, exist_ok=True)

    # Extract mesh from SDF using marching cubes
    print("\nExtracting mesh from SDF using marching cubes...")
    level = 0.0  # Surface is at SDF value 0

    vertices, faces, normals, values = measure.marching_cubes(sdf_grid, level=level)
    print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Transform vertices back to world coordinates
    sdf_size = sdf_grid.shape[0]
    vertices_normalized = vertices / sdf_size * 2 - 1  # Map to [-1, 1]
    vertices_world = vertices_normalized * (bbox_size / 2) + bbox_center

    print(
        f"Vertex bounds: min={vertices_world.min(axis=0)}, max={vertices_world.max(axis=0)}"
    )

    # Get camera matrices for multiple views
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix,
        get_camera_extrinsic_matrix,
    )

    # Render from multiple camera views
    camera_names = ["birdview", "frontview", "sideview"]
    for camera_name in camera_names:
        try:
            intrinsic = get_camera_intrinsic_matrix(env.sim, camera_name, 128, 128)
            extrinsic = get_camera_extrinsic_matrix(env.sim, camera_name)

            # Render and save (RGB)
            rgb = render_mesh(
                vertices_world,
                faces,
                extrinsic,
                intrinsic,
                resolution=(128, 128),
                grayscale=False,
            )
            output_path = f"{path_to_save}sdf_mesh_{camera_name}.png"
            plt.imsave(output_path, rgb)
            print(f"Saved {camera_name} rendering to '{output_path}'")
        except Exception as e:
            print(f"Warning: Could not render from {camera_name}: {e}")

    # Create trimesh object for mesh export
    mesh_trimesh = trimesh.Trimesh(
        vertices=vertices_world, faces=faces, vertex_normals=normals
    )

    # Save mesh as OBJ
    obj_path = path_to_save + "sdf_mesh.obj"
    mesh_trimesh.export(obj_path)
    print(f"Saved mesh as OBJ to '{obj_path}'")

    print("\nSDF mesh rendering complete!")
