"""
Test function to compute and visualize SDF from environment mesh.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from arm_nbv.utils.plot_sdf_slices import plot_sdf_slices


def test_compute_sdf(env, path_to_save="./data/test_reconstruct3D_env/sdf_slices/"):
    """
    Compute SDF using the environment's built-in method and visualize slices.

    Args:
        env: Reconstruct3D environment instance
        sdf_size: Optional SDF grid resolution. If provided, temporarily overrides env.sdf_size.
        path_to_save: Directory to save SDF grid and visualization

    Returns:
        tuple: (sdf_grid, bbox_center, bbox_size) from the environment's object variables
    """
    print("\n" + "=" * 60)
    print("TEST 2: Computing SDF from Environment Mesh")
    print("=" * 60)

    os.makedirs(path_to_save, exist_ok=True)

    # Use the environment's built-in method to compute SDF
    print("\nComputing SDF using env.compute_static_env_sdf()...")
    sdf_grid, bbox_center, bbox_size = env.compute_static_env_sdf()
    sdf_size = env.sdf_size

    print(f"SDF grid shape: {sdf_grid.shape}")
    print(f"SDF value range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
    print(f"Bounding box center: {bbox_center}, size: {bbox_size}")
    os.makedirs(path_to_save, exist_ok=True)
    plot_sdf_slices(sdf_grid, path_to_save + "sdf_slices.png")

    print("\nSDF computation complete!")

    return
