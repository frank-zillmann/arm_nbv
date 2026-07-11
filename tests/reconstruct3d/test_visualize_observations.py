"""
Test function to visualize observations from Reconstruct3D environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from robosuite.utils.camera_utils import get_real_depth_map


def test_visualize_observations(env, path_to_save="./data/test_reconstruct3D_env/obs/"):
    """
    Visualize RGB and depth observations from all cameras.

    Args:
        env: Reconstruct3D environment instance
        path_to_save: Directory to save visualization images
    """
    print("\n" + "=" * 60)
    print("TEST 1: Visualizing Observations")
    print("=" * 60)

    os.makedirs(path_to_save, exist_ok=True)

    # Get current observation
    obs = env._get_observations()

    print(f"\nObservation keys: {list(obs.keys())}")

    # Process each observation
    for key in obs:
        if isinstance(obs[key], np.ndarray):
            print(f"  {key}: shape {obs[key].shape}, dtype {obs[key].dtype}")

            # Print small arrays
            if obs[key].shape.__len__() == 1 and obs[key].shape[0] <= 50:
                print(f"    Values: {obs[key]}")

            # Save RGB images
            if obs[key].shape.__len__() == 3 and obs[key].shape[2] == 3:
                plt.figure(figsize=(6, 6))
                plt.imshow(obs[key])
                plt.title(f"{key} - RGB Image")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(path_to_save + f"initial_{key}.png")
                print(f"    Saved to '{path_to_save}initial_{key}.png'")
                plt.close()

            # Save depth maps
            elif obs[key].shape.__len__() == 3 and obs[key].shape[2] == 1:
                # Convert normalized depth to real depth values (in meters)
                depth_normalized = obs[key][:, :, 0]
                depth_real = get_real_depth_map(env.sim, depth_normalized)

                # Print depth statistics
                print(
                    f"    Depth range (normalized): [{depth_normalized.min():.3f}, {depth_normalized.max():.3f}]"
                )
                print(
                    f"    Depth range (real meters): [{depth_real.min():.3f}, {depth_real.max():.3f}]"
                )

                # Save depth map with real values
                plt.figure(figsize=(6, 6))
                im = plt.imshow(depth_real, cmap="viridis", vmin=0, vmax=None)
                plt.title(f"{key} - Depth Map (meters)")
                plt.axis("off")
                plt.colorbar(im, label="Depth (m)")
                plt.tight_layout()
                plt.savefig(path_to_save + f"initial_{key}.png")
                print(f"    Saved to '{path_to_save}initial_{key}.png'")
                plt.close()
            else:
                print(f"    (omitted values)")

    print("\nObservation visualization complete!")
