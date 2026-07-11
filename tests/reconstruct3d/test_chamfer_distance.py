"""
Test Chamfer distance computation for Reconstruct3D environment.
"""

import numpy as np


def test_chamfer_distance(env):
    """Test Chamfer distance computation with mesh reconstructions."""
    print("\n" + "=" * 60)
    print("TEST: Chamfer Distance")
    print("=" * 60)

    # Get ground truth mesh from environment
    gt_vertices = env.static_env_vertices
    gt_faces = env.static_env_faces

    print(
        f"Ground truth mesh: {gt_vertices.shape[0]} vertices, {gt_faces.shape[0]} faces"
    )

    # Test 1: Identical mesh should have near-zero Chamfer distance
    print("\n--- Test 1: Identical mesh ---")
    chamfer_identical = env.compute_chamfer_distance(
        gt_vertices, gt_faces, n_samples=5000
    )
    print(f"Chamfer distance (identical mesh): {chamfer_identical:.6f}")
    assert (
        chamfer_identical < 0.01
    ), f"Identical mesh should have near-zero Chamfer distance, got {chamfer_identical}"

    # Test 2: Slightly offset mesh should have small but non-zero distance
    print("\n--- Test 2: Slightly offset mesh (0.01 units) ---")
    offset_vertices = gt_vertices + 0.01
    chamfer_small_offset = env.compute_chamfer_distance(
        offset_vertices, gt_faces, n_samples=5000
    )
    print(f"Chamfer distance (offset by 0.01): {chamfer_small_offset:.6f}")

    # Test 3: Larger offset should have larger distance
    print("\n--- Test 3: Larger offset mesh (0.1 units) ---")
    large_offset_vertices = gt_vertices + 0.1
    chamfer_large_offset = env.compute_chamfer_distance(
        large_offset_vertices, gt_faces, n_samples=5000
    )
    print(f"Chamfer distance (offset by 0.1): {chamfer_large_offset:.6f}")

    # Test 4: Even larger offset
    print("\n--- Test 4: Very large offset mesh (0.5 units) ---")
    very_large_offset_vertices = gt_vertices + 0.5
    chamfer_very_large = env.compute_chamfer_distance(
        very_large_offset_vertices, gt_faces, n_samples=5000
    )
    print(f"Chamfer distance (offset by 0.5): {chamfer_very_large:.6f}")

    # Verify ordering: distances should increase with offset
    print("\n--- Verifying distance ordering ---")
    assert (
        chamfer_identical < chamfer_small_offset
    ), "Identical should be less than small offset"
    assert (
        chamfer_small_offset < chamfer_large_offset
    ), "Small offset should be less than large offset"
    assert (
        chamfer_large_offset < chamfer_very_large
    ), "Large offset should be less than very large offset"
    print("Distance ordering verified: identical < 0.01 < 0.1 < 0.5")

    # Test 5: Test reward function with mesh reconstruction
    print("\n--- Test 5: Reward function with mesh reconstruction ---")
    reward_identical = env.reward(reconstruction=(gt_vertices, gt_faces))
    reward_offset = env.reward(reconstruction=(large_offset_vertices, gt_faces))
    print(f"Reward (identical mesh): {reward_identical:.4f}")
    print(f"Reward (offset mesh): {reward_offset:.4f}")
    assert reward_identical > reward_offset, "Identical mesh should have higher reward"

    print("\nChamfer distance test PASSED!")
