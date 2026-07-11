"""
Main test script for Reconstruct3D environment.
Runs all test functions in sequence.
"""

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

import numpy as np

# Import test functions
from test_visualize_observations import test_visualize_observations
from test_compute_sdf import test_compute_sdf
from test_render_sdf_mesh import test_render_sdf_mesh
from test_run_dummy_steps import test_run_dummy_steps
from test_chamfer_distance import test_chamfer_distance


def test_reward(env):
    """Test reward function with ground truth and noisy SDF."""
    print("\n" + "=" * 60)
    print("TEST: Reward Function")
    print("=" * 60)

    # Perfect match should give max reward
    gt_sdf = env.sdf_grid
    reward_perfect = env.reward(reconstruction=gt_sdf)
    print(
        f"Reward (perfect match): {reward_perfect:.4f} (expected: {env.reward_scale})"
    )
    assert np.isclose(
        reward_perfect, env.reward_scale
    ), "Perfect match should give max reward"

    # Noisy SDF should give lower reward
    noisy_sdf = gt_sdf + np.random.randn(*gt_sdf.shape) * 0.1
    reward_noisy = env.reward(reconstruction=noisy_sdf)
    print(f"Reward (noisy SDF): {reward_noisy:.4f}")
    assert reward_noisy < reward_perfect, "Noisy SDF should give lower reward"

    print("Reward test passed!")


def main():
    """Run all Reconstruct3D tests."""

    print("=" * 60)
    print("RECONSTRUCT3D ENVIRONMENT TEST SUITE")
    print("=" * 60)

    # Create environment with default Panda controller (OSC_POSE)
    print("\nCreating environment...")
    controller_config = load_composite_controller_config(
        robot="Panda",
    )

    env = suite.make(
        env_name="Reconstruct3D",
        robots="Panda",
        controller_configs=controller_config,
        camera_names=[
            "frontview",
            "birdview",
            "sideview",
            "robot0_eye_in_hand",
        ],
    )

    print("Environment created successfully!")

    # Reset environment
    print("\nResetting environment...")
    obs = env.reset()
    print("Environment reset complete!")

    # Run all tests
    try:
        # Test 1: Visualize observations
        test_visualize_observations(env)

        # Test 2: Compute SDF (uses env.compute_static_env_sdf internally)
        test_compute_sdf(env)

        # Test 3: Reward function
        test_reward(env)

        # Test 4: Chamfer distance
        test_chamfer_distance(env)

        # Test 5: Render mesh from SDF
        test_render_sdf_mesh(env)

        # Test 6: Run dummy steps
        test_run_dummy_steps(env, n_steps=3)

    finally:
        # Close environment
        env.close()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
