"""
Test function to run dummy steps in the environment.
"""

import numpy as np


def test_run_dummy_steps(env, n_steps=3):
    """
    Run a few dummy steps with simple actions.

    Args:
        env: Reconstruct3D environment instance
        n_steps: Number of steps to run
    """
    print("\n" + "=" * 60)
    print(f"TEST 4: Running {n_steps} Dummy Steps")
    print("=" * 60)

    print(f"\nAction space: {env.action_spec}")
    print(f"Action dim: {env.action_dim}")

    total_reward = 0
    for i in range(n_steps):
        # Simple linear action progression
        action = np.array([i / n_steps] * env.action_dim)

        print(f"\nStep {i+1}/{n_steps}:")
        print(
            f"  Action: {action[:5]}..." if len(action) > 5 else f"  Action: {action}"
        )

        # Step environment
        obs, reward, done, info = env.step(action)

        print(f"  Reward: {reward}")
        print(f"  Done: {done}")

    print("\nDummy steps complete!")
