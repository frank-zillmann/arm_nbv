"""Evaluate trained 3D reconstruction policy."""

import argparse
import numpy as np
from stable_baselines3 import PPO
from arm_nbv.reconstruct3d_gym_wrapper import Reconstruct3DGymWrapper


def evaluate(checkpoint: str, n_episodes: int = 10):
    """Evaluate trained model."""
    print(f"Loading: {checkpoint}")
    model = PPO.load(checkpoint)
    env = Reconstruct3DGymWrapper()

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward, done = 0, False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"Episode {ep + 1}: {total_reward:.4f}")

    print(f"\nMean: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.n_episodes)
