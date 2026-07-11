"""Evaluate the scripted table edge policy for 3D reconstruction."""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

from arm_nbv.robot_policies import ScriptedPolicy
from arm_nbv.utils.env_factory import create_env
from arm_nbv.config import TrainConfig


def evaluate_scripted(config: TrainConfig, n_episodes: int = 1):
    """Evaluate the scripted table edge policy."""

    # Setup logging directory
    run_name = f"scripted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(log_dir / "config.yaml"))

    # Create environment with eval logging
    eval_log_dir = log_dir / "eval_data"
    env = create_env(config, seed=42, collect_timing=True, eval_log_dir=eval_log_dir)

    # Create scripted policy (uses default table geometry)
    policy = ScriptedPolicy(horizon=config.horizon)

    print(f"Evaluating scripted table edge policy")
    print(f"Logging to: {log_dir}")
    print(f"Running {n_episodes} episodes with horizon {config.horizon}")

    all_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        policy.reset()

        episode_reward = 0.0
        done = False
        step = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated

            if step % 10 == 0:
                print(f"  Episode {ep+1}, Step {step}: reward={reward:.6f}")

        all_rewards.append(episode_reward)
        print(
            f"Episode {ep + 1}/{n_episodes}: reward={episode_reward:.6f}, steps={step}"
        )

    # Summary
    print(f"\n{'='*50}")
    print(f"Mean reward: {np.mean(all_rewards):.6f} ± {np.std(all_rewards):.6f}")

    # Save results
    with open(log_dir / "results.txt", "w") as f:
        f.write(f"Scripted Table Edge Policy\n{'='*40}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(
            f"Mean reward: {np.mean(all_rewards):.6f} ± {np.std(all_rewards):.6f}\n"
        )
        for i, r in enumerate(all_rewards):
            f.write(f"  Episode {i+1}: {r:.6f}\n")

    print(f"\nResults saved to: {log_dir / 'results.txt'}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate scripted table edge policy")
    parser.add_argument(
        "--config", type=str, nargs="+", default=[], help="Config YAML(s)"
    )
    parser.add_argument("--n_episodes", type=int, default=1, help="Number of episodes")
    args = parser.parse_args()

    config = TrainConfig.load(args.config)
    evaluate_scripted(config, args.n_episodes)
