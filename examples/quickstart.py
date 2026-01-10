#!/usr/bin/env python3
"""
PDE-Gym Quickstart Example

This script demonstrates basic usage of the PDE-Gym environment.
"""

import numpy as np


def random_agent_example():
    """Run a random agent in the environment."""
    from pde_gym import PDEEnv

    print("Creating PDE-Gym environment...")
    env = PDEEnv(
        preset="GrayScott",
        obs_width=128,
        obs_height=128,
        timesteps_per_step=50,
        max_episode_steps=100,
        headless=True,
        use_swiftshader=True,  # CPU-only WebGL
    )

    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Preset: {info['preset']}")

    print("\nRunning 10 random steps...")
    total_reward = 0
    for step in range(10):
        # Random action: [x, y, species, value, radius]
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"  Step {step + 1}: sim_time={info['sim_time']:.2f}, reward={reward:.4f}")

        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.4f}")

    env.close()
    print("Environment closed.")


def custom_reward_example():
    """Example with custom reward function."""
    from pde_gym import PDEEnv
    from pde_gym.rewards import VarianceReward

    print("\n--- Custom Reward Example ---")

    # Create reward function that encourages pattern variance
    reward_fn = VarianceReward(maximize=True, scale=100.0)

    env = PDEEnv(
        preset="GrayScott",
        obs_width=64,
        obs_height=64,
        timesteps_per_step=100,
        max_episode_steps=50,
        reward_fn=reward_fn,
        headless=True,
        use_swiftshader=True,
    )

    obs, info = env.reset(seed=123)

    print("Running with variance reward...")
    total_reward = 0
    for step in range(10):
        # Apply brush to center
        action = np.array([0.5, 0.5, 0.0, 0.8, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step + 1}: reward={reward:.4f}")

    print(f"Total variance reward: {total_reward:.4f}")
    env.close()


def gymnasium_make_example():
    """Example using gymnasium.make()."""
    import gymnasium as gym
    # Import to register environments
    import pde_gym  # noqa

    print("\n--- Using gymnasium.make() ---")

    env = gym.make("PDEEnv-GrayScott-v0", obs_width=64, obs_height=64)

    obs, info = env.reset()
    print(f"Created environment via gym.make()")
    print(f"Observation shape: {obs.shape}")

    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()
    print("Done!")


def save_frames_example():
    """Example saving simulation frames to disk."""
    from pathlib import Path
    from PIL import Image
    from pde_gym import PDEEnv

    print("\n--- Saving Frames Example ---")

    output_dir = Path("pde_frames")
    output_dir.mkdir(exist_ok=True)

    env = PDEEnv(
        preset="BrusselatorPDE",
        obs_width=256,
        obs_height=256,
        timesteps_per_step=100,
        render_mode="rgb_array",
        headless=True,
        use_swiftshader=True,
    )

    obs, info = env.reset(seed=42)

    # Save initial frame
    Image.fromarray(obs).save(output_dir / "frame_000.png")
    print(f"Saved frame 0")

    # Run a few steps and save frames
    for i in range(5):
        # Apply different brush strokes
        x = 0.3 + i * 0.1
        y = 0.5
        action = np.array([x, y, 0.0, 0.9, 0.3], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        # Save frame
        Image.fromarray(obs).save(output_dir / f"frame_{i+1:03d}.png")
        print(f"Saved frame {i + 1}")

    env.close()
    print(f"Frames saved to {output_dir}/")


if __name__ == "__main__":
    print("=" * 50)
    print("PDE-Gym Quickstart")
    print("=" * 50)

    random_agent_example()

    # Uncomment to run other examples:
    # custom_reward_example()
    # gymnasium_make_example()
    # save_frames_example()
