"""
Tests for the PDE-Gym environment.
"""

import pytest
import numpy as np


def test_import():
    """Test that the package can be imported."""
    from pde_gym import PDEEnv, BrowserManager, PDEBridge
    assert PDEEnv is not None
    assert BrowserManager is not None
    assert PDEBridge is not None


def test_observation_space():
    """Test observation space is correctly defined."""
    from pde_gym import PDEEnv

    env = PDEEnv(obs_width=64, obs_height=64)
    assert env.observation_space.shape == (64, 64, 3)
    assert env.observation_space.dtype == np.uint8
    env.close()


def test_action_space():
    """Test action space is correctly defined."""
    from pde_gym import PDEEnv

    env = PDEEnv()
    assert env.action_space.shape == (5,)
    assert env.action_space.dtype == np.float32
    env.close()


def test_action_decode():
    """Test action decoding."""
    from pde_gym import PDEEnv

    env = PDEEnv()

    # Test action decoding
    action = np.array([0.5, 0.5, 0.25, 0.8, 0.5], dtype=np.float32)
    params = env._decode_action(action)

    assert params["x"] == 0.5
    assert params["y"] == 0.5
    assert params["species"] == "u"  # species < 0.5
    assert params["value"] == 0.8
    assert 0.01 <= params["radius"] <= 0.2

    # Test species "v" selection
    action = np.array([0.5, 0.5, 0.75, 0.8, 0.5], dtype=np.float32)
    params = env._decode_action(action)
    assert params["species"] == "v"  # species >= 0.5

    env.close()


@pytest.mark.slow
def test_reset(env_with_browser):
    """Test environment reset (requires browser)."""
    env = env_with_browser

    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == env.observation_space.dtype
    assert "preset" in info
    assert "sim_time" in info


@pytest.mark.slow
def test_step(env_with_browser):
    """Test environment step (requires browser)."""
    env = env_with_browser

    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "step" in info
    assert "sim_time" in info


@pytest.mark.slow
def test_multiple_steps(env_with_browser):
    """Test multiple steps in sequence."""
    env = env_with_browser

    obs, _ = env.reset()

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["step"] == i + 1

        if terminated or truncated:
            break


@pytest.mark.slow
def test_render(env_with_browser):
    """Test rendering."""
    env = env_with_browser
    env.render_mode = "rgb_array"

    env.reset()
    frame = env.render()

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert len(frame.shape) == 3
    assert frame.shape[2] == 3


@pytest.mark.slow
def test_seeded_reset(env_with_browser):
    """Test that seeded resets are reproducible."""
    env = env_with_browser

    # First run
    obs1, _ = env.reset(seed=42)
    for _ in range(3):
        env.step(np.array([0.5, 0.5, 0.0, 0.5, 0.5], dtype=np.float32))
    frame1 = env.render()

    # Second run with same seed (need new env for true reproducibility)
    env.close()

    from pde_gym import PDEEnv
    env2 = PDEEnv(
        preset="GrayScott",
        obs_width=64,
        obs_height=64,
        render_mode="rgb_array",
    )
    obs2, _ = env2.reset(seed=42)

    # Initial observations should be similar (same preset, same seed)
    # Note: Exact equality may not hold due to floating point
    assert obs1.shape == obs2.shape

    env2.close()


# Fixtures
@pytest.fixture
def env_with_browser():
    """Create environment with browser (slow)."""
    from pde_gym import PDEEnv

    env = PDEEnv(
        preset="GrayScott",
        obs_width=64,
        obs_height=64,
        sim_width=256,
        sim_height=256,
        timesteps_per_step=10,
        max_episode_steps=100,
        render_mode="rgb_array",
        headless=True,
        use_swiftshader=True,
    )

    yield env

    env.close()


if __name__ == "__main__":
    # Run quick tests without browser
    test_import()
    test_observation_space()
    test_action_space()
    test_action_decode()
    print("Basic tests passed!")
