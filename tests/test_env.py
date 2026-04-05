"""Tests for the BipedalEnv Gymnasium environment."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2real.envs.bipedal_env import BipedalEnv


@pytest.fixture
def env():
    e = BipedalEnv(enable_domain_rand=False, seed=42)
    yield e
    e.close()


def test_observation_shape(env):
    obs, _ = env.reset()
    assert obs.shape == (env.OBS_DIM,), f"Expected obs dim {env.OBS_DIM}, got {obs.shape}"


def test_action_shape(env):
    assert env.action_space.shape == (env.ACT_DIM,)


def test_reset_determinism(env):
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)


def test_step_returns_correct_shapes(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (env.OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_runs_multiple_times(env):
    env.reset(seed=0)
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()


def test_termination_on_fall(env):
    env.reset(seed=0)
    terminated = False
    for _ in range(2000):
        action = np.ones(env.ACT_DIM) * 1.0
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert terminated or truncated, "Expected episode to end within 2000 steps with max torque"


def test_reward_info_contains_components(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    assert "reward_components" in info
    expected_keys = {"velocity_tracking", "orientation", "energy", "torque_stability",
                     "joint_velocity", "foot_clearance", "alive", "symmetry"}
    assert expected_keys == set(info["reward_components"].keys())


def test_observation_values_are_finite(env):
    env.reset(seed=0)
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert np.all(np.isfinite(obs)), f"Non-finite values in obs: {obs}"
        if terminated or truncated:
            obs, _ = env.reset()
