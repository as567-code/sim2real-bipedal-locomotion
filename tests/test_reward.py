"""Tests for the AMP-inspired reward function."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2real.envs.reward import RewardComputer


@pytest.fixture
def reward():
    r = RewardComputer(cmd_velocity=0.5)
    r.reset()
    return r


def _default_kwargs():
    return dict(
        torso_velocity=np.array([0.5, 0.0, 0.0]),
        torso_orientation_rpy=np.array([0.0, 0.0, 0.0]),
        joint_torques=np.zeros(8),
        joint_velocities=np.zeros(8),
        left_foot_height=0.0,
        right_foot_height=0.0,
        left_foot_contact=True,
        right_foot_contact=True,
        left_joint_positions=np.zeros(4),
        right_joint_positions=np.zeros(4),
    )


def test_perfect_tracking_gives_max_velocity_reward(reward):
    total, components = reward.compute(**_default_kwargs())
    assert components["velocity_tracking"] == pytest.approx(1.0, abs=0.01)


def test_zero_velocity_penalizes_tracking(reward):
    kwargs = _default_kwargs()
    kwargs["torso_velocity"] = np.array([0.0, 0.0, 0.0])
    _, components = reward.compute(**kwargs)
    assert components["velocity_tracking"] < 0.5


def test_upright_gives_max_orientation_reward(reward):
    _, components = reward.compute(**_default_kwargs())
    assert components["orientation"] == pytest.approx(1.0, abs=0.01)


def test_tilted_penalizes_orientation(reward):
    kwargs = _default_kwargs()
    kwargs["torso_orientation_rpy"] = np.array([0.0, 0.5, 0.0])
    _, components = reward.compute(**kwargs)
    assert components["orientation"] < 0.5


def test_zero_torque_gives_zero_energy_penalty(reward):
    _, components = reward.compute(**_default_kwargs())
    assert components["energy"] == 0.0


def test_high_torque_gives_negative_energy(reward):
    kwargs = _default_kwargs()
    kwargs["joint_torques"] = np.ones(8) * 10.0
    _, components = reward.compute(**kwargs)
    assert components["energy"] < 0.0


def test_torque_stability_zero_on_first_step(reward):
    _, components = reward.compute(**_default_kwargs())
    assert components["torque_stability"] == 0.0


def test_torque_stability_penalizes_change(reward):
    kwargs = _default_kwargs()
    reward.compute(**kwargs)
    kwargs["joint_torques"] = np.ones(8) * 5.0
    _, components = reward.compute(**kwargs)
    assert components["torque_stability"] < 0.0


def test_alive_bonus_is_positive(reward):
    _, components = reward.compute(**_default_kwargs())
    assert components["alive"] == 1.0


def test_symmetric_joints_give_zero_symmetry_penalty(reward):
    kwargs = _default_kwargs()
    kwargs["left_joint_positions"] = np.array([0.1, 0.2, -0.5, 0.0])
    kwargs["right_joint_positions"] = np.array([0.1, 0.2, -0.5, 0.0])
    _, components = reward.compute(**kwargs)
    assert components["symmetry"] == pytest.approx(0.0)


def test_asymmetric_joints_penalize(reward):
    kwargs = _default_kwargs()
    kwargs["left_joint_positions"] = np.array([0.5, 0.0, 0.0, 0.0])
    kwargs["right_joint_positions"] = np.array([-0.5, 0.0, 0.0, 0.0])
    _, components = reward.compute(**kwargs)
    assert components["symmetry"] < 0.0


def test_total_reward_is_weighted_sum(reward):
    total, components = reward.compute(**_default_kwargs())
    expected = sum(reward.weights[k] * v for k, v in components.items())
    assert total == pytest.approx(expected)
