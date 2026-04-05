"""Tests for domain randomization."""
import sys
from pathlib import Path

import numpy as np
import mujoco
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2real.envs.domain_randomization import DomainRandomizer, DomainRandConfig, DRParamRange

_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(str(_ASSET_DIR / "biped.xml"))


@pytest.fixture
def randomizer(model):
    return DomainRandomizer(model, DomainRandConfig())


def test_randomize_returns_sampled_values(randomizer):
    rng = np.random.default_rng(42)
    sampled = randomizer.randomize(rng)
    assert "body_mass_scale" in sampled
    assert "ground_friction" in sampled
    assert "actuator_delay_steps" in sampled
    assert "joint_damping_scale" in sampled
    assert "obs_noise_std" in sampled
    assert "gravity_z" in sampled
    assert "actuator_gain_scale" in sampled
    assert "terrain_roughness" in sampled


def test_body_mass_within_range(model, randomizer):
    rng = np.random.default_rng(0)
    original_mass = model.body_mass.copy()

    for _ in range(50):
        randomizer.restore_defaults()
        sampled = randomizer.randomize(rng)
        scale = sampled["body_mass_scale"]
        assert 0.8 <= scale <= 1.2
        np.testing.assert_allclose(model.body_mass, original_mass * scale, rtol=1e-6)


def test_gravity_within_range(model, randomizer):
    rng = np.random.default_rng(0)
    for _ in range(50):
        sampled = randomizer.randomize(rng)
        gz = sampled["gravity_z"]
        assert -10.3 <= gz <= -9.5
        assert model.opt.gravity[2] == pytest.approx(gz)


def test_actuator_delay_buffers_actions(randomizer):
    rng = np.random.default_rng(42)
    randomizer.config.actuator_delay_steps = DRParamRange(True, 2, 2)
    randomizer.randomize(rng)

    action1 = np.ones(8) * 0.5
    action2 = np.ones(8) * 1.0

    out1 = randomizer.apply_action_delay(action1)
    np.testing.assert_array_equal(out1, np.zeros(8))

    out2 = randomizer.apply_action_delay(action2)
    np.testing.assert_array_equal(out2, np.zeros(8))

    out3 = randomizer.apply_action_delay(np.zeros(8))
    np.testing.assert_array_equal(out3, action1)


def test_obs_noise_changes_observation(randomizer):
    rng = np.random.default_rng(42)
    randomizer.config.obs_noise_std = DRParamRange(True, 0.01, 0.01)
    randomizer.randomize(rng)

    obs = np.zeros(32)
    noisy = randomizer.apply_obs_noise(obs, rng)
    assert not np.allclose(noisy, obs)


def test_restore_defaults_resets_model(model, randomizer):
    rng = np.random.default_rng(0)
    original_mass = model.body_mass.copy()
    original_gravity = model.opt.gravity.copy()

    randomizer.randomize(rng)
    assert not np.allclose(model.body_mass, original_mass)

    randomizer.restore_defaults()
    np.testing.assert_array_equal(model.body_mass, original_mass)
    np.testing.assert_array_equal(model.opt.gravity, original_gravity)


def test_disabled_params_not_changed(model):
    cfg = DomainRandConfig(
        body_mass=DRParamRange(False, 0.5, 2.0),
        ground_friction=DRParamRange(False, 0.1, 3.0),
        actuator_delay_steps=DRParamRange(False, 0, 10),
        joint_damping=DRParamRange(False, 0.1, 3.0),
        obs_noise_std=DRParamRange(False, 0.0, 1.0),
        gravity=DRParamRange(False, -20.0, -5.0),
        actuator_gain=DRParamRange(False, 0.1, 3.0),
        terrain_roughness=DRParamRange(False, 0.0, 0.1),
    )
    randomizer = DomainRandomizer(model, cfg)

    original_mass = model.body_mass.copy()
    original_gravity = model.opt.gravity.copy()

    rng = np.random.default_rng(0)
    randomizer.randomize(rng)

    np.testing.assert_array_equal(model.body_mass, original_mass)
    np.testing.assert_array_equal(model.opt.gravity, original_gravity)
