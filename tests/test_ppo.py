"""Tests for PPO algorithm components."""
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2real.algo.actor_critic import ActorCritic
from sim2real.algo.normalizer import RunningMeanStd
from sim2real.algo.rollout_buffer import RolloutBuffer


OBS_DIM = 32
ACT_DIM = 8


@pytest.fixture
def policy():
    return ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM)


@pytest.fixture
def normalizer():
    return RunningMeanStd(shape=(OBS_DIM,))


def test_actor_critic_forward_shapes(policy):
    obs = torch.randn(16, OBS_DIM)
    mean, value = policy(obs)
    assert mean.shape == (16, ACT_DIM)
    assert value.shape == (16,)


def test_act_returns_correct_shapes(policy):
    obs = torch.randn(4, OBS_DIM)
    action, log_prob, value = policy.act(obs)
    assert action.shape == (4, ACT_DIM)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)


def test_act_deterministic_is_consistent(policy):
    obs = torch.randn(1, OBS_DIM)
    a1, _, _ = policy.act(obs, deterministic=True)
    a2, _, _ = policy.act(obs, deterministic=True)
    torch.testing.assert_close(a1, a2)


def test_evaluate_actions_shapes(policy):
    obs = torch.randn(32, OBS_DIM)
    actions = torch.randn(32, ACT_DIM)
    log_prob, entropy, value = policy.evaluate_actions(obs, actions)
    assert log_prob.shape == (32,)
    assert entropy.shape == (32,)
    assert value.shape == (32,)


def test_normalizer_update_changes_stats(normalizer):
    initial_mean = normalizer.mean.clone()
    batch = torch.randn(100, OBS_DIM) * 5 + 3
    normalizer.update(batch)
    assert not torch.allclose(normalizer.mean, initial_mean)
    assert normalizer.count.item() == 100


def test_normalizer_normalize_clips(normalizer):
    batch = torch.randn(1000, OBS_DIM)
    normalizer.update(batch)
    extreme = torch.ones(1, OBS_DIM) * 1000.0
    normed = normalizer.normalize(extreme)
    assert normed.max().item() <= normalizer.clip


def test_rollout_buffer_insert_and_returns():
    buf = RolloutBuffer(num_envs=4, rollout_length=8, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    rng = np.random.default_rng(0)

    for t in range(8):
        buf.insert(
            obs=rng.standard_normal((4, OBS_DIM)).astype(np.float32),
            actions=rng.standard_normal((4, ACT_DIM)).astype(np.float32),
            rewards=rng.standard_normal(4).astype(np.float32),
            dones=np.zeros(4, dtype=np.float32),
            log_probs=rng.standard_normal(4).astype(np.float32),
            values=rng.standard_normal(4).astype(np.float32),
        )

    last_vals = np.zeros(4, dtype=np.float32)
    last_dones = np.zeros(4, dtype=np.float32)
    buf.compute_returns(last_vals, last_dones)

    assert np.all(np.isfinite(buf.advantages))
    assert np.all(np.isfinite(buf.returns))


def test_rollout_buffer_minibatches():
    buf = RolloutBuffer(num_envs=4, rollout_length=16, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    rng = np.random.default_rng(0)

    for t in range(16):
        buf.insert(
            obs=rng.standard_normal((4, OBS_DIM)).astype(np.float32),
            actions=rng.standard_normal((4, ACT_DIM)).astype(np.float32),
            rewards=rng.standard_normal(4).astype(np.float32),
            dones=np.zeros(4, dtype=np.float32),
            log_probs=rng.standard_normal(4).astype(np.float32),
            values=rng.standard_normal(4).astype(np.float32),
        )

    buf.compute_returns(np.zeros(4), np.zeros(4))
    batches = buf.get_minibatches(minibatch_size=16, rng=rng)

    assert len(batches) > 0
    for mb in batches:
        assert mb["obs"].shape[0] == 16
        assert mb["actions"].shape == (16, ACT_DIM)
        assert mb["advantages"].shape == (16,)


def test_ppo_single_update_runs():
    """Verify a single PPO update cycle doesn't crash and produces finite loss."""
    from sim2real.algo.ppo import PPOTrainer, PPOConfig

    policy = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    normalizer = RunningMeanStd(shape=(OBS_DIM,))

    normalizer.update(torch.randn(100, OBS_DIM))

    cfg = PPOConfig(minibatch_size=32, epochs_per_rollout=2)
    trainer = PPOTrainer(policy, normalizer, config=cfg, device="cpu")

    buf = RolloutBuffer(num_envs=4, rollout_length=16, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    rng = np.random.default_rng(0)
    for t in range(16):
        obs = torch.randn(4, OBS_DIM)
        obs_norm = normalizer.normalize(obs)
        action, log_prob, value = policy.act(obs_norm)
        buf.insert(
            obs=obs.numpy(),
            actions=action.detach().numpy(),
            rewards=np.random.randn(4).astype(np.float32),
            dones=np.zeros(4, dtype=np.float32),
            log_probs=log_prob.detach().numpy(),
            values=value.detach().numpy(),
        )

    buf.compute_returns(np.zeros(4), np.zeros(4))
    info = trainer.train_on_rollout(buf)

    assert np.isfinite(info["policy_loss"])
    assert np.isfinite(info["value_loss"])
    assert info["entropy"] > 0
