from __future__ import annotations

import torch
import numpy as np


class RolloutBuffer:
    """Fixed-size buffer for on-policy rollout data across N parallel envs.

    Stores observations, actions, rewards, dones, log_probs, and values
    for T timesteps x N envs. Computes GAE-lambda returns after collection.
    """

    def __init__(
        self,
        num_envs: int,
        rollout_length: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.obs = np.zeros((rollout_length, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, num_envs, act_dim), dtype=np.float32)
        self.rewards = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_length, num_envs), dtype=np.float32)

        self.advantages = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_length, num_envs), dtype=np.float32)

        self._ptr = 0

    def insert(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = actions
        self.rewards[self._ptr] = rewards
        self.dones[self._ptr] = dones
        self.log_probs[self._ptr] = log_probs
        self.values[self._ptr] = values
        self._ptr += 1

    def compute_returns(self, last_values: np.ndarray, last_dones: np.ndarray) -> None:
        """Compute GAE-lambda advantages and discounted returns."""
        gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_values * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_minibatches(self, minibatch_size: int, rng: np.random.Generator) -> list[dict[str, torch.Tensor]]:
        """Flatten rollout data and yield shuffled minibatches as tensors."""
        total = self.rollout_length * self.num_envs
        flat_obs = self.obs.reshape(total, -1)
        flat_actions = self.actions.reshape(total, -1)
        flat_log_probs = self.log_probs.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_values = self.values.reshape(total)

        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        indices = rng.permutation(total)
        batches = []
        for start in range(0, total, minibatch_size):
            end = start + minibatch_size
            if end > total:
                break
            mb_idx = indices[start:end]
            batches.append({
                "obs": torch.as_tensor(flat_obs[mb_idx], dtype=torch.float32, device=self.device),
                "actions": torch.as_tensor(flat_actions[mb_idx], dtype=torch.float32, device=self.device),
                "old_log_probs": torch.as_tensor(flat_log_probs[mb_idx], dtype=torch.float32, device=self.device),
                "returns": torch.as_tensor(flat_returns[mb_idx], dtype=torch.float32, device=self.device),
                "advantages": torch.as_tensor(flat_advantages[mb_idx], dtype=torch.float32, device=self.device),
                "old_values": torch.as_tensor(flat_values[mb_idx], dtype=torch.float32, device=self.device),
            })
        return batches

    def reset(self) -> None:
        self._ptr = 0
