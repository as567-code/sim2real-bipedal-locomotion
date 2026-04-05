from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _ortho_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, final_gain: float = 1.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        linear = nn.Linear(prev, h)
        _ortho_init(linear)
        layers += [linear, nn.ELU()]
        prev = h
    final = nn.Linear(prev, output_dim)
    _ortho_init(final, gain=final_gain)
    layers.append(final)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate MLP actor-critic with diagonal Gaussian policy.

    Actor: obs -> 3x256 + ELU -> action_mean
    Critic: obs -> 3x256 + ELU -> value scalar
    log_std is a learnable parameter (not state-dependent).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] | None = None,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 256]

        self.actor_mean = _build_mlp(obs_dim, hidden_dims, act_dim, final_gain=0.01)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        self.critic = _build_mlp(obs_dim, hidden_dims, 1, final_gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_mean, value)."""
        return self.actor_mean(obs), self.critic(obs).squeeze(-1)

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log_prob, entropy, and value for given obs-action pairs."""
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, return (action, log_prob, value)."""
        dist = self.get_distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
