from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from sim2real.algo.actor_critic import ActorCritic
from sim2real.algo.rollout_buffer import RolloutBuffer
from sim2real.algo.normalizer import RunningMeanStd


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs_per_rollout: int = 5
    minibatch_size: int = 4096
    target_kl: float | None = 0.02
    lr_schedule: str = "linear"
    num_iterations: int = 12000

    entropy_coef_final: float = 0.001


class PPOTrainer:
    """Proximal Policy Optimization with clipped surrogate, GAE, and LR annealing."""

    def __init__(
        self,
        policy: ActorCritic,
        obs_normalizer: RunningMeanStd,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        # Normalizer stays on CPU (Welford needs float64, unsupported on MPS)
        self.obs_normalizer = obs_normalizer
        self.config = config or PPOConfig()
        self.device = device

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr, eps=1e-5)
        self._initial_lr = self.config.lr
        self._rng = np.random.default_rng()

        self.iteration = 0

    def update_lr(self, iteration: int) -> float:
        """Linear or constant LR schedule."""
        if self.config.lr_schedule == "linear":
            frac = 1.0 - iteration / self.config.num_iterations
            lr = self._initial_lr * max(frac, 0.0)
        else:
            lr = self._initial_lr

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _entropy_coef(self) -> float:
        """Linearly anneal entropy coefficient."""
        frac = min(self.iteration / self.config.num_iterations, 1.0)
        return self.config.entropy_coef + frac * (
            self.config.entropy_coef_final - self.config.entropy_coef
        )

    @torch.no_grad()
    def collect_rollout(
        self,
        envs,
        buffer: RolloutBuffer,
        obs: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Collect a full rollout of T steps across N envs.

        Args:
            envs: vectorized env with step()/reset() returning batched arrays
            buffer: rollout buffer to fill
            obs: current observations (num_envs, obs_dim)

        Returns:
            next_obs, episode_infos dict with aggregated stats
        """
        buffer.reset()
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        successes: list[bool] = []

        for _ in range(buffer.rollout_length):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_norm = self.obs_normalizer.normalize(obs_t)

            action, log_prob, value = self.policy.act(obs_norm)
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()

            next_obs, rewards, terminateds, truncateds, infos = envs.step(action_np)
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)

            buffer.insert(obs, action_np, rewards, dones, log_prob_np, value_np)

            self.obs_normalizer.update(obs_t)

            if "final_info" in infos:
                for fi in infos["final_info"]:
                    if fi is not None and "episode_length" in fi:
                        episode_lengths.append(fi["episode_length"])
                        if "episode" in fi:
                            episode_returns.append(fi["episode"]["r"])
                        if "is_success" in fi:
                            successes.append(fi["is_success"])
            else:
                for i in range(len(dones)):
                    if dones[i]:
                        info_i = infos[i] if isinstance(infos, list) else {}
                        if isinstance(infos, dict):
                            for key in ("episode_length", "is_success"):
                                if key in infos and hasattr(infos[key], "__getitem__"):
                                    info_i[key] = infos[key][i]
                        if "episode_length" in info_i:
                            episode_lengths.append(info_i["episode_length"])
                        if "is_success" in info_i:
                            successes.append(info_i["is_success"])

            obs = next_obs

        last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        last_obs_norm = self.obs_normalizer.normalize(last_obs_t)
        last_values = self.policy.get_value(last_obs_norm).cpu().numpy()
        last_dones = dones

        buffer.compute_returns(last_values, last_dones)

        ep_info = {
            "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "success_rate": np.mean(successes) if successes else 0.0,
            "num_episodes": len(episode_lengths),
        }
        return obs, ep_info

    def train_on_rollout(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update epochs on the collected rollout."""
        cfg = self.config
        entropy_coef = self._entropy_coef()

        all_policy_loss = []
        all_value_loss = []
        all_entropy = []
        all_kl = []

        for _epoch in range(cfg.epochs_per_rollout):
            batches = buffer.get_minibatches(cfg.minibatch_size, self._rng)

            for mb in batches:
                obs_norm = self.obs_normalizer.normalize(mb["obs"])
                log_prob, entropy, values = self.policy.evaluate_actions(obs_norm, mb["actions"])

                ratio = (log_prob - mb["old_log_probs"]).exp()
                adv = mb["advantages"]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if cfg.value_clip > 0:
                    values_clipped = mb["old_values"] + torch.clamp(
                        values - mb["old_values"], -cfg.value_clip, cfg.value_clip
                    )
                    vf_loss1 = (values - mb["returns"]) ** 2
                    vf_loss2 = (values_clipped - mb["returns"]) ** 2
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * ((values - mb["returns"]) ** 2).mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss + entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb["old_log_probs"] - log_prob).mean().item()

                all_policy_loss.append(policy_loss.item())
                all_value_loss.append(value_loss.item())
                all_entropy.append(-entropy_loss.item())
                all_kl.append(approx_kl)

            if cfg.target_kl is not None and np.mean(all_kl) > cfg.target_kl:
                break

        self.iteration += 1

        return {
            "policy_loss": float(np.mean(all_policy_loss)),
            "value_loss": float(np.mean(all_value_loss)),
            "entropy": float(np.mean(all_entropy)),
            "approx_kl": float(np.mean(all_kl)),
            "entropy_coef": entropy_coef,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
