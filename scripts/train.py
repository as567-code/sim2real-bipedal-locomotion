#!/usr/bin/env python3
"""Main training entry point for bipedal locomotion PPO."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2real.algo.actor_critic import ActorCritic
from sim2real.algo.normalizer import RunningMeanStd
from sim2real.algo.ppo import PPOTrainer
from sim2real.algo.rollout_buffer import RolloutBuffer
from sim2real.envs.bipedal_env import BipedalEnv
from sim2real.envs.domain_randomization import DomainRandConfig
from sim2real.utils.checkpoint import save_checkpoint, load_checkpoint
from sim2real.utils.config import load_train_config, TrainConfig
from sim2real.utils.logger import Logger


def make_env(cfg: TrainConfig, rank: int, eval_mode: bool = False):
    """Factory for creating a single env instance (used by SubprocVecEnv or sync wrapper)."""
    def _init():
        env = BipedalEnv(
            xml_path=cfg.env.xml_path,
            cmd_velocity=cfg.env.cmd_velocity,
            max_episode_steps=cfg.env.max_episode_steps,
            reward_weights=cfg.env.reward_weights,
            domain_rand_config=cfg.domain_rand if not eval_mode else None,
            enable_domain_rand=cfg.env.enable_domain_rand and not eval_mode,
            terrain_type=cfg.env.terrain_type,
            seed=cfg.seed + rank,
        )
        return env
    return _init


class SyncVectorEnv:
    """Minimal synchronous vectorized env wrapper (avoids subprocess overhead for debugging)."""

    def __init__(self, env_fns: list):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, **kwargs):
        obs_list, info_list = [], []
        for env in self.envs:
            o, i = env.reset(**kwargs)
            obs_list.append(o)
            info_list.append(i)
        return np.stack(obs_list), info_list

    def step(self, actions: np.ndarray):
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            if term or trunc:
                final_info = dict(info)
                o, reset_info = env.reset()
                info = reset_info
                info["episode_length"] = final_info.get("episode_length", 0)
                info["is_success"] = final_info.get("is_success", False)
            obs_list.append(o)
            rew_list.append(r)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(term_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            info_list,
        )

    def close(self):
        for env in self.envs:
            env.close()


def evaluate_policy(
    cfg: TrainConfig, policy: ActorCritic, normalizer: RunningMeanStd, device: str
) -> dict[str, float]:
    """Run evaluation episodes without domain randomization."""
    env = BipedalEnv(
        xml_path=cfg.env.xml_path,
        cmd_velocity=cfg.env.cmd_velocity,
        max_episode_steps=cfg.env.max_episode_steps,
        enable_domain_rand=False,
        seed=cfg.seed + 99999,
    )

    returns, lengths, successes = [], [], []
    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=cfg.seed + 99999 + ep)
        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_norm = normalizer.normalize(obs_t)
            with torch.no_grad():
                action, _, _ = policy.act(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            ep_return += reward
            steps += 1
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(steps)
        successes.append(not terminated)

    env.close()
    return {
        "eval/mean_return": float(np.mean(returns)),
        "eval/mean_length": float(np.mean(lengths)),
        "eval/success_rate": float(np.mean(successes)),
    }


def get_device(device_str: str) -> str:
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def main():
    parser = argparse.ArgumentParser(description="Train bipedal walking policy with PPO")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--env-config", type=str, default="configs/env.yaml")
    parser.add_argument("--dr-config", type=str, default="configs/domain_rand.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_train_config(args.config, args.env_config, args.dr_config)
    if args.device:
        cfg.device = args.device
    if args.num_envs:
        cfg.num_envs = args.num_envs

    device = get_device(cfg.device)
    print(f"Using device: {device}")
    print(f"Training for {cfg.ppo.num_iterations} iterations with {cfg.num_envs} envs")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env_fns = [make_env(cfg, rank=i) for i in range(cfg.num_envs)]
    envs = SyncVectorEnv(env_fns)

    policy = ActorCritic(
        obs_dim=cfg.env.obs_dim,
        act_dim=cfg.env.act_dim,
        hidden_dims=cfg.policy.hidden_dims,
        init_log_std=cfg.policy.init_log_std,
    )
    obs_normalizer = RunningMeanStd(shape=(cfg.env.obs_dim,))

    trainer = PPOTrainer(policy, obs_normalizer, config=cfg.ppo, device=device)

    start_iter = 0
    if args.resume:
        # Checkpoint stores the train-loop index that just finished. train_on_rollout
        # increments trainer.iteration each step, so after completing loop iter K,
        # trainer.iteration == K + 1. Resume at loop K+1 with matching schedule state.
        finished_iter = load_checkpoint(
            args.resume, policy, obs_normalizer, trainer.optimizer, device
        )
        start_iter = finished_iter + 1
        trainer.iteration = finished_iter + 1
        print(
            f"Resumed after completing iteration {finished_iter}; "
            f"continuing from iteration {start_iter}"
        )

    buffer = RolloutBuffer(
        num_envs=cfg.num_envs,
        rollout_length=cfg.rollout_length,
        obs_dim=cfg.env.obs_dim,
        act_dim=cfg.env.act_dim,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        device=device,
    )

    logger = Logger(cfg.log_dir, run_name=f"ppo_biped_{cfg.seed}")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    obs, _ = envs.reset()
    best_eval_return = -float("inf")

    pbar = tqdm(range(start_iter, cfg.ppo.num_iterations), desc="Training", unit="iter")
    for iteration in pbar:
        t0 = time.time()

        lr = trainer.update_lr(iteration)
        obs, ep_info = trainer.collect_rollout(envs, buffer, obs)
        train_info = trainer.train_on_rollout(buffer)

        elapsed = time.time() - t0
        total_steps = (iteration + 1) * cfg.num_envs * cfg.rollout_length
        fps = cfg.num_envs * cfg.rollout_length / elapsed

        if iteration % cfg.log_interval == 0:
            metrics = {
                **train_info,
                "episode/mean_return": ep_info["mean_return"],
                "episode/mean_length": ep_info["mean_length"],
                "episode/success_rate": ep_info["success_rate"],
                "episode/num_episodes": ep_info["num_episodes"],
                "perf/fps": fps,
                "perf/total_steps": total_steps,
            }
            logger.log_scalars(metrics, step=iteration, prefix="train")
            pbar.set_postfix(
                ret=f"{ep_info['mean_return']:.1f}",
                succ=f"{ep_info['success_rate']:.2f}",
                fps=f"{fps:.0f}",
            )

        if iteration % cfg.checkpoint_interval == 0 and iteration > 0:
            ckpt_path = Path(cfg.checkpoint_dir) / f"iter_{iteration:06d}.pt"
            save_checkpoint(ckpt_path, policy, obs_normalizer, trainer.optimizer, iteration)

        if iteration % cfg.eval_interval == 0:
            eval_metrics = evaluate_policy(cfg, policy, obs_normalizer, device)
            logger.log_scalars(eval_metrics, step=iteration, prefix="")
            print(f"\n[Eval iter {iteration}] return={eval_metrics['eval/mean_return']:.1f} "
                  f"success={eval_metrics['eval/success_rate']:.2%}")

            if eval_metrics["eval/mean_return"] > best_eval_return:
                best_eval_return = eval_metrics["eval/mean_return"]
                best_path = Path(cfg.checkpoint_dir) / "best.pt"
                save_checkpoint(best_path, policy, obs_normalizer, trainer.optimizer, iteration)

    final_path = Path(cfg.checkpoint_dir) / "final.pt"
    save_checkpoint(final_path, policy, obs_normalizer, trainer.optimizer, cfg.ppo.num_iterations)

    envs.close()
    logger.close()
    print(f"\nTraining complete. Best eval return: {best_eval_return:.1f}")


if __name__ == "__main__":
    main()
