#!/usr/bin/env python3
"""Visualize trained policy using MuJoCo's passive viewer."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2real.algo.actor_critic import ActorCritic
from sim2real.algo.normalizer import RunningMeanStd
from sim2real.envs.bipedal_env import BipedalEnv
from sim2real.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Visualize bipedal policy in MuJoCo viewer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cmd-velocity", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    policy = ActorCritic(obs_dim=32, act_dim=8)
    normalizer = RunningMeanStd(shape=(32,))
    load_checkpoint(args.checkpoint, policy, normalizer, device=args.device)
    policy.train(False)

    env = BipedalEnv(
        cmd_velocity=args.cmd_velocity,
        enable_domain_rand=False,
        seed=args.seed,
    )

    model = env.model
    data = env.data
    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            ep_return = 0.0
            steps = 0

            while not done and viewer.is_running():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
                obs_norm = normalizer.normalize(obs_t)
                with torch.no_grad():
                    action, _, _ = policy.act(obs_norm, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                ep_return += reward
                steps += 1
                done = terminated or truncated

                viewer.sync()
                time.sleep(dt / args.speed)

            if not viewer.is_running():
                break
            status = "OK" if not terminated else "FALL"
            print(f"Episode {ep}: return={ep_return:.1f}  length={steps}  [{status}]")
            time.sleep(0.5)

    env.close()


if __name__ == "__main__":
    main()
