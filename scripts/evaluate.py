#!/usr/bin/env python3
"""Run evaluation episodes for a trained bipedal policy and produce metrics + plots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2real.algo.actor_critic import ActorCritic
from sim2real.algo.normalizer import RunningMeanStd
from sim2real.envs.bipedal_env import BipedalEnv
from sim2real.utils.checkpoint import load_checkpoint


def run_episode(
    env: BipedalEnv,
    policy: ActorCritic,
    normalizer: RunningMeanStd,
    device: str,
    seed: int,
    record_frames: bool = False,
) -> dict:
    obs, _ = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    steps = 0
    reward_components_agg: dict[str, list[float]] = {}
    frames = []

    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        obs_norm = normalizer.normalize(obs_t)
        with torch.no_grad():
            action, _, _ = policy.act(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        ep_return += reward
        steps += 1
        done = terminated or truncated

        if "reward_components" in info:
            for k, v in info["reward_components"].items():
                reward_components_agg.setdefault(k, []).append(v)

        if record_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    return {
        "return": ep_return,
        "length": steps,
        "success": not terminated,
        "reward_components": {k: np.mean(v) for k, v in reward_components_agg.items()},
        "frames": frames,
    }


def plot_results(results: list[dict], out_path: Path) -> None:
    returns = [r["return"] for r in results]
    lengths = [r["length"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(range(len(returns)), returns, color="steelblue")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].set_title(f"Episode Returns (mean={np.mean(returns):.1f})")
    axes[0].axhline(np.mean(returns), color="red", linestyle="--", linewidth=1)

    axes[1].bar(range(len(lengths)), lengths, color="seagreen")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title(f"Episode Lengths (mean={np.mean(lengths):.0f})")

    if results[0]["reward_components"]:
        comp_names = list(results[0]["reward_components"].keys())
        comp_means = [np.mean([r["reward_components"][c] for r in results]) for c in comp_names]
        axes[2].barh(comp_names, comp_means, color="coral")
        axes[2].set_xlabel("Mean Value")
        axes[2].set_title("Reward Components")
    else:
        successes = [r["success"] for r in results]
        axes[2].text(0.5, 0.5, f"Success: {np.mean(successes):.0%}", ha="center", va="center",
                     fontsize=18, transform=axes[2].transAxes)
        axes[2].set_title("Success Rate")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for trained bipedal policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--render", action="store_true", help="Render to rgb_array and save plot")
    parser.add_argument("--output", type=str, default="eval_results.png")
    parser.add_argument("--cmd-velocity", type=float, default=0.5)
    args = parser.parse_args()

    policy = ActorCritic(obs_dim=32, act_dim=8)
    normalizer = RunningMeanStd(shape=(32,))
    load_checkpoint(args.checkpoint, policy, normalizer, device=args.device)
    policy.train(False)

    env = BipedalEnv(
        cmd_velocity=args.cmd_velocity,
        enable_domain_rand=False,
        render_mode="rgb_array" if args.render else None,
    )

    results = []
    for ep in range(args.episodes):
        result = run_episode(env, policy, normalizer, args.device, seed=args.seed + ep, record_frames=False)
        results.append(result)
        status = "OK" if result["success"] else "FALL"
        print(f"Episode {ep:3d}: return={result['return']:7.1f}  length={result['length']:4d}  [{status}]")

    env.close()

    returns = [r["return"] for r in results]
    successes = [r["success"] for r in results]
    print(f"\n--- Summary ({args.episodes} episodes) ---")
    print(f"  Mean return:   {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"  Success rate:  {np.mean(successes):.1%}")
    print(f"  Mean length:   {np.mean([r['length'] for r in results]):.0f}")

    plot_results(results, Path(args.output))


if __name__ == "__main__":
    main()
