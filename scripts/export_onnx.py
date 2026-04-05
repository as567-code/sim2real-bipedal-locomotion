#!/usr/bin/env python3
"""CLI to export a trained policy checkpoint to ONNX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2real.algo.actor_critic import ActorCritic
from sim2real.algo.normalizer import RunningMeanStd
from sim2real.export.onnx_export import export_to_onnx
from sim2real.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Export trained policy to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="policy.onnx")
    parser.add_argument("--obs-dim", type=int, default=32)
    parser.add_argument("--act-dim", type=int, default=8)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    policy = ActorCritic(obs_dim=args.obs_dim, act_dim=args.act_dim)
    normalizer = RunningMeanStd(shape=(args.obs_dim,))
    iteration = load_checkpoint(args.checkpoint, policy, normalizer, device=args.device)
    print(f"Loaded checkpoint from iteration {iteration}")

    policy.train(False)

    onnx_path = export_to_onnx(
        policy=policy,
        normalizer=normalizer,
        output_path=args.output,
        obs_dim=args.obs_dim,
        opset_version=args.opset,
        validate=not args.no_validate,
    )
    print(f"\nDone. ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()
