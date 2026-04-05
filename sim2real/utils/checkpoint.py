from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sim2real.algo.actor_critic import ActorCritic
from sim2real.algo.normalizer import RunningMeanStd


def save_checkpoint(
    path: str | Path,
    policy: ActorCritic,
    normalizer: RunningMeanStd,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    extra: dict[str, Any] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy_state_dict": policy.state_dict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    policy: ActorCritic,
    normalizer: RunningMeanStd,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> int:
    """Load checkpoint and return the iteration number."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy_state_dict"])
    normalizer.load_state_dict(ckpt["normalizer_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("iteration", 0)
