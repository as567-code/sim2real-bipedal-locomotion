from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sim2real.algo.ppo import PPOConfig
from sim2real.envs.domain_randomization import DomainRandConfig


@dataclass
class PolicyConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    init_log_std: float = -0.5


@dataclass
class EnvConfig:
    xml_path: str = "assets/biped.xml"
    cmd_velocity: float = 0.5
    max_episode_steps: int = 1000
    enable_domain_rand: bool = True
    terrain_type: str | None = None
    reward_weights: dict[str, float] = field(default_factory=dict)
    obs_dim: int = 32
    act_dim: int = 8


@dataclass
class TrainConfig:
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)

    num_envs: int = 64
    rollout_length: int = 256

    checkpoint_interval: int = 100
    eval_interval: int = 200
    eval_episodes: int = 10
    log_interval: int = 10

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    seed: int = 42
    device: str = "auto"


def load_train_config(
    train_yaml: str | Path,
    env_yaml: str | Path | None = None,
    dr_yaml: str | Path | None = None,
) -> TrainConfig:
    """Load and merge YAML config files into a TrainConfig."""
    with open(train_yaml) as f:
        train_dict: dict[str, Any] = yaml.safe_load(f)

    env_dict: dict[str, Any] = {}
    if env_yaml:
        with open(env_yaml) as f:
            env_dict = yaml.safe_load(f) or {}

    dr_dict: dict[str, Any] = {}
    if dr_yaml:
        with open(dr_yaml) as f:
            dr_dict = yaml.safe_load(f) or {}

    ppo_cfg = PPOConfig()
    for k, v in train_dict.get("ppo", {}).items():
        if hasattr(ppo_cfg, k):
            setattr(ppo_cfg, k, v)

    policy_cfg = PolicyConfig()
    for k, v in train_dict.get("policy", {}).items():
        if hasattr(policy_cfg, k):
            setattr(policy_cfg, k, v)

    env_section = env_dict.get("env", {})
    env_cfg = EnvConfig()
    for k, v in env_section.items():
        if hasattr(env_cfg, k):
            setattr(env_cfg, k, v)
    env_cfg.reward_weights = env_dict.get("reward_weights", {})
    if "obs_dim" in env_dict:
        env_cfg.obs_dim = env_dict["obs_dim"]
    if "act_dim" in env_dict:
        env_cfg.act_dim = env_dict["act_dim"]

    domain_rand_cfg = DomainRandConfig.from_dict(dr_dict)

    cfg = TrainConfig(
        ppo=ppo_cfg,
        env=env_cfg,
        policy=policy_cfg,
        domain_rand=domain_rand_cfg,
        num_envs=train_dict.get("num_envs", 64),
        rollout_length=train_dict.get("rollout_length", 256),
        checkpoint_interval=train_dict.get("checkpoint_interval", 100),
        eval_interval=train_dict.get("eval_interval", 200),
        eval_episodes=train_dict.get("eval_episodes", 10),
        log_interval=train_dict.get("log_interval", 10),
        checkpoint_dir=train_dict.get("checkpoint_dir", "checkpoints"),
        log_dir=train_dict.get("log_dir", "runs"),
        seed=train_dict.get("seed", 42),
        device=train_dict.get("device", "auto"),
    )
    return cfg
