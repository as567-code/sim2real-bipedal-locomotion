from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any

import numpy as np
import mujoco


@dataclass
class DRParamRange:
    enabled: bool = True
    low: float = 0.0
    high: float = 1.0


@dataclass
class DomainRandConfig:
    body_mass: DRParamRange = field(default_factory=lambda: DRParamRange(True, 0.8, 1.2))
    ground_friction: DRParamRange = field(default_factory=lambda: DRParamRange(True, 0.4, 1.5))
    actuator_delay_steps: DRParamRange = field(default_factory=lambda: DRParamRange(True, 0, 4))
    joint_damping: DRParamRange = field(default_factory=lambda: DRParamRange(True, 0.7, 1.3))
    obs_noise_std: DRParamRange = field(default_factory=lambda: DRParamRange(True, 0.0, 0.02))
    gravity: DRParamRange = field(default_factory=lambda: DRParamRange(True, -10.3, -9.5))
    actuator_gain: DRParamRange = field(default_factory=lambda: DRParamRange(True, 0.85, 1.15))
    terrain_roughness: DRParamRange = field(default_factory=lambda: DRParamRange(True, 0.0, 0.03))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DomainRandConfig":
        cfg = cls()
        for key, val in d.items():
            if hasattr(cfg, key) and isinstance(val, dict):
                setattr(cfg, key, DRParamRange(**val))
        return cfg


class DomainRandomizer:
    """Applies domain randomization over 8 physics parameters at episode reset.

    Stores original model values at init and applies uniform-sampled
    perturbations within configured bounds.
    """

    def __init__(self, model: mujoco.MjModel, config: DomainRandConfig | None = None):
        self.model = model
        self.config = config or DomainRandConfig()

        self._orig_body_mass = model.body_mass.copy()
        self._orig_geom_friction = model.geom_friction.copy()
        self._orig_dof_damping = model.dof_damping.copy()
        self._orig_gravity = model.opt.gravity.copy()
        self._orig_actuator_gainprm = model.actuator_gainprm.copy()

        self._action_buffer: deque[np.ndarray] = deque()
        self._delay_steps: int = 0
        self._obs_noise_std: float = 0.0

    def randomize(self, rng: np.random.Generator) -> dict[str, float]:
        """Randomize all enabled parameters. Returns sampled values for logging."""
        sampled: dict[str, float] = {}
        cfg = self.config

        if cfg.body_mass.enabled:
            scale = rng.uniform(cfg.body_mass.low, cfg.body_mass.high)
            self.model.body_mass[:] = self._orig_body_mass * scale
            sampled["body_mass_scale"] = scale

        if cfg.ground_friction.enabled:
            fric = rng.uniform(cfg.ground_friction.low, cfg.ground_friction.high)
            self.model.geom_friction[:, 0] = self._orig_geom_friction[:, 0] * (fric / 0.8)
            sampled["ground_friction"] = fric

        if cfg.actuator_delay_steps.enabled:
            self._delay_steps = int(rng.integers(
                int(cfg.actuator_delay_steps.low),
                int(cfg.actuator_delay_steps.high) + 1,
            ))
            self._action_buffer.clear()
            sampled["actuator_delay_steps"] = float(self._delay_steps)

        if cfg.joint_damping.enabled:
            scale = rng.uniform(cfg.joint_damping.low, cfg.joint_damping.high)
            self.model.dof_damping[:] = self._orig_dof_damping * scale
            sampled["joint_damping_scale"] = scale

        if cfg.obs_noise_std.enabled:
            self._obs_noise_std = rng.uniform(cfg.obs_noise_std.low, cfg.obs_noise_std.high)
            sampled["obs_noise_std"] = self._obs_noise_std

        if cfg.gravity.enabled:
            gz = rng.uniform(cfg.gravity.low, cfg.gravity.high)
            self.model.opt.gravity[2] = gz
            sampled["gravity_z"] = gz

        if cfg.actuator_gain.enabled:
            scale = rng.uniform(cfg.actuator_gain.low, cfg.actuator_gain.high)
            self.model.actuator_gainprm[:] = self._orig_actuator_gainprm * scale
            sampled["actuator_gain_scale"] = scale

        if cfg.terrain_roughness.enabled:
            sampled["terrain_roughness"] = rng.uniform(
                cfg.terrain_roughness.low, cfg.terrain_roughness.high
            )

        return sampled

    def apply_action_delay(self, action: np.ndarray) -> np.ndarray:
        """Buffer actions to simulate actuator communication delay."""
        if self._delay_steps == 0:
            return action
        self._action_buffer.append(action.copy())
        if len(self._action_buffer) > self._delay_steps:
            return self._action_buffer.popleft()
        return np.zeros_like(action)

    def apply_obs_noise(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self._obs_noise_std > 0:
            return obs + rng.normal(0.0, self._obs_noise_std, size=obs.shape)
        return obs

    def restore_defaults(self) -> None:
        """Reset model parameters to original values."""
        self.model.body_mass[:] = self._orig_body_mass
        self.model.geom_friction[:] = self._orig_geom_friction
        self.model.dof_damping[:] = self._orig_dof_damping
        self.model.opt.gravity[:] = self._orig_gravity
        self.model.actuator_gainprm[:] = self._orig_actuator_gainprm
        self._delay_steps = 0
        self._obs_noise_std = 0.0
        self._action_buffer.clear()
