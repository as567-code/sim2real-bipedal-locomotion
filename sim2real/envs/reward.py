from __future__ import annotations

import numpy as np


class RewardComputer:
    """AMP-inspired decomposed reward for bipedal locomotion.

    Each component returns a scalar reward; the final reward is a weighted sum.
    Weights are loaded from the env config dict at construction time.
    """

    DEFAULT_WEIGHTS = {
        "velocity_tracking": 1.5,
        "orientation": 0.8,
        "energy": 1.0,
        "torque_stability": 0.5,
        "joint_velocity": 0.3,
        "foot_clearance": 0.4,
        "alive": 1.0,
        "symmetry": 0.3,
    }

    def __init__(self, weights: dict[str, float] | None = None, cmd_velocity: float = 0.5):
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.cmd_velocity = cmd_velocity
        self._prev_torques: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_torques = None

    def compute(
        self,
        torso_velocity: np.ndarray,
        torso_orientation_rpy: np.ndarray,
        joint_torques: np.ndarray,
        joint_velocities: np.ndarray,
        left_foot_height: float,
        right_foot_height: float,
        left_foot_contact: bool,
        right_foot_contact: bool,
        left_joint_positions: np.ndarray,
        right_joint_positions: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Compute total reward and per-component breakdown."""
        components: dict[str, float] = {}

        vx = torso_velocity[0]
        components["velocity_tracking"] = float(np.exp(-4.0 * (vx - self.cmd_velocity) ** 2))

        pitch, roll = torso_orientation_rpy[1], torso_orientation_rpy[0]
        components["orientation"] = float(np.exp(-3.0 * (pitch ** 2 + roll ** 2)))

        components["energy"] = float(-0.001 * np.sum(joint_torques ** 2))

        if self._prev_torques is not None:
            torque_diff = joint_torques - self._prev_torques
            components["torque_stability"] = float(-0.01 * np.sum(torque_diff ** 2))
        else:
            components["torque_stability"] = 0.0
        self._prev_torques = joint_torques.copy()

        components["joint_velocity"] = float(-0.001 * np.sum(joint_velocities ** 2))

        swing_clearance = 0.0
        if not left_foot_contact:
            swing_clearance += max(0.0, left_foot_height - 0.02)
        if not right_foot_contact:
            swing_clearance += max(0.0, right_foot_height - 0.02)
        components["foot_clearance"] = float(swing_clearance)

        components["alive"] = 1.0

        mirror_diff = left_joint_positions - right_joint_positions
        components["symmetry"] = float(-0.5 * np.sum(mirror_diff ** 2))

        total = sum(self.weights[k] * v for k, v in components.items())
        return total, components
