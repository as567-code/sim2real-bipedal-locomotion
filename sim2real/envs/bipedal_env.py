from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from sim2real.envs.reward import RewardComputer
from sim2real.envs.domain_randomization import DomainRandomizer, DomainRandConfig
from sim2real.envs.terrain import TerrainGenerator

_ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
_DEFAULT_XML = _ASSET_DIR / "biped.xml"

JOINT_NAMES = [
    "left_hip_flexion", "left_hip_abduction", "left_knee", "left_ankle",
    "right_hip_flexion", "right_hip_abduction", "right_knee", "right_ankle",
]

ACTUATOR_NAMES = [n + "_motor" for n in JOINT_NAMES]


class BipedalEnv(gym.Env):
    """Gymnasium environment for a 3D bipedal robot in MuJoCo.

    Observation (dim=32):
        joint_pos(8) + joint_vel(8) + quat(4) + ang_vel(3) + lin_vel(3) +
        foot_contact(2) + phase_clock(2) + prev_action(2, hip_flexion only)

    Action (dim=8): normalized [-1, 1] mapped to actuator torques.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    OBS_DIM = 32
    ACT_DIM = 8

    def __init__(
        self,
        xml_path: str | Path | None = None,
        render_mode: str | None = None,
        cmd_velocity: float = 0.5,
        max_episode_steps: int = 1000,
        reward_weights: dict[str, float] | None = None,
        domain_rand_config: DomainRandConfig | None = None,
        enable_domain_rand: bool = True,
        terrain_type: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        xml_path = str(xml_path or _DEFAULT_XML)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None

        self.cmd_velocity = cmd_velocity
        self.max_episode_steps = max_episode_steps

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.ACT_DIM,), dtype=np.float64,
        )

        self.reward_computer = RewardComputer(weights=reward_weights, cmd_velocity=cmd_velocity)
        self.enable_domain_rand = enable_domain_rand
        self.domain_randomizer = DomainRandomizer(self.model, domain_rand_config)
        self.terrain_generator = TerrainGenerator(self.model)

        self._terrain_type = terrain_type
        self._step_count = 0
        self._phase = 0.0
        self._phase_freq = 1.25
        self._prev_action = np.zeros(2)

        self._rng = np.random.default_rng(seed)

        self._joint_qpos_adr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES
        ])
        self._joint_dof_adr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES
        ])

        self._left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self._right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        self._left_foot_contact_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_contact"
        )
        self._right_foot_contact_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_contact"
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        if self.enable_domain_rand:
            dr_info = self.domain_randomizer.randomize(self._rng)
        else:
            self.domain_randomizer.restore_defaults()
            dr_info = {}

        self.terrain_generator.generate(self._terrain_type, self._rng)

        qpos_init = self.data.qpos.copy()
        qpos_init[2] = 0.85
        qpos_init[3:7] = [1, 0, 0, 0]
        self.data.qpos[:] = qpos_init
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._phase = 0.0
        self._prev_action = np.zeros(2)
        self.reward_computer.reset()

        obs = self._get_obs()
        info = {"domain_rand": dr_info}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0)

        if self.enable_domain_rand:
            delayed_action = self.domain_randomizer.apply_action_delay(action)
        else:
            delayed_action = action

        self.data.ctrl[:] = delayed_action
        mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        dt = self.model.opt.timestep
        self._phase += 2.0 * np.pi * self._phase_freq * dt

        obs = self._get_obs()
        if self.enable_domain_rand:
            obs = self.domain_randomizer.apply_obs_noise(obs, self._rng)

        reward, reward_info = self._compute_reward(action)

        terminated = self._check_termination()
        truncated = self._step_count >= self.max_episode_steps

        self._prev_action = action[:2].copy()

        info: dict[str, Any] = {
            "reward_components": reward_info,
            "step": self._step_count,
        }
        if terminated or truncated:
            info["episode_length"] = self._step_count
            info["is_success"] = not terminated and self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self._joint_qpos_adr]
        joint_vel = self.data.qvel[self._joint_dof_adr]

        torso_quat = self.data.xquat[self._torso_id]

        torso_ang_vel = self.data.cvel[self._torso_id, 3:]
        torso_lin_vel = self.data.cvel[self._torso_id, :3]

        left_contact = float(self._foot_in_contact(self._left_foot_id))
        right_contact = float(self._foot_in_contact(self._right_foot_id))

        phase_sin = np.sin(self._phase)
        phase_cos = np.cos(self._phase)

        obs = np.concatenate([
            joint_pos,              # 8
            joint_vel,              # 8
            torso_quat,             # 4
            torso_ang_vel,          # 3
            torso_lin_vel,          # 3
            [left_contact, right_contact],  # 2
            [phase_sin, phase_cos],         # 2
            self._prev_action,              # 2
        ])
        return obs

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        torso_vel = self.data.cvel[self._torso_id, :3]

        quat = self.data.xquat[self._torso_id]
        rpy = self._quat_to_rpy(quat)

        joint_torques = self.data.qfrc_actuator[self._joint_dof_adr]
        joint_vel = self.data.qvel[self._joint_dof_adr]

        left_foot_h = float(self.data.xpos[self._left_foot_id][2])
        right_foot_h = float(self.data.xpos[self._right_foot_id][2])

        left_contact = self._foot_in_contact(self._left_foot_id)
        right_contact = self._foot_in_contact(self._right_foot_id)

        joint_pos = self.data.qpos[self._joint_qpos_adr]
        left_joints = joint_pos[:4]
        right_joints = joint_pos[4:]

        return self.reward_computer.compute(
            torso_velocity=torso_vel,
            torso_orientation_rpy=rpy,
            joint_torques=joint_torques,
            joint_velocities=joint_vel,
            left_foot_height=left_foot_h,
            right_foot_height=right_foot_h,
            left_foot_contact=left_contact,
            right_foot_contact=right_contact,
            left_joint_positions=left_joints,
            right_joint_positions=right_joints,
        )

    def _check_termination(self) -> bool:
        torso_z = self.data.xpos[self._torso_id][2]
        if torso_z < 0.4:
            return True

        quat = self.data.xquat[self._torso_id]
        rpy = self._quat_to_rpy(quat)
        if abs(rpy[1]) > 1.0 or abs(rpy[0]) > 0.8:
            return True

        return False

    def _foot_in_contact(self, body_id: int) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]
            if body_id in (geom1_body, geom2_body):
                return True
        return False

    @staticmethod
    def _quat_to_rpy(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to roll-pitch-yaw."""
        w, x, y, z = quat
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
