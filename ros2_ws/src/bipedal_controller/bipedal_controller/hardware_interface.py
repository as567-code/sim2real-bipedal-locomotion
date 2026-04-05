"""Hardware abstraction for joint command/state topics."""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


JOINT_NAMES = [
    "left_hip_flexion",
    "left_hip_abduction",
    "left_knee",
    "left_ankle",
    "right_hip_flexion",
    "right_hip_abduction",
    "right_knee",
    "right_ankle",
]

TORQUE_LIMITS = np.array([80.0, 60.0, 80.0, 40.0, 80.0, 60.0, 80.0, 40.0])


class HardwareInterface:
    """Manages ROS2 subscriptions/publications for joint states and commands.

    Subscribes to /joint_states for sensor feedback and publishes
    JointTrajectory commands to /joint_commands.
    """

    def __init__(self, node: Node):
        self._node = node
        self._joint_positions = np.zeros(8, dtype=np.float64)
        self._joint_velocities = np.zeros(8, dtype=np.float64)
        self._joint_efforts = np.zeros(8, dtype=np.float64)
        self._last_stamp = None
        self._state_received = False

        self._joint_index_map: dict[str, int] = {
            name: i for i, name in enumerate(JOINT_NAMES)
        }

        self._state_sub = node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
        )

        self._cmd_pub = node.create_publisher(
            JointTrajectory,
            "/joint_commands",
            10,
        )

    @property
    def state_received(self) -> bool:
        return self._state_received

    @property
    def joint_positions(self) -> np.ndarray:
        return self._joint_positions.copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        return self._joint_velocities.copy()

    def _joint_state_callback(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name in self._joint_index_map:
                idx = self._joint_index_map[name]
                if i < len(msg.position):
                    self._joint_positions[idx] = msg.position[i]
                if i < len(msg.velocity):
                    self._joint_velocities[idx] = msg.velocity[i]
                if i < len(msg.effort):
                    self._joint_efforts[idx] = msg.effort[i]
        self._last_stamp = msg.header.stamp
        self._state_received = True

    def send_joint_commands(self, actions: np.ndarray) -> None:
        """Convert normalized [-1, 1] actions to torque commands and publish.

        Applies torque limits and safety clipping before publishing.
        """
        torques = np.clip(actions, -1.0, 1.0) * TORQUE_LIMITS

        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.effort = torques.tolist()
        point.time_from_start = Duration(sec=0, nanosec=20_000_000)
        msg.points = [point]

        self._cmd_pub.publish(msg)

    def build_observation(
        self,
        torso_quaternion: np.ndarray,
        torso_angular_velocity: np.ndarray,
        torso_linear_velocity: np.ndarray,
        foot_contacts: np.ndarray,
        phase_clock: np.ndarray,
        prev_action: np.ndarray,
    ) -> np.ndarray:
        """Assemble the 32-dim observation vector from hardware sensor data."""
        return np.concatenate([
            self._joint_positions,       # 8
            self._joint_velocities,      # 8
            torso_quaternion,            # 4
            torso_angular_velocity,      # 3
            torso_linear_velocity,       # 3
            foot_contacts,               # 2
            phase_clock,                 # 2
            prev_action[:2],             # 2
        ]).astype(np.float32)
