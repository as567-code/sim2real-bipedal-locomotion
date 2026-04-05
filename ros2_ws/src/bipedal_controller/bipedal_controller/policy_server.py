"""ROS2 action server that runs ONNX policy inference for bipedal walking."""
from __future__ import annotations

import math
import time

import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from bipedal_controller.hardware_interface import HardwareInterface

# Walk.action must be generated via rosidl; for deployment the import would be:
# from bipedal_controller.action import Walk
# During development we define a placeholder dataclass structure.
try:
    from bipedal_controller.action import Walk
except ImportError:
    Walk = None


class PolicyServerNode(Node):
    """Action server that accepts Walk goals and runs the ONNX policy at 50 Hz."""

    def __init__(self):
        super().__init__("bipedal_policy_server")

        self.declare_parameter("model_path", "policy.onnx")
        self.declare_parameter("control_rate", 50.0)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self._control_rate = self.get_parameter("control_rate").get_parameter_value().double_value

        self.get_logger().info(f"Loading ONNX model from {model_path}")
        self._session = ort.InferenceSession(model_path)
        self._input_name = self._session.get_inputs()[0].name

        self._hw = HardwareInterface(self)

        self._phase = 0.0
        self._phase_freq = 1.25
        self._prev_action = np.zeros(2, dtype=np.float32)

        self._callback_group = ReentrantCallbackGroup()

        if Walk is not None:
            self._action_server = ActionServer(
                self,
                Walk,
                "walk",
                execute_callback=self._execute_callback,
                goal_callback=self._goal_callback,
                cancel_callback=self._cancel_callback,
                callback_group=self._callback_group,
            )
            self.get_logger().info("Walk action server started on /walk")
        else:
            self.get_logger().warn(
                "Walk action type not found (rosidl not built). "
                "Running in standalone inference mode."
            )
            self._timer = self.create_timer(
                1.0 / self._control_rate,
                self._standalone_step,
                callback_group=self._callback_group,
            )

    def _goal_callback(self, goal_request):
        self.get_logger().info(
            f"Walk goal received: v={goal_request.target_velocity:.2f} m/s, "
            f"dur={goal_request.duration:.1f} s"
        )
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info("Walk goal cancelled")
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        """Main control loop for a Walk action goal."""
        target_vel = goal_handle.request.target_velocity
        duration = goal_handle.request.duration

        dt = 1.0 / self._control_rate
        total_steps = int(duration / dt)
        distance = 0.0
        self._phase = 0.0

        feedback_msg = Walk.Feedback()
        result_msg = Walk.Result()

        self.get_logger().info(f"Executing walk: {total_steps} steps at {self._control_rate} Hz")

        for step in range(total_steps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result_msg.success = False
                result_msg.distance_walked = distance
                result_msg.average_velocity = distance / max((step + 1) * dt, 1e-6)
                return result_msg

            if not self._hw.state_received:
                self.get_logger().warn("Waiting for joint states...", throttle_duration_sec=1.0)
                time.sleep(dt)
                continue

            obs = self._build_observation()
            action = self._run_inference(obs)
            self._hw.send_joint_commands(action)

            self._phase += 2.0 * math.pi * self._phase_freq * dt
            self._prev_action = action[:2].copy()
            distance += target_vel * dt

            feedback_msg.current_velocity = target_vel
            feedback_msg.phase = self._phase % (2.0 * math.pi)
            feedback_msg.elapsed_time = (step + 1) * dt
            goal_handle.publish_feedback(feedback_msg)

            time.sleep(dt)

        goal_handle.succeed()
        result_msg.success = True
        result_msg.distance_walked = distance
        result_msg.average_velocity = distance / max(duration, 1e-6)
        return result_msg

    def _standalone_step(self) -> None:
        """Timer callback for inference without the action interface."""
        if not self._hw.state_received:
            return

        obs = self._build_observation()
        action = self._run_inference(obs)
        self._hw.send_joint_commands(action)

        dt = 1.0 / self._control_rate
        self._phase += 2.0 * math.pi * self._phase_freq * dt
        self._prev_action = action[:2].copy()

    def _build_observation(self) -> np.ndarray:
        """Construct 32-dim observation from hardware state.

        In a full deployment, torso IMU data and foot contact sensors
        would come from additional subscribers. Here we use placeholder
        values that would be replaced with real sensor topics.
        """
        torso_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        torso_ang_vel = np.zeros(3, dtype=np.float32)
        torso_lin_vel = np.zeros(3, dtype=np.float32)
        foot_contacts = np.array([1.0, 1.0], dtype=np.float32)
        phase_clock = np.array([
            math.sin(self._phase),
            math.cos(self._phase),
        ], dtype=np.float32)

        return self._hw.build_observation(
            torso_quaternion=torso_quat,
            torso_angular_velocity=torso_ang_vel,
            torso_linear_velocity=torso_lin_vel,
            foot_contacts=foot_contacts,
            phase_clock=phase_clock,
            prev_action=self._prev_action,
        )

    def _run_inference(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = obs.reshape(1, -1).astype(np.float32)
        result = self._session.run(None, {self._input_name: obs_batch})
        return result[0].squeeze(0)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
