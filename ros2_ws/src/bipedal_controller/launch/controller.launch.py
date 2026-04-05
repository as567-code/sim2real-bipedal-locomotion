"""Launch file for the bipedal controller action server."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="policy.onnx",
        description="Path to the ONNX policy model",
    )

    control_rate_arg = DeclareLaunchArgument(
        "control_rate",
        default_value="50.0",
        description="Policy inference rate in Hz",
    )

    policy_server_node = Node(
        package="bipedal_controller",
        executable="policy_server",
        name="bipedal_policy_server",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "control_rate": LaunchConfiguration("control_rate"),
            }
        ],
        output="screen",
    )

    return LaunchDescription([
        model_path_arg,
        control_rate_arg,
        policy_server_node,
    ])
