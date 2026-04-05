from setuptools import setup, find_packages

package_name = "bipedal_controller"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/controller.launch.py"]),
        ("share/" + package_name + "/action", ["action/Walk.action"]),
    ],
    install_requires=["setuptools", "onnxruntime", "numpy"],
    zip_safe=True,
    maintainer="developer",
    maintainer_email="dev@example.com",
    description="ROS2 action server for bipedal locomotion policy deployment",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_server = bipedal_controller.policy_server:main",
        ],
    },
)
