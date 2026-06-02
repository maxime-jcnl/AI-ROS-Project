"""
demo.launch.py
==============
  1. robot_state_publisher : connait la forme de la pince (URDF)
  2. emg_replay            : rejoue le signal musculaire
  3. emg_classifier        : devine le geste
  4. gripper_controller    : fait bouger la pince
  5. rviz2                 : la fenêtre où tu VOIS la pince bouger


"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory("ros2_emg_gripper")
    urdf = os.path.join(pkg, "urdf", "simple_gripper.urdf")
    with open(urdf, "r") as f:
        robot_desc = f.read()

    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_desc}]),

        Node(package="ros2_emg_gripper", executable="emg_replay",
             parameters=[{"mat_file": "data/DB2_s1/S1_E1_A1.mat"}]),

        Node(package="ros2_emg_gripper", executable="emg_classifier",
             parameters=[{"model_path": "model/emg_model.joblib"}]),

        Node(package="ros2_emg_gripper", executable="gripper_controller"),

        Node(package="rviz2", executable="rviz2", name="rviz2"),
    ])
