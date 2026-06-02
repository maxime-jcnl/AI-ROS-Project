from setuptools import setup
import os
from glob import glob

package_name = "ros2_emg_gripper"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.urdf")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ton Nom",
    maintainer_email="toi@exemple.com",
    description="Contrôle myoélectrique d'une pince robotique (EMG -> ML -> ROS2).",
    license="MIT",
    entry_points={
        "console_scripts": [
            # nom_de_commande = package.fichier:fonction_main
            "emg_replay = ros2_emg_gripper.emg_replay_node:main",
            "emg_classifier = ros2_emg_gripper.emg_classifier_node:main",
            "gripper_controller = ros2_emg_gripper.gripper_controller_node:main",
        ],
    },
)
