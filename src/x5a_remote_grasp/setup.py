from glob import glob

from setuptools import find_packages, setup


package_name = "x5a_remote_grasp"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="x5a user",
    maintainer_email="user@example.com",
    description="Plan-only remote SAM and GraspNet integration for ARX X5A",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "remote_grasp_node = x5a_remote_grasp.remote_grasp_node:main",
            "remote_grasp_health = x5a_remote_grasp.health_check:main",
            "remote_grasp_offline = x5a_remote_grasp.offline_request:main",
            "remote_grasp_executor = x5a_remote_grasp.remote_grasp_executor:main",
        ]
    },
)
