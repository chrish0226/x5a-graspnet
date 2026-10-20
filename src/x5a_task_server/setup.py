from glob import glob
from setuptools import setup

package_name = "x5a_task_server"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools", "paho-mqtt>=1.5"],
    zip_safe=True,
    maintainer="x5a user",
    maintainer_email="user@example.com",
    description="Persistent X5A pick-place task server",
    license="BSD",
    entry_points={
        "console_scripts": [
            "task_server_node = x5a_task_server.wrist_task_server:main",
            "wrist_task_server = x5a_task_server.wrist_task_server:main",
        ]
    },
)
