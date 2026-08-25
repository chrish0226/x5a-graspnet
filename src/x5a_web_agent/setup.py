from glob import glob
from setuptools import setup

package_name = "x5a_web_agent"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/static", glob("static/*")),
        ("share/" + package_name + "/certs", glob("certs/*")),
    ],
    install_requires=["setuptools", "fastapi", "uvicorn", "paho-mqtt>=1.5"],
    zip_safe=True,
    maintainer="x5a user",
    maintainer_email="user@example.com",
    description="Phone voice/web agent for X5A pick-place",
    license="BSD",
    entry_points={
        "console_scripts": [
            "web_agent = x5a_web_agent.app:main",
        ]
    },
)
