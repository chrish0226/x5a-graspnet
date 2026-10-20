from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    share = get_package_share_directory("x5a_remote_grasp")
    config = os.path.join(share, "config", "remote_grasp.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument('config_file', default_value=config),
            DeclareLaunchArgument("base_url", default_value="http://118.253.179.80:8300"),
            DeclareLaunchArgument("planning_enabled", default_value="true"),
            DeclareLaunchArgument("start_ik_server", default_value="true"),
            DeclareLaunchArgument("ik_workers", default_value="4"),
            DeclareLaunchArgument(
                "tcp_mapping_confirmed",
                default_value="true",
                description="URDF tool0 matches GraspNet axes; set false to stop after RViz markers",
            ),
            Node(
                package="x5a_parallel_ik",
                executable="parallel_ik_server",
                name="x5a_parallel_ik",
                output="screen",
                parameters=[{'workers': ParameterValue(LaunchConfiguration('ik_workers'), value_type=int)}],
                condition=IfCondition(LaunchConfiguration('start_ik_server')),
            ),
            Node(
                package="x5a_remote_grasp",
                executable="remote_grasp_node",
                name="x5a_remote_grasp",
                output="screen",
                parameters=[
                    LaunchConfiguration('config_file'),
                    {
                        "remote.base_url": LaunchConfiguration("base_url"),
                        "planning.enabled": ParameterValue(
                            LaunchConfiguration("planning_enabled"), value_type=bool
                        ),
                        "safety.tcp_mapping_confirmed": ParameterValue(
                            LaunchConfiguration("tcp_mapping_confirmed"),
                            value_type=bool,
                        ),
                        "safety.plan_only": True,
                        "planning.parallel_ik_workers": ParameterValue(LaunchConfiguration('ik_workers'), value_type=int),
                    },
                ],
            ),
        ]
    )
