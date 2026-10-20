#!/usr/bin/env python3
"""Real X5A MoveIt bringup using the official ARX command/status API.

Starts the official ARX X5Controller (optional), MoveIt planning, and RViz.

The standard trajectory adapter exposes FollowJointTrajectory to MoveIt and
converts it to the vendor /arm_cmd RobotCmd topic. Feedback and RViz state are
derived from the vendor /arm_status RobotStatus topic. No x5a_control_bridge.

Does NOT start joint_state_publisher_gui or mock controllers.
NOT VALIDATED FOR FULL HARDWARE RANGE.
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_rviz = LaunchConfiguration("use_rviz")
    start_driver = LaunchConfiguration("start_driver")
    start_adapter = LaunchConfiguration("start_adapter")
    start_readonly_feedback = LaunchConfiguration('start_readonly_feedback')
    robot_description_file = LaunchConfiguration('robot_description_file')
    allow_execution = LaunchConfiguration('allow_execution')
    can_id = LaunchConfiguration("arm_can_id")
    status_topic = LaunchConfiguration("arm_pub_topic_name")
    cmd_topic = LaunchConfiguration("arm_sub_topic_name")

    moveit_config = (
        MoveItConfigsBuilder("X5A", package_name="x5a_moveit_config")
        .robot_description(file_path="config/X5A.urdf.xacro")
        .robot_description_semantic(file_path="config/X5A.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    # All consumers must use the same model (especially a calibrated candidate).
    moveit_config.robot_description = {
        'robot_description': ParameterValue(Command(['xacro ', robot_description_file]), value_type=str)
    }

    # Prefer description without depending on mock controllers running.
    # robot_description still may include ros2_control block; unused without controller_manager.
    arx_config = os.path.join(
        get_package_share_directory("arx_x5_controller"), "config", "single_arm.yaml"
    )
    arx_driver = Node(
        package="arx_x5_controller",
        executable="X5Controller",
        name="arm",
        output="screen",
        parameters=[
            arx_config,
            {
                "arm_can_id": can_id,
                "arm_control_type": "normal",
                "arm_pub_topic_name": status_topic,
                "arm_sub_topic_name": cmd_topic,
                "arm_end_type": 0,
            },
        ],
        condition=IfCondition(start_driver),
    )

    adapter_config = os.path.join(
        get_package_share_directory("x5a_moveit_official_adapter"), "config", "adapter.yaml"
    )
    official_adapter = Node(
        package="x5a_moveit_official_adapter",
        executable="official_trajectory_adapter",
        name="x5a_official_trajectory_adapter",
        output="screen",
        parameters=[adapter_config],
        condition=IfCondition(start_adapter),
    )
    readonly_feedback = Node(
        package='x5a_handeye', executable='readonly_joint_states',
        name='x5a_readonly_joint_states', output='screen',
        parameters=[{'status_topic': status_topic}],
        condition=IfCondition(start_readonly_feedback),
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base",
        arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                # Humble MTC Task::execute() sends complete solutions through
                # this capability; controller selection remains in MoveIt.
                "capabilities": "move_group/ExecuteTaskSolutionCapability",
                "publish_robot_description_semantic": True,
                "allow_trajectory_execution": ParameterValue(allow_execution, value_type=bool),
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
                "trajectory_execution.allowed_execution_duration_scaling": 2.0,
                "trajectory_execution.allowed_goal_duration_margin": 2.0,
                "trajectory_execution.allowed_start_tolerance": 0.1,
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", str(moveit_config.package_path / "config" / "moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument('robot_description_file',
                                  default_value=str(moveit_config.package_path / 'config' / 'X5A.urdf.xacro'),
                                  description='Shared URDF/xacro for MoveIt, RViz and robot_state_publisher'),
            DeclareLaunchArgument('allow_execution', default_value='true',
                                  description='Set false for planning-only model validation; also disable driver/adapter startup'),
            DeclareLaunchArgument('start_readonly_feedback', default_value='false',
                                  description='Display raw RobotStatus joints without any command/action interface'),
            DeclareLaunchArgument(
                "start_driver",
                default_value="true",
                description="Start official X5Controller; false when 04single_arm.sh is already running",
            ),
            DeclareLaunchArgument(
                "start_adapter",
                default_value="true",
                description="Enable MoveIt execution through official RobotCmd/RobotStatus topics",
            ),
            # ARX single-arm defaults to can1 in vendor yaml; override if using can0.
            DeclareLaunchArgument("arm_can_id", default_value="can1"),
            DeclareLaunchArgument("arm_pub_topic_name", default_value="arm_status"),
            DeclareLaunchArgument("arm_sub_topic_name", default_value="arm_cmd"),
            static_tf,
            rsp,
            arx_driver,
            readonly_feedback,
            TimerAction(period=1.0, actions=[official_adapter]),
            TimerAction(period=3.0, actions=[move_group]),
            TimerAction(period=5.0, actions=[rviz]),
        ]
    )
