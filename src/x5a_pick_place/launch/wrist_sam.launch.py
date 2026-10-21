"""Gemini + calibrated wrist MoveIt + original SAM/GraspNet interaction.

Startup submits no tasks. This mode owns the arm instead of the color task server.
"""
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch import logging as launch_logging
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from x5a_pick_place.wrist_launch import build_task_nodes
from x5a_remote_grasp.gripper_model import GripperModel
from x5a_remote_grasp.scene_policy import contact_limit_box


MOVEIT_REMAPS = [(f'/{name}', f'/wrist_pick/{name}') for name in (
    'move_action', 'compute_ik', 'compute_fk', 'check_state_validity',
    'apply_planning_scene', 'get_planning_scene', 'compute_cartesian_path')]
MOVEIT_REMAPS += [('/move_group/get_parameters', '/wrist_pick/move_group/get_parameters')]
FEEDBACK_REMAPS = [('/joint_states', '/wrist_preview/joint_states')]


def sam_nodes(context):
    root = Path(LaunchConfiguration('workspace').perform(context))
    config = root/'config/sam_grasp.yaml'
    settings = yaml.safe_load(config.read_text())['/**']['ros__parameters']
    enabled = LaunchConfiguration('execution_enabled').perform(context) == 'true'
    verbose = context.launch_configurations.get('verbose_logs', 'false').lower() == 'true'
    backend_output = 'screen' if verbose else 'log'
    # Validate configuration and the calibrated model before constructing hardware.
    if (settings['frames']['base'] != 'wrist_preview/base_link'
            or settings['planning']['tcp_frame'] != 'wrist_preview/tool0'
            or settings['scene']['camera_model_link'] != 'wrist_preview/camera_payload'
            or settings['scene']['camera_mount'] != 'eye_in_hand'
            or not settings['safety']['require_table_collision']
            or not settings['safety']['require_camera_collision']
            or not settings['execution']['plan_joint_moves']):
        raise ValueError('SAM wrist mode requires calibrated frames, camera/table collision and planned ready/home')
    candidate = root/'config/calibration/handeye_model_candidate.yaml'
    scene = settings['scene']
    contact_limit_box(scene['table_center'], scene['table_size'],
                      scene.get('table_contact_tolerance_m', 0.0))
    if set(scene.get('table_touch_links', ['wrist_preview/link7', 'wrist_preview/link8'])) != {
            'wrist_preview/link7', 'wrist_preview/link8'}:
        raise ValueError('SAM table contact is restricted to the two wrist finger links')
    model_path = settings.get('gripper', {}).get('model_path')
    if model_path:
        GripperModel.load(model_path).validate_sources(root)
    planning = settings['planning']
    moveit_settings = settings.get('moveit', {})
    pipeline = planning.get('pipeline_id', 'ompl')
    pipelines = moveit_settings.get('planning_pipelines', ['ompl'])
    if pipeline not in pipelines:
        raise ValueError('planning.pipeline_id must name a loaded MoveIt pipeline')
    ik_solver = context.launch_configurations.get('ik_solver', 'config')
    if ik_solver == 'config':
        ik_solver = moveit_settings.get('ik_solver', 'kdl')
    moveit = build_task_nodes(candidate, root/'config/task_execute.yaml',
        accepted=True, plan_only=False, include_task_server=False,
        gripper_model_path=model_path, ik_solver=ik_solver,
        ik_solve_type=moveit_settings.get('ik_solve_type', 'Speed'),
        planning_pipelines=pipelines, output=backend_output)
    preview = Path(get_package_share_directory('x5a_vision'))/'launch/wrist_localization_preview.launch.py'
    feedback = FEEDBACK_REMAPS
    return [
        Node(package='x5a_wrist_driver', executable='wrist_controller', name='arm',
             parameters=[str(root/'config/driver.yaml')], output='screen',
             arguments=['--ros-args', '--log-level', 'info' if verbose else 'warn'],
             condition=IfCondition(LaunchConfiguration('start_driver'))),
        Node(package='x5a_moveit_official_adapter', executable='official_trajectory_adapter',
             name='x5a_official_trajectory_adapter', output='screen',
             arguments=['--ros-args', '--log-level', 'info' if verbose else 'warn'],
             parameters=[str(root/'config/adapter.yaml'), {'require_driver_health': True}],
             condition=IfCondition(LaunchConfiguration('start_adapter'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(preview)), launch_arguments={
            'candidate_result': str(candidate), 'vision_config': str(root/'config/vision.yaml'),
            'allow_failed_candidate': 'true', 'start_camera': LaunchConfiguration('start_camera'),
            'color_width': '1920', 'color_height': '1080', 'color_fps': '15', 'depth_fps': '15',
            'node_output': backend_output,
        }.items()),
        *moveit,
        Node(package='x5a_parallel_ik', executable='parallel_ik_server', name='wrist_sam_parallel_ik',
             parameters=[{'workers': 4, 'source_node': '/wrist_pick/move_group',
                          'service_name': '/wrist_sam/parallel_ik/solve'}], output=backend_output),
        Node(package='x5a_remote_grasp', executable='remote_grasp_node', name='wrist_sam_planner',
             parameters=[str(config), {'safety.plan_only': True}],
             remappings=MOVEIT_REMAPS+feedback, output=backend_output),
        Node(package='x5a_remote_grasp', executable='remote_grasp_executor', name='wrist_sam_executor',
             parameters=[str(config), {'execution.enabled': enabled}],
             remappings=MOVEIT_REMAPS+feedback, output=backend_output),
        Node(package='x5a_web_agent', executable='sam_web_agent', name='wrist_sam_web_agent',
             arguments=['--host', '0.0.0.0', '--port', '8000', '--https-port', '8443', '--terminal'],
             output='screen', output_format='{line}', additional_env={
                 'X5A_GRASP_ROUTE': 'remote', 'X5A_AUTO_EXECUTE': '1' if enabled else '0',
                 'X5A_MQTT_ENABLED': '0', 'X5A_TEXT_FORWARD': '0', 'X5A_CLOUD_PARSE': '1',
                 'X5A_CARRY_TO_READY': '1' if enabled else '0',
                 'X5A_TASK_LOG_DIR': str(launch_logging.launch_config.log_dir),
                 'X5A_VERBOSE_TASK_LOGS': '1' if verbose else '0'}),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('workspace', default_value=str(Path.home()/'arx/wrist_ws')),
        DeclareLaunchArgument('execution_enabled', default_value='false'),
        DeclareLaunchArgument('verbose_logs', default_value='false',
                              description='Show backend planning logs on screen instead of only in log files'),
        DeclareLaunchArgument('ik_solver', default_value='config',
                              choices=['config', 'kdl', 'trac_ik']),
        DeclareLaunchArgument('start_driver', default_value='true'),
        DeclareLaunchArgument('start_adapter', default_value='true'),
        DeclareLaunchArgument('start_camera', default_value='true'),
        OpaqueFunction(function=sam_nodes),
    ])
