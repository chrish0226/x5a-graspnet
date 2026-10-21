"""Independent wrist color stack. Startup never submits a robot task."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from x5a_pick_place.wrist_launch import build_task_nodes


def stack(context):
    root = Path(LaunchConfiguration('workspace').perform(context))
    candidate = str(root/'config/calibration/handeye_model_candidate.yaml')
    # Evaluate the task's configuration and calibrated model before returning
    # any driver/camera actions. A late included-launch exception would shut
    # down a driver that was already initializing and disable its motors.
    task_nodes = build_task_nodes(candidate, root/'config/task_execute.yaml',
                                  accepted=True, plan_only=False)
    def include(package, file, arguments):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory(package))/'launch'/file)),
            launch_arguments=arguments.items())
    driver_config = str(root/'config/driver.yaml')
    return [
        Node(package='x5a_wrist_driver', executable='wrist_controller', name='arm',
             parameters=[driver_config], output='screen', condition=IfCondition(LaunchConfiguration('start_driver'))),
        Node(package='x5a_moveit_official_adapter', executable='official_trajectory_adapter',
             name='x5a_official_trajectory_adapter', parameters=[str(root/'config/adapter.yaml'),
                 {'require_driver_health': True}],
             output='screen', condition=IfCondition(LaunchConfiguration('start_adapter'))),
        include('x5a_vision', 'wrist_localization_preview.launch.py', {
            'candidate_result': candidate, 'vision_config': str(root/'config/vision.yaml'),
            'allow_failed_candidate': 'true', 'start_camera': LaunchConfiguration('start_camera'),
            'color_width': '1920', 'color_height': '1080', 'color_fps': '15', 'depth_fps': '15'}),
        *task_nodes,
        Node(package='x5a_web_agent', executable='wrist_web_agent', name='wrist_web_agent',
             arguments=['--legacy-ports'], respawn=True, respawn_delay=2.0,
             output='screen', additional_env={'X5A_MQTT_ENABLED': '0', 'X5A_TEXT_FORWARD': '0'}),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('workspace', default_value=str(Path.home()/'arx/wrist_ws')),
        DeclareLaunchArgument('start_driver', default_value='true'),
        DeclareLaunchArgument('start_adapter', default_value='true'),
        DeclareLaunchArgument('start_camera', default_value='true'),
        OpaqueFunction(function=stack),
    ])
