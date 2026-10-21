"""Read-only wrist RGB-D localization using an isolated candidate robot tree.

No MoveIt, ARX driver, trajectory adapter or command interface is started.
The camera retains its driver optical frames. Its root is attached to the
preview tool frame, disconnected from the production base/robot TF tree.
All detected-object topics are private to /wrist_preview.
"""
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from x5a_handeye.calibrated_tf import calibrated_transform
from x5a_handeye.robot_model import require_matching_model


def preview_nodes(context):
    result_file = Path(LaunchConfiguration('candidate_result').perform(context)).expanduser()
    result = yaml.safe_load(result_file.read_text())
    allow_failed = LaunchConfiguration('allow_failed_candidate', default='false').perform(context).lower() == 'true'
    calibrated_transform(result, allow_model_candidate=True, allow_failed_candidate=allow_failed)
    if result.get('calibration_type') != 'eye_in_hand':
        raise ValueError('wrist localization preview requires an eye_in_hand candidate')
    profile = result['robot_calibration']
    model = Path(profile['candidate_urdf']).read_text()
    require_matching_model(model, profile['expected_model'])
    vision_config = LaunchConfiguration('vision_config').perform(context)
    output = context.launch_configurations.get('node_output', 'screen')
    remaps = [(f'/x5a_vision/{name}', f'/wrist_preview/{name}') for name in (
        'red_cube_pose', 'red_cube_stable', 'white_cube_pose', 'white_cube_stable',
        'orange_cube_pose', 'orange_cube_stable', 'object_point_camera', 'object_pose',
        'detection_stable', 'debug_image', 'detection_details', 'box_pose', 'box_stable')]
    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='wrist_preview_robot_state_publisher', output=output,
             parameters=[{'robot_description': ParameterValue(model, value_type=str),
                          'frame_prefix': 'wrist_preview/'}],
             remappings=[('/robot_description', '/wrist_preview/robot_description'),
                         ('/joint_states', '/wrist_preview/joint_states')]),
        Node(package='x5a_handeye', executable='readonly_joint_states',
             name='wrist_preview_readonly_feedback', output=output,
             parameters=[{'status_topic': '/arm_status',
                          'joint_state_topic': '/wrist_preview/joint_states'}]),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='wrist_preview_root', output=output,
             arguments=['--frame-id', 'wrist_preview/world',
                        '--child-frame-id', 'wrist_preview/base_link']),
        Node(package='x5a_handeye', executable='publish_handeye_tf',
             name='wrist_preview_camera_tf', output=output,
             parameters=[{'result_yaml': str(result_file), 'allow_model_candidate': True,
                          'allow_failed_candidate': allow_failed,
                          'robot_frame_prefix': 'wrist_preview/',
                          'robot_description_topic': '/wrist_preview/robot_description'}]),
        Node(package='x5a_vision', executable='cube_detector',
             name='wrist_preview_cube_detector', output=output, respawn=True, respawn_delay=2.0,
             parameters=[vision_config, {'base_frame': 'wrist_preview/base_link',
                          'calibration_label': ('Experimental calibration - 3mm validation FAIL'
                                                if result.get('verdict') == 'FAIL' else 'Calibration candidate')}],
             remappings=remaps),
    ]


def generate_launch_description():
    handeye_share = Path(get_package_share_directory('x5a_handeye'))
    vision_share = Path(get_package_share_directory('x5a_vision'))
    return LaunchDescription([
        DeclareLaunchArgument('candidate_result', default_value=str(
            Path.home()/'.ros/x5a_handeye/model_candidate_20261006/handeye_model_candidate.yaml')),
        DeclareLaunchArgument('vision_config', default_value=str(vision_share/'config/vision_wrist.yaml')),
        DeclareLaunchArgument('start_camera', default_value='true',
                              description='False when the wrist Gemini driver is already running'),
        DeclareLaunchArgument('node_output', default_value='screen'),
        DeclareLaunchArgument('allow_failed_candidate', default_value='false',
                              description='Explicitly accept a failed candidate for isolated localization preview'),
        DeclareLaunchArgument('color_width', default_value='640'),
        DeclareLaunchArgument('color_height', default_value='480'),
        DeclareLaunchArgument('color_fps', default_value='30'),
        DeclareLaunchArgument('depth_fps', default_value='30'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(handeye_share/'launch/eye_in_hand.launch.py')),
            launch_arguments={'start_camera': LaunchConfiguration('start_camera'),
                              'start_board': 'false',
                              'color_width': LaunchConfiguration('color_width'),
                              'color_height': LaunchConfiguration('color_height'),
                              'color_fps': LaunchConfiguration('color_fps'),
                              'depth_fps': LaunchConfiguration('depth_fps')}.items()),
        OpaqueFunction(function=preview_nodes),
    ])
