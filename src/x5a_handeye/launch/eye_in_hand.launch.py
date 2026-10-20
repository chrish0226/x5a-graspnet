"""Wrist Gemini camera and board detector; no robot motion or extrinsic TF."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    handeye = get_package_share_directory('x5a_handeye')
    driver = get_package_share_directory('orbbec_camera')
    return LaunchDescription([
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_board', default_value='true'),
        DeclareLaunchArgument('serial_number', default_value='CPMY85P003P'),
        DeclareLaunchArgument('color_width', default_value='640'),
        DeclareLaunchArgument('color_height', default_value='480'),
        DeclareLaunchArgument('color_fps', default_value='30'),
        DeclareLaunchArgument('depth_fps', default_value='30'),
        DeclareLaunchArgument('board_config', default_value=os.path.join(handeye, 'config', 'board_wrist_30mm.yaml')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(driver, 'launch', 'gemini345_lg.launch.py')),
            condition=IfCondition(LaunchConfiguration('start_camera')),
            launch_arguments={
                'camera_name': 'wrist_camera', 'serial_number': LaunchConfiguration('serial_number'),
                'color_width': LaunchConfiguration('color_width'),
                'color_height': LaunchConfiguration('color_height'),
                'color_fps': LaunchConfiguration('color_fps'),
                'depth_width': '640', 'depth_height': '480',
                'depth_fps': LaunchConfiguration('depth_fps'),
                'enable_point_cloud': 'false', 'depth_registration': 'true',
                'align_mode': 'SW', 'enable_color_undistortion': 'true',
                'enable_frame_sync': 'true',
            }.items(),
        ),
        Node(package='x5a_handeye', executable='board_detector', name='wrist_board_detector',
             condition=IfCondition(LaunchConfiguration('start_board')), output='screen',
             parameters=[{'board_config': LaunchConfiguration('board_config'),
                          'image_topic': '/wrist_camera/color/image_raw',
                          'camera_info_topic': '/wrist_camera/color/camera_info',
                          'min_charuco_corners': 12,
                          'max_reprojection_error_px': 1.0,
                          'max_board_tilt_deg': 60.0,
                          'publish_tf': False}],
             remappings=[('/calibration_board/pose', '/wrist_calibration/pose'),
                         ('/calibration_board/observation', '/wrist_calibration/observation'),
                         ('/x5a_handeye/debug_image', '/wrist_calibration/debug_image')]),
    ])
