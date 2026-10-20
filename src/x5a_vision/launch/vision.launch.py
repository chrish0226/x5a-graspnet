from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    vision_share = get_package_share_directory("x5a_vision")
    handeye_share = get_package_share_directory("x5a_handeye")
    config = os.path.join(vision_share, "config", "vision.yaml")
    result = os.path.join(handeye_share, "config", "handeye_result.yaml")
    start_camera = LaunchConfiguration("start_camera")
    return LaunchDescription([
        DeclareLaunchArgument("start_camera", default_value="true"),
        DeclareLaunchArgument("start_handeye_tf", default_value="true"),
        DeclareLaunchArgument("handeye_result_yaml", default_value=result),
        DeclareLaunchArgument("vision_config", default_value=config),
        Node(package="realsense2_camera", executable="realsense2_camera_node",
             namespace="camera", name="camera", output="screen",
             parameters=[{"serial_no": "_342522073696", "enable_color": True,
                          "enable_depth": True, "enable_infra1": False,
                          "enable_infra2": False, "enable_gyro": False,
                          "enable_accel": False,
                          "rgb_camera.color_profile": "640x480x30",
                          "depth_module.depth_profile": "640x480x30",
                          "align_depth.enable": True}],
             condition=IfCondition(start_camera)),
        Node(package="x5a_handeye", executable="publish_handeye_tf",
             name="x5a_handeye_tf", output="screen",
             parameters=[{"result_yaml": LaunchConfiguration("handeye_result_yaml")}],
             condition=IfCondition(LaunchConfiguration("start_handeye_tf"))),
        TimerAction(period=2.0, actions=[Node(
            package="x5a_vision", executable="cube_detector", name="cube_detector",
            output="screen", parameters=[LaunchConfiguration("vision_config")])]),
    ])
