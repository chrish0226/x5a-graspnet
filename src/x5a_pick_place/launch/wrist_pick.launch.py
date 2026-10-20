"""Private MoveIt/task endpoints; startup never submits a robot task."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from x5a_pick_place.wrist_launch import build_task_nodes


def nodes(context):
    return build_task_nodes(
        LaunchConfiguration('candidate_result').perform(context),
        LaunchConfiguration('task_config').perform(context),
        accepted=LaunchConfiguration('experimental_model_accepted').perform(context) == 'true',
        plan_only=LaunchConfiguration('plan_only').perform(context) == 'true')


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('candidate_result'),
        DeclareLaunchArgument('task_config'),
        DeclareLaunchArgument('experimental_model_accepted', default_value='false'),
        DeclareLaunchArgument('plan_only', default_value='true'),
        OpaqueFunction(function=nodes),
    ])
