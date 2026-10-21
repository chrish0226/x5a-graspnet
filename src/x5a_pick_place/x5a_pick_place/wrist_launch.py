"""Shared construction of wrist task actions, before hardware startup."""
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from x5a_handeye.calibrated_tf import calibrated_transform
from x5a_pick_place.wrist_model import prefixed_models


def build_task_nodes(candidate_result, task_config, *, accepted=False, plan_only=True,
                     include_task_server=True, gripper_model_path=None,
                     ik_solver='kdl', ik_solve_type='Speed', planning_pipelines=None,
                     output='screen'):
    """Construct and validate task actions without starting any processes."""
    result_path = Path(candidate_result).expanduser()
    result = yaml.safe_load(result_path.read_text())
    calibrated_transform(result, allow_model_candidate=True, allow_failed_candidate=accepted)
    share = Path(get_package_share_directory('x5a_moveit_config'))
    solvers = {'kdl': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
               'trac_ik': 'trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin'}
    if ik_solver not in solvers or ik_solve_type not in ('Speed', 'Distance'):
        raise ValueError('IK solver must be kdl or trac_ik; solve type must be Speed or Distance')
    if ik_solver == 'trac_ik':
        # Fail before any hardware actions if the optional plugin is missing.
        get_package_share_directory('trac_ik_kinematics_plugin')
    pipelines = list(planning_pipelines or ['ompl'])
    if 'ompl' not in pipelines or any(p not in ('ompl', 'pilz_industrial_motion_planner', 'chomp')
                                      for p in pipelines):
        raise ValueError('wrist planning requires ompl; optional pipelines are pilz_industrial_motion_planner and chomp')
    robot_xml = Path(result['robot_calibration']['candidate_urdf']).read_text()
    if gripper_model_path:
        from x5a_remote_grasp.gripper_model import GripperModel
        model = GripperModel.load(gripper_model_path)
        robot_xml = model.apply_to_urdf(robot_xml, Path(gripper_model_path).resolve().parents[1])
    urdf, srdf = prefixed_models(
        robot_xml,
        (share/'config/X5A.srdf').read_text(),
        result['robot_calibration']['expected_model'], result['T_tool_camera'])
    cfg = (MoveItConfigsBuilder('X5A', package_name='x5a_moveit_config')
           .robot_description(file_path='config/X5A.urdf.xacro')
           .robot_description_semantic(file_path='config/X5A.srdf')
           .robot_description_kinematics(file_path='config/kinematics.yaml')
           .joint_limits(file_path='config/joint_limits.yaml')
           .trajectory_execution(file_path='config/moveit_controllers.yaml')
           .planning_pipelines(pipelines=pipelines).to_moveit_configs())
    params = cfg.to_dict()
    params['robot_description'] = urdf
    params['robot_description_semantic'] = srdf
    arm_kinematics = params['robot_description_kinematics']['arm']
    arm_kinematics['kinematics_solver'] = solvers[ik_solver]
    if ik_solver == 'trac_ik':
        arm_kinematics['solve_type'] = ik_solve_type
        arm_kinematics['position_only_ik'] = False
        arm_kinematics['epsilon'] = 1.0e-5
    # The actual controller can overshoot its target by a few milliradians.
    # Keep planned wrist pitches away from the commissioned active lower stop.
    params['robot_description_planning']['joint_limits']['joint4']['min_position'] = -1.22
    controllers = params['moveit_simple_controller_manager']
    names = list(controllers['controller_names'])
    controllers['controller_names'] = ['/' + name for name in names]
    for name in names:
        controllers['/' + name] = controllers.pop(name)
    params.update({'publish_robot_description_semantic': True,
                   'allow_trajectory_execution': True,
                   'moveit_manage_controllers': False,
                   'publish_planning_scene': True,
                   'publish_geometry_updates': True,
                   'publish_state_updates': True,
                   'trajectory_execution.allowed_start_tolerance': 0.05,
                   'trajectory_execution.allowed_execution_duration_scaling': 2.0})
    remaps = [('/x5a_vision/'+name, '/wrist_preview/'+name) for name in (
        'red_cube_pose', 'red_cube_stable', 'white_cube_pose', 'white_cube_stable',
        'orange_cube_pose', 'orange_cube_stable', 'detection_details', 'box_pose', 'box_stable')]
    remaps += [('/x5a/task_stage', '/wrist_pick/task_stage'),
               ('/x5a/task_busy', '/wrist_pick/task_busy'),
               ('/planning_scene', '/wrist_pick/planning_scene'),
               ('/compute_cartesian_path', '/wrist_pick/compute_cartesian_path'),
               ('/compute_ik', '/wrist_pick/compute_ik')]
    task_params = str(Path(task_config).expanduser())
    if not plan_only:
        settings = yaml.safe_load(Path(task_params).read_text())['/**']['ros__parameters']
        if not settings.get('execution', {}).get('require_servo_bounds', False):
            raise ValueError('wrist execution requires active servo bounds checks')
        # Box geometry is optional: the demo explicitly disables object/box
        # collision models. Servo limits and the table/self checks remain active.
    nodes = [
        Node(package='moveit_ros_move_group', executable='move_group',
             namespace='wrist_pick', name='move_group', output=output,
             parameters=[params],
             remappings=[('joint_states', '/wrist_preview/joint_states'),
                         ('/joint_states', '/wrist_preview/joint_states')]),
    ]
    if not include_task_server:
        return nodes
    nodes.append(Node(package='x5a_task_server', executable='wrist_task_server',
             namespace='wrist_pick', name='task_server', output='screen', respawn=True, respawn_delay=2.0,
             parameters=[task_params, {'plan_only': plan_only,
                                      'task.action_name': '/wrist_pick/pick_place'}], remappings=remaps,
             additional_env={'X5A_MQTT_ENABLED': '0'}))
    return nodes
