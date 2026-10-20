"""Wrist SAM wiring and rejection paths; no physical controller goals."""
import importlib.util
import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import rclpy
import yaml
from launch import LaunchContext
from std_srvs.srv import Trigger
from rclpy.qos import ReliabilityPolicy

from x5a_remote_grasp.remote_grasp_executor import RemoteGraspExecutor
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode

ROOT = Path(__file__).resolve().parents[1]


def sam_launch():
    path = ROOT/'src/x5a_pick_place/launch/wrist_sam.launch.py'
    spec = importlib.util.spec_from_file_location('wrist_sam_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sam_mode_has_one_task_owner_and_routes_private_moveit(monkeypatch):
    launch = sam_launch()
    monkeypatch.setattr(launch, 'Node', lambda **kwargs: kwargs)
    context = LaunchContext()
    context.launch_configurations.update(workspace=str(ROOT), execution_enabled='false')
    actions = launch.sam_nodes(context)
    nodes = [a for a in actions if isinstance(a, dict)]
    executables = [n['executable'] for n in nodes]
    assert 'remote_grasp_node' in executables and 'remote_grasp_executor' in executables
    assert 'wrist_task_server' not in executables and 'wrist_web_agent' not in executables
    executor = next(n for n in nodes if n['executable'] == 'remote_grasp_executor')
    assert executor['parameters'][-1]['execution.enabled'] is False
    assert ('/joint_states', '/wrist_preview/joint_states') in executor['remappings']
    assert ('/get_planning_scene', '/wrist_pick/get_planning_scene') in executor['remappings']
    web = next(n for n in nodes if n['executable'] == 'sam_web_agent')
    assert web['additional_env']['X5A_AUTO_EXECUTE'] == '0'
    assert web['additional_env']['X5A_CLOUD_PARSE'] == '1'


def test_mismatched_camera_model_fails_before_hardware(tmp_path, monkeypatch):
    launch = sam_launch()
    settings = yaml.safe_load((ROOT/'config/sam_grasp.yaml').read_text())
    settings['/**']['ros__parameters']['frames']['base'] = 'base_link'
    (tmp_path/'config').mkdir()
    (tmp_path/'config/sam_grasp.yaml').write_text(yaml.safe_dump(settings))
    def no_hardware(**kwargs):
        pytest.fail('constructed hardware before rejecting mismatched calibration')
    monkeypatch.setattr(launch, 'Node', no_hardware)
    context = LaunchContext()
    context.launch_configurations.update(workspace=str(tmp_path), execution_enabled='true')
    with pytest.raises(ValueError, match='calibrated frames'):
        launch.sam_nodes(context)


@pytest.fixture
def ros_context():
    # Root conftest selects a separate DDS domain from the physical arm.
    rclpy.init()
    yield
    rclpy.try_shutdown()


def test_plan_only_executor_rejects_all_motion_services(ros_context):
    node = RemoteGraspExecutor()
    try:
        def forbidden(*args):
            pytest.fail('plan-only executor attempted a controller goal')
        node._follow = node._gripper = forbidden
        for method in (node._on_execute, node._on_release, node._on_go_home, node._on_go_ready):
            response = method(Trigger.Request(), Trigger.Response())
            assert not response.success
            assert '执行未启用' in json.loads(response.message)['message']
    finally:
        node.destroy_node()


def test_stop_prevents_subsequent_arm_and_release_goals(ros_context):
    node = RemoteGraspExecutor()
    try:
        response = node._on_cancel(Trigger.Request(), Trigger.Response())
        assert response.success
        assert node._follow(None) == (False, -1)
        assert node._gripper(0.044, 3.0) is False
    finally:
        node.destroy_node()


def test_actual_ros_parameter_file_and_hd_qos():
    rclpy.init(args=['--ros-args', '--params-file', str(ROOT/'config/sam_grasp.yaml')])
    node = executor = None
    try:
        node = RemoteGraspNode()
        executor = RemoteGraspExecutor()
        assert node.base_frame == executor.base_frame == 'wrist_preview/base_link'
        assert executor.tcp_frame == 'wrist_preview/tool0'
        assert executor.pose_planner is not None and not executor.enabled
        assert executor.pose_planner.move_client._action_name == '/wrist_pick/move_action'
        # Exercise the planner's stage dispatch, not just the YAML values:
        # a hardcoded legacy stage table must not override the configured profile.
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.msg import RobotTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        task = yaml.safe_load((ROOT/'config/task_execute.yaml').read_text())['/**']['ros__parameters']['planning']
        sam = yaml.safe_load((ROOT/'config/sam_grasp.yaml').read_text())['/**']['ros__parameters']['planning']
        assert executor.velocity_scaling == task['transit_velocity_scaling']
        assert executor.acceleration_scaling == task['transit_acceleration_scaling']
        observed = []
        def plan_stage(stage, pose, start, velocity, acceleration):
            observed.append((stage, velocity, acceleration))
            trajectory = RobotTrajectory()
            trajectory.joint_trajectory.joint_names = [f'joint{i}' for i in range(1, 7)]
            trajectory.joint_trajectory.points = [JointTrajectoryPoint(positions=[0.0]*6)]
            return trajectory, node.planner.joints_to_state([0.0]*6), ''
        node.planner._plan_stage = plan_stage
        node.planner.verify_endpoint = lambda *args: (True, '')
        stages = [('PREGRASP', 'transit'), ('GRASP', 'precision'), ('LIFT', 'lift_retreat'),
                  ('TRANSPORT', 'transit'), ('PLACE', 'precision'), ('RETREAT', 'lift_retreat')]
        plan, reason = node.planner.plan_sequence([(stage, PoseStamped()) for stage, _ in stages])
        assert plan is not None, reason
        assert observed == [(stage, sam[f'{profile}_velocity_scaling'],
                              sam[f'{profile}_acceleration_scaling']) for stage, profile in stages]
        assert node.camera_model_link == 'wrist_preview/camera_payload'
        for subscriber in (node.color_sub, node.depth_sub, node.info_sub):
            assert subscriber.sub.qos_profile.reliability == ReliabilityPolicy.RELIABLE
    finally:
        if node is not None:
            node.destroy_node()
        if executor is not None:
            executor.destroy_node()
        rclpy.try_shutdown()


def test_failed_ready_planning_preserves_holding_without_execution():
    fake = NS(holding=True, running=True, _done=threading.Event(),
        count_publishers=lambda _: 1, joints_monotonic=__import__('time').monotonic(),
        velocity_scaling=.1, acceleration_scaling=.1,
        pose_planner=NS(joints_to_state=lambda x: x,
                        apply_scene=lambda _: (True, ''),
                        plan_joints=lambda *args: (None, 'collision: camera payload')),
        _pose_scene=lambda: None,
        _follow=lambda _: pytest.fail('failed plan was executed'),
        get_logger=lambda: NS(error=lambda _: None))
    RemoteGraspExecutor._run_planned_joints(fake, [0.0]*6, 'READY', 'ready')
    assert not fake._outcome['ok'] and fake._outcome['holding']
    assert 'collision' in fake._outcome['message']


def test_measured_urdf_camera_does_not_get_duplicate_attachment():
    node = object.__new__(RemoteGraspNode)
    node.base_frame = 'wrist_preview/base_link'
    node.require_table = False
    node.require_camera = True
    node.camera_model_link = 'wrist_preview/camera_payload'
    node.place_support_hull = None
    scene = node._safety_scene(None)
    assert not scene.robot_state.attached_collision_objects
    assert not any(o.id == 'remote_grasp_camera' for o in scene.world.collision_objects)


def test_remote_ui_health_does_not_require_color_task_server(monkeypatch):
    from x5a_web_agent.ros_bridge import RosBridge
    monkeypatch.setenv('X5A_GRASP_ROUTE', 'remote')
    fake = NS(dry_run=False, _client=None,
        _remote_client=NS(wait_for_server=lambda **kwargs: True),
        _state_client=NS(service_is_ready=lambda: True))
    assert RosBridge.server_ready(fake)


def test_voice_commands_keep_original_sam_parser_and_two_step_place(monkeypatch):
    from x5a_web_agent import app
    monkeypatch.setenv('X5A_GRASP_ROUTE', 'remote')
    monkeypatch.setenv('X5A_CLOUD_PARSE', '1')
    monkeypatch.setattr(app, 'cloud_parse_text', lambda _: {
        'object': 'bottle', 'color': 'blue', 'action': 'pick', 'sam_prompt': 'blue bottle'})
    assert asyncio.run(app._cloud_parse_command('准备'))['action'] == 'go_ready'
    assert asyncio.run(app._cloud_parse_command('镜像准备'))['action'] == 'unknown'
    grasp = asyncio.run(app._cloud_parse_command('抓蓝色瓶子'))
    assert grasp['action'] == 'remote_grasp' and grasp['prompt'] == 'blue bottle'
    assert asyncio.run(app._cloud_parse_command('放到白色方块上'))['action'] == 'remote_place'
    assert asyncio.run(app._cloud_parse_command('停止'))['action'] == 'cancel'


@pytest.mark.parametrize('empty', [None, 'null', ' Null ', 'None', 'undefined', ''])
def test_pliers_translation_and_nullable_fields_from_gateway(monkeypatch, empty):
    from x5a_web_agent import app
    monkeypatch.setenv('X5A_GRASP_ROUTE', 'remote')
    monkeypatch.setenv('X5A_CLOUD_PARSE', '1')
    cloud = dict(object='clamp', action='pick', sam_prompt='clamp', color=empty,
                 reference_prompt=empty, relation=empty, place_prompt=empty, place_rel=empty)
    monkeypatch.setattr(app, 'cloud_parse_text', lambda _: cloud)
    task = asyncio.run(app._cloud_parse_command('抓去钳子'))
    assert task['object'] == 'pliers' and task['prompt'] == 'pliers'
    assert task['color'] is None
    # Real clamps and commands using another object as a reference keep meaning.
    assert asyncio.run(app._cloud_parse_command('抓取夹具'))['prompt'] == 'clamp'
    assert asyncio.run(app._cloud_parse_command('抓钳子旁边的夹具'))['prompt'] == 'clamp'
    cloud.update(object='cube', sam_prompt='red cube', color='red',
                 reference_prompt='pliers', relation='next_to',
                 place_prompt='black box', place_rel='in')
    assert asyncio.run(app._cloud_parse_command('把钳子旁边的红方块放进黑盒子'))['prompt'] == (
        'red cube | ref: pliers; rel: next_to | place: black box; prel: in')


def test_sam_speech_default_matches_fast_wrist_backend():
    import os
    import subprocess
    import sys
    # Importing the SAM entry point in the normal environment must select the
    # same local recognizer as the wrist app, before any audio request arrives.
    env = dict(os.environ)
    env.pop('X5A_ASR_BACKEND', None)
    subprocess.run([sys.executable, '-c',
        'from x5a_web_agent import app; '
        'from x5a_web_agent.asr import SenseVoiceSpeechToText; '
        'assert isinstance(app.asr_backend, SenseVoiceSpeechToText)'],
        env=env, check=True, capture_output=True, timeout=15)


def test_prepare_round_trip_uses_namespaced_action_without_hardware(monkeypatch):
    """Real ROS transport reproduces the endpoint bug missed by mocked plans."""
    from rclpy.node import Node
    from rclpy.action import ActionServer
    from rclpy.executors import MultiThreadedExecutor
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.srv import ApplyPlanningScene
    from trajectory_msgs.msg import JointTrajectoryPoint
    from x5a_remote_grasp.moveit_plan import PlanOnlyMoveIt

    rclpy.init(args=['--ros-args', '--params-file', str(ROOT/'config/sam_grasp.yaml'),
                     '-p', 'execution.enabled:=true'])
    node = backend = spin = worker = action = None
    received, followed = [], []
    try:
        node = RemoteGraspExecutor()
        backend = Node('fake_namespaced_moveit')
        def plan(handle):
            received.append(handle.request)
            result = MoveGroup.Result()
            result.error_code.val = 1
            trajectory = result.planned_trajectory.joint_trajectory
            trajectory.joint_names = [f'joint{i}' for i in range(1, 7)]
            positions = [c.position for c in handle.request.request.goal_constraints[0].joint_constraints]
            trajectory.points = [JointTrajectoryPoint(positions=positions)]
            trajectory.points[0].time_from_start.sec = 1
            handle.succeed()
            return result
        action = ActionServer(backend, MoveGroup, '/wrist_pick/move_action', plan)
        service = backend.create_service(ApplyPlanningScene, '/wrist_pick/apply_planning_scene',
            lambda request, response: _scene_ok(response))
        def fresh_feedback():
            node.latest_joints = ([f'joint{i}' for i in range(1, 9)], [0.0]*8)
            node.joints_monotonic = time.monotonic()
        fresh_feedback()
        timer = node.create_timer(.02, fresh_feedback)
        monkeypatch.setattr(node, 'count_publishers', lambda _: 1)
        monkeypatch.setattr(node, '_follow', lambda trajectory: (followed.append(trajectory) or True, 0))
        spin = MultiThreadedExecutor(num_threads=4)
        spin.add_node(node)
        spin.add_node(backend)
        worker = threading.Thread(target=spin.spin, daemon=True)
        worker.start()
        assert node.pose_planner.scene_client.wait_for_service(timeout_sec=3.)
        client = backend.create_client(Trigger, '/x5a_remote_grasp/go_ready')
        assert client.wait_for_service(timeout_sec=3.)
        future = client.call_async(Trigger.Request())
        assert PlanOnlyMoveIt.wait_future(future, 12.)
        response = future.result()
        assert response.success, response.message
        assert len(received) == len(followed) == 1
        assert received[0].planning_options.plan_only is True
        assert received[0].request.group_name == 'arm'
        assert received[0].request.max_velocity_scaling_factor == pytest.approx(.82)
        assert received[0].request.max_acceleration_scaling_factor == pytest.approx(.43)
    finally:
        if spin is not None:
            spin.shutdown(timeout_sec=3.)
        if worker is not None:
            worker.join(timeout=3.)
        if action is not None:
            action.destroy()
        if node is not None:
            node.destroy_node()
        if backend is not None:
            backend.destroy_node()
        rclpy.try_shutdown()


def _scene_ok(response):
    response.success = True
    return response
