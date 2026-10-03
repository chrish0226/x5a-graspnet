"""Exercise real ROS message contracts with stub MoveIt responses only."""
from types import SimpleNamespace as NS
import time
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from x5a_remote_grasp.moveit_plan import PlanOnlyMoveIt


class Future:
    def __init__(self, value):
        self.value = value
    def done(self):
        return True
    def result(self):
        return self.value


class Client:
    def __init__(self, value):
        self.value = value
    def call_async(self, request):
        self.request = request
        return Future(self.value)
    def wait_for_service(self, timeout_sec):
        return True


def trajectory():
    result = RobotTrajectory()
    result.joint_trajectory.joint_names = ['joint1']
    result.joint_trajectory.points = [JointTrajectoryPoint(positions=[0.0]),
        JointTrajectoryPoint(positions=[0.1])]
    result.joint_trajectory.points[1].time_from_start.sec = 1
    return result


def planner():
    result = object.__new__(PlanOnlyMoveIt)
    result.deadline = None
    result.group_name = 'arm'
    result.tcp_frame = 'tool0'
    result.cartesian_step = 0.003
    result.jump_threshold = 0.0
    result.velocity_scaling = result.acceleration_scaling = 0.1
    result.min_cartesian_fraction = 0.95  # even legacy settings must reject 95%
    result.position_tolerance = 0.003
    result.orientation_tolerance = 0.05
    result.node = NS(get_logger=lambda: NS(warn=lambda _: None, info=lambda _: None))
    return result


@pytest.mark.parametrize('fraction,code', [(0.95, 1), (0.999, 1), (1.0, -1)])
def test_partial_or_failed_cartesian_never_reaches_grasp(fraction, code):
    p = planner()
    p.cartesian_client = Client(NS(fraction=fraction, error_code=NS(val=code), solution=trajectory()))
    result, reason = p.plan_cartesian(PoseStamped(), RobotState())
    assert result is None and reason
    # A failed straight approach cannot become an arbitrary joint-space path.
    result, _, reason = p._plan_stage('GRASP', PoseStamped(), RobotState(), 0.1, 0.1)
    assert result is None and 'complete straight approach' in reason


def test_endpoint_fk_rejects_short_grasp_and_accepts_quaternion_sign():
    p = planner()
    target, actual = PoseStamped(), PoseStamped()
    target.pose.orientation.w = 1.0
    actual.pose.orientation.w = -1.0
    actual.pose.position.x = 0.006
    p.fk_client = Client(NS(error_code=NS(val=1), pose_stamped=[actual]))
    assert not p.verify_endpoint(trajectory(), RobotState(), target)[0]
    actual.pose.position.x = 0.002
    assert p.verify_endpoint(trajectory(), RobotState(), target)[0]


def test_budget_caps_service_waits_and_time_scaling_normalizes_nanoseconds():
    p = planner()
    p.deadline = time.monotonic()+0.03
    assert 0 < p._timeout(30) <= 0.03
    t = trajectory()
    t.joint_trajectory.points[1].time_from_start.sec = 0
    t.joint_trajectory.points[1].time_from_start.nanosec = 333333333
    p._apply_stage_scaling(t, 0.3, 0.1)
    stamp = t.joint_trajectory.points[1].time_from_start
    assert 0 <= stamp.nanosec < 1000000000
    assert stamp.sec+stamp.nanosec/1e9 == pytest.approx(1.11111111)


def test_arm_only_ik_seed_preserves_current_finger_state():
    seed = PlanOnlyMoveIt.joints_to_state([0.0]*6)
    assert seed.is_diff and 'joint7' not in seed.joint_state.name


def test_join_retreat_keeps_continuous_joints_and_strictly_increasing_time():
    first, second = trajectory(), trajectory()
    second.joint_trajectory.points[0].positions = [.1]
    second.joint_trajectory.points[1].positions = [.2]
    first.joint_trajectory.points[-1].time_from_start.sec = 0
    first.joint_trajectory.points[-1].time_from_start.nanosec = 900000000
    result = PlanOnlyMoveIt.join_trajectories(first, second)
    points = result.joint_trajectory.points
    assert [p.positions[0] for p in points] == [0., .1, .2]
    assert points[-1].time_from_start.sec == 1
    assert points[-1].time_from_start.nanosec == 900000000
    second.joint_trajectory.points[0].positions = [.12]
    with pytest.raises(ValueError):
        PlanOnlyMoveIt.join_trajectories(first, second)


@pytest.mark.parametrize('exists', [True, False])
def test_obstacle_cleanup_only_removes_support_if_present(exists):
    from moveit_msgs.msg import PlanningScene, CollisionObject
    p = planner()
    world = PlanningScene()
    if exists:
        world.world.collision_objects = [CollisionObject(id='support')]
    p.scene_query_client = Client(NS(scene=world))
    p.scene_client = Client(NS(success=True))
    change = PlanningScene()
    change.world.collision_objects = [CollisionObject(id='table', operation=CollisionObject.ADD),
                                      CollisionObject(id='support', operation=CollisionObject.REMOVE)]
    assert p.apply_scene(change)[0]
    actual = p.scene_client.request.scene.world.collision_objects
    assert [o.id for o in actual] == (['table', 'support'] if exists else ['table'])


def test_place_collision_checks_use_closed_fingers_after_grasp():
    p = planner()
    initial = RobotState()
    initial.joint_state.name = ['joint'+str(i) for i in range(1, 9)]
    initial.joint_state.position = [0.0]*6+[.044, .044]
    starts = {}
    def stage(name, pose, start, *_):
        starts[name] = start
        return trajectory(), initial if start is None else start, ''
    p._plan_stage = stage
    p.verify_endpoint = lambda *_: (True, '')
    p.expected_grasp_finger_position = .014
    p.expected_place_open_position = .044
    result, reason = p.plan_sequence([('GRASP', PoseStamped()), ('PLACE', PoseStamped()), ('RETREAT', PoseStamped())])
    assert result is not None, reason
    placed = starts['PLACE'].joint_state
    assert placed.position[placed.name.index('joint7')] == .014
    assert placed.position[placed.name.index('joint8')] == .014
    retreat = starts['RETREAT'].joint_state
    assert retreat.position[retreat.name.index('joint7')] == .044
    assert retreat.position[retreat.name.index('joint8')] == .044


@pytest.mark.parametrize('workers', [1, 2])
def test_second_native_candidate_precedes_geometric_reorientation(workers):
    import numpy as np
    from x5a_remote_grasp.geometry import RemoteGrasp
    from x5a_remote_grasp.grasp_quality import GripperGeometry
    from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
    from test_grasp_quality import contact_cloud
    calls = []
    plan = NS(stages=['PREGRASP', 'GRASP', 'LIFT', 'RETREAT'])
    def sequence(poses):
        calls.append(poses[1][1][:3, 3].copy())
        return (None, 'first native branch unavailable') if len(calls) == 1 else (plan, '')
    p = NS(ready=lambda: (True, ''), apply_scene=lambda _: (True, ''),
           plan_sequence=sequence, last_solution_seed=None)
    target = contact_cloud(x_span=(0.0, 0.02))+[0.2, 0.3, 0.02]
    mapping = np.eye(4)
    mapping[0, 3] = -0.015
    fake = NS(target_cloud=target, scene_cloud=target, planner=p, planning_enabled=True,
        geometry_workers=workers,
        _safety_scene=lambda _: None, plan_budget_s=30.0, native_budget_fraction=0.45,
        min_score=0.15, max_approach_z=0.35, allow_reachable_fallback=True,
        template_fallback=True, orientation_mode='graspnet', gripper_geometry=GripperGeometry(),
        _orientation_variants=lambda t: [(t[:3, :3], t[:3, :3], 'graspnet')],
        _template_orientation_variants=lambda _: [(np.eye(3), np.eye(3), 'template')],
        _make_tf=RemoteGraspNode._make_tf,
        reachability=NS(order_pitches=lambda *_: []), pitch_ladder=[], transit_pitches=[1.15],
        transform_grasp_tool=mapping, pregrasp_distance=0.05, lift_distance=0.08,
        retreat_distance=0.08, max_pregrasp_z=0.11, table_center=[0, 0, -0.026],
        table_size=[1, 1, 0.04], _waypoint_in_bounds=lambda *_: True,
        _ik_joint_seeds=lambda *_: [], _pose_stamped=lambda matrix, _: matrix,
        _record_reject=lambda *_: None, _feedback=lambda *_: None)
    transforms, grasps = [], []
    for index, x in enumerate([0.2, 0.21]):
        transform = np.eye(4)
        transform[:3, 3] = [x, 0.3, 0.02]
        transforms.append(transform)
        grasps.append(RemoteGrasp(index, 0.9-index*0.1, 0.04, 0.02, 0.01, transform))
    chosen = RemoteGraspNode._select_candidate(fake, NS(is_cancel_requested=False),
        grasps, transforms, None, True)
    assert len(calls) == 2
    assert chosen[0].index == 1 and chosen[0].score == pytest.approx(0.8)
    assert fake.selection_kind == 'native'
