import copy
from types import SimpleNamespace as NS

import pytest
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from x5a_remote_grasp.moveit_plan import PlanOnlyMoveIt


def trajectory(start=0., end=.1):
    result = RobotTrajectory()
    result.joint_trajectory.joint_names = [f'joint{i}' for i in range(1, 7)]
    result.joint_trajectory.points = [JointTrajectoryPoint(positions=[start]*6),
                                      JointTrajectoryPoint(positions=[end]*6)]
    result.joint_trajectory.points[-1].time_from_start.sec = 1
    return result


def planner():
    p = object.__new__(PlanOnlyMoveIt)
    p.deadline = None
    p.velocity_scaling = p.acceleration_scaling = .1
    p.sequence_cache_enabled = True
    p.grasp_branch_recovery_enabled = False
    p.reset_sequence_cache()
    p.verify_endpoint = lambda *_: (True, '')
    state = RobotState()
    state.joint_state.name = [f'joint{i}' for i in range(1, 9)]
    state.joint_state.position = [0.]*6+[.044, .044]
    p._plan_stage = lambda stage, pose, start, *_: (trajectory(), state if start is None else start, '')
    return p


def poses(place_x):
    out = [(name, PoseStamped()) for name in ('PREGRASP', 'GRASP', 'LIFT', 'PLACE')]
    out[-1][1].pose.position.x = place_x
    return out


def test_placement_variants_reuse_verified_pick_prefix_with_closed_fingers():
    p = planner()
    p.expected_grasp_finger_position = .014
    calls = []
    starts = []
    original = p._plan_stage
    def stage(name, *args):
        calls.append(name)
        if name == 'PLACE':
            starts.append(copy.deepcopy(args[1]))
        return original(name, *args)
    p._plan_stage = stage
    first, _ = p.plan_sequence(poses(.3))
    second, _ = p.plan_sequence(poses(.32))
    assert first and second
    assert calls == ['PREGRASP', 'GRASP', 'LIFT', 'PLACE', 'PLACE']
    assert p.sequence_cache_stats['hits'] == 3
    assert all(state.joint_state.position[state.joint_state.name.index('joint7')] == .014 for state in starts)
    # Mutating a returned plan cannot corrupt the cache for another variant.
    first.trajectories[0].joint_trajectory.points[-1].positions[0] = 99.
    third, _ = p.plan_sequence(poses(.34))
    assert third.trajectories[0].joint_trajectory.points[-1].positions[0] == .1


def test_failed_approach_is_not_repeated_for_each_place_height():
    p = planner()
    calls = []
    original = p._plan_stage
    def stage(name, *args):
        calls.append(name)
        return (None, None, 'incomplete straight approach') if name == 'GRASP' else original(name, *args)
    p._plan_stage = stage
    assert p.plan_sequence(poses(.3))[0] is None
    assert p.plan_sequence(poses(.32))[0] is None
    assert calls == ['PREGRASP', 'GRASP']
    assert p.sequence_cache_stats['negative_hits'] == 1


def test_new_request_or_changed_gripper_opening_cannot_reuse_old_collision_results():
    p = planner()
    p.expected_pregrasp_open_position = .044
    assert p.plan_sequence(poses(.3))[0]
    p.expected_pregrasp_open_position = .04
    assert p.plan_sequence(poses(.3))[0]
    assert p.sequence_cache_stats['hits'] == 0
    p.reset_sequence_cache()
    assert p.plan_sequence(poses(.3))[0]
    assert p.sequence_cache_stats['hits'] == 0


def test_timeouts_are_not_memoized_as_unreachable_geometry():
    p = planner()
    original = p._plan_stage
    p._plan_stage = lambda *_: (None, None, 'IK timed out')
    assert p.plan_sequence(poses(.3))[0] is None
    p._plan_stage = original
    assert p.plan_sequence(poses(.3))[0]


def test_reversing_approach_preserves_positions_timing_and_derivative_signs():
    t = trajectory(.2, .1)
    t.joint_trajectory.points[0].velocities = [-.1]*6
    t.joint_trajectory.points[-1].velocities = [-.2]*6
    t.joint_trajectory.points[0].accelerations = [.3]*6
    reverse = PlanOnlyMoveIt.reverse_trajectory(t)
    assert list(reverse.joint_trajectory.points[0].positions) == [.1]*6
    assert list(reverse.joint_trajectory.points[-1].positions) == [.2]*6
    assert list(reverse.joint_trajectory.points[0].velocities) == [.2]*6
    assert list(reverse.joint_trajectory.points[-1].accelerations) == [.3]*6
    assert reverse.joint_trajectory.points[0].time_from_start.sec == 0
    assert reverse.joint_trajectory.points[-1].time_from_start.sec == 1
    assert list(t.joint_trajectory.points[0].positions) == [.2]*6


def test_recovery_connects_contact_branch_and_rejects_a_discontinuity():
    p = planner()
    initial = PlanOnlyMoveIt.joints_to_state([0.]*6)
    contact = PlanOnlyMoveIt.joints_to_state([.2]*6)
    p.collision_aware_ik = lambda *_, **__: (contact, '')
    p.plan_cartesian = lambda *_: (trajectory(.2, .1), '')
    p.plan_joints = lambda *_: (NS(planned_trajectory=trajectory(0., .1), trajectory_start=initial), '')
    assert p._recover_approach(PoseStamped(), PoseStamped(), initial, initial) is not None
    p.plan_joints = lambda *_: (NS(planned_trajectory=trajectory(0., .11), trajectory_start=initial), '')
    assert p._recover_approach(PoseStamped(), PoseStamped(), initial, initial) is None
