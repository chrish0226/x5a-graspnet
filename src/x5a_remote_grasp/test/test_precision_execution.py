"""Executor regressions using messages and fakes, never controller actions."""
import json
import threading
from types import SimpleNamespace as NS
import pytest
import numpy as np

pytest.importorskip('rclpy')
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from x5a_remote_grasp.grasp_quality import trajectory_fingerprint
from x5a_remote_grasp.remote_grasp_executor import RemoteGraspExecutor


def run_fake(stages, observation='empty', fail_stage=None, validation=True, place_error=False, contact_error=False,
             tcp_error_m=None, enforce_tcp_tolerance=True, stale_tf=False, plan_fingerprint=None):
    plan = DisplayTrajectory(trajectory=[RobotTrajectory() for _ in stages])
    checked = NS(success=validation, message=json.dumps({
        'plan_fingerprint': (trajectory_fingerprint(plan.trajectory)
                             if plan_fingerprint is None else plan_fingerprint),
        'tool': np.eye(4).tolist(),
        'place_tool': np.eye(4).tolist(), 'object_cloud_base': np.zeros((30, 3)).tolist()}))
    calls = []
    counter = iter(stages)
    def follow(_):
        name = next(counter)
        calls.append(name)
        return (False, -4) if name == fail_stage else (True, 0)
    def verify_place():
        calls.append('VERIFY_PLACE')
        if place_error:
            raise RuntimeError('actual release TCP missed by 8mm')
        if tcp_error_m is not None:
            return RemoteGraspExecutor._verify_tool_pose(fake, np.eye(4), .003, .05, '放置')
    fake = NS(validation_client=NS(wait_for_service=lambda **_: True,
        call_async=lambda _: NS(result=lambda: checked)),
        _wait_future=lambda *_: True, _follow=follow,
        _gripper=lambda *_: calls.append('GRIPPER') or True,
        _verify_grasp_pose=lambda: np.eye(4), _verify_place_pose=verify_place,
        _holding_observation=lambda: observation,
        get_logger=lambda: NS(info=lambda _: None, warn=lambda _: None, error=lambda _: None),
        holding=not ('GRASP' in stages), holding_state='unknown', latest_plan=plan,
        held_geometry=None, open_width=0.044, close_width=0.0,
        running=True, _done=threading.Event())
    fake.enforce_tcp_tolerance = enforce_tcp_tolerance
    fake.base_frame, fake.tcp_frame = 'base', 'tool0'
    fake.get_clock = lambda: NS(now=lambda: NS(nanoseconds=10_000_000_000))
    fake.tf_buffer = NS(lookup_transform=lambda *_: NS(
        header=NS(stamp=NS(sec=9 if stale_tf else 10, nanosec=0)),
        transform=NS(translation=NS(x=tcp_error_m, y=0., z=0.),
                     rotation=NS(x=0., y=0., z=0., w=1.))))
    def verify_contact(checked, actual):
        if contact_error:
            raise RuntimeError('actual TCP has insufficient inset contact')
        return RemoteGraspExecutor._verify_contact_geometry(fake, checked, actual)
    fake._verify_contact_geometry = verify_contact
    RemoteGraspExecutor._run(fake, plan, stages)
    return fake, calls


def test_empty_close_does_not_lift_or_report_success():
    fake, calls = run_fake(['PREGRASP', 'GRASP', 'LIFT', 'RETREAT'])
    assert calls == ['GRIPPER', 'PREGRASP', 'GRASP', 'GRIPPER']
    assert not fake._outcome['ok'] and not fake.holding
    assert fake._outcome['holding_state'] == 'empty'
    assert fake.latest_plan is None


def test_unknown_gripper_feedback_does_not_report_success():
    fake, calls = run_fake(['PREGRASP', 'GRASP', 'LIFT', 'RETREAT'], 'unknown')
    assert not fake._outcome['ok'] and 'LIFT' not in calls
    assert fake._outcome['holding_state'] == 'unknown'


def test_failed_plan_metadata_check_sends_no_arm_or_gripper_goals():
    fake, calls = run_fake(['PREGRASP', 'GRASP', 'LIFT', 'RETREAT'], validation=False)
    assert not fake._outcome['ok'] and calls == []


def test_changed_plan_fingerprint_sends_no_arm_or_gripper_goals():
    fake, calls = run_fake(['PREGRASP', 'GRASP', 'LIFT', 'RETREAT'],
                           plan_fingerprint='different-plan')
    assert not fake._outcome['ok'] and calls == []
    assert '规划已经更换' in fake._outcome['message']


def test_place_tracking_error_does_not_release_held_object():
    fake, calls = run_fake(['TRANSPORT', 'PLACE', 'RETREAT'], fail_stage='PLACE')
    assert not fake._outcome['ok'] and fake.holding
    assert calls == ['TRANSPORT', 'PLACE']


def test_joint_success_with_inaccurate_release_tcp_does_not_open():
    fake, calls = run_fake(['TRANSPORT', 'PLACE', 'RETREAT'], place_error=True)
    assert not fake._outcome['ok'] and fake.holding
    assert calls == ['TRANSPORT', 'PLACE', 'VERIFY_PLACE']


def test_actual_grasp_attaches_cloud_and_verified_release_clears_it():
    fake, calls = run_fake(['PREGRASP', 'GRASP', 'LIFT', 'RETREAT'], 'held')
    assert fake._outcome['ok'] and fake.held_geometry
    np.testing.assert_allclose(fake.held_geometry['cloud_tool'], np.zeros((30, 3)))
    fake, calls = run_fake(['TRANSPORT', 'PLACE', 'RETREAT'])
    assert fake._outcome['ok'] and not fake.holding and fake.held_geometry is None
    assert calls.index('VERIFY_PLACE') < calls.index('GRIPPER')


def test_actual_contact_rejection_never_closes_or_lifts():
    fake, calls = run_fake(['PREGRASP', 'GRASP', 'LIFT', 'RETREAT'], contact_error=True)
    assert calls == ['GRIPPER', 'PREGRASP', 'GRASP']
    assert not fake._outcome['ok'] and 'insufficient inset' in fake._outcome['message']


def test_advisory_tcp_tolerance_releases_and_retreats_after_4_4mm_offset():
    fake, calls = run_fake(['TRANSPORT', 'PLACE', 'RETREAT'],
                          tcp_error_m=.0044, enforce_tcp_tolerance=False)
    assert fake._outcome['ok'] and not fake.holding
    assert calls == ['TRANSPORT', 'PLACE', 'VERIFY_PLACE', 'GRIPPER', 'RETREAT']


def test_strict_tcp_tolerance_still_rejects_same_4_4mm_offset(monkeypatch):
    import x5a_remote_grasp.remote_grasp_executor as module
    ticks = iter([0., 2.])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    fake, calls = run_fake(['TRANSPORT', 'PLACE', 'RETREAT'], tcp_error_m=.0044)
    assert not fake._outcome['ok'] and fake.holding
    assert calls == ['TRANSPORT', 'PLACE', 'VERIFY_PLACE']
    assert '4.4mm' in fake._outcome['message']


def test_advisory_tcp_tolerance_still_requires_fresh_feedback():
    fake, calls = run_fake(['TRANSPORT', 'PLACE', 'RETREAT'],
                          tcp_error_m=.0044, enforce_tcp_tolerance=False, stale_tf=True)
    assert not fake._outcome['ok'] and fake.holding
    assert calls == ['TRANSPORT', 'PLACE', 'VERIFY_PLACE']
    assert '实际 TCP 状态已过期' in fake._outcome['message']
