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


def run_fake(stages, observation='empty', fail_stage=None, validation=True, place_error=False):
    plan = DisplayTrajectory(trajectory=[RobotTrajectory() for _ in stages])
    checked = NS(success=validation, message=json.dumps({
        'plan_fingerprint': trajectory_fingerprint(plan.trajectory), 'tool': np.eye(4).tolist(),
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


def test_failed_perception_recheck_sends_no_arm_or_gripper_goals():
    fake, calls = run_fake(['PREGRASP', 'GRASP', 'LIFT', 'RETREAT'], validation=False)
    assert not fake._outcome['ok'] and calls == []


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
