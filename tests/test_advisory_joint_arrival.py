"""Controller arrival residuals, using a simulated clock and fake feedback."""
from types import SimpleNamespace as NS

import pytest
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from x5a_moveit_official_adapter.official_trajectory_adapter import OfficialTrajectoryAdapter


def execute(monkeypatch, residual, strict=False, fault='', target=.2):
    import x5a_moveit_official_adapter.official_trajectory_adapter as module
    clock = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    command, logs = [0.]*6, []
    def snapshot():
        actual = list(command)
        if command[-1] > 0:
            actual[-1] -= residual
        return actual, [0.]*6, 0., clock[0]
    first = JointTrajectoryPoint(positions=[0.]*6)
    last = JointTrajectoryPoint(positions=[target]*6)
    last.time_from_start.nanosec = 200_000_000
    outcome = []
    handle = NS(request=NS(trajectory=NS(points=[first, last])), is_cancel_requested=False,
                succeed=lambda: outcome.append('success'), abort=lambda: outcome.append('abort'))
    fake = NS(snapshot=snapshot, start_tolerance=.15, max_speed=1., rate_hz=50.,
        status_timeout=.5, command_topic='/arm_cmd', tracking_grace=.75,
        goal_time_tolerance=2., goal_tolerance=.02, enforce_goal_tolerance=strict,
        goal_advisory_tolerance=.06, goal_advisory_settle=.15,
        response_command_delta=.03, response_actual_delta=.002,
        count_publishers=lambda _: 1, readiness_fault=lambda: fault,
        driver_health_fault=lambda: fault, goal_active=True,
        _desired_at=OfficialTrajectoryAdapter._desired_at, _feedback=lambda *_: None,
        _fmt_joints=OfficialTrajectoryAdapter._fmt_joints,
        _publish_command=lambda q, *_: command.__setitem__(slice(None), q),
        get_logger=lambda: NS(info=logs.append, warn=logs.append, error=logs.append))
    fake._arm_abort = lambda *args, **kwargs: OfficialTrajectoryAdapter._arm_abort(fake, *args, **kwargs)
    result = OfficialTrajectoryAdapter._execute(fake, handle)
    return result, outcome, clock[0], logs


def test_observed_joint6_residual_is_advisory_and_does_not_wait_two_seconds(monkeypatch):
    result, outcome, elapsed, logs = execute(monkeypatch, .0209)
    assert result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
    assert outcome == ['success'] and elapsed < .5
    assert any('仅提示' in line for line in logs)


def test_strict_mode_preserves_original_goal_tolerance(monkeypatch):
    result, outcome, elapsed, _ = execute(monkeypatch, .0209, strict=True)
    assert result.error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
    assert outcome == ['abort'] and elapsed >= 2.


def test_large_residual_and_driver_fault_are_not_reported_as_arrival(monkeypatch):
    result, outcome, _, _ = execute(monkeypatch, .10)
    assert result.error_code != 0 and outcome == ['abort']


def test_unresponsive_wrist_is_not_hidden_by_advisory_accuracy(monkeypatch):
    result, outcome, _, _ = execute(monkeypatch, .04, target=.04)
    assert result.error_code != 0 and outcome == ['abort']
    result, outcome, _, _ = execute(monkeypatch, .0209, fault='motor feedback lost')
    assert result.error_code != 0 and outcome == ['abort']
