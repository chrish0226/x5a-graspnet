import math
from types import SimpleNamespace

import pytest

from x5a_pick_place.reobserve import ReobserveWindow, bounded_correction
from x5a_pick_place.pick_place_node import PickPlaceNode


def test_requires_new_frames_after_stop_and_rejects_duplicate_cache():
    window = ReobserveWindow(1_000_000_000)
    xyz = (0.2, 0.3, 0.03)
    assert window.add(999_000_000, xyz, True) is None
    for _ in range(20):
        assert window.add(1_100_000_000, xyz, True) is None
    for ns in (1_170_000_000, 1_240_000_000, 1_310_000_000):
        assert window.add(ns, xyz, True) is None
    assert window.add(1_380_000_000, xyz, True) == xyz


def test_unstable_frames_and_detection_gaps_restart_window():
    window = ReobserveWindow(0)
    xyz = (0.2, 0.3, 0.03)
    for i in range(1, 5):
        window.add(i * 70_000_000, xyz, True)
    assert window.add(280_000_000, xyz, False) is None
    assert window.add(350_000_000, xyz, True) is None
    assert window.add(900_000_000, xyz, True) is None
    assert len(window.samples) == 1


def test_jitter_and_nonfinite_depth_cannot_pass():
    window = ReobserveWindow(0)
    for i in range(1, 10):
        assert window.add(i * 70_000_000, (0.2 + (-1)**i * 0.005, 0.3, 0.03), True) is None
    assert window.add(700_000_000, (math.nan, 0.3, 0.03), True) is None
    assert not window.samples


def test_cross_view_change_is_bounded_in_all_three_dimensions():
    assert bounded_correction((0.2, 0.3, 0.03), (0.205, 0.297, 0.032), 0.015) == pytest.approx((0.005, -0.003, 0.002))
    with pytest.raises(ValueError):
        bounded_correction((0.2, 0.3, 0.03), (0.21, 0.31, 0.04), 0.015)


def correction_fixture(observations):
    events = []
    iterator = iter(observations)
    n = SimpleNamespace(
        reobserve_max_delta=0.015, obj_sz=0.03, obj_sx=0.03, obj_sy=0.03,
        gx=0., gy=0., gz=0.03, base_frame='base_link', transit_vel=0.82, transit_acc=0.43,
        last_joints=[1]*6,
        reobserve_backoff=0.,
        reobserve_verify_tolerance=.003,
        wait_for_reobserved_cube=lambda: next(iterator),
        report_stage=lambda stage: events.append(stage),
        log_stage=lambda *args, **kwargs: None,
        publish_box=lambda *args, **kwargs: None,
        pose_rpy=lambda x, y, z, rpy: (x, y, z, tuple(rpy)),
    )
    def move(pose, stage, vel, acc):
        events.append((pose, stage, vel, acc, n.last_joints))
        return True
    n.move_pose = move
    return n, events


@pytest.mark.parametrize('observed', [None, (0.24, 0.3, 0.03)])
def test_missing_or_large_correction_sends_no_motion(observed):
    n, events = correction_fixture([observed])
    assert PickPlaceNode.correct_pregrasp(n, (0.2, 0.3, 0.03), [0., 1.45, 0.], 0.18) is None
    assert not any(isinstance(e, tuple) for e in events)


def test_correction_replans_from_feedback_keeps_speed_and_checks_again():
    n, events = correction_fixture([(0.205, 0.3, 0.03), (0.206, 0.3, 0.03)])
    result = PickPlaceNode.correct_pregrasp(n, (0.2, 0.3, 0.03), [0., 1.45, 0.], 0.18)
    assert result == (0.206, 0.3, 0.03)
    motion = next(e for e in events if isinstance(e, tuple))
    assert motion[0][:3] == pytest.approx((0.205, 0.3, 0.24))
    assert motion[0][3] == (0., 1.45, 0.)
    assert motion[2:4] == (0.82, 0.43)
    assert motion[4] is None
    assert 'VERIFYING_CORRECTION' in events


def test_five_hz_frames_with_timestamp_jitter_form_a_new_window():
    window = ReobserveWindow(0, max_gap_s=.4)
    for index in range(1, 5):
        assert window.add(index*200050000, (.2, .3, .03), True) is None
    assert window.add(5*200050000, (.2, .3, .03), True) == (.2, .3, .03)
    assert window.add(2000000000, (.2, .3, .03), True) is None


def test_correction_preserves_observation_backoff_and_reports_cube_position():
    n, events = correction_fixture([(0.205, 0.3, 0.03), (0.206, 0.3, 0.03)])
    n.reobserve_backoff = .05
    actual = PickPlaceNode.correct_pregrasp(n, (.2, .3, .03), [0., 1.1, 0.], .18)
    assert actual == (.206, .3, .03)
    motion = next(e for e in events if isinstance(e, tuple))
    assert motion[0][:3] == pytest.approx((.155, .3, .24))


def test_residual_after_correction_blocks_grasp_target():
    n, events = correction_fixture([(0.205, 0.3, 0.03), (0.210, 0.3, 0.03)])
    assert PickPlaceNode.correct_pregrasp(n, (0.2, 0.3, 0.03), [0., 1.45, 0.], 0.18) is None


def test_user_five_mm_tolerance_accepts_four_mm_but_still_blocks_large_residual():
    for residual, allowed in ((.004, True), (.006, False)):
        n, _ = correction_fixture([(.205, .3, .03), (.205+residual, .3, .03)])
        n.reobserve_verify_tolerance = .005
        result = PickPlaceNode.correct_pregrasp(n, (.2, .3, .03), [0., 1.1, 0.], .18)
        assert (result is not None) == allowed


@pytest.mark.parametrize('corrected', [None, (0.205, 0.3, 0.032)])
def test_task_blocks_descent_on_failure_and_uses_corrected_target_on_success(corrected):
    calls = []
    n = SimpleNamespace(
        vision_enabled=True, plan_only=False, reobserve_enabled=True,
        require_servo_bounds=False,
        pre_grasp_ready_enabled=True,
        approach_precheck=False,
        initial_observation_enabled=False,
        reobserve_backoff=0.,
        reobserve_height=0.18, pre_grasp_h=0.06,
        obj_sz=0.03, obj_sx=0.03, obj_sy=0.03,
        gx=0., gy=0., gz=0.03, place_z=0.144,
        transit_vel=0.82, transit_acc=0.43, precision_vel=0.33, precision_acc=0.19,
        vision_pose=None,
        reset_task_state=lambda *a: None, clear_planning_scene_objects=lambda: None,
        wait_ready=lambda: True, log_calibration=lambda: None, handeye_tf_ok=lambda: True,
        report_stage=lambda stage, **kw: calls.append(stage),
        wait_for_frozen_vision=lambda **kw: ((0.2, 0.3, 0.03), (0.3, 0.35, 0.)),
        setup_scene=lambda: None, setup_box_walls=lambda *a: None,
        grasp_orientation_candidates=lambda *a: [[0., 1.45, 0.]],
        place_orientation_candidates=lambda *a: [[0., 1.45, 0.]],
        log_stage=lambda *a, **kw: None, _cancelled=lambda: False,
        call_gripper=lambda opening: calls.append(('gripper', opening)) or True,
        move_pre_grasp_ready=lambda *a: True,
        move_pose_candidates=lambda *a, **kw: [0., 1.45, 0.],
        correct_pregrasp=lambda *a: corrected,
        pose_rpy=lambda x, y, z, rpy: (x, y, z),
        publish_box=lambda *a, **kw: None,
        # Stop at the first descent instead of simulating the rest of a task.
        vertical_move=lambda pose, *a: (calls.append(('descent', pose)) or False, 0.),
        _fail=lambda message: (False, message),
    )
    ok, _ = PickPlaceNode.execute_pick_place(n, 'red')
    assert not ok
    assert ('gripper', False) not in calls
    descents = [c for c in calls if isinstance(c, tuple) and c[0] == 'descent']
    if corrected is None:
        assert not descents and 'APPROACH' not in calls
    else:
        assert descents[0][1] == pytest.approx((0.205, 0.3, 0.062))
