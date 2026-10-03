"""A held cube must clear the fingers before release over a small support."""
from functools import partial
from types import SimpleNamespace as NS
import time
import numpy as np
import pytest
pytest.importorskip('rclpy')
from x5a_remote_grasp.geometry import rpy_rotation
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode


def test_final_collision_uses_smallest_clear_release_height_and_same_center():
    x, y, z = np.meshgrid(np.linspace(-.015, .015, 8), np.linspace(-.015, .015, 8), np.linspace(-.04, -.01, 8))
    cloud = np.c_[x.ravel(), y.ravel(), z.ravel()]
    R = rpy_rotation(0, 1.45, 1.1)
    surface = np.array([.2, .32, .034])
    attempts = []
    def plan(poses):
        tool = poses[1][1]
        gap = RemoteGraspNode._place_clearance(tool, surface, cloud)
        attempts.append(gap)
        world = cloud @ tool[:3, :3].T+tool[:3, 3]
        lo, hi = np.percentile(world, [2, 98], axis=0)
        np.testing.assert_allclose((lo[:2]+hi[:2])/2, surface[:2], atol=1e-12)
        if gap < .017:
            return None, 'PLACE: collision in last millimetres'
        return NS(stages=['TRANSPORT', 'PLACE', 'RETREAT']), ''
    fake = NS(place_object_cloud_tool=cloud, place_grasp_rotation=R, place_relation='on',
        place_release_clearances=[.010, .015, .018, .020], place_pitches=[1.45],
        reachability=NS(order_pitches=lambda values, *_: values), transit_pitches=[1.15],
        lift_distance=.08, _rpy_matrix=RemoteGraspNode._rpy_matrix,
        _make_tf=RemoteGraspNode._make_tf, _pose_stamped=lambda t, _: t,
        _in_workspace_place=lambda _: True, _place_point_candidates=lambda _: [surface],
        _budget_exhausted=lambda _: False, _ik_joint_seeds=lambda *_: [],
        planner=NS(plan_sequence=plan),
        get_logger=lambda: NS(info=lambda _: None), _record_reject=lambda *_, **kw: None)
    fake._place_pose_variants = partial(RemoteGraspNode._place_pose_variants, fake)
    selected = RemoteGraspNode._plan_place_only(fake, NS(is_cancel_requested=False),
        (surface, 'on'), None, time.monotonic()+10)
    assert selected is not None
    assert RemoteGraspNode._place_clearance(selected[0]['place'], surface, cloud) == pytest.approx(.018)
    assert min(attempts) == pytest.approx(.010)
    assert max(attempts) < .019
    assert all(a <= b+1e-12 for a, b in zip(attempts, attempts[1:]))


def test_place_failures_survive_budget_exit_without_overwriting_planning_feedback():
    surface = np.array([.2, .32, .078])
    calls = []
    fake = NS(planner=NS(plan_sequence=lambda _: (None, 'PLACE: incomplete approach')),
              transit_pitches=[1.15], _place_point_candidates=lambda _: [surface],
              _place_pose_variants=lambda *args: iter([([], {'retreat_clearance': np.eye(4)})]*3),
              _budget_exhausted=lambda _: len(calls) >= 2,
              _ik_joint_seeds=lambda *_: [], _pose_stamped=lambda t, _: t,
              _feedback=lambda *args: pytest.fail('Planning phase must stay visible'))
    def record(report, *args, **kwargs):
        calls.append(args[2])
        RemoteGraspNode._record_reject(fake, report, *args, **kwargs)
    fake._record_reject = record
    report = []
    selected = RemoteGraspNode._plan_place_only(fake, NS(is_cancel_requested=False),
        (surface, 'on'), None, time.monotonic()+10, report)
    assert selected is None
    assert [entry['reason'] for entry in report] == ['place_plan', 'place_plan', 'place_budget']
    assert 'budget exhausted' in report[-1]['detail']
