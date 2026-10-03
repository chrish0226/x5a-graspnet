"""Parallel checks must preserve grasp preference and fail closed."""
import threading

import numpy as np
import pytest

from x5a_remote_grasp.parallel_geometry import ordered_geometry_map
from x5a_remote_grasp.grasp_quality import (
    PreparedGraspScene, GripperGeometry, assess_tool_grasp,
)
from test_grasp_quality import contact_cloud


@pytest.mark.parametrize('obstacle', ['none', 'approach', 'closing'])
def test_prepared_scene_preserves_contact_and_collision_decisions(obstacle):
    target = contact_cloud()
    scene = target.copy()
    if obstacle == 'approach':
        scene = np.vstack([scene, contact_cloud(x_span=(-.125, -.115))])
    if obstacle == 'closing':
        scene = np.vstack([scene, contact_cloud(y_span=(.03, .038))])
    reference = assess_tool_grasp(target, scene, np.eye(4), GripperGeometry())
    prepared = PreparedGraspScene(scene)
    assert assess_tool_grasp(target, prepared, np.eye(4), GripperGeometry()) == reference
    # Subsequent capture updates must not mutate a scene used by running checks.
    scene[:] = 9.
    assert assess_tool_grasp(target, prepared, np.eye(4), GripperGeometry()) == reference
    with pytest.raises(ValueError):
        prepared.points[0, 0] = 9.


def test_parallel_completion_order_does_not_change_candidate_order():
    second_done = threading.Event()
    finished = []
    def check(index):
        if index == 0:
            assert second_done.wait(timeout=2)
        finished.append(index)
        if index == 1:
            second_done.set()
        return index
    assert list(ordered_geometry_map(check, range(5), workers=2)) == list(range(5))
    assert finished.index(1) < finished.index(0)


@pytest.mark.parametrize('workers', [1, 2])
def test_cancelled_check_cannot_yield_an_accepted_candidate(workers):
    cancelled = threading.Event()
    seen = []
    def check(index):
        seen.append(index)
        cancelled.set()
        return {'valid': True}
    assert list(ordered_geometry_map(check, range(100), workers, cancelled.is_set)) == []
    assert len(seen) <= workers


def test_worker_exception_aborts_selection():
    def check(index):
        if index == 1:
            raise ValueError('invalid captured geometry')
        return index
    results = ordered_geometry_map(check, range(100), workers=2)
    assert next(results) == 0
    with pytest.raises(ValueError, match='invalid captured geometry'):
        next(results)


def test_prepared_scene_rejects_nonfinite_obstacles():
    with pytest.raises(ValueError):
        PreparedGraspScene(np.array([[float('nan'), 0., 0.]]))
