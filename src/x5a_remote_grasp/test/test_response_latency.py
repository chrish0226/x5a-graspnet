"""Independent same-frame inference must overlap without changing results."""
import threading
from types import SimpleNamespace as NS
import pytest
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode


def test_combined_grasp_and_place_requests_overlap_and_keep_both_results(tmp_path):
    grasp_started, place_started = threading.Event(), threading.Event()
    encoded, transform = object(), object()
    payload, destination = {'grasps': []}, ('surface', 'on')
    def grasp(frame, fields, matrix, directory):
        assert frame is encoded and matrix is transform and directory == tmp_path
        grasp_started.set()
        assert place_started.wait(2), 'placement was serialized after grasp inference'
        return payload
    def place(goal, frame, matrix, prompt, relation, minimum, maximum, directory):
        assert frame is encoded and matrix is transform
        assert (prompt, relation, minimum, maximum) == ('white cube', 'on', .1, 1.5)
        place_started.set()
        assert grasp_started.wait(2), 'grasp inference waited for placement'
        return destination
    fake = NS(_infer_target_grasps=grasp, _locate_place_target=place)
    actual = RemoteGraspNode._infer_scene_targets(fake, encoded, {}, transform,
        tmp_path, None, 'white cube', 'on', .1, 1.5)
    assert actual[0] is payload and actual[1] is destination


def test_single_grasp_does_not_request_an_unrelated_place():
    payload = {'grasps': []}
    def forbidden(*_):
        pytest.fail('single grasp requested a place')
    fake = NS(_infer_target_grasps=lambda *_: payload, _locate_place_target=forbidden)
    assert RemoteGraspNode._infer_scene_targets(fake, None, {}, None, None,
        None, None, None, .1, 1.5) == (payload, None)


def test_parallel_placement_failure_is_not_ignored():
    def fail(*_):
        raise RuntimeError('no valid destination depth')
    fake = NS(_infer_target_grasps=lambda *_: {'grasps': []}, _locate_place_target=fail)
    with pytest.raises(RuntimeError, match='destination depth'):
        RemoteGraspNode._infer_scene_targets(fake, None, {}, None, None,
            None, 'white cube', 'on', .1, 1.5)
