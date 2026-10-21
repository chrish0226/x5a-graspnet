"""Foreground-depth and motion-gate regressions; never send controller goals."""
import base64
import json
import time
import threading
from types import MethodType, SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from x5a_remote_grasp.api_client import GraspAPIClient, HTTPResult
from x5a_remote_grasp.grasp_quality import depth_cloud, height_statistics
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
from x5a_remote_grasp.target_depth import (
    target_surface, target_motion_landmarks, stable_target_motion,
)


def frame():
    depth = np.full((80, 100), .48)
    mask = np.zeros(depth.shape, bool)
    mask[10:45, 20:50] = True
    depth[10:45, 20:50] = .25
    mask[45:78, 50:98] = True  # background has more points than the target
    mask[:5, :5] = True
    depth[:5, :5] = .15  # tiny near-camera speck must not win
    transform = np.diag([1., -1., -1., 1.])
    transform[:3, 3] = [.1, .2, .6]
    return depth, mask, transform


def filtered(depth, mask, transform):
    return target_surface(depth, mask, (400., 400., 50., 40.), transform,
                          table_top=0.)


def encode(array):
    return cv2.imencode('.png', array)[1].tobytes()


def encoded(depth):
    return NS(rgb_png=b'original RGB bytes', depth_png=encode((depth*1000).astype('uint16')),
              fx=400., fy=400., cx=50., cy=40., depth_scale_m_per_unit=.001,
              depth_scale_divisor=1000.)


def payload(mask):
    return dict(quality_schema=2, target_mask_png_b64=base64.b64encode(
        encode(mask.astype('uint8')*255)).decode(), instance={'sam_score': .98})


def fake_node():
    node = NS(min_depth_default=.1, max_depth_default=1.5,
              table_center=np.array([.1, .2, -.02]), table_size=np.array([.5, .5, .04]),
              workspace={'z_max': .5, 'r_max': .54},
              get_logger=lambda: NS(info=lambda _: None, warn=lambda _: None))
    for name in ('_target_depth', '_target_surface', '_quality_clouds', '_infer_target_grasps'):
        setattr(node, name, MethodType(getattr(RemoteGraspNode, name), node))
    return node


def test_foreground_beats_larger_background_and_near_speck_without_filling_holes():
    depth, mask, transform = frame()
    depth[20:24, 30:34] = 0
    clean, report = filtered(depth, mask, transform)
    expected = np.zeros(mask.shape, bool)
    expected[10:45, 20:50] = True
    expected[20:24, 30:34] = False
    np.testing.assert_array_equal(clean, expected)
    assert report['rejected_points'] > report['selected_points']
    assert not np.any(clean & ~mask)
    np.testing.assert_array_equal(filtered(depth, clean, transform)[0], clean)


def test_background_depth_change_cannot_fake_motion_of_the_foreground():
    depth, mask, transform = frame()
    first, quality = filtered(depth, mask, transform)
    changed = depth.copy()
    changed[45:78, 50:98] = .8  # projects below the table
    second, checked = filtered(changed, mask, transform)
    np.testing.assert_array_equal(first, second)
    assert stable_target_motion(quality['center_base'],
        [checked['center_base'], checked['center_base']])[0] == pytest.approx(0.)
    old_center = height_statistics(depth_cloud(depth, 400, 400, 50, 40)[mask], transform)[1]
    changed_center = height_statistics(depth_cloud(changed, 400, 400, 50, 40)[mask], transform)[1]
    assert np.linalg.norm(old_center-changed_center) > .1


def test_missing_or_scattered_depth_fails_closed():
    depth, mask, transform = frame()
    with pytest.raises(ValueError, match='有效深度不足'):
        filtered(np.zeros_like(depth), mask, transform)
    with pytest.raises(ValueError, match='有效深度不足'):
        filtered(np.full_like(depth, 1.4), mask, transform)


def test_many_impossible_background_returns_do_not_dilute_real_target_support():
    depth, mask, transform = frame()
    depth[:] = 1.4  # thousands of returns behind/below the physical table
    mask[:] = True
    depth[10:35, 20:50] = .25
    clean, report = filtered(depth, mask, transform)
    assert report['selected_points'] == 750
    assert report['selected_points'] < .1*report['raw_points']
    assert report['plausible_coverage'] == 1.
    assert clean.sum() == 750


def test_depth_holes_group_nearby_fragments_without_adding_contact_points():
    depth, mask, transform = frame()
    depth[:] = .25
    mask[:] = False
    mask[10:45, 10:35] = True
    mask[10:45, 50:75] = True  # observed halves separated by a small depth hole
    clean, report = filtered(depth, mask, transform)
    np.testing.assert_array_equal(clean, mask)
    assert report['selected_points'] == mask.sum()
    assert report['components'] == 1


@pytest.mark.parametrize('axis', range(3))
def test_real_11mm_target_motion_is_still_rejected(axis):
    moved = np.zeros(3)
    moved[axis] = .011
    with pytest.raises(ValueError, match='超过 10.0mm'):
        stable_target_motion(np.zeros(3), [moved, moved])


def test_unstable_depth_is_not_averaged_into_a_safe_position():
    with pytest.raises(ValueError, match='定位不稳定'):
        stable_target_motion([0, 0, 0], [[.012, 0, 0], [0, 0, 0]])
    assert stable_target_motion([0, 0, 0], [[.001, 0, 0], [.002, 0, 0]]) == pytest.approx((.002, .001))


def motion_fixture():
    mask = np.zeros((160, 300), bool)
    mask[40:100, 80:160] = True
    return mask, (1000., 1000., 150., 80.)


def test_missing_deeper_body_points_do_not_move_the_near_surface_reference():
    mask, intrinsics = motion_fixture()
    original = np.r_[np.full(400, .21), np.full(600, .22)]
    changed = np.r_[np.full(400, .21), np.full(600, .26)]
    a = target_motion_landmarks(mask, original, intrinsics, np.eye(4))
    b = target_motion_landmarks(mask, changed, intrinsics, np.eye(4))
    np.testing.assert_array_equal(a, b)
    assert stable_target_motion(a, [b, b]) == (0., 0.)
    # The old depth-extent midpoint would falsely shift by 20 mm.
    assert abs(np.mean(np.percentile(original, [5, 95]))-
               np.mean(np.percentile(changed, [5, 95]))) == pytest.approx(.02)


@pytest.mark.parametrize('axis', range(3))
def test_landmarks_still_detect_real_11mm_motion(axis):
    mask, intrinsics = motion_fixture()
    depths = np.full(1000, .25)
    reference = target_motion_landmarks(mask, depths, intrinsics, np.eye(4))
    moved = np.zeros_like(mask)
    if axis == 0:
        moved[:, 44:] = mask[:, :-44]  # 44 pixels * .25 / 1000 = 11 mm
    elif axis == 1:
        moved[44:, :] = mask[:-44, :]
    else:
        moved = mask
        depths = depths+.011
    actual = target_motion_landmarks(moved, depths, intrinsics, np.eye(4))
    with pytest.raises(ValueError, match='超过 10.0mm'):
        stable_target_motion(reference, [actual, actual])


def test_silhouette_edge_change_is_checked_even_when_center_shift_is_small():
    mask, intrinsics = motion_fixture()
    depth = np.full(1000, .25)
    reference = target_motion_landmarks(mask, depth, intrinsics, np.eye(4))
    mask[40:100, 160:220] = True  # right edge moves 15 mm; center only 7.5 mm
    actual = target_motion_landmarks(mask, depth, intrinsics, np.eye(4))
    assert np.linalg.norm(actual[0]-reference[0]) < .01
    with pytest.raises(ValueError, match='超过 10.0mm'):
        stable_target_motion(reference, [actual, actual])


def test_uncertain_near_surface_or_invalid_limits_cannot_pass_motion_gate():
    mask, intrinsics = motion_fixture()
    a = target_motion_landmarks(mask, np.full(100, .21), intrinsics, np.eye(4))
    b = target_motion_landmarks(mask, np.full(100, .215), intrinsics, np.eye(4))
    with pytest.raises(ValueError, match='定位不稳定'):
        stable_target_motion(a, [a, b])
    with pytest.raises(ValueError):
        stable_target_motion(a, [a, a], max_motion=float('nan'))


def test_clean_target_mask_reaches_graspnet_while_original_depth_and_scene_survive(tmp_path):
    depth, mask, transform = frame()
    depth[45:78, 50:98] = .45  # obstacle within the existing 22 cm scene crop
    node = fake_node()
    image = encoded(depth)
    requests = []
    def grasp(rgb, original_depth, fields, mask_png):
        requests.append((rgb, original_depth, fields, mask_png))
        return {'instance': {'selected_by': 'client_mask'}}
    node._api_client = lambda: NS(locate=lambda *args: payload(mask), grasp=grasp)
    fields = dict(prompt='bottle', fx=400., fy=400., cx=50., cy=40., depth_scale=1000.,
                  threshold=.45, min_depth=.1, max_depth=1.5, mask_dilation_px=2,
                  collision_thresh=.01)
    result = node._infer_target_grasps(image, fields, transform, tmp_path)
    rgb, original_depth, sent, png = requests[0]
    assert rgb == image.rgb_png and original_depth == image.depth_png
    assert sent['mask_dilation_px'] == 0 and sent['collision_thresh'] == .01
    assert fields['mask_dilation_px'] == 2  # caller configuration is immutable
    clean = cv2.imdecode(np.frombuffer(png, np.uint8), 0) > 127
    np.testing.assert_array_equal(clean, filtered(depth, mask, transform)[0])
    assert result['instance']['sam_score'] == .98
    target, scene, _, _ = node._quality_clouds(image, payload(mask), transform)
    # The rejected background is still an obstacle in the original scene.
    assert len(scene) and np.min(scene[:, 2]) < np.min(target[:, 2])-.1
    assert (tmp_path/'target_depth_quality.json').exists()


def test_api_client_really_uploads_optional_mask(monkeypatch):
    client = GraspAPIClient('http://localhost:8300')
    calls = []
    monkeypatch.setattr(client, '_json_request', lambda *args, **kwargs:
        (calls.append(kwargs) or HTTPResult(200, {'grasps': []})))
    client.grasp(b'RGB', b'DEPTH', {'prompt': 'bottle'}, mask_png=b'EXACT-MASK')
    assert b'name="mask"' in calls[0]['body'] and b'EXACT-MASK' in calls[0]['body']
    assert b'DEPTH' in calls[0]['body']


@pytest.mark.parametrize('mode', ['stable', 'moving', 'unstable', 'stale'])
def test_execution_gate_requires_two_fresh_stable_frames_and_keeps_motion_limit(mode, tmp_path, monkeypatch):
    import x5a_remote_grasp.remote_grasp_node as module
    centers = {'stable': [[.001, 0, 0], [.002, 0, 0]],
               'moving': [[.011, 0, 0], [.011, 0, 0]],
               'unstable': [[.001, 0, 0], [.012, 0, 0]],
               'stale': [[.001, 0, 0], [.001, 0, 0]]}[mode]
    frames = iter([NS(stamp=NS(sec=1, nanosec=0)),
                   NS(stamp=NS(sec=1 if mode == 'stale' else 2, nanosec=0))])
    for_frame = []
    remote_calls = []
    second_requested = threading.Event()
    def locate(*args):
        remote_calls.append(True)
        if len(remote_calls) == 2:
            second_requested.set()
        return {}
    def image():
        item = next(frames)
        item.source_frame = 'camera'
        item.rgb_png = b'RGB'; item.depth_png = b'DEPTH'
        item.fx = item.fy = 400.; item.cx = item.cy = 0.
        item.depth_scale_divisor = 1000.
        return item
    samples = iter(centers)
    current_center = [None]
    target = np.zeros((50, 3))
    def cloud(*args, **kwargs):
        for_frame.append(kwargs['filter_target'])
        if mode == 'stable' and len(for_frame) == 1:
            assert second_requested.wait(2.)  # second request overlaps first geometry
        current_center[0] = np.array(next(samples))
        result = target, target, current_center[0], (0., .1)
        if kwargs.get('with_landmarks'):
            result += (np.tile(current_center[0], (5, 1)),)
        return result
    node = NS(last_plan_validation=dict(captured_monotonic=time.monotonic(),
        center=np.zeros(3), motion_landmarks=np.zeros((5, 3)),
        prompt='bottle', tool=np.eye(4), fingerprint='same-plan',
        capture_directory=tmp_path), goal_active=False, max_capture_to_execute=60.,
        pre_execution_visual_recheck=True,
        max_target_motion=.01, threshold=.45, min_depth_default=.1, max_depth_default=1.5,
        base_frame='base', _encode_latest_frame=image,
        _lookup_matrix=lambda *args: np.eye(4), _quality_clouds=cloud,
        _motion_landmarks=lambda *args: np.tile(current_center[0], (5, 1)),
        _api_client=lambda: NS(locate=locate),
        _save_capture=lambda *args: None, gripper_geometry=None, pregrasp_distance=.05,
        get_logger=lambda: NS(warn=lambda _: None))
    monkeypatch.setattr(module, 'assess_tool_grasp', lambda *args: {'valid': True})
    result = RemoteGraspNode._validate_latest_callback(node, None, NS())
    assert result.success == (mode == 'stable')
    if result.success:
        assert json.loads(result.message)['plan_fingerprint'] == 'same-plan'
        assert for_frame == [True, True]
    assert (tmp_path/'execution_validation.json').exists()


def test_cloud_model_id_mismatch_is_rejected_before_candidates(tmp_path):
    depth,mask,transform=frame();node=fake_node();node.gripper_model=NS(model_id='expected-model')
    def grasp(rgb,original_depth,fields,mask_png):
        assert fields['gripper_model_id']=='expected-model'
        return {'geometry_model_id':'wrong-model'}
    node._api_client=lambda:NS(locate=lambda *args:payload(mask),grasp=grasp)
    fields=dict(prompt='bottle',fx=400.,fy=400.,cx=50.,cy=40.,depth_scale=1000.,
        threshold=.45,min_depth=.1,max_depth=1.5,mask_dilation_px=2,collision_thresh=.01)
    with pytest.raises(RuntimeError,match='夹爪模型不一致'):
        node._infer_target_grasps(encoded(depth),fields,transform,tmp_path)


def static_execution_node(tmp_path, *, grasp=True):
    def no_visual_work(*args, **kwargs):
        pytest.fail('static execution must not capture, locate, or rebuild scene geometry')
    cloud = np.arange(90, dtype=float).reshape(30, 3)*.002
    expected = dict(captured_monotonic=time.monotonic(), fingerprint='static-plan',
        tool=np.eye(4) if grasp else None, place_tool=np.eye(4),
        object_cloud_base=cloud.copy() if grasp else None,
        place_surface=np.zeros(3), place_prompt='box', capture_directory=tmp_path)
    return NS(last_plan_validation=expected, goal_active=False,
        pre_execution_visual_recheck=False, max_capture_to_execute=60.,
        # A later mutable cloud must never replace the saved plan's observation.
        target_cloud=np.full((30, 3), 10.), gripper_model=NS(model_id='model-A'),
        _encode_latest_frame=no_visual_work, _api_client=no_visual_work,
        _lookup_matrix=no_visual_work, _quality_clouds=no_visual_work,
        get_logger=lambda: NS(warn=lambda _: None))


@pytest.mark.parametrize('grasp', [True, False])
def test_static_execution_keeps_plan_contract_without_visual_calls(grasp, tmp_path):
    node = static_execution_node(tmp_path, grasp=grasp)
    result = RemoteGraspNode._validate_latest_callback(node, None, NS())
    assert result.success, result.message
    checked = json.loads(result.message)
    assert checked['plan_fingerprint'] == 'static-plan'
    assert checked['visual_rechecked'] is False and checked['motion_m'] is None
    assert checked['place_tool'] == np.eye(4).tolist()
    assert checked['geometry_model_id'] == 'model-A'
    assert checked['capture_directory'] == str(tmp_path)
    if grasp:
        assert checked['tool'] == np.eye(4).tolist()
        np.testing.assert_allclose(checked['object_cloud_base'],
                                   node.last_plan_validation['object_cloud_base'])
    else:
        assert checked['tool'] is None and checked['object_cloud_base'] is None
    report = json.loads((tmp_path/'execution_validation.json').read_text())
    assert report['success'] and report['observations'] == []
    assert report['source'] == 'planned_observation'
    assert report['visual_recheck_enabled'] is False


@pytest.mark.parametrize('problem', ['expired', 'planning', 'no_plan', 'missing_cloud'])
def test_static_execution_still_rejects_unusable_plan(problem, tmp_path):
    node = static_execution_node(tmp_path)
    if problem == 'expired':
        node.last_plan_validation['captured_monotonic'] -= 61.
    elif problem == 'planning':
        node.goal_active = True
    elif problem == 'no_plan':
        node.last_plan_validation = None
    else:
        node.last_plan_validation.pop('object_cloud_base')
    result = RemoteGraspNode._validate_latest_callback(node, None, NS())
    assert not result.success
    assert result.message
