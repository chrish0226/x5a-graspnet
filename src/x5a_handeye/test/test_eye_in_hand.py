import json
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

from x5a_handeye.solve_handeye import solve_opencv_handeye, evaluate, motion_diversity_report
from x5a_handeye.transforms import inv_T
from x5a_handeye.calibrated_tf import calibrated_transform, camera_root_transform


def transform(r, t):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(r, float))[0]
    T[:3, 3] = t
    return T


def dataset(mode):
    rng = np.random.default_rng(12)
    X = transform([.2, -.5, .4], [.06, -.02, .09])
    Y = transform([-.4, .2, .1], [.35, .1, .07])
    samples = []
    for i in range(30):
        B = transform(rng.uniform(-1, 1, 3), rng.uniform(-.3, .3, 3))
        C = inv_T(X) @ (inv_T(B) @ Y if mode == 'eye_in_hand' else B @ Y)
        samples.append({'index': i, 'T_base_tool': B, 'T_camera_board': C})
    return samples, X, Y


@pytest.mark.parametrize('mode', ['eye_in_hand', 'eye_to_hand'])
def test_recovers_known_extrinsic_and_independent_board_poses(mode):
    samples, X, Y = dataset(mode)
    actual_X, actual_Y = solve_opencv_handeye(samples[:24], cv2.CALIB_HAND_EYE_PARK, mode)
    np.testing.assert_allclose(actual_X, X, atol=1e-8)
    np.testing.assert_allclose(actual_Y, Y, atol=1e-8)
    error = evaluate(samples[24:], actual_X, actual_Y, mode)
    assert error['translation_max_m'] < 1e-8
    assert error['rotation_mean_deg'] < 1e-4


@pytest.mark.parametrize('mode', ['eye_in_hand', 'eye_to_hand'])
def test_cli_mode_specific_schema_and_tf_parent(tmp_path, mode):
    samples, X, _ = dataset(mode)
    raw = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()} for s in samples]
    source, output = tmp_path/'samples.json', tmp_path/'result.yaml'
    source.write_text(json.dumps({'calibration_type': mode, 'camera_frame': 'wrist_camera_color_optical_frame', 'samples': raw}))
    proc = subprocess.run([sys.executable, '-m', 'x5a_handeye.solve_handeye', '--samples', str(source),
                           '--output', str(output), '--holdout', '6', '--camera-root-frame', 'wrist_camera_link'],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = yaml.safe_load(output.read_text())
    assert result['verdict'] == 'PASS'
    assert result['samples']['holdout'] == 6
    parent, root, optical, actual = calibrated_transform(result)
    assert parent == ('tool0' if mode == 'eye_in_hand' else 'base_link')
    assert root == 'wrist_camera_link' and optical == 'wrist_camera_color_optical_frame'
    np.testing.assert_allclose(actual, X, atol=1e-8)
    proc = subprocess.run([sys.executable, '-m', 'x5a_handeye.solve_handeye', '--samples', str(source),
                           '--calibration-type', 'eye_to_hand' if mode == 'eye_in_hand' else 'eye_in_hand'],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and 'differs' in proc.stderr


def test_tf_optical_conversion_and_camera_motion():
    samples, X, Y = dataset('eye_in_hand')
    L = transform([-.7, .3, .9], [.01, .02, -.01])
    root = camera_root_transform(X, L)
    np.testing.assert_allclose(root @ L, X, atol=1e-10)
    for s in samples:
        np.testing.assert_allclose(s['T_base_tool'] @ root @ L @ s['T_camera_board'], Y, atol=1e-10)


@pytest.mark.parametrize('fault', ['failed', 'bad_parent', 'reflection', 'nan'])
def test_rejects_unsafe_calibration_results(fault):
    result = {'calibration_type': 'eye_in_hand', 'T_tool_camera': np.eye(4).tolist()}
    if fault == 'failed': result['verdict'] = 'FAIL'
    elif fault == 'bad_parent': result['parent_frame'] = 'base_link'
    elif fault == 'reflection': result['T_tool_camera'][0][0] = -1
    else: result['T_tool_camera'][0][3] = float('nan')
    with pytest.raises(ValueError): calibrated_transform(result)


def test_rotation_diversity_rejects_translation_only_and_parallel_axes():
    samples, _, _ = dataset('eye_in_hand')
    assert motion_diversity_report(samples)['ok']
    for i, s in enumerate(samples): s['T_base_tool'] = transform([0, 0, 0], [i*.01, 0, 0])
    assert not motion_diversity_report(samples)['ok']
    for i, s in enumerate(samples): s['T_base_tool'] = transform([0, 0, i*.15], [0, 0, 0])
    assert not motion_diversity_report(samples)['ok']
