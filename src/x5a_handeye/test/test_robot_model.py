from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from x5a_handeye.calibrated_tf import calibrated_transform
from x5a_handeye.refine_robot_model import fit_offsets, matrix, split
from x5a_handeye.robot_model import (
    CHAIN_NAMES, corrected_samples, export_candidate_urdf, model_signature,
    require_matching_model,
)
from x5a_handeye.solve_handeye import average_T, evaluate
from x5a_handeye.transforms import inv_T
from x5a_handeye.x5a_fk import fk_base_tool0, _rpy_matrix, _axis_matrix


def source_xml():
    return (Path(__file__).resolve().parents[2] / 'x5a_moveit_config/config/X5A.urdf.xacro').read_text()


def xml_fk(xml, joints):
    root = ET.fromstring(xml)
    T = np.eye(4)
    for i, name in enumerate(CHAIN_NAMES):
        joint = root.find(f"joint[@name='{name}']")
        origin = joint.find('origin')
        O = np.eye(4)
        O[:3, 3] = [float(x) for x in origin.get('xyz').split()]
        O[:3, :3] = _rpy_matrix(*[float(x) for x in origin.get('rpy').split()])
        T = T @ O
        if joint.get('type') != 'fixed':
            R = np.eye(4)
            R[:3, :3] = _axis_matrix([float(x) for x in joint.find('axis').get('xyz').split()], joints[i])
            T = T @ R
    return T


def test_export_matches_offset_fk_and_preserves_mass_limits_and_feedback():
    nominal = source_xml()
    offsets = np.deg2rad([0., .75, -.62, 2.73, 1.18, 0.])
    candidate = export_candidate_urdf(nominal, offsets)
    rng = np.random.default_rng(14)
    for q in rng.uniform(-2., 2., (52, 6)):
        before = q.copy()
        np.testing.assert_allclose(xml_fk(candidate, q), fk_base_tool0(q + offsets), atol=1e-12)
        np.testing.assert_array_equal(q, before)
    original_root, candidate_root = ET.fromstring(nominal), ET.fromstring(candidate)
    for element in original_root.findall('link'):
        assert ET.tostring(element) == ET.tostring(candidate_root.find(f"link[@name='{element.get('name')}']"))
    for element in original_root.findall('joint'):
        for tag in ('axis', 'parent', 'child', 'limit', 'mimic'):
            old, new = element.find(tag), candidate_root.find(f"joint[@name='{element.get('name')}']/{tag}")
            assert (None if old is None else old.attrib) == (None if new is None else new.attrib)


def test_tf_candidate_guard_rejects_default_and_mismatched_robot_model():
    nominal = source_xml()
    candidate = export_candidate_urdf(nominal, np.deg2rad([0., 1., -1., 2., 1., 0.]))
    signature = model_signature(candidate)
    result = dict(calibration_type='eye_in_hand', T_tool_camera=np.eye(4).tolist(),
                  verdict='PASS', validation_status='offline_model_candidate',
                  robot_calibration={'expected_model': signature})
    with pytest.raises(ValueError, match='allow_model_candidate'):
        calibrated_transform(result)
    calibrated_transform(result, allow_model_candidate=True)
    require_matching_model(candidate, signature)
    with pytest.raises(ValueError, match='differs'):
        require_matching_model(nominal, signature)
    with pytest.raises(ValueError):
        export_candidate_urdf(candidate, np.zeros(6))  # prevent double correction
    damaged = candidate.replace('xyz="0.11277 0 0"', 'xyz="0.1 0 0"')
    with pytest.raises(ValueError, match='differs'):
        require_matching_model(damaged, signature)


def test_recovers_model_offsets_and_predicts_separate_board_location_without_refitting_camera():
    offsets = np.deg2rad([0., .8, -.6, 2.5, 1.2, 0.])
    camera = matrix([.06, -.02, .09, .2, -.5, .4])
    rng = np.random.default_rng(27)
    sessions = []
    for board in [matrix([.3, .1, .03, -.4, .2, .1]), matrix([.25, -.1, .02, .3, -.1, .2])]:
        samples = []
        for i, q in enumerate(rng.uniform(-1.5, 1.5, (24, 6))):
            samples.append(dict(index=i, joints=q.tolist(), T_base_tool=fk_base_tool0(q),
                                T_camera_board=inv_T(camera) @ inv_T(fk_base_tool0(q + offsets)) @ board))
        sessions.append(samples)
    train, held = split(sessions[0])
    actual_offsets, actual_camera, board_a, diagnostic = fit_offsets(train)
    assert diagnostic['converged'] and diagnostic['jacobian_rank'] == 16
    assert not diagnostic['offset_at_bound']
    np.testing.assert_allclose(actual_offsets, offsets, atol=1e-7)
    np.testing.assert_allclose(actual_camera, camera, atol=1e-7)
    assert evaluate(corrected_samples(held, actual_offsets), actual_camera, board_a,
                    'eye_in_hand')['translation_max_m'] < 1e-7
    train_b, held_b = split(sessions[1])
    board_b = average_T([s['T_base_tool'] @ actual_camera @ s['T_camera_board']
                         for s in corrected_samples(train_b, actual_offsets)])
    assert evaluate(corrected_samples(held_b, actual_offsets), actual_camera, board_b,
                    'eye_in_hand')['translation_max_m'] < 1e-7
    with pytest.raises(ValueError, match='raw six-joint'):
        corrected_samples([dict(joints=None)], offsets)
