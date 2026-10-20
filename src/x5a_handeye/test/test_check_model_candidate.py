import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from x5a_handeye.check_model_candidate import check_candidate, check_against_reference, reject_old_session
from x5a_handeye.refine_robot_model import matrix
from x5a_handeye.robot_model import export_candidate_urdf, model_signature
from x5a_handeye.transforms import inv_T
from x5a_handeye.x5a_fk import fk_base_tool0


def validation_session():
    camera = matrix([-.056, -.016, .047, .2, -.5, .4])
    offsets = np.deg2rad([0., .8, -.6, 2.5, 1.2, 0.])
    board = matrix([.3, -.1, .02, .3, -.1, .2])
    optical = 'wrist_camera_color_optical_frame'
    geometry = dict(board={'square_length_m': .03, 'marker_length_m': .022},
                    detector_version='intrinsics_quality_v2', opencv_version='4.5.4',
                    camera_frame=optical, image_size=[640, 480],
                    K=[[397., 0., 320.], [0., 397., 257.], [0., 0., 1.]], D=[0.]*8)
    nominal = (Path(__file__).resolve().parents[2]/'x5a_moveit_config/config/X5A.urdf.xacro').read_text()
    xml = export_candidate_urdf(nominal, offsets)
    result = dict(calibration_type='eye_in_hand', verdict='PASS',
                  validation_status='offline_model_candidate', base_frame='base_link', tool_frame='tool0',
                  camera_frame=optical, camera_root_frame='wrist_camera_link', T_tool_camera=camera.tolist(),
                  T_base_board=matrix([1., 1., 1., 0., 0., 0.]).tolist(),
                  observation_geometry=geometry,
                  robot_calibration=dict(offsets_rad=offsets.tolist(), expected_model=model_signature(xml),
                                         joint_names=[f'joint{i}' for i in range(1, 7)],
                                         convention='nominal_fk(raw_joint_feedback + offsets_rad)'))
    samples = []
    for i, q in enumerate(np.random.default_rng(92).uniform(-1.4, 1.4, (8, 6))):
        C = inv_T(camera) @ inv_T(fk_base_tool0(q + offsets)) @ board
        samples.append(dict(index=i, joints=q.tolist(), T_base_tool=np.eye(4).tolist(),
                            T_camera_board=C.tolist(), source={'board_frame_id': optical}))
    data = dict(calibration_type='eye_in_hand', base_frame='base_link', tool_frame='tool0',
                camera_frame=optical, observation_geometry=geometry, samples=samples)
    return result, data, xml


def test_checks_new_board_location_from_raw_joints_without_refitting_or_using_saved_fk():
    result, data, _ = validation_session()
    before = copy.deepcopy(result)
    report = check_candidate(result, data)
    assert report['verdict'] == 'PASS'
    assert report['reference_indices'] == [0, 1, 2]
    assert report['test_indices'] == [3, 4, 5, 6, 7]
    assert report['test_error']['translation_max_m'] < 1e-9
    assert result == before


@pytest.mark.parametrize('fault', ['camera', 'offsets', 'board_move', 'bad_geometry', 'frame', 'nan'])
def test_rejects_wrong_camera_offsets_board_motion_and_incompatible_observations(fault):
    result, data, _ = validation_session()
    if fault == 'camera':
        result['T_tool_camera'][0][3] += .02
    elif fault == 'offsets':
        result['robot_calibration']['offsets_rad'] = [0.]*6
    elif fault == 'board_move':
        data['samples'][6]['T_camera_board'][0][3] += .02
    elif fault == 'bad_geometry':
        data['observation_geometry'] = copy.deepcopy(data['observation_geometry'])
        data['observation_geometry']['board']['square_length_m'] = .02
    elif fault == 'frame':
        data['samples'][5]['source']['board_frame_id'] = 'other_camera'
    else:
        data['samples'][5]['T_camera_board'][0][3] = float('nan')
    if fault in ('bad_geometry', 'frame', 'nan'):
        with pytest.raises(ValueError):
            check_candidate(result, data)
    else:
        report = check_candidate(result, data)
        assert report['verdict'] == 'FAIL'
        assert report['test_error']['translation_max_m'] > .003


def test_insufficient_and_unchanged_poses_cannot_pass():
    result, data, _ = validation_session()
    with pytest.raises(ValueError):
        check_candidate(result, {**data, 'samples': data['samples'][:5]})
    repeated = {**data, 'samples': [copy.deepcopy(data['samples'][0]) for _ in range(8)]}
    assert check_candidate(result, repeated)['verdict'] == 'FAIL'


def test_supplemental_check_freezes_old_board_reference_and_preserves_previous_fail():
    result, data, _ = validation_session()
    original = check_candidate(result, data)
    original['verdict'] = 'FAIL'  # A previous bad test pose must remain a failure.
    before = copy.deepcopy(original)
    additional = {**data, 'samples': copy.deepcopy(data['samples'][5:])}
    report = check_against_reference(result, additional, original)
    assert report['verdict'] == 'PASS'
    assert report['previous_validation_verdict'] == 'FAIL'
    assert report['validation_status'] == 'supplemental_board_consistency'
    assert original == before
    # Moving the board consistently in the new file cannot be hidden by fitting
    # another board position on the supplemental poses.
    moved_board = np.asarray(original['T_base_board_reference']).copy()
    moved_board[0, 3] += .02
    camera = np.asarray(result['T_tool_camera'])
    offsets = np.asarray(result['robot_calibration']['offsets_rad'])
    for s in additional['samples']:
        s['T_camera_board'] = (inv_T(camera) @ inv_T(fk_base_tool0(np.asarray(s['joints']) + offsets))
                               @ moved_board).tolist()
    assert check_against_reference(result, additional, original)['verdict'] == 'FAIL'


def test_supplemental_check_requires_valid_reference_and_diverse_poses():
    result, data, _ = validation_session()
    reference = check_candidate(result, data)
    short = {**data, 'samples': data['samples'][:2]}
    with pytest.raises(ValueError):
        check_against_reference(result, short, reference)
    invalid = copy.deepcopy(reference)
    invalid['reference_error']['translation_max_m'] = .02
    with pytest.raises(ValueError, match='reference poses'):
        check_against_reference(result, data, invalid)
    repeated = {**data, 'samples': [data['samples'][0]]*3}
    assert check_against_reference(result, repeated, reference)['verdict'] == 'FAIL'


def test_cli_writes_report_without_changing_candidate_and_refuses_training_file(tmp_path):
    result, data, xml = validation_session()
    candidate, samples = tmp_path/'candidate.yaml', tmp_path/'samples.json'
    output, training_report, model = tmp_path/'report.json', tmp_path/'training_report.json', tmp_path/'model.urdf'
    model.write_text(xml)
    result['robot_calibration']['candidate_urdf'] = str(model)
    result['validation_report'] = str(training_report)
    provenance = {'sources': {name: {'path': str(tmp_path/(name+'.json')), 'sha256': 'unrelated'}
                             for name in ('training_samples', 'validation_samples')}}
    training_report.write_text(json.dumps(provenance))
    candidate.write_text(yaml.safe_dump(result))
    samples.write_text(json.dumps(data))
    before = candidate.read_bytes()
    proc = subprocess.run([sys.executable, '-m', 'x5a_handeye.check_model_candidate', '--candidate', str(candidate),
                           '--samples', str(samples), '--output', str(output)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(output.read_text())['verdict'] == 'PASS'
    assert candidate.read_bytes() == before
    # The installed entry point also supports a separate supplemental report.
    extra_samples = tmp_path/'extra.json'
    extra_samples.write_text(json.dumps({**data, 'samples': data['samples'][5:]}))
    extra_output = tmp_path/'extra_report.json'
    reference_bytes = output.read_bytes()
    proc = subprocess.run([sys.executable, '-m', 'x5a_handeye.check_model_candidate', '--candidate', str(candidate),
                           '--samples', str(extra_samples), '--reference-report', str(output),
                           '--output', str(extra_output)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(extra_output.read_text())['validation_status'] == 'supplemental_board_consistency'
    assert candidate.read_bytes() == before and output.read_bytes() == reference_bytes
    bad_reference = json.loads(output.read_text())
    bad_reference['sources']['candidate']['sha256'] = 'another_model'
    output.write_text(json.dumps(bad_reference))
    proc = subprocess.run([sys.executable, '-m', 'x5a_handeye.check_model_candidate', '--candidate', str(candidate),
                           '--samples', str(extra_samples), '--reference-report', str(output),
                           '--output', str(extra_output)], capture_output=True, text=True)
    assert proc.returncode != 0 and 'different candidate' in proc.stderr
    provenance['sources']['validation_samples']['path'] = str(samples)
    training_report.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='used to build'):
        reject_old_session(result, samples, candidate)
