"""Read-only, offline consistency check of a frozen model on new observations.

The first three poses locate the fixed board for this session. All remaining
poses test the same frozen camera transform and joint offsets. Neither the
camera calibration nor robot model is fitted or written by this command.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from x5a_handeye.calibrated_tf import calibrated_transform
from x5a_handeye.observations import observation_geometry
from x5a_handeye.robot_model import corrected_samples, require_matching_model, valid_offsets
from x5a_handeye.solve_handeye import average_T, evaluate, motion_diversity_report, normalize_samples


def prepared_samples(result, data, minimum_count):
    """Validate observation/model compatibility, then recompute raw-feedback FK."""
    if result.get('calibration_type') != 'eye_in_hand' or result.get('verdict') != 'PASS':
        raise ValueError('need a validated eye_in_hand candidate')
    _, _, optical, camera = calibrated_transform(result, allow_model_candidate=True)
    profile = result['robot_calibration']
    if profile.get('convention') != 'nominal_fk(raw_joint_feedback + offsets_rad)':
        raise ValueError('unsupported robot offset convention')
    if profile.get('joint_names') != [f'joint{i}' for i in range(1, 7)]:
        raise ValueError('robot calibration joint order differs from raw feedback')
    offsets = valid_offsets(profile['offsets_rad'])
    if (data.get('calibration_type') != 'eye_in_hand'
            or data.get('camera_frame') != optical
            or data.get('base_frame') != 'base_link' or result.get('base_frame') != 'base_link'
            or data.get('tool_frame') != 'tool0' or result.get('tool_frame') != 'tool0'):
        raise ValueError('validation sample mode/frames differ from the candidate')
    expected_geometry = result.get('observation_geometry')
    if not expected_geometry or data.get('observation_geometry') != expected_geometry:
        raise ValueError('board/camera geometry differs or is missing; use the same detector configuration')
    observation_geometry(expected_geometry)
    raw = data['samples']
    samples = normalize_samples(raw)
    if len(samples) != len(raw) or len(samples) < minimum_count:
        raise ValueError(f'need at least {minimum_count} valid poses with raw joints')
    for item, sample in zip(raw, samples):
        if item.get('source', {}).get('board_frame_id') != optical:
            raise ValueError('a sampled camera optical frame differs from the candidate')
        T = sample['T_camera_board']
        if (not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-5)):
            raise ValueError('sampled board pose is not a finite rigid transform')
    # Always recompute from raw feedback; saved T_base_tool is nominal and must
    # not be mixed with a model-corrected extrinsic or corrected twice.
    return corrected_samples(samples, offsets), camera


def check_candidate(result, data, reference_count=3, pass_mm=3., pass_deg=1.):
    if reference_count < 3 or pass_mm <= 0 or pass_deg <= 0:
        raise ValueError('need at least 3 reference poses and positive thresholds')
    samples, camera = prepared_samples(result, data, reference_count + 3)
    reference, test = samples[:reference_count], samples[reference_count:]
    board = average_T([s['T_base_tool'] @ camera @ s['T_camera_board'] for s in reference])
    reference_error = evaluate(reference, camera, board, 'eye_in_hand')
    test_error = evaluate(test, camera, board, 'eye_in_hand')
    diversity = motion_diversity_report(test)
    passed = (diversity.get('ok', False)
              and reference_error['translation_max_m']*1000 <= pass_mm
              and reference_error['rotation_max_deg'] <= pass_deg
              and test_error['translation_max_m']*1000 <= pass_mm
              and test_error['rotation_max_deg'] <= pass_deg)
    return {
        'validation_status': 'independent_board_consistency',
        'verdict': 'PASS' if passed else 'FAIL',
        'camera_and_offsets': 'frozen', 'robot_fk': 'recomputed_from_raw_feedback',
        'reference_indices': [s['index'] for s in reference],
        'test_indices': [s['index'] for s in test],
        'T_base_board_reference': board.tolist(),
        'reference_error': reference_error, 'test_error': test_error,
        'motion_diversity': diversity,
        'criteria': {'maximum_translation_mm': pass_mm, 'maximum_rotation_deg': pass_deg},
        'limitations': 'Checks fixed-board consistency; absolute TCP/grasp accuracy still needs a physical target check.',
    }


def check_against_reference(result, data, reference_report, pass_mm=3., pass_deg=1.):
    """Test additional poses against the already established board position.

    Requires the physical board to remain fixed between sessions. The original
    report and its verdict remain intact even when this supplemental check passes.
    """
    if pass_mm <= 0 or pass_deg <= 0:
        raise ValueError('validation thresholds must be positive')
    samples, camera = prepared_samples(result, data, 3)
    if (reference_report.get('validation_status') != 'independent_board_consistency'
            or reference_report.get('camera_and_offsets') != 'frozen'):
        raise ValueError('reference report must be an independent frozen-parameter check')
    anchor = reference_report['reference_error']
    if anchor['translation_max_m']*1000 > pass_mm or anchor['rotation_max_deg'] > pass_deg:
        raise ValueError('original reference poses did not meet the requested thresholds')
    board = np.asarray(reference_report['T_base_board_reference'], dtype=float)
    if (board.shape != (4, 4) or not np.isfinite(board).all()
            or not np.allclose(board[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(board[:3, :3].T @ board[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(board[:3, :3]), 1., atol=1e-5)):
        raise ValueError('reference board pose is not a finite rigid transform')
    error = evaluate(samples, camera, board, 'eye_in_hand')
    diversity = motion_diversity_report(samples)
    passed = (diversity.get('ok', False) and error['translation_max_m']*1000 <= pass_mm
              and error['rotation_max_deg'] <= pass_deg)
    return {
        'validation_status': 'supplemental_board_consistency',
        'verdict': 'PASS' if passed else 'FAIL',
        'camera_and_offsets': 'frozen', 'robot_fk': 'recomputed_from_raw_feedback',
        'reference_indices': reference_report['reference_indices'],
        'test_indices': [s['index'] for s in samples],
        'T_base_board_reference': board.tolist(), 'reference_error': anchor,
        'test_error': error, 'motion_diversity': diversity,
        'previous_validation_verdict': reference_report['verdict'],
        'criteria': {'maximum_translation_mm': pass_mm, 'maximum_rotation_deg': pass_deg},
        'limitations': 'Supplemental check assumes the board has not moved; it does not erase the original verdict or establish absolute grasp accuracy.',
    }


def reject_old_session(result, sample_path, candidate_path):
    report_path = Path(result['validation_report'])
    if not report_path.is_absolute():
        report_path = Path(candidate_path).parent / report_path
    report = json.loads(report_path.read_text())
    digest = hashlib.sha256(Path(sample_path).read_bytes()).hexdigest()
    sources = [report['sources'][name] for name in ('training_samples', 'validation_samples')]
    # A remounted camera may retain robot offsets learned from older sessions.
    # Those older fitting sessions are not fresh validation data either.
    prior = report.get('frozen_model_provenance', {}).get('sources', {})
    sources.extend(prior[name] for name in ('training_samples', 'validation_samples') if name in prior)
    for source in sources:
        if Path(source['path']).resolve() == Path(sample_path).resolve() or source['sha256'] == digest:
            raise ValueError('these samples were used to build the candidate; collect a new validation file')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-count', type=int, default=3)
    parser.add_argument('--reference-report',
                        help='Reuse an existing board reference for >=3 additional poses; board must not have moved')
    parser.add_argument('--pass-mm', type=float, default=3.)
    parser.add_argument('--pass-deg', type=float, default=1.)
    args = parser.parse_args()
    result = yaml.safe_load(Path(args.candidate).read_text())
    output = Path(args.output).resolve()
    protected = [args.candidate, args.samples, result['validation_report'],
                 result['robot_calibration']['candidate_urdf']]
    if args.reference_report:
        protected.append(args.reference_report)
    if output in [Path(p).resolve() for p in protected]:
        raise SystemExit('output must not overwrite samples, model, candidate or training report')
    try:
        reject_old_session(result, args.samples, args.candidate)
        profile = result['robot_calibration']
        require_matching_model(Path(profile['candidate_urdf']).read_text(), profile['expected_model'])
        data = json.loads(Path(args.samples).read_text())
        if args.reference_report:
            reference = json.loads(Path(args.reference_report).read_text())
            digest = hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest()
            if reference.get('sources', {}).get('candidate', {}).get('sha256') != digest:
                raise ValueError('reference report was computed with a different candidate')
            old_samples = reference['sources']['samples']
            sample_digest = hashlib.sha256(Path(args.samples).read_bytes()).hexdigest()
            if (Path(old_samples['path']).resolve() == Path(args.samples).resolve()
                    or old_samples['sha256'] == sample_digest):
                raise ValueError('supplemental check requires a new sample file')
            report = check_against_reference(result, data, reference, args.pass_mm, args.pass_deg)
        else:
            report = check_candidate(result, data, args.reference_count, args.pass_mm, args.pass_deg)
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc))
    report['sources'] = {
        name: {'path': str(Path(path).resolve()),
               'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for name, path in [('candidate', args.candidate), ('samples', args.samples)]
    }
    if args.reference_report:
        report['sources']['reference_report'] = {
            'path': str(Path(args.reference_report).resolve()),
            'sha256': hashlib.sha256(Path(args.reference_report).read_bytes()).hexdigest(),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    err = report['test_error']
    print(f"reference={len(report['reference_indices'])} test={len(report['test_indices'])}")
    anchor = report['reference_error']
    print(f"reference max={anchor['translation_max_m']*1000:.2f} mm "
          f"rotation_max={anchor['rotation_max_deg']:.2f} deg")
    print(f"test mean={err['translation_mean_m']*1000:.2f} mm "
          f"max={err['translation_max_m']*1000:.2f} mm "
          f"rotation_mean={err['rotation_mean_deg']:.2f} deg max={err['rotation_max_deg']:.2f} deg")
    for index, translation, rotation in zip(report['test_indices'], err['per_sample_t'], err['per_sample_r']):
        status = 'EXCEEDS' if translation*1000 > args.pass_mm or rotation > args.pass_deg else 'OK'
        print(f"sample #{index+1}: {translation*1000:.2f} mm {rotation:.2f} deg {status}")
    if not report['motion_diversity'].get('ok'):
        print('Insufficient motion diversity: add poses with rotations about different directions.')
    label = 'SUPPLEMENTAL BOARD CONSISTENCY' if args.reference_report else 'INDEPENDENT BOARD CONSISTENCY'
    print(label + ':', report['verdict'])
    print('saved', output)
    if report['verdict'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
