"""Recalibrate a remounted camera without fitting the frozen robot offsets.

Writes a separate offline candidate; never changes feedback, commands or the
SDK gravity model. A fresh physical validation is required after this solve.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from x5a_handeye.observations import observation_geometry
from x5a_handeye.refine_robot_model import load, session_report, split
from x5a_handeye.robot_model import corrected_samples, require_matching_model, valid_offsets
from x5a_handeye.solve_handeye import evaluate, motion_diversity_report, solve_opencv_handeye
from x5a_handeye.transforms import T_to_pose


def recalibrate(samples, offsets, holdout=6, seed=7, pass_mm=3., pass_deg=1.):
    if pass_mm <= 0 or pass_deg <= 0:
        raise ValueError('validation thresholds must be positive')
    offsets = valid_offsets(offsets)
    train, held = split(samples, holdout, seed)
    corrected = corrected_samples(train, offsets)
    if not motion_diversity_report(corrected).get('ok'):
        raise ValueError('training poses lack rotational diversity')
    candidates = []
    failures = {}
    for name in ('TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS'):
        try:
            X, Y = solve_opencv_handeye(corrected, getattr(cv2, 'CALIB_HAND_EYE_'+name), 'eye_in_hand')
            error = evaluate(corrected, X, Y, 'eye_in_hand')
            candidates.append((error['translation_mean_m'], name, X, Y, error))
        except (ValueError, cv2.error) as exc:
            failures[name] = str(exc)
    if not candidates:
        raise ValueError('all fixed-model hand-eye solvers failed')
    # Selection uses training data only. No joint fitting or outlier rejection.
    candidates.sort(key=lambda item: item[0])
    _, name, camera, board, _ = candidates[0]
    report = session_report(samples, train, held, offsets, camera, board)
    tr, ho = report['training'], report['holdout']
    passed = (tr['translation_mean_m']*1000 <= pass_mm
              and tr['rotation_mean_deg'] <= pass_deg
              and ho['translation_max_m']*1000 <= pass_mm
              and ho['rotation_max_deg'] <= pass_deg)
    report.update(
        solver=name, verdict='PASS' if passed else 'FAIL', offsets='frozen',
        solver_selection='minimum training translation mean; holdout not used for selection',
        training_methods={item[1]: item[4] for item in candidates}, solver_failures=failures,
        criteria={'training_mean_translation_mm': pass_mm, 'training_mean_rotation_deg': pass_deg,
                  'holdout_max_translation_mm': pass_mm, 'holdout_max_rotation_deg': pass_deg},
    )
    return camera, report


def source(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--robot-profile', required=True)
    parser.add_argument('--output-dir', required=True, help='new directory; existing directories are refused')
    parser.add_argument('--holdout', type=int, default=6)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--pass-mm', type=float, default=3.)
    parser.add_argument('--pass-deg', type=float, default=1.)
    args = parser.parse_args()
    meta, samples = load(args.samples)
    observation_geometry(meta['observation_geometry'])
    if meta.get('base_frame') != 'base_link' or meta.get('tool_frame') != 'tool0':
        raise ValueError('fixed model requires base_link/tool0 sample frames')
    if meta.get('camera_frame') != meta['observation_geometry']['camera_frame']:
        raise ValueError('sample and observation camera frames differ')
    profile = json.loads(Path(args.robot_profile).read_text())
    if (profile.get('joint_names') != [f'joint{i}' for i in range(1, 7)]
            or profile.get('convention') != 'nominal_fk(raw_joint_feedback + offsets_rad)'):
        raise ValueError('unsupported frozen robot profile')
    offsets = valid_offsets(profile['offsets_rad'])
    model = Path(profile['candidate_urdf']).read_text()
    require_matching_model(model, profile['expected_model'])
    camera, report = recalibrate(samples, offsets, args.holdout, args.seed, args.pass_mm, args.pass_deg)
    report.update(validation_status='offline_model_candidate', independent_validation='required',
                  sources={'training_samples': source(args.samples), 'validation_samples': source(args.samples),
                           'frozen_robot_profile': source(args.robot_profile),
                           'frozen_robot_urdf': source(profile['candidate_urdf'])},
                  limitations=['Training/holdout poses come from one new mounting session.',
                               'Frozen robot offsets are a hypothesis, not measured encoder zero corrections.',
                               'Training maximum residuals are retained, not rejected.',
                               'Fresh independent board and physical target checks are required.'])
    prior_report = Path(args.robot_profile).resolve().parent/'validation_report.json'
    if prior_report.exists():
        previous = json.loads(prior_report.read_text())
        report['frozen_model_provenance'] = {'report': source(prior_report), 'sources': previous['sources']}
    dest = Path(args.output_dir).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    model_path = dest/'robot_model_candidate.urdf'
    model_path.write_text(model)
    profile = {**profile, 'candidate_urdf': str(model_path)}
    (dest/'robot_model_candidate.json').write_text(json.dumps(profile, indent=2)+'\n')
    report['robot_calibration'] = profile
    report_path = dest/'validation_report.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    xyz, quat = T_to_pose(camera)
    result = {
        'calibration_type': 'eye_in_hand', 'validation_status': 'offline_model_candidate',
        'verdict': report['verdict'], 'parent_frame': meta['tool_frame'], 'tool_frame': meta['tool_frame'],
        'base_frame': meta['base_frame'], 'camera_frame': meta['camera_frame'],
        'camera_root_frame': 'wrist_camera_link', 'board_frame': 'calibration_board',
        'T_tool_camera': camera.tolist(), 'T_base_board': report['T_base_board'],
        'translation': dict(zip(('x', 'y', 'z'), xyz)), 'rotation': dict(zip(('x', 'y', 'z', 'w'), quat)),
        'solver': 'frozen_robot_offsets_opencv_'+report['solver'], 'robot_calibration': profile,
        'error': report['training'], 'holdout_error': report['holdout'], 'all_samples_error': report['all'],
        'observation_geometry': meta['observation_geometry'], 'validation_report': str(report_path),
    }
    (dest/'handeye_model_candidate.yaml').write_text(yaml.safe_dump(result, sort_keys=False))
    for label, error in [('train', report['training']), ('holdout', report['holdout'])]:
        print(f"{label}: mean={error['translation_mean_m']*1000:.3f} mm "
              f"max={error['translation_max_m']*1000:.3f} mm rotation_max={error['rotation_max_deg']:.3f} deg")
    print('FROZEN MODEL / NEW CAMERA CANDIDATE:', report['verdict'], 'saved', dest)
    print('Fresh independent validation is still required.')
    if report['verdict'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
