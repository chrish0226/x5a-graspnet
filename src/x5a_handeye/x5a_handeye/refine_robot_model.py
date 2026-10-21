"""Offline joint-axis offset hypothesis with a separate-session validation.

No ROS node, publisher, driver or hardware command is created. Only joint2..5
are fitted: joint1/joint6 are gauge freedoms of the board/camera transforms.
The camera transform and offsets learned on session A stay frozen on B; only
the fixed board's new position is estimated from B's training subset.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from x5a_handeye.robot_model import corrected_samples, export_candidate_urdf, model_signature
from x5a_handeye.solve_handeye import (
    average_T, evaluate, motion_diversity_report, normalize_samples, solve_opencv_handeye,
)
from x5a_handeye.transforms import T_to_pose


def vector(T):
    return np.r_[T[:3, 3], cv2.Rodrigues(T[:3, :3])[0].reshape(3)]


def matrix(p):
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = cv2.Rodrigues(np.asarray(p[3:6], dtype=float))[0]
    return T


def rotation_log(R):
    # OpenCV rounds very small rotations to zero; a smooth small-angle branch
    # keeps the numerical Jacobian informative as a synthetic exact fit converges.
    v = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]]) * .5
    cosine = (np.trace(R)-1.) * .5
    theta = np.arctan2(np.linalg.norm(v), cosine)
    if theta < 1e-4:
        return v * (1. + theta*theta/6.)
    return cv2.Rodrigues(R)[0].reshape(3)


def split(samples, count=6, seed=7):
    if count < 1 or len(samples) - count < 12:
        raise ValueError('need a holdout and at least 12 training samples per session')
    idx = np.arange(len(samples))
    np.random.default_rng(seed).shuffle(idx)
    held = set(idx[:count].tolist())
    return ([s for i, s in enumerate(samples) if i not in held],
            [s for i, s in enumerate(samples) if i in held])


def fit_offsets(train, bound_deg=3., loss='linear'):
    # Imported lazily: normal hand-eye calibration does not require SciPy.
    from scipy.optimize import least_squares
    if not motion_diversity_report(train).get('ok'):
        raise ValueError('training poses lack rotational diversity')
    X, Y = solve_opencv_handeye(train, cv2.CALIB_HAND_EYE_HORAUD, 'eye_in_hand')
    initial = np.r_[vector(X), vector(Y), np.zeros(4)]

    def residual(p):
        camera, board = matrix(p[:6]), matrix(p[6:12])
        out = []
        for s in corrected_samples(train, np.r_[0., p[12:16], 0.]):
            predicted = s['T_base_tool'] @ camera @ s['T_camera_board']
            out.extend((predicted[:3, 3] - board[:3, 3]) / .003)
            out.extend(rotation_log(board[:3, :3].T @ predicted[:3, :3]) / np.deg2rad(1.))
        return out

    bound = np.deg2rad(bound_deg)
    fit = least_squares(residual, initial,
                        bounds=(np.r_[np.full(12, -np.inf), np.full(4, -bound)],
                                np.r_[np.full(12, np.inf), np.full(4, bound)]),
                        loss=loss, f_scale=1.,
                        max_nfev=200, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    diagnostics = {
        'converged': bool(fit.success), 'message': fit.message,
        'jacobian_rank': int(np.linalg.matrix_rank(fit.jac)),
        'singular_min_max_ratio': float(singular[-1] / singular[0]),
        'offset_bound_deg': bound_deg,
        'offset_at_bound': bool(np.any(np.abs(fit.x[12:]) >= bound - 1e-5)),
        'translation_residual_scale_m': .003, 'rotation_residual_scale_deg': 1.,
        'loss': loss,
    }
    return np.r_[0., fit.x[12:16], 0.], matrix(fit.x[:6]), matrix(fit.x[6:12]), diagnostics


def session_report(samples, train, hold, offsets, camera, board):
    corrected_train = corrected_samples(train, offsets)
    corrected_hold = corrected_samples(hold, offsets)
    return {
        'training_indices': [s['index'] for s in train],
        'holdout_indices': [s['index'] for s in hold],
        'motion_diversity': motion_diversity_report(corrected_train),
        'training': evaluate(corrected_train, camera, board, 'eye_in_hand'),
        'holdout': evaluate(corrected_hold, camera, board, 'eye_in_hand'),
        'all': evaluate(corrected_samples(samples, offsets), camera, board, 'eye_in_hand'),
        'T_base_board': board.tolist(),
    }


def load(path):
    data = json.loads(Path(path).read_text())
    if data.get('calibration_type') != 'eye_in_hand':
        raise ValueError('robot model refinement only supports eye_in_hand samples')
    samples = normalize_samples(data['samples'])
    if len(samples) != len(data['samples']):
        raise ValueError('invalid sample poses; refusing to silently discard samples')
    corrected_samples(samples, np.zeros(6))  # validate all raw joints before fitting
    return data, samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--training-samples', required=True)
    ap.add_argument('--validation-samples', required=True)
    ap.add_argument('--urdf', required=True, help='nominal project URDF/xacro; not the SDK gravity model')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--holdout', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    if Path(args.training_samples).resolve() == Path(args.validation_samples).resolve():
        raise SystemExit('validation requires a separately collected session')
    a_meta, a = load(args.training_samples)
    b_meta, b = load(args.validation_samples)
    for key in ('camera_frame', 'tool_frame', 'base_frame'):
        if not a_meta.get(key) or a_meta[key] != b_meta.get(key):
            raise SystemExit(f'the two sessions must use the same {key}')
    a_train, a_hold = split(a, args.holdout, args.seed)
    b_train, b_hold = split(b, args.holdout, args.seed)
    offsets, camera, a_board, fit = fit_offsets(a_train)
    b_board = average_T([s['T_base_tool'] @ camera @ s['T_camera_board']
                         for s in corrected_samples(b_train, offsets)])
    a_report = session_report(a, a_train, a_hold, offsets, camera, a_board)
    b_report = session_report(b, b_train, b_hold, offsets, camera, b_board)
    passed = (fit['converged'] and fit['jacobian_rank'] == 16 and not fit['offset_at_bound']
              and all(r['motion_diversity'].get('ok')
                      and r['training']['translation_mean_m'] < .003
                      and r['training']['rotation_mean_deg'] < 1.
                      and r['holdout']['translation_mean_m'] < .003
                      and r['holdout']['translation_max_m'] < .003
                      and r['holdout']['rotation_max_deg'] < 1.
                      for r in (a_report, b_report)))
    import xacro
    nominal_xml = xacro.process_file(args.urdf).toxml()
    candidate_xml = export_candidate_urdf(nominal_xml, offsets)
    dest = Path(args.output_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    model_file = dest / 'robot_model_candidate.urdf'
    model_file.write_text(candidate_xml)
    profile = {
        'joint_names': [f'joint{i}' for i in range(1, 7)],
        'offsets_rad': offsets.tolist(), 'offsets_deg': np.rad2deg(offsets).tolist(),
        'convention': 'nominal_fk(raw_joint_feedback + offsets_rad)',
        'candidate_urdf': str(model_file), 'expected_model': model_signature(candidate_xml),
        'encoder_feedback_and_commands': 'unchanged',
        'sdk_gravity_model': 'unchanged',
    }
    report = {
        'validation_status': 'offline_model_candidate', 'fit': fit,
        'training_session': a_report, 'validation_session_frozen_camera_and_offsets': b_report,
        'robot_calibration': profile, 'verdict': 'PASS' if passed else 'FAIL',
        'limitations': ['Model offsets are a hypothesis, not measured encoder zero corrections.',
                        'These sessions were also inspected during diagnosis; independent physical validation is still required.',
                        'Raw SDK joint feedback must not be offset again with the candidate URDF.'],
        'sources': {key: {'path': str(Path(path).resolve()),
                          'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                    for key, path in [('training_samples', args.training_samples),
                                      ('validation_samples', args.validation_samples), ('nominal_urdf', args.urdf)]},
    }
    (dest / 'robot_model_candidate.json').write_text(json.dumps(profile, indent=2))
    (dest / 'validation_report.json').write_text(json.dumps(report, indent=2))
    xyz, quat = T_to_pose(camera)
    result = {
        'calibration_type': 'eye_in_hand', 'validation_status': 'offline_model_candidate',
        'parent_frame': b_meta['tool_frame'], 'tool_frame': b_meta['tool_frame'],
        'base_frame': b_meta['base_frame'], 'camera_frame': b_meta['camera_frame'],
        'camera_root_frame': 'wrist_camera_link', 'board_frame': 'calibration_board',
        'T_tool_camera': camera.tolist(), 'T_base_board': b_board.tolist(),
        'translation': dict(zip(('x', 'y', 'z'), xyz)),
        'rotation': dict(zip(('x', 'y', 'z', 'w'), quat)),
        'solver': 'joint2_to_5_axis_offsets_and_handeye_nonlinear_fit',
        'robot_calibration': profile, 'error': b_report['training'],
        'holdout_error': b_report['holdout'], 'all_samples_error': b_report['all'],
        'observation_geometry': b_meta.get('observation_geometry'),
        'verdict': report['verdict'], 'validation_report': str(dest / 'validation_report.json'),
    }
    (dest / 'handeye_model_candidate.yaml').write_text(yaml.safe_dump(result, sort_keys=False))
    print('offsets_deg:', np.round(np.rad2deg(offsets), 6).tolist())
    for name, r in [('A', a_report), ('B frozen camera/offsets', b_report)]:
        e = r['holdout']
        print(f"{name} holdout: mean={e['translation_mean_m']*1000:.3f} mm "
              f"max={e['translation_max_m']*1000:.3f} mm rotation_mean={e['rotation_mean_deg']:.3f} deg")
    print('OFFLINE MODEL CANDIDATE:', report['verdict'], 'saved', dest)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
