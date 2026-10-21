"""Trim high-residual training poses once; retain every holdout observation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from x5a_handeye.refine_robot_model import load, split, fit_offsets
from x5a_handeye.robot_model import corrected_samples, export_candidate_urdf, model_signature
from x5a_handeye.solve_handeye import evaluate, motion_diversity_report
from x5a_handeye.transforms import T_to_pose


def optimize(samples, holdout=6, seed=7, cutoff_mm=4., cutoff_deg=1., max_reject_fraction=.3):
    if cutoff_mm <= 0 or cutoff_deg <= 0 or not 0 <= max_reject_fraction < 1:
        raise ValueError('invalid training rejection limits')
    train, held = split(samples, holdout, seed)
    initial_offsets, initial_camera, initial_board, initial_fit = fit_offsets(train, loss='soft_l1')
    if not initial_fit['converged'] or initial_fit['offset_at_bound']:
        raise ValueError('initial robust model fit did not converge within the offset bounds')
    initial_error = evaluate(corrected_samples(train, initial_offsets), initial_camera, initial_board, 'eye_in_hand')
    retained, rejected, reasons = [], [], []
    for s, te, re in zip(train, initial_error['per_sample_t'], initial_error['per_sample_r']):
        if te*1000 > cutoff_mm or re > cutoff_deg:
            rejected.append(s)
            reasons.append({'index': s['index'], 'translation_mm': te*1000, 'rotation_deg': re})
        else:
            retained.append(s)
    if len(retained) < 12 or len(rejected)/len(train) > max_reject_fraction:
        raise ValueError('too many training poses would be rejected; refusing a small cherry-picked fit')
    diversity = motion_diversity_report(retained)
    if not diversity['ok']:
        raise ValueError('retained training poses lack rotational diversity')
    offsets, camera, board, fit = fit_offsets(retained)
    def error(group):
        return evaluate(corrected_samples(group, offsets), camera, board, 'eye_in_hand')
    tr, ho = error(retained), error(held)
    passed = (fit['converged'] and fit['jacobian_rank'] == 16 and not fit['offset_at_bound']
              and tr['translation_mean_m']*1000 <= 3. and tr['rotation_mean_deg'] <= 1.
              and ho['translation_max_m']*1000 <= 3. and ho['rotation_max_deg'] <= 1.)
    report = {
        'verdict': 'PASS' if passed else 'FAIL', 'validation_status': 'offline_model_candidate',
        'fit': fit, 'initial_fit': initial_fit, 'motion_diversity': diversity,
        'training_indices': [s['index'] for s in retained],
        'original_training_indices': [s['index'] for s in train],
        'holdout_indices': [s['index'] for s in held],
        'rejected_training_indices': [s['index'] for s in rejected],
        'rejection': {'translation_mm': cutoff_mm, 'rotation_deg': cutoff_deg,
                      'max_fraction': max_reject_fraction, 'passes': 1, 'reasons': reasons,
                      'selection_data': 'training only; holdout never removed or used for fitting'},
        'training': tr, 'training_all': error(train), 'holdout': ho, 'all': error(samples),
        'rejected_training_error': error(rejected) if rejected else None,
        'T_base_board': board.tolist(),
        'criteria': {'training_mean_translation_mm': 3., 'training_mean_rotation_deg': 1.,
                     'holdout_max_translation_mm': 3., 'holdout_max_rotation_deg': 1.},
        'limitations': ['Training poses are filtered once after a robust initial fit.',
                       'All original samples and all six holdout poses are retained in reports.',
                       'These data were inspected during diagnosis; fresh independent validation is required.',
                       'Joint offsets are an estimated model, not measured encoder zeros.',
                       'Static board consistency does not establish grasp accuracy.'],
    }
    return offsets, camera, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', required=True)
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--cutoff-mm', type=float, default=4.)
    args = ap.parse_args()
    meta, samples = load(args.samples)
    offsets, camera, report = optimize(samples, cutoff_mm=args.cutoff_mm)
    import xacro
    nominal = xacro.process_file(args.urdf).toxml()
    xml = export_candidate_urdf(nominal, offsets)
    dest = Path(args.output_dir).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    model_path = dest/'robot_model_candidate.urdf'
    model_path.write_text(xml)
    profile = {'joint_names': [f'joint{i}' for i in range(1, 7)],
               'offsets_rad': offsets.tolist(), 'offsets_deg': np.rad2deg(offsets).tolist(),
               'convention': 'nominal_fk(raw_joint_feedback + offsets_rad)',
               'candidate_urdf': str(model_path), 'expected_model': model_signature(xml),
               'encoder_feedback_and_commands': 'unchanged', 'sdk_gravity_model': 'unchanged'}
    def source(path):
        p = Path(path).resolve()
        return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    report.update(robot_calibration=profile, sources={
        'training_samples': source(args.samples), 'validation_samples': source(args.samples),
        'nominal_urdf': source(args.urdf)})
    (dest/'robot_model_candidate.json').write_text(json.dumps(profile, indent=2)+'\n')
    report_path = dest/'validation_report.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    xyz, quat = T_to_pose(camera)
    result = {'calibration_type': 'eye_in_hand', 'validation_status': 'offline_model_candidate',
              'verdict': report['verdict'], 'parent_frame': meta['tool_frame'],
              'tool_frame': meta['tool_frame'], 'base_frame': meta['base_frame'],
              'camera_frame': meta['camera_frame'], 'camera_root_frame': 'wrist_camera_link',
              'T_tool_camera': camera.tolist(), 'T_base_board': report['T_base_board'],
              'translation': dict(zip(('x', 'y', 'z'), xyz)),
              'rotation': dict(zip(('x', 'y', 'z', 'w'), quat)),
              'solver': 'training_only_robust_trim_then_joint2_to_5_nonlinear_fit',
              'robot_calibration': profile, 'error': report['training'],
              'holdout_error': report['holdout'], 'all_samples_error': report['all'],
              'observation_geometry': meta['observation_geometry'], 'validation_report': str(report_path)}
    (dest/'handeye_model_candidate.yaml').write_text(yaml.safe_dump(result, sort_keys=False))
    print('retained training:', len(report['training_indices']), 'rejected:',
          [i+1 for i in report['rejected_training_indices']], 'holdout:', len(report['holdout_indices']))
    print('offsets_deg:', np.round(np.rad2deg(offsets), 6).tolist())
    for name in ('training', 'holdout', 'all'):
        e=report[name]
        print(f"{name}: mean={e['translation_mean_m']*1000:.3f} mm max={e['translation_max_m']*1000:.3f} mm rotation_max={e['rotation_max_deg']:.3f} deg")
    print('ORIGINAL 3mm CRITERIA:', report['verdict'], 'saved', dest)


if __name__ == '__main__':
    main()
