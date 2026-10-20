import json

import numpy as np
import pytest

from x5a_handeye.check_model_candidate import reject_old_session
from x5a_handeye.recalibrate_fixed_model import recalibrate
from x5a_handeye.refine_robot_model import matrix, split
from x5a_handeye.transforms import inv_T
from x5a_handeye.x5a_fk import fk_base_tool0


def remounted_samples():
    offsets = np.deg2rad([0., .75, -.62, 2.73, 1.18, 0.])
    camera = matrix([-.06, .02, .05, 3.1, .15, -.04])
    board = matrix([.2, .15, .01, .05, .03, -.1])
    rng = np.random.default_rng(328)
    samples = [dict(index=i, joints=q.tolist(), T_base_tool=fk_base_tool0(q),
                    T_camera_board=inv_T(camera) @ inv_T(fk_base_tool0(q+offsets)) @ board)
               for i, q in enumerate(rng.uniform(-1.5, 1.5, (27, 6)))]
    return samples, offsets, camera


def test_recovers_remounted_camera_with_frozen_offsets_and_raw_feedback():
    samples, offsets, camera = remounted_samples()
    raw_before = np.array([s['joints'] for s in samples])
    actual, report = recalibrate(samples, offsets)
    np.testing.assert_allclose(actual, camera, atol=1e-9)
    np.testing.assert_array_equal([s['joints'] for s in samples], raw_before)
    assert report['verdict'] == 'PASS'
    assert report['holdout']['translation_max_m'] < 1e-9
    _, nominal = recalibrate(samples, np.zeros(6))
    assert nominal['training']['translation_mean_m'] > .003


def test_bad_holdout_cannot_change_selected_camera_or_be_silently_discarded():
    samples, offsets, _ = remounted_samples()
    baseline, original = recalibrate(samples, offsets)
    _, held = split(samples)
    held[0]['T_camera_board'][:3, 3] += [.05, 0., 0.]
    actual, report = recalibrate(samples, offsets)
    np.testing.assert_array_equal(actual, baseline)
    assert report['solver'] == original['solver']
    assert report['verdict'] == 'FAIL'
    assert report['holdout']['translation_max_m'] > .04
    assert len(report['holdout']['per_sample_t']) == 6


def test_remount_check_refuses_samples_used_to_learn_frozen_robot_model(tmp_path):
    old = tmp_path/'old_samples.json'
    old.write_text('{}')
    report = tmp_path/'validation_report.json'
    report.write_text(json.dumps({
        'sources': {'training_samples': {'path': str(tmp_path/'new.json'), 'sha256': 'new'},
                    'validation_samples': {'path': str(tmp_path/'new.json'), 'sha256': 'new'}},
        'frozen_model_provenance': {'sources': {
            'training_samples': {'path': str(old), 'sha256': 'old'}}},
    }))
    with pytest.raises(ValueError, match='used to build'):
        reject_old_session({'validation_report': str(report)}, old, tmp_path/'candidate.yaml')
