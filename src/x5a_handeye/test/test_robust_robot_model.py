import copy
import numpy as np
import pytest

from x5a_handeye.robust_robot_model import optimize
from x5a_handeye.refine_robot_model import matrix,split
from x5a_handeye.transforms import inv_T
from x5a_handeye.x5a_fk import fk_base_tool0
from x5a_handeye.calibrated_tf import calibrated_transform,require_failed_preview_context


@pytest.fixture(scope='module')
def robust_example():
    offsets=np.deg2rad([0.,.8,-.6,2.3,1.1,0.])
    camera=matrix([-.06,.01,.05,.2,-.5,.4]);board=matrix([.3,.1,.03,-.4,.2,.1])
    rng=np.random.default_rng(27)
    samples=[dict(index=i,joints=q.tolist(),T_base_tool=fk_base_tool0(q),
                  T_camera_board=inv_T(camera)@inv_T(fk_base_tool0(q+offsets))@board)
             for i,q in enumerate(rng.uniform(-1.5,1.5,(30,6)))]
    train,_=split(samples)
    bad=train[2]['index'];samples[bad]['T_camera_board'][:3,3]+=[.04,-.025,.02]
    result=optimize(samples)
    return samples,offsets,camera,bad,result


def test_robust_trim_recovers_known_model_without_altering_raw_feedback(robust_example):
    samples,expected_offsets,expected_camera,bad,(offsets,camera,report)=robust_example
    assert report['rejected_training_indices']==[bad]
    assert not set(report['rejected_training_indices'])&set(report['holdout_indices'])
    np.testing.assert_allclose(offsets,expected_offsets,atol=1e-7)
    np.testing.assert_allclose(camera,expected_camera,atol=1e-7)
    assert report['verdict']=='PASS'
    assert report['all']['translation_max_m']>.03
    assert report['holdout']['translation_max_m']<1e-7


def test_bad_holdout_cannot_be_removed_or_change_the_optimized_model(robust_example):
    samples,_,_,_,baseline=robust_example
    changed=copy.deepcopy(samples);_,held=split(changed)
    held[0]['T_camera_board'][:3,3]+=[.05,0,0]
    raw_before=np.array([s['joints'] for s in changed])
    offsets,camera,report=optimize(changed)
    np.testing.assert_array_equal(offsets,baseline[0]);np.testing.assert_array_equal(camera,baseline[1])
    np.testing.assert_array_equal([s['joints'] for s in changed],raw_before)
    assert report['rejected_training_indices']==baseline[2]['rejected_training_indices']
    assert report['verdict']=='FAIL' and len(report['holdout']['per_sample_t'])==6
    assert report['holdout']['translation_max_m']>.04


def test_failed_candidate_requires_explicit_acceptance_and_still_checks_geometry():
    result=dict(calibration_type='eye_in_hand',verdict='FAIL',validation_status='offline_model_candidate',
                T_tool_camera=np.eye(4).tolist(),robot_calibration={'expected_model':{'test':'signature'}})
    with pytest.raises(ValueError):calibrated_transform(result,True)
    with pytest.raises(ValueError):calibrated_transform(result,False,True)
    calibrated_transform(result,True,True)
    damaged=copy.deepcopy(result);damaged['T_tool_camera'][0][0]=-1
    with pytest.raises(ValueError):calibrated_transform(damaged,True,True)
    damaged=copy.deepcopy(result);damaged['validation_status']='ordinary_failed_calibration'
    with pytest.raises(ValueError):calibrated_transform(damaged,True,True)
    require_failed_preview_context(True,'wrist_preview/','/wrist_preview/robot_description')
    for prefix,topic in [('','/robot_description'),('wrist_preview/','/robot_description'),('robot/','/wrist_preview/robot_description')]:
        with pytest.raises(ValueError):require_failed_preview_context(True,prefix,topic)
