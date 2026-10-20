from types import SimpleNamespace

import pytest
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
from x5a_pick_place.pick_place_node import ARM_JOINTS, PickPlaceNode


def candidate_fixture(feasibility, restore=True):
    calls = []
    results = iter(feasibility)
    n = SimpleNamespace(
        reobserve_enabled=True, reobserve_backoff=.05, transit_vel=.82, transit_acc=.43,
        _cancelled=lambda: False,
        pose_rpy=lambda x,y,z,rpy: (x,y,z,tuple(rpy)),
        ik_joints_for_pose=lambda pose: [pose[3][2]]*6,
        precheck_approach_path=lambda *a: next(results),
        set_preflight_target_present=lambda present: calls.append(('target',present)) or (restore if present else True),
        log_stage=lambda *a,**kw: None,
        move_joints=lambda q,stage,v,a: calls.append(('move',q,stage,v,a)) or True,
    )
    return n,calls


def test_partial_approach_is_skipped_before_any_observation_motion():
    n,calls=candidate_fixture([False,True])
    selected=PickPlaceNode.move_pregrasp_candidates(n,.2,.3,.044,
                                                  [[0.,1.1,.2],[0.,1.05,1.57]],.15)
    assert selected == [0.,1.05,1.57]
    assert calls[:2] == [('target',False),('target',True)]
    assert len(calls) == 3 and calls[2][1] == [1.57]*6
    assert calls[2][3:] == (.82,.43)


@pytest.mark.parametrize('feasibility,restore', [([False,False],True),([True,True],False)])
def test_no_motion_when_no_full_approach_or_scene_restore_fails(feasibility,restore):
    n,calls=candidate_fixture(feasibility,restore)
    assert PickPlaceNode.move_pregrasp_candidates(n,.2,.3,.044,
                                                [[0.,1.1,.2],[0.,1.05,1.57]],.15) is None
    assert calls == [('target',False),('target',True)]


def test_exception_restores_target_before_propagation():
    n,calls=candidate_fixture([])
    def fail(*args): raise RuntimeError('service failed')
    n.precheck_approach_path=fail
    with pytest.raises(RuntimeError):
        PickPlaceNode.move_pregrasp_candidates(n,.2,.3,.044,[[0.,1.1,.2]],.15)
    assert calls == [('target',False),('target',True)]


def test_no_camera_backoff_when_reobservation_is_disabled():
    n,calls=candidate_fixture([True])
    n.reobserve_enabled=False
    poses=[]
    def ik(pose):
        poses.append(pose)
        return [0.]*6
    n.ik_joints_for_pose=ik
    assert PickPlaceNode.move_pregrasp_candidates(n,.2,.3,.044,[[0.,1.1,0.]],.06)
    assert poses[0][:3] == pytest.approx((.2,.3,.104))


@pytest.mark.parametrize('fraction,midpoint,accepted', [(1.,0.,True),(.9375,0.,False),(1.,4.,False)])
def test_requires_complete_path_and_bounds_uses_actual_open_fingers(fraction,midpoint,accepted):
    requests=[]
    t=RobotTrajectory();t.joint_trajectory.joint_names=list(ARM_JOINTS)
    for value in (0.,midpoint,0.):
        t.joint_trajectory.points.append(JointTrajectoryPoint(positions=[value]*6))
    result=SimpleNamespace(error_code=SimpleNamespace(val=1),fraction=fraction,solution=t)
    def call(req):
        requests.append(req)
        return SimpleNamespace(result=lambda:result)
    n=SimpleNamespace(
        base_frame='base_link',group='arm',tcp_frame='tool0',cart_step=.01,
        jump_threshold=0.,gripper_position=.042,min_frac=.9999,
        servo_bounds_fault=lambda q: 'bad' if any(abs(v)>2. for v in q) else '',
        cart_cli=SimpleNamespace(call_async=call),wait_future=lambda *a:True,
        log_stage=lambda *a,**kw:None,
    )
    assert PickPlaceNode.precheck_approach_path(n,[0.]*6,Pose()) == accepted
    assert requests[0].avoid_collisions
    assert list(requests[0].start_state.joint_state.position[-2:]) == [.042,.042]
