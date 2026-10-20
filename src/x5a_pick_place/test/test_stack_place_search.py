"""Aligned placement must search alternatives without moving rejected paths."""
from types import SimpleNamespace as NS
import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from x5a_pick_place.pick_place_node import PickPlaceNode, pose_to_xyz
from x5a_pick_place.stacking import aligned_stack_candidates


def fixture(ik_results, descent_results=(True,), plan_results=(True,)):
    ik=iter(ik_results);desc=iter(descent_results);plans=iter(plan_results);calls=[]
    node=NS(_held_object_offset=[.033326576,.001541879,-.017067999],pre_place_h=.06,
            transit_vel=.82,transit_acc=.43,_cancelled=lambda:False,
            log_stage=lambda *args,**kw:None,
            pose_rpy=lambda *args:PickPlaceNode.pose_rpy(None,*args))
    def solve(pose,collision_aware_only=False):
        assert collision_aware_only
        calls.append(('ik',pose))
        return next(ik)
    def precheck(joints,pose):calls.append(('descent',list(joints),pose));return next(desc)
    def move(joints,stage,v,a):calls.append(('transit',list(joints),stage));return next(plans)
    node.ik_joints_for_pose=solve;node.precheck_approach_path=precheck;node.move_joints=move
    return node,calls


def candidates():
    return aligned_stack_candidates([0.,1.1,1.5551078555320985],1.5551078555320985,2.276)


def test_recorded_failed_yaw_falls_back_to_equivalent_quarter_turn():
    node,calls=fixture([None,[.1]*6])
    center=[.342,.256,.041]
    result=PickPlaceNode.move_stack_pre_place(node,center,candidates())
    xyz,rpy=result
    assert rpy[2]==pytest.approx(.7052036732051032)
    assert [call[0] for call in calls]==['ik','ik','descent','transit']
    contact=calls[-2][-1]
    np.testing.assert_allclose(pose_to_xyz(contact),xyz)
    np.testing.assert_allclose(np.array(xyz)+Rotation.from_euler('xyz',rpy).apply(node._held_object_offset),center)
    assert calls[-1][-1]=='MOVE_PRE_PLACE_2_IK'


def test_partial_descent_is_rejected_before_transit():
    node,calls=fixture([[.1]*6,[.2]*6],(False,True))
    selected=PickPlaceNode.move_stack_pre_place(node,[.342,.256,.041],candidates())
    assert selected[1]==candidates()[1]
    assert [call[0] for call in calls]==['ik','descent','ik','descent','transit']
    assert calls[-1][1]==[.2]*6


def test_global_planning_failure_tries_next_aligned_pose():
    node,calls=fixture([[.1]*6,[.2]*6],(True,True),(False,True))
    assert PickPlaceNode.move_stack_pre_place(node,[.342,.256,.041],candidates())[1]==candidates()[1]
    assert [call[0] for call in calls]==['ik','descent','transit','ik','descent','transit']


def test_all_ik_failures_leave_arm_and_gripper_untouched():
    node,calls=fixture([None]*4)
    assert PickPlaceNode.move_stack_pre_place(node,[.342,.256,.041],candidates()) is None
    assert [call[0] for call in calls]==['ik']*4


def test_execution_interruption_is_not_retried():
    node,calls=fixture([[.1]*6])
    def interrupted(*args):raise RuntimeError('partial execution; retain holding')
    node.move_joints=interrupted
    with pytest.raises(RuntimeError,match='retain holding'):
        PickPlaceNode.move_stack_pre_place(node,[.342,.256,.041],candidates())
    assert [call[0] for call in calls]==['ik','descent']


def test_cancel_after_precheck_does_not_start_transit():
    node,calls=fixture([[.1]*6])
    def precheck(*args):node._cancelled=lambda:True;return True
    node.precheck_approach_path=precheck
    assert PickPlaceNode.move_stack_pre_place(node,[.342,.256,.041],candidates()) is None
    assert [call[0] for call in calls]==['ik']
