from types import SimpleNamespace

import pytest
from x5a_pick_place.pick_place_node import PickPlaceNode


@pytest.mark.parametrize('publishers,plan_ok,observation,expected', [
    (2,True,(.2,.3,.03),False),
    (1,False,(.2,.3,.03),False),
    (1,True,None,False),
    (1,True,(.2,.3,.03),True),
])
def test_home_observation_requires_one_writer_plan_and_new_detection(publishers,plan_ok,observation,expected):
    calls=[]
    n=SimpleNamespace(
        plan_only=False,reobserve_enabled=True,
        table_x=.25,table_y=.3,table_z=-.026,
        table_sx=.52,table_sy=.72,table_sz=.04,
        ready_joints=[.8,.67,.83,-1.04,.05,.06],transit_vel=.82,transit_acc=.43,
        last_joints=[0.]*6,
        q=[0.]*6,
        _cancelled=lambda:False,count_publishers=lambda topic:publishers,
        log_stage=lambda *a,**kw:None,report_stage=lambda *a,**kw:None,
        publish_box=lambda *a,**kw:calls.append('table'),
        move_joints=lambda *a:calls.append(('planned_move',a)) or plan_ok,
        wait_for_reobserved_cube=lambda:calls.append('new_frames') or observation,
    )
    assert PickPlaceNode.move_initial_observation(n) == expected
    if publishers != 1:
        assert not calls
    elif not plan_ok:
        assert 'new_frames' not in calls
    else:
        assert calls[0]=='table' and calls[-1]=='new_frames'
        assert calls[1][1][2:]==(.82,.43)
        assert n.last_joints is None


def test_plan_only_home_observation_does_not_require_an_actual_camera_move():
    calls=[]
    n=SimpleNamespace(
        plan_only=True,reobserve_enabled=True,
        table_x=.25,table_y=.3,table_z=-.026,
        table_sx=.52,table_sy=.72,table_sz=.04,
        ready_joints=[.8,.67,.83,-1.04,.05,.06],transit_vel=.82,transit_acc=.43,
        q=[0.]*6,
        _cancelled=lambda:False,log_stage=lambda *a,**kw:None,report_stage=lambda *a,**kw:None,
        publish_box=lambda *a,**kw:None,
        move_joints=lambda *a:True,
        wait_for_reobserved_cube=lambda:calls.append('unexpected_camera_wait'),
    )
    assert PickPlaceNode.move_initial_observation(n)
    assert not calls


def test_repeat_at_observation_pose_skips_zero_motion_but_requires_new_frames():
    calls=[]
    ready=[.8,.67,.83,-1.04,.05,.06]
    n=SimpleNamespace(
        plan_only=False,reobserve_enabled=True,q=ready,ready_joints=ready,
        table_x=.25,table_y=.3,table_z=-.026,table_sx=.52,table_sy=.72,table_sz=.04,
        transit_vel=.82,transit_acc=.43,_cancelled=lambda:False,
        count_publishers=lambda topic:1,log_stage=lambda *a,**kw:None,
        report_stage=lambda *a,**kw:None,publish_box=lambda *a,**kw:None,
        move_joints=lambda *a:calls.append('unexpected_motion') or True,
        wait_for_reobserved_cube=lambda:calls.append('new_frames') or (.2,.3,.03),
    )
    assert PickPlaceNode.move_initial_observation(n)
    assert calls==['new_frames']
