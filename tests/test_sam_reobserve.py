"""Bounded retries and slow release with fake controllers only."""
import asyncio
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
from x5a_moveit_official_adapter.official_trajectory_adapter import OfficialTrajectoryAdapter
from x5a_remote_grasp.place_geometry import cube_placement_cloud, object_in_tool, placement_tool
from x5a_web_agent import app
from x5a_web_agent.session import Session


def test_cube_hidden_bottom_is_completed_for_placement_without_altering_contact_points():
    x,y=np.meshgrid(np.linspace(.25,.28,40),np.linspace(.30,.33,40))
    measured=np.c_[x.ravel(),y.ravel(),np.full(x.size,.055)]
    before=measured.copy()
    completed=cube_placement_cloud(measured)
    np.testing.assert_array_equal(measured,before)
    target=placement_tool(object_in_tool(completed,np.eye(4)),np.eye(3),[.3,.2,.055],.01)
    # Real cube bottom at .025, so its top must be at .095 after release.
    assert np.median((completed+target[:3,3])[:len(measured),2]) == pytest.approx(.095,abs=.001)
    narrow=measured.copy();narrow[:,0]=.25+(narrow[:,0]-.25)*.3
    np.testing.assert_array_equal(cube_placement_cloud(narrow),narrow)


def test_opening_interpolates_and_holds_arm_target_instead_of_following_drift(monkeypatch):
    import x5a_moveit_official_adapter.official_trajectory_adapter as module
    t=[0.];commands=[];outcome=[]
    monkeypatch.setattr(module.time,'monotonic',lambda:t[0])
    monkeypatch.setattr(module.time,'sleep',lambda v:t.__setitem__(0,t[0]+v))
    def snapshot():
        gripper=commands[-1][1] if commands else 1.5
        return [.18]*6,[0.]*6,gripper,t[0]
    n=NS(finger_max=.044,gripper_open=5.,gripper_closed=0.,gripper_duration=.8,
        gripper_open_ramp=.65,gripper_open_threshold=.03,gripper_tolerance=.004,
        start_tolerance=.15,tracking_tolerance=.4,status_timeout=.5,rate_hz=50.,
        last_arm_target=[.2]*6,goal_active=True,snapshot=snapshot,
        command_topic='/arm_cmd',count_publishers=lambda _:1,driver_health_fault=lambda:None,
        get_logger=lambda:NS(info=lambda _:None,error=lambda _:None),
        _publish_command=lambda q,e,g:commands.append((list(q),g)))
    n.finger_position=lambda x:OfficialTrajectoryAdapter.finger_position(n,x)
    n.gripper_readout=lambda x:OfficialTrajectoryAdapter.gripper_readout(n,x)
    goal=NS(request=NS(command=NS(position=.044)),is_cancel_requested=False,
        succeed=lambda:outcome.append('success'),abort=lambda:outcome.append('abort'),publish_feedback=lambda _:None)
    result=OfficialTrajectoryAdapter._execute_gripper(n,goal)
    assert outcome==['success'] and result.reached_goal
    assert all(q==[.2]*6 for q,g in commands)
    assert commands[0][1]==pytest.approx(1.5)
    assert 1.5 < commands[15][1] < 5.
    assert commands[-1][1]==pytest.approx(5.)
    assert t[0] >= 1.45


@pytest.mark.parametrize('forever,move_ok,expected_plans,expected_moves',[
    (False,True,2,2),(True,True,2,1),(False,False,1,1)])
def test_retry_executes_only_one_observation_then_requires_a_new_manipulation_plan(
        monkeypatch,forever,move_ok,expected_plans,expected_moves):
    calls=[];plans=[0]
    class Bridge:
        dry_run=False
        def send_remote_grasp(self,*args,**kwargs):
            plans[0]+=1;calls.append('plan')
            return dict(success=True,plan_valid=True,recovery_only=forever or plans[0]==1)
        def execute_latest(self):
            calls.append('move')
            return dict(success=move_ok,holding=False,stage='SUCCESS' if move_ok else 'FAILED')
    monkeypatch.setattr(app,'bridge',Bridge())
    monkeypatch.setattr(app,'hub',app.ConnectionHub())
    monkeypatch.setattr(app,'session',Session())
    monkeypatch.setenv('X5A_AUTO_EXECUTE','1')
    monkeypatch.setenv('X5A_CARRY_TO_READY','0')
    result=asyncio.run(app._run_remote_grasp('orange cube',{}))
    assert calls.count('plan')==expected_plans and calls.count('move')==expected_moves
    assert result['success']==(not forever and move_ok)
