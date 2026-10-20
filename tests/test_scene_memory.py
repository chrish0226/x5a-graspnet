"""Robot operation memory is evidence for clearing, never a hidden grasp pose."""
import time

import pytest

from x5a_task_server.scene_memory import SceneMemory
from x5a_task_server.clearing import ClearingMixin


def recorded(upper='white', lower='orange'):
    memory=SceneMemory()
    memory.record_stack(upper,lower,(.30,.30,.041),.03,.002,'base','current',100)
    return memory


@pytest.mark.parametrize('upper,lower',[('white','orange'),('orange','white'),('red','white')])
def test_hidden_lower_uses_executed_relationship_and_live_upper(upper,lower):
    memory=recorded(upper,lower)
    scene={lower:{'ready':False,'xyz':None},upper:{'ready':True,'xyz':[.30,.30,.056]}}
    assert memory.candidate(lower,scene,'base','current')==upper
    scene[upper]['ready']=False
    assert memory.candidate(lower,scene,'base','current') is None


@pytest.mark.parametrize('point',[(.33,.30,.056),(.30,.30,.024),(.30,.35,.056)])
def test_moved_or_unstacked_upper_invalidates_memory(point):
    memory=recorded()
    assert memory.candidate('orange',{'white':{'ready':True,'xyz':point}},'base','current') is None
    # A single mismatched frame suspends use, without destroying the record.
    assert memory.snapshot()
    for stamp in (101, 102, 103):
        memory.observe('white',point,'base','current',stamp)
    assert not memory.snapshot()


def test_lower_visible_before_release_cannot_clear_new_memory():
    memory=recorded()
    memory.observe('orange',(.30,.30,.024),'base','current',99)
    assert memory.has_lower('orange')
    memory.observe('orange',(.30,.30,.024),'base','current',101)
    assert memory.has_lower('orange')
    memory.observe('orange',(.30,.30,.024),'base','current',102)
    memory.observe('orange',(.30,.30,.024),'base','current',103)
    assert not memory.has_lower('orange')


def test_partial_lower_detection_cannot_erase_a_live_matching_stack():
    memory=recorded()
    upper={'ready':True,'xyz':[.30,.30,.056]}
    for stamp in range(101, 111):
        memory.observe('orange',(.30,.30,.024),'base','current',stamp,upper)
    assert memory.candidate('orange',{'white':upper},'base','current')=='white'


def test_transient_wrong_upper_recovers_and_repeated_stamp_is_not_evidence():
    memory=recorded()
    for _ in range(5):
        memory.observe('white',(.26,.36,.021),'base','current',101)
    assert memory.has_lower('orange')
    memory.observe('white',(.30,.30,.056),'base','current',102)
    assert memory.candidate('orange',{'white':{'ready':True,'xyz':[.30,.30,.056]}},'base','current')=='white'


def test_visible_lower_still_clears_when_upper_is_confirmed():
    from test_clearing import Session
    robot=Session()
    robot.cube_poses['white']=object()  # lower is reported ready at command entry
    assert robot.vision_snapshot()['white']['ready']
    assert robot.clear_target_if_needed('white')[0]
    assert any(isinstance(c,tuple) and c[:2]==('pick','orange') for c in robot.calls)


def test_pick_and_new_stack_replace_old_relationships():
    memory=recorded()
    memory.moved('white');assert not memory.snapshot()
    memory=recorded()
    memory.record_stack('white','red',(.20,.30,.041),.03,.002,'base','current',200)
    assert not memory.has_lower('orange') and memory.has_lower('red')
    memory.moved('red');assert not memory.snapshot()


def test_memory_cannot_cross_frame_side_or_expiry():
    memory=recorded()
    scene={'white':{'ready':True,'xyz':[.30,.30,.056]}}
    assert memory.candidate('orange',scene,'other','current') is None
    assert memory.candidate('orange',scene,'base','mirror') is None
    memory.ttl_s=0
    assert memory.candidate('orange',scene,'base','current') is None


def test_resolver_needs_no_live_lower_color_relation():
    class Robot(ClearingMixin):
        base_frame='base';observation_side='current'
        scene_memory=recorded()
        def _clearing_details(self):return {'blocked_by':{}}
        def vision_snapshot(self):return {'orange':{'ready':False,'xyz':None},'white':{'ready':True,'xyz':[.30,.30,.056]}}
    assert Robot()._resolve_blocker('orange')=='white'


def test_complete_clearing_flow_accepts_completely_invisible_lower_from_memory():
    from test_clearing import Session
    class MemorySession(Session):
        def __init__(self):
            super().__init__()
            self.base_frame='base';self.observation_side='current'
            self.scene_memory=SceneMemory()
            self.scene_memory.record_stack('orange','white',(.36,.30,.041),.03,.002,'base','current',1)
        def _clearing_details(self,after=0.):return {'blocked_by':{},'stamp_ns':100}
        def vision_snapshot(self):
            result=super().vision_snapshot()
            if not self.cube_poses:
                result['white'].update(status='missing',xyz=None)
            return result
    robot=MemorySession()
    assert robot.clear_target_if_needed('white')[0]
    assert any(isinstance(c,tuple) and c[:2]==('pick','orange') for c in robot.calls)
