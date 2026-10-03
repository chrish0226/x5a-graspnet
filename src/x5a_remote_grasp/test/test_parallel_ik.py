"""Parallel IK preserves seed preference, live collision checks and fail-closed contracts."""
from types import SimpleNamespace as NS
import copy
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import PlanningScene, RobotState
from x5a_remote_grasp.parallel_ik import ParallelIKClient
from x5a_remote_grasp.moveit_plan import PlanOnlyMoveIt


def state(value):
    result=RobotState()
    result.joint_state.name=['joint1']
    result.joint_state.position=[float(value)]
    return result


class Future:
    def __init__(self, result):self.value=result
    def done(self):return True
    def result(self):return self.value


class Client:
    def __init__(self, result):self.result=result;self.requests=[]
    def service_is_ready(self):return True
    def call_async(self, request):
        self.requests.append(copy.deepcopy(request))
        return Future(self.result(request))


def backend(response):
    result=object.__new__(ParallelIKClient)
    result.workers=4;result.model_digest='digest';result.wait=lambda future,timeout:True
    scene=PlanningScene();scene.robot_state.joint_state.name=['joint7','joint8']
    scene.robot_state.joint_state.position=[.032,.031]
    result._snapshot=lambda *_:scene
    result.client=Client(response)
    return result


def response(codes):
    return NS(success=True,message='',model_digest='digest',
              error_codes=[NS(val=code) for code in codes],solutions=[state(i) for i in range(len(codes))])


def test_first_seed_success_does_not_launch_expensive_fallback_batch():
    b=backend(lambda req:response([1]*len(req.requests)))
    iterator=b.solutions(PoseStamped(),[state(0),state(1),None],'arm','tool0',.15,lambda t:t)
    assert next(iterator).joint_state.position[0]==0
    iterator.close()
    assert len(b.client.requests)==1 and b.client.requests[0].workers==1


def test_fallbacks_share_snapshot_and_preserve_seed_order():
    b=backend(lambda req:response([-31] if req.workers==1 else [1,1,-31]))
    solutions=list(b.solutions(PoseStamped(),[state(0),state(1),state(2),None],
                             'arm','tool0',.15,lambda t:t))
    assert [s.joint_state.position[0] for s in solutions]==[0,1]
    assert [r.workers for r in b.client.requests]==[1,4]
    first,second=b.client.requests
    assert first.scene==second.scene
    assert list(second.scene.robot_state.joint_state.position)==[.032,.031]
    assert [list(r.robot_state.joint_state.position) for r in second.requests]==[[1.],[2.],[]]
    assert second.requests[-1].robot_state.is_diff


@pytest.mark.parametrize('fault',['digest','length','success','error'])
def test_bad_batch_never_silently_falls_back_to_serial(fault):
    def broken(req):
        r=response([1])
        if fault=='digest':r.model_digest='other model'
        elif fault=='length':r.solutions=[]
        elif fault=='success':r.success=False
        else:r.error_codes[0].val=-17
        return r
    b=backend(broken)
    with pytest.raises(RuntimeError):
        list(b.solutions(PoseStamped(),[None],'arm','tool0',.15,lambda t:t))


def test_parallel_success_still_requires_live_scene_collision_validation():
    p=object.__new__(PlanOnlyMoveIt)
    p.deadline=None;p.group_name='arm';p.tcp_frame='tool0';p.ik_timeout=.15
    p.parallel_ik=NS(ready=lambda:True,solutions=lambda *args:iter([state(0),state(1)]))
    checked=[]
    def valid(candidate):
        checked.append(candidate.joint_state.position[0])
        return (checked[-1]==1,'new obstacle')
    p.state_valid=valid
    chosen,reason=p.collision_aware_ik(PoseStamped())
    assert chosen.joint_state.position[0]==1 and checked==[0,1]


def test_required_parallel_service_missing_aborts():
    p=object.__new__(PlanOnlyMoveIt)
    p.deadline=None
    p.parallel_ik=NS(ready=lambda:False);p.parallel_ik_fallback_serial=False
    with pytest.raises(RuntimeError,match='unavailable'):
        p.collision_aware_ik(PoseStamped())


@pytest.mark.parametrize('planned_gap', [.014, .044])
def test_all_fallback_seeds_keep_planned_closed_or_open_fingers(planned_gap):
    b=backend(lambda req:response([-31]*len(req.requests)))
    primary=state(0)
    primary.joint_state.name+=['joint7','joint8']
    primary.joint_state.position=[0.,planned_gap,planned_gap]
    list(b.solutions(PoseStamped(),[primary,None],'arm','tool0',.15,lambda t:t))
    assert len(b.client.requests)==2
    for request in b.client.requests:
        assert list(request.scene.robot_state.joint_state.position)==[planned_gap,planned_gap]


@pytest.mark.parametrize('budget_exhausted', [False, True])
def test_batch_timeout_distinguishes_global_budget_from_service_failure(budget_exhausted):
    b=backend(lambda req:response([1]))
    expired=[False]
    def wait(*args):
        expired[0]=budget_exhausted
        return False
    b.wait=wait
    remaining=lambda t: 0.0 if expired[0] else t
    result=b.solutions(PoseStamped(),[None],'arm','tool0',.15,remaining)
    if budget_exhausted:
        assert list(result)==[]
    else:
        with pytest.raises(RuntimeError,match='batch timed out'):
            list(result)
