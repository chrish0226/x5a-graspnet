import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import GoalResponse
from std_srvs.srv import Trigger
from x5a_task_interfaces.action import X5aPickPlace
from x5a_task_server.wrist_task_server import WristTaskServer
from x5a_pick_place.pick_place_node import PickPlaceNode
from x5a_web_agent.wrist_commands import parse_wrist_command


def test_clear_and_target_share_one_action_and_busy_lock(robot, monkeypatch):
    robot._prepared = True
    calls = []
    def clear(color):
        assert robot._busy
        calls.append(('clear', color))
        assert robot.goal_callback(goal('orange')) == GoalResponse.REJECT
        return True, ''
    monkeypatch.setattr(robot, 'clear_target_if_needed', clear)
    monkeypatch.setattr(robot, 'execute_pick_place', lambda color, **kwargs: calls.append(('pick',color)) or (True,''))
    monkeypatch.setattr(robot, '_wait_observation', lambda **kwargs: True)
    request = goal('white')
    request.destination_color = 'hold'
    assert robot.goal_callback(request) == GoalResponse.ACCEPT
    assert robot.execute_callback(handle(request)).success
    assert calls == [('clear','white'), ('pick','white')] and not robot._busy


def test_clear_failure_prevents_target_grasp_and_preserves_holding(robot, monkeypatch):
    robot._prepared = True
    def clear(color):
        robot._holding = True
        return False, '暂放位被占用；夹持状态保留'
    monkeypatch.setattr(robot, 'clear_target_if_needed', clear)
    monkeypatch.setattr(robot, 'execute_pick_place', lambda *a, **kw: pytest.fail('must not grasp after clearing failure'))
    request = goal('white')
    assert robot.goal_callback(request) == GoalResponse.ACCEPT
    result = robot.execute_callback(handle(request))
    assert not result.success and '夹持状态保留' in result.message
    assert robot._holding and not robot._busy


def test_buffer_check_can_only_select_and_never_grasp_or_place(robot, monkeypatch):
    from geometry_msgs.msg import PoseStamped
    pose=PoseStamped();pose.pose.position.x=.28;pose.pose.position.y=.30;pose.pose.position.z=.054
    robot.cube_poses['white']=pose
    robot.cube_stable['white']=True
    robot.cube_pose_arrivals['white']=time.monotonic()
    monkeypatch.setattr(robot,'_clearing_details',lambda *a: {'blocked_by':{'orange':'white'}})
    monkeypatch.setattr(robot,'_select_buffer',lambda *a: ((.20,.30),''))
    monkeypatch.setattr(robot,'execute_pick_place',lambda *a,**kw: pytest.fail('read-only check must not grasp'))
    monkeypatch.setattr(robot,'execute_place_motion',lambda *a,**kw: pytest.fail('read-only check must not place'))
    reply=robot.check_buffer_callback(Trigger.Request(),Trigger.Response(),'orange')
    assert reply.success and json.loads(reply.message)['motion_sent'] is False
    assert robot._stage=='IDLE' and not robot._busy


@pytest.fixture(scope='module')
def robot_context():
    config = str(Path(__file__).resolve().parents[1]/'config/task_execute.yaml')
    rclpy.init(args=['--ros-args', '--params-file', config])
    yield
    rclpy.try_shutdown()


@pytest.fixture
def robot(monkeypatch, robot_context):
    node = WristTaskServer()
    node.q = list(node.ready_joints)
    node.joint_state_arrival = time.monotonic()
    node.dq = [0.0]*6
    monkeypatch.setattr(node, 'wait_ready', lambda: True)
    monkeypatch.setattr(node, 'handeye_tf_ok', lambda: True)
    yield node
    node.destroy_node()


def goal(color='red'):
    request = X5aPickPlace.Goal()
    request.target_color = color
    return request


@pytest.mark.parametrize('text,action,color', [
    ('准备', 'prepare', None), ('准备。', 'prepare', None), ('就位', 'prepare', None),
    ('抓红色', 'pick', 'red'), ('把白色方块放进盒子里', 'pick_place', 'white'),
    ('抓橙色', 'pick', 'orange'), ('停止抓红色', 'cancel', None),
    ('抓起一个杯子', 'unknown', None), ('红色和白色都抓取', 'unknown', None),
    ('松开夹爪', 'release', None), ('抓最近的', 'unknown', None),
    ('准备然后抓红色', 'unknown', None), ('红色白色', 'unknown', None),
    ('不要抓红色', 'cancel', None), ('放到白色方块上', 'place', None),
    ('回原点', 'go_home', None), ('归原位', 'go_home', None), ('返回原位', 'go_home', None),
    ('请回到原点一下', 'go_home', None), ('回到观察位', 'prepare', None),
])
def test_local_color_commands(text, action, color):
    parsed = parse_wrist_command(text)
    assert parsed['action'] == action
    assert parsed.get('target_color') == color


@pytest.mark.parametrize('text,expected', [
    ('抓白色方块', {'action':'pick','target_color':'white'}),
    ('请拿起白色积木', {'action':'pick','target_color':'white'}),
    ('放到红色方块', {'action':'place','destination_color':'red'}),
    ('放到红色方块上', {'action':'place','destination_color':'red'}),
    ('把它放进盒子里', {'action':'place','destination_color':'box'}),
    ('放到黑盒子里面', {'action':'place','destination_color':'box'}),
    ('把白色方块放到红色方块上', {'action':'pick_place','target_color':'white','destination_color':'red'}),
    ('抓白色方块放进盒子里', {'action':'pick_place','target_color':'white'}),
])
def test_split_and_complete_commands(text,expected):
    assert parse_wrist_command(text) == expected


def test_prepare_is_independent_of_color_and_does_not_operate_gripper(robot, monkeypatch):
    calls = []
    monkeypatch.setattr(robot, 'move_initial_observation',
                        lambda observe_target: calls.append(observe_target) or True)
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: True)
    monkeypatch.setattr(robot, 'call_gripper', lambda *_: pytest.fail('prepare must not change the gripper'))
    robot.cube_poses.clear()
    reply = robot.prepare_callback(Trigger.Request(), Trigger.Response())
    assert reply.success and robot._prepared and calls == [False]
    assert not robot._busy
    assert robot.goal_callback(goal()) == GoalResponse.ACCEPT


def test_prepare_motion_skips_target_detection_but_keeps_planner_and_table():
    calls = []
    n = SimpleNamespace(
        plan_only=False, reobserve_enabled=True, q=[0.0]*6, ready_joints=[0.5]*6,
        table_x=.25, table_y=.3, table_z=-.026, table_sx=.52, table_sy=.72, table_sz=.04,
        transit_vel=.82, transit_acc=.43, _cancelled=lambda: False,
        count_publishers=lambda _: 1, log_stage=lambda *a, **k: None,
        report_stage=lambda *a, **k: None, publish_box=lambda *a, **k: calls.append('table'),
        move_joints=lambda *a: calls.append('plan_and_execute') or True,
        wait_for_reobserved_cube=lambda: pytest.fail('prepare cannot require a red block'),
    )
    assert PickPlaceNode.move_initial_observation(n, observe_target=False)
    assert calls == ['table', 'plan_and_execute']


def test_failed_prepare_does_not_enable_grasp(robot, monkeypatch):
    robot._prepared = True
    monkeypatch.setattr(robot, 'move_initial_observation', lambda **kw: False)
    reply = robot.prepare_callback(Trigger.Request(), Trigger.Response())
    assert not reply.success and not robot._prepared
    assert robot.goal_callback(goal()) == GoalResponse.REJECT


def test_prepare_cannot_move_when_motor_recovery_fails(robot, monkeypatch):
    monkeypatch.setattr(robot, 'recover_arm_for_prepare', lambda: '保护尚未复位')
    monkeypatch.setattr(robot, 'move_initial_observation',
                        lambda **kw: pytest.fail('must not move before recovery'))
    reply = robot.prepare_callback(Trigger.Request(), Trigger.Response())
    assert not reply.success and reply.message == '保护尚未复位'


def test_motor_protection_blocks_a_new_pick_despite_fresh_joint_state(robot):
    robot._prepared = True
    robot._on_motor_health(SimpleNamespace(data=json.dumps(
        {'online': True, 'protected': True, 'ready': False})))
    assert robot.goal_callback(goal()) == GoalResponse.REJECT
    assert '保护尚未复位' in robot.servo_readiness_fault()


@pytest.mark.parametrize('protected', [False, True])
def test_place_execution_abort_stops_all_pose_and_offset_retries(robot, monkeypatch, protected):
    robot._on_motor_health(SimpleNamespace(data=json.dumps(
        {'online': True, 'protected': protected, 'ready': not protected})))
    calls = []
    monkeypatch.setattr(robot, 'ik_joints_for_pose', lambda _: list(robot.q))
    monkeypatch.setattr(PickPlaceNode, 'execute_moveit_trajectory', lambda *args: False)
    def execute(*args):
        calls.append(args[1])
        return robot.execute_moveit_trajectory(None, args[1] + '_EXEC')
    monkeypatch.setattr(robot, 'move_joints', execute)
    with pytest.raises(RuntimeError, match='执行中断.*本次任务已停止'):
        robot.move_pose_candidates(.3, .3, .2, [[0.,1.,0.], [0.,1.15,0.]],
                                   'MOVE_PRE_PLACE', .5, .3, xy_offsets=[(0.,0.),(.01,0.)])
    assert calls == ['MOVE_PRE_PLACE_1_IK']
    reply = robot.state_callback(Trigger.Request(), Trigger.Response())
    assert bool(json.loads(reply.message)['arm_fault']) == protected


def test_pose_planning_failure_can_still_try_another_orientation(robot, monkeypatch):
    calls = []
    monkeypatch.setattr(robot, 'move_pose', lambda *a, **kw: calls.append(a[1]) or len(calls) == 2)
    found = robot.move_pose_candidates(.3, .3, .2, [[0.,1.,0.],[0.,1.15,0.]], 'MOVE_PRE_PLACE', .5, .3)
    assert found == [0.,1.15,0.] and len(calls) == 2


def test_demo_scene_omits_blocks_and_box_but_keeps_tables_and_removals(robot, monkeypatch):
    from moveit_msgs.msg import CollisionObject
    calls = []
    monkeypatch.setattr(PickPlaceNode, 'publish_box', lambda self, name, *a, **kw: calls.append((name,kw['operation'])))
    assert not robot.scene_objects_collision_enabled
    for name in ('object','source_support','cube_obstacle_red','box_wall_0','table','table_mirror'):
        robot.publish_box(name,[0.,0.,0.],[.03]*3)
    robot.publish_box('cube_obstacle_red',[0.,0.,0.],[.03]*3,operation=CollisionObject.REMOVE)
    assert calls == [('table',CollisionObject.ADD),('table_mirror',CollisionObject.ADD),('cube_obstacle_red',CollisionObject.REMOVE)]
    monkeypatch.setattr(PickPlaceNode,'set_preflight_target_present',
                        lambda *a: pytest.fail('No absent object may be removed or restored'))
    assert robot.set_preflight_target_present(False)
    assert robot.set_preflight_target_present(True)


def test_disabled_object_models_do_not_abort_approach_before_ik(robot, monkeypatch):
    calls=[]
    monkeypatch.setattr(PickPlaceNode,'set_preflight_target_present',lambda *a: False)
    monkeypatch.setattr(robot,'ik_joints_for_pose',lambda pose: calls.append('ik') or list(robot.q))
    monkeypatch.setattr(robot,'precheck_approach_path',lambda *a: calls.append('descent_plan') or True)
    monkeypatch.setattr(robot,'move_joints',lambda *a: calls.append('transit') or True)
    selected=robot.move_pregrasp_candidates(.245,.268,.044,[[0.,1.1,1.56]],.06)
    assert selected == [0.,1.1,1.56]
    assert calls == ['ik','descent_plan','transit']


def test_enabled_object_models_still_require_successful_scene_update(robot,monkeypatch):
    robot.scene_objects_collision_enabled=True
    calls=[]
    monkeypatch.setattr(PickPlaceNode,'set_preflight_target_present',lambda self,p: calls.append(p) or False)
    monkeypatch.setattr(robot,'ik_joints_for_pose',lambda *a: pytest.fail('failed scene update must stop planning'))
    assert robot.move_pregrasp_candidates(.245,.268,.044,[[0.,1.1,1.56]],.06) is None
    assert calls == [False,True]


def test_mirror_prepare_switches_return_pose_and_keeps_selected_side(robot, monkeypatch):
    chosen=[]
    monkeypatch.setattr(robot, 'move_initial_observation',
                        lambda **kw: chosen.append(list(robot.ready_joints)) or True)
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: True)
    reply=robot.prepare_mirror_callback(Trigger.Request(), Trigger.Response())
    assert reply.success and robot.observation_side == 'mirror'
    assert chosen[-1] == robot.mirror_ready_joints and robot.table_y < 0
    reply=robot.prepare_callback(Trigger.Request(), Trigger.Response())
    assert reply.success and chosen[-1] == robot.mirror_ready_joints
    reply=robot.prepare_current_callback(Trigger.Request(), Trigger.Response())
    assert reply.success and chosen[-1] == robot.current_ready_joints and robot.table_y > 0


def test_side_change_clears_old_detection_without_releasing_held_object(robot):
    robot._holding=True
    robot.box_pose=PoseStamped()
    robot.cube_stable['red']=True
    robot.select_observation_side('mirror')
    assert robot.box_pose is None and not robot.cube_stable['red']
    assert robot._holding


def test_plan_only_prepare_cannot_authorize_real_grasp(robot, monkeypatch):
    robot.plan_only = True
    monkeypatch.setattr(robot, 'move_initial_observation', lambda **kw: True)
    reply = robot.prepare_callback(Trigger.Request(), Trigger.Response())
    assert reply.success and not robot._prepared
    assert robot.goal_callback(goal()) == GoalResponse.REJECT


def test_initial_prepare_required_and_stale_feedback_rejected(robot):
    assert robot.goal_callback(goal()) == GoalResponse.REJECT
    robot._prepared = True
    robot.joint_state_arrival -= 2
    assert robot.goal_callback(goal()) == GoalResponse.REJECT


def test_operation_lock_survives_terminal_status(robot, monkeypatch):
    robot._prepared = True
    assert robot.goal_callback(goal()) == GoalResponse.ACCEPT
    robot._on_stage('SUCCESS')
    assert robot._busy
    assert robot.goal_callback(goal()) == GoalResponse.REJECT
    move = Mock(return_value=True)
    monkeypatch.setattr(robot, 'move_initial_observation', move)
    reply = robot.prepare_callback(Trigger.Request(), Trigger.Response())
    assert not reply.success
    move.assert_not_called()


def test_holding_is_informational_and_prepare_keeps_gripper(robot, monkeypatch):
    robot._holding = True
    robot._prepared = True
    assert robot.goal_callback(goal()) == GoalResponse.ACCEPT
    robot._finish(False)
    move = Mock(return_value=True)
    monkeypatch.setattr(robot, 'move_initial_observation', move)
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: True)
    monkeypatch.setattr(robot, 'call_gripper', lambda *_: pytest.fail('prepare must retain the gripper'))
    assert robot.prepare_callback(Trigger.Request(), Trigger.Response()).success
    assert robot._holding
    state = json.loads(robot.state_callback(Trigger.Request(), Trigger.Response()).message)
    assert state['prepared'] and state['can_accept'] and state['holding']


def handle(request=None):
    return SimpleNamespace(request=request or goal(), is_cancel_requested=False,
                           succeed=Mock(), abort=Mock(), canceled=Mock())


def test_complete_cycle_and_failed_return_keep_initial_preparation(robot, monkeypatch):
    robot._prepared = True
    monkeypatch.setattr(robot, 'execute_pick_place', lambda color: (True, '任务完成'))
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: True)
    assert robot.goal_callback(goal()) == GoalResponse.ACCEPT
    h = handle()
    reply = robot.execute_callback(h)
    assert reply.success and robot._prepared and not robot._busy
    h.succeed.assert_called_once()
    assert robot.goal_callback(goal()) == GoalResponse.ACCEPT
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: False)
    h = handle()
    reply = robot.execute_callback(h)
    assert not reply.success and robot._prepared
    h.abort.assert_called_once()


def test_direct_clients_cannot_skip_return_or_reuse_old_box(robot):
    robot._prepared = True
    request = goal()
    request.skip_return_home = True
    assert robot.goal_callback(request) == GoalResponse.REJECT
    request.skip_return_home = False
    request.reuse_frozen_box = True
    assert robot.goal_callback(request) == GoalResponse.REJECT


def test_cancellation_before_execute_sends_no_motion(robot, monkeypatch):
    robot._prepared = True
    assert robot.goal_callback(goal()) == GoalResponse.ACCEPT
    execute = Mock()
    monkeypatch.setattr(robot, 'execute_pick_place', execute)
    h = handle()
    h.is_cancel_requested = True
    reply = robot.execute_callback(h)
    assert not reply.success and robot._prepared and not robot._busy
    execute.assert_not_called()
    h.canceled.assert_called_once()


def test_vision_failure_can_retry_without_another_prepare(robot, monkeypatch):
    robot._prepared = True
    monkeypatch.setattr(robot, 'execute_pick_place', lambda *_: (False, '定位暂不可用'))
    movement = Mock()
    monkeypatch.setattr(robot, 'move_initial_observation', movement)
    assert robot.goal_callback(goal()) == GoalResponse.ACCEPT
    assert not robot.execute_callback(handle()).success
    state = json.loads(robot.state_callback(Trigger.Request(), Trigger.Response()).message)
    assert state['initialized'] and state['prepared'] and state['can_accept']
    assert robot.goal_callback(goal('white')) == GoalResponse.ACCEPT
    movement.assert_not_called()


def test_next_grasp_recovers_observation_pose_automatically(robot, monkeypatch):
    robot._prepared = True
    robot.q[0] += .2
    calls = []
    def recover(**kwargs):
        calls.append(('recover',kwargs))
        robot.q = list(robot.ready_joints)
        return True
    monkeypatch.setattr(robot, 'move_initial_observation', recover)
    monkeypatch.setattr(robot, '_wait_observation', lambda **kwargs: True)
    monkeypatch.setattr(robot, 'execute_pick_place', lambda *_: calls.append('pick') or (True, '完成'))
    assert robot.goal_callback(goal()) == GoalResponse.ACCEPT
    assert robot.execute_callback(handle()).success
    assert calls == [('recover', {'observe_target':False}), 'pick']


def test_retry_preserves_live_pose_and_box_measurement(robot):
    pose = PoseStamped()
    robot.cube_poses['white'] = pose
    robot.cube_pose_arrivals['white'] = time.monotonic()
    robot.cube_stable['white'] = True
    robot.box_pose = pose
    robot.box_pose_arrival = time.monotonic()
    robot.box_stable = True
    arrival = robot.box_pose_arrival
    robot.reset_task_state('white')
    assert robot.vision_pose is pose and robot.box_pose is pose
    assert robot.box_pose_arrival == arrival
    assert robot.wait_for_frozen_vision() is not None


def test_box_pose_accepts_short_gap_but_expires_without_new_measurement(robot):
    pose = PoseStamped()
    robot.box_pose, robot.box_stable = pose, True
    robot.box_pose_arrival = time.monotonic()-2.5
    assert robot.vision_snapshot()['box']['ready']
    robot.box_pose_arrival -= 1
    assert robot.vision_snapshot()['box']['status'] == 'stale'


def test_bridge_does_not_release_busy_on_response_timeout():
    from rclpy.task import Future
    from x5a_web_agent.wrist_bridge import WristBridge
    bridge = WristBridge()
    future = Future()
    def timed_out():
        bridge._pending = future
        raise TimeoutError('still running')
    assert not bridge._operate(timed_out)['success']
    assert bridge.busy
    # Even without a recent ROS busy topic, a pending operation cannot be
    # replaced by a prepare/grasp request until the real response arrives.
    assert not bridge.prepare()['success']
    future.set_result(True)
    assert not bridge.busy


def test_prepare_requires_post_stop_exposure(robot):
    robot._prepare_timeout = .04
    robot.reobserve_settle = .01
    robot._image_stamp = robot.get_clock().now().nanoseconds - 100_000_000
    robot._image_arrival = time.monotonic()
    assert not robot._wait_observation()
    robot._prepare_timeout = .15
    def exposure():
        time.sleep(.025)
        robot._image_stamp = robot.get_clock().now().nanoseconds
        robot._image_arrival = time.monotonic()
    thread = threading.Thread(target=exposure)
    thread.start()
    assert robot._wait_observation()
    thread.join()


class FakeBridge:
    dry_run = False
    def __init__(self):
        self.prepared = False
        self.held_color = None
        self.calls = []
        self.block = None
    def state(self):
        return {'prepared': self.prepared, 'server_ready': True, 'busy': False,
                'holding':self.held_color is not None,'holding_color':self.held_color,
                'placement_ready':self.held_color is not None}
    def prepare(self):
        self.calls.append('prepare')
        if self.block:
            self.block.wait(1)
        self.prepared = True
        return {'success': True, 'message': '已到观察位'}
    def send_pick_place(self, color, destination_color='box'):
        self.calls.append(color if destination_color == 'box' else (color, destination_color))
        return {'success': True, 'message': '抓放完成'}
    def send_pick(self, color):
        self.calls.append(color)
        self.held_color=color
        return {'success':True,'message':'已抓起并等待放置'}
    def place(self, destination_color):
        self.calls.append(('place',destination_color))
        self.held_color=None
        return {'success':True,'message':'已放置'}
    def go_home(self):
        self.calls.append('home')
        self.prepared = False
        return {'success': True, 'message': '已收起'}
    def cancel(self):
        return {'success': True, 'message': '已取消'}


def web_fixture(monkeypatch):
    from x5a_web_agent import wrist_app as web
    fake = FakeBridge()
    monkeypatch.setattr(web, 'bridge', fake)
    monkeypatch.setattr(web, '_job', None)
    monkeypatch.setattr(web, '_submission_lock', asyncio.Lock())
    monkeypatch.setattr(web, 'clients', set())
    monkeypatch.setattr(web, 'observers', [])
    monkeypatch.setattr(web, 'state', {'busy': False, 'stage': 'IDLE'})
    return web, fake


def test_web_prepare_then_color_and_no_remote_bypass(monkeypatch):
    web, fake = web_fixture(monkeypatch)
    async def run():
        assert not (await web.handle_text('抓红色'))['ok']
        assert fake.calls == []
        assert (await web.handle_text('准备'))['ok']
        await web._job
        assert (await web.handle_text('抓红色'))['ok']
        await web._job
        assert fake.calls == ['prepare', 'red']
        assert not (await web.dispatch({'action': 'remote_grasp', 'prompt': 'cup'}))['ok']
    asyncio.run(run())


def test_concurrent_web_requests_do_not_queue_grasp_behind_prepare(monkeypatch):
    web, fake = web_fixture(monkeypatch)
    fake.block = threading.Event()
    async def run():
        results = await asyncio.gather(web.handle_text('准备'), web.handle_text('抓红色'))
        assert results[0]['ok'] and not results[1]['ok']
        fake.block.set()
        await web._job
        assert fake.calls == ['prepare']
    asyncio.run(run())


def test_vision_timeout_names_missing_box_and_preserves_grasp_gate(robot):
    pose = PoseStamped()
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = .28, .31, .024
    robot.cube_poses['red'] = pose
    robot.cube_pose_arrivals['red'] = time.monotonic()
    robot.cube_stable['red'] = True
    robot.vision_wait_timeout = .01
    assert robot.vision_snapshot()['red']['ready']
    assert robot.wait_for_frozen_vision() is None
    assert '红块定位稳定' in robot._vision_wait_failure
    assert '盒子未收到有效坐标' in robot._vision_wait_failure
    assert robot._session_box_xyz is None
    robot.box_pose = pose
    robot.box_pose_arrival = time.monotonic()
    robot.box_stable = True
    robot.vision_wait_timeout = .1
    assert robot.wait_for_frozen_vision() is not None
    assert robot._vision_wait_failure == ''


def test_coordinates_alone_do_not_mean_ready(robot):
    pose = PoseStamped()
    robot.cube_poses['red'] = pose
    robot.cube_pose_arrivals['red'] = time.monotonic()
    robot.cube_stable['red'] = False
    assert robot.vision_snapshot()['red']['status'] == 'unstable'
    robot.cube_stable['red'] = True
    robot.cube_pose_arrivals['red'] -= robot.vision_max_age + 1
    assert robot.vision_snapshot()['red']['status'] == 'stale'
    pose.pose.position.x = float('nan')
    snapshot = robot.vision_snapshot()['red']
    assert snapshot['status'] == 'invalid' and snapshot['xyz'] is None


def test_phone_separates_last_failure_from_current_vision(monkeypatch):
    web, _ = web_fixture(monkeypatch)
    vision = {'red': {'status_zh': '定位稳定'}, 'box': {'status_zh': '定位稳定'}}
    async def run():
        await web.update(stage='FAILED', busy=False, prepared=False, vision=vision,
                         last_result={'success': False, 'message': '盒子未收到有效坐标'})
        assert web.state['last_result']['message'] == '上次任务失败：盒子未收到有效坐标'
        assert '当前定位：红块定位稳定；盒子定位稳定' in web.state['stage_zh']
        await web.update(vision=vision)
        assert web.state['last_result']['message'].count('上次任务失败：') == 1
        await web.update(stage='ACCEPTED', busy=True, last_result=None)
        assert web.state['last_result'] is None
        assert '当前定位' not in web.state['stage_zh']
        await web.update(stage='IDLE', busy=False, last_result=None)
        assert web.state['last_result']['kind'] == 'status'
        assert '请先说“准备”' in web.state['last_result']['message']
    asyncio.run(run())


def test_phone_protection_message_does_not_offer_automatic_return(monkeypatch):
    web, _ = web_fixture(monkeypatch)
    asyncio.run(web.update(stage='FAILED', busy=False, initialized=True, at_observation=False,
                          arm_fault='电机保护尚未复位，请先取消任务再说准备',
                          vision={'red': {'status_zh':'定位稳定'},'box':{'status_zh':'定位稳定'}}))
    assert web.state['stage_zh'].startswith('机械臂状态异常：电机保护尚未复位')
    assert '下次抓取会先自动返回观察位' not in web.state['stage_zh']


@pytest.mark.parametrize('text,source,destination', [
    ('把红色方块放到白色方块上', 'red', 'white'),
    ('请把橘色积木叠到红色积木上面', 'orange', 'red'),
    ('拿起白色方块放在橙色方块顶部', 'white', 'orange'),
])
def test_stack_grammar_keeps_source_and_destination(text, source, destination):
    assert parse_wrist_command(text) == {'action': 'pick_place', 'target_color': source,
                                         'destination_color': destination}


@pytest.mark.parametrize('text', [
    '把红色方块放到红色方块上', '红色和白色都抓取',
    '把红色方块放到蓝色方块上', '把红色方块放到白色方块里面',
    '回原点然后抓红色', '准备然后抓红色', '别回原点',
])
def test_unsupported_or_ambiguous_voice_never_starts_a_task(text):
    assert parse_wrist_command(text)['action'] in ('unknown', 'cancel')


def test_home_plans_to_folded_pose_and_invalidates_prepare(robot, monkeypatch):
    robot._prepared = True
    planned = Mock(return_value=True)
    gripper = Mock()
    monkeypatch.setattr(robot, 'count_publishers', lambda _: 1)
    monkeypatch.setattr(robot, 'move_joints', planned)
    monkeypatch.setattr(robot, 'call_gripper', gripper)
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: kw['target'] == [0.]*6 and not kw['require_image'])
    reply = robot.home_callback(Trigger.Request(), Trigger.Response())
    assert reply.success and not robot._prepared and robot._stage == 'HOME'
    assert planned.call_args.args[0] == [0.]*6
    gripper.assert_not_called()
    assert robot.goal_callback(goal()) == GoalResponse.REJECT


def test_home_allows_holding_but_refuses_busy_without_opening_gripper(robot, monkeypatch):
    planned = Mock(return_value=True)
    monkeypatch.setattr(robot, 'move_joints', planned)
    monkeypatch.setattr(robot, 'count_publishers', lambda _: 1)
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: True)
    monkeypatch.setattr(robot, 'call_gripper', lambda *_: pytest.fail('home must retain the gripper'))
    robot._holding = True
    assert robot.home_callback(Trigger.Request(), Trigger.Response()).success
    assert robot._holding
    planned.assert_called_once()
    robot._busy = True
    assert not robot.home_callback(Trigger.Request(), Trigger.Response()).success
    assert planned.call_count == 1


def test_stacking_needs_destination_cube_but_not_box(robot):
    pose = PoseStamped()
    robot.cube_poses['red'] = pose
    robot.cube_pose_arrivals['red'] = time.monotonic()
    robot.cube_stable['red'] = True
    robot.vision_wait_timeout = .01
    assert robot.wait_for_frozen_vision(destination_color='white') is None
    assert '目标白块未收到有效坐标' in robot._vision_wait_failure
    robot.cube_poses['white'] = pose
    robot.cube_pose_arrivals['white'] = time.monotonic()
    robot.cube_stable['white'] = True
    robot.vision_wait_timeout = .1
    assert robot.box_pose is None
    assert robot.wait_for_frozen_vision(destination_color='white') is not None


def test_action_rejects_invalid_stack_destination(robot):
    robot._prepared = True
    for destination in ('red', 'blue'):
        request = goal()
        request.destination_color = destination
        assert robot.goal_callback(request) == GoalResponse.REJECT
    request.destination_color = 'white'
    assert robot.goal_callback(request) == GoalResponse.ACCEPT


def test_web_routes_home_and_stacking_through_same_lock(monkeypatch):
    web, fake = web_fixture(monkeypatch)
    async def run():
        assert (await web.handle_text('准备'))['ok']
        await web._job
        assert (await web.handle_text('把红色方块放到白色方块上'))['ok']
        await web._job
        assert ('red', 'white') in fake.calls
        assert (await web.handle_text('返回原位'))['ok']
        await web._job
        assert fake.calls[-1] == 'home' and web.state['stage'] == 'HOME'
        assert not (await web.handle_text('抓红色'))['ok']
    asyncio.run(run())


def test_web_allows_initialized_robot_to_continue_without_reprepare(monkeypatch):
    web,fake=web_fixture(monkeypatch)
    fake.state=lambda:{'server_ready':True,'busy':False,'holding':False,'prepared':False,
                       'initialized':True,'at_observation':False,'can_accept':True}
    async def run():
        assert (await web.handle_text('抓红色'))['ok']
        await web._job
        assert (await web.handle_text('抓白色'))['ok']
        await web._job
        assert fake.calls==['red','white']
    asyncio.run(run())


def test_complete_stack_uses_measured_offset_keeps_base_cube_and_skips_reobservation(robot, monkeypatch):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from x5a_pick_place.pick_place_node import pose_to_xyz
    robot.plan_only = True
    assert not robot.reobserve_enabled
    source, destination = (.18,.33,.024), (.30,.30,.024)
    stages, boxes, motions, placements = [], [], [], []
    def freeze(**kwargs):
        assert kwargs['destination_color'] == 'white'
        robot.vision_pose = PoseStamped()
        robot.vision_pose.pose.orientation.w = 1.
        robot._destination_yaw = .1
        return source, destination
    def attach(grasp):
        q = grasp.orientation
        rotation = Rotation.from_quat([q.x,q.y,q.z,q.w])
        center = np.array([source[0],source[1],source[2]-.015])
        robot._held_object_offset = tuple(rotation.inv().apply(center-np.array(pose_to_xyz(grasp))))
        return True
    def move_place(center,rpys):
        from x5a_pick_place.stacking import tcp_for_object_center
        assert len(rpys)==4
        np.testing.assert_allclose(center,[.30,.30,.041])
        return tcp_for_object_center(center,robot._held_object_offset,rpys[0]),rpys[0]
    def vertical(pose,stage,*args):
        motions.append((stage,pose))
        return True, 1.
    monkeypatch.setattr(robot,'wait_for_frozen_vision',freeze)
    monkeypatch.setattr(robot,'clear_planning_scene_objects',lambda:None)
    monkeypatch.setattr(robot,'log_calibration',lambda:None)
    monkeypatch.setattr(robot,'setup_scene',lambda:None)
    monkeypatch.setattr(robot,'publish_box',lambda *args,**kw:boxes.append((args,kw)))
    monkeypatch.setattr(robot,'report_stage',lambda stage,**kw:stages.append(stage))
    monkeypatch.setattr(robot,'call_gripper',lambda *_:True)
    monkeypatch.setattr(robot,'move_pregrasp_candidates',lambda *args:[0.,1.1,0.])
    monkeypatch.setattr(robot,'correct_pregrasp',lambda *args:pytest.fail('second localization is disabled'))
    monkeypatch.setattr(robot,'attach_object',attach)
    monkeypatch.setattr(robot,'move_stack_pre_place',move_place)
    monkeypatch.setattr(robot,'vertical_move',vertical)
    monkeypatch.setattr(robot,'detach_object',lambda xyz:placements.append(xyz))
    monkeypatch.setattr(robot,'move_joints',lambda *args:True)
    ok,message=robot.execute_pick_place('red',destination_color='white')
    assert ok
    assert stages == ['WAITING_VISION','TARGET_FOUND','MOVE_PRE_GRASP','APPROACH','GRASPING',
                      'LIFTING','MOVE_PRE_PLACE','DESCENDING','RELEASING','RETREATING','RETURN_HOME','SUCCESS']
    assert any(args[0]=='cube_obstacle_white' for args,kwargs in boxes)
    assert not any(args[0].startswith('box_wall') for args,kwargs in boxes)
    release = next(pose for stage,pose in motions if stage=='DESCEND')
    q=release.orientation
    center=np.array(pose_to_xyz(release))+Rotation.from_quat([q.x,q.y,q.z,q.w]).apply(robot._held_object_offset)
    np.testing.assert_allclose(center,[.30,.30,.041],atol=1e-9)
    np.testing.assert_allclose(placements[0],center)

@pytest.mark.parametrize('text,colors', [
    ('把橙色和白色方块都放进盒子里。', ['orange', 'white']),
    ('请把白色方块以及橙色方块全部放进黑盒子里', ['white', 'orange']),
    ('把红色、白色和橙色方块一起放入盒子', ['red', 'white', 'orange']),
])
def test_batch_box_grammar(text, colors):
    parsed = parse_wrist_command(text)
    assert parsed == {'action': 'sequence', 'tasks': [
        {'action': 'pick_place', 'target_color': c} for c in colors]}

@pytest.mark.parametrize('text', [
    '把橙色和橙色方块都放进盒子里', '把红色和蓝色方块都放进盒子里',
    '把红色和白色放到橙色方块上', '准备然后把红色和白色方块放进盒子里',
])
def test_ambiguous_batch_is_rejected(text):
    assert parse_wrist_command(text)['action'] == 'unknown'


def batch_goal(colors=('orange', 'white')):
    request = goal(colors[0])
    request.target_colors = list(colors)
    return request


def test_batch_keeps_robot_lock_and_order(robot, monkeypatch):
    robot._prepared = True
    calls = []
    def execute(color):
        assert robot._busy
        assert robot.goal_callback(goal('red')) == GoalResponse.REJECT
        calls.append(color)
        return True, '完成'
    monkeypatch.setattr(robot, 'execute_pick_place', execute)
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: True)
    request = batch_goal()
    assert robot.goal_callback(request) == GoalResponse.ACCEPT
    h = handle(request)
    h.publish_feedback = Mock()
    reply = robot.execute_callback(h)
    assert reply.success and calls == ['orange', 'white']
    assert robot._prepared and not robot._busy
    assert '橙块、白块' in reply.message
    h.succeed.assert_called_once()
    assert h.publish_feedback.call_count == 2


def test_batch_failure_stops_remaining_items(robot, monkeypatch):
    robot._prepared = True
    calls = []
    def execute(color):
        calls.append(color)
        return (True, '完成') if color == 'orange' else (False, '未识别到完整顶面')
    monkeypatch.setattr(robot, 'execute_pick_place', execute)
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: True)
    request = batch_goal(('orange', 'white', 'red'))
    assert robot.goal_callback(request) == GoalResponse.ACCEPT
    h = handle(request)
    h.publish_feedback = Mock()
    reply = robot.execute_callback(h)
    assert not reply.success and calls == ['orange', 'white']
    assert '2/3' in reply.message and '已放置：橙块' in reply.message
    assert robot._prepared and not robot._busy
    h.abort.assert_called_once()


def test_batch_cancellation_between_items_stops_next_grasp(robot, monkeypatch):
    robot._prepared = True
    calls = []
    def execute(color):
        calls.append(color)
        robot.request_cancel()
        return True, '完成'
    monkeypatch.setattr(robot, 'execute_pick_place', execute)
    monkeypatch.setattr(robot, '_wait_observation', lambda **kw: True)
    request = batch_goal()
    assert robot.goal_callback(request) == GoalResponse.ACCEPT
    h = handle(request)
    h.publish_feedback = Mock()
    reply = robot.execute_callback(h)
    assert not reply.success and calls == ['orange']
    assert '已放置：橙块' in reply.message
    h.canceled.assert_called_once()


def test_malformed_batch_cannot_authorize_grasp(robot):
    robot._prepared = True
    for colors in (['orange'], ['orange', 'orange'], ['orange', 'blue'], ['orange', 'white', 'red', 'orange']):
        request = batch_goal(colors)
        assert robot.goal_callback(request) == GoalResponse.REJECT
    request = batch_goal()
    request.destination_color = 'red'
    assert robot.goal_callback(request) == GoalResponse.REJECT


def test_web_batch_uses_one_bridge_call_and_initial_preparation(monkeypatch):
    web, fake = web_fixture(monkeypatch)
    fake.send_sequence = Mock(return_value={'success': True, 'message': '队列完成'})
    async def run():
        assert not (await web.handle_text('把橙色和白色方块都放进盒子里'))['ok']
        fake.send_sequence.assert_not_called()
        fake.prepared = True
        assert (await web.handle_text('把橙色和白色方块都放进盒子里'))['ok']
        await web._job
        fake.send_sequence.assert_called_once_with(['orange', 'white'])
        assert fake.calls == []
    asyncio.run(run())


def test_current_detection_explains_occlusion_without_reusing_stale_pose(robot):
    from std_msgs.msg import String
    pose = PoseStamped()
    pose.pose.position.x = .25
    robot.cube_poses['orange'] = pose
    robot.cube_pose_arrivals['orange'] = time.monotonic()-100
    robot._on_detection_details(String(data='{"orange":"no_top_face"}'))
    snapshot = robot.vision_snapshot()['orange']
    assert not snapshot['ready'] and snapshot['status'] == 'no_top_face'
    assert snapshot['xyz'] is None and snapshot['last_xyz'][0] == .25
    robot._details_arrival -= 6
    assert robot.vision_snapshot()['orange']['status'] == 'stale'


def test_reused_physical_block_accepts_new_pose_after_old_one_expired(robot):
    from std_msgs.msg import Bool, String
    pose = PoseStamped()
    pose.header.frame_id = robot.base_frame
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = .25, .3, .054
    robot.cube_poses['red'] = pose
    robot.cube_pose_arrivals['red'] = time.monotonic()-100
    assert not robot.vision_snapshot()['red']['ready']
    robot._cube_pose_cb('red', pose)
    robot._cube_stable_cb('red', Bool(data=True))
    robot._on_detection_details(String(data='{"red":"stable"}'))
    assert robot.vision_snapshot()['red']['ready']


def test_upper_cube_keeps_lower_support_as_collision_obstacle(robot, monkeypatch):
    calls = []
    robot.scene_obj_x, robot.scene_obj_y, robot.scene_obj_z = .25, .3, .039
    robot.vision_pose = PoseStamped()
    monkeypatch.setattr(robot, 'publish_box', lambda *a, **kw: calls.append((a, kw)))
    robot.setup_scene()
    support = next(a for a, kw in calls if a[0] == 'source_support')
    assert support[1] == pytest.approx([.25, .3, .009])
    assert support[2] == pytest.approx([.03, .03, .03])


def test_box_depth_failure_is_distinguished_from_old_coordinate_expiry(robot):
    from std_msgs.msg import String
    pose = PoseStamped()
    robot.box_pose = pose
    robot.box_pose_arrival = time.monotonic()-10
    robot._on_detection_details(String(data='{"box":{"status":"no_valid_depth"}}'))
    snapshot = robot.vision_snapshot()['box']
    assert not snapshot['ready'] and snapshot['status'] == 'no_valid_depth'
    assert snapshot['xyz'] is None and snapshot['last_xyz'] is not None
    robot.box_pose_arrival = time.monotonic()
    robot.box_stable = True
    robot._on_detection_details(String(data='{"box":{"status":"stable","rim_z_m":0.059,"tilt_deg":3.9,"plane_residual_m":0.0015}}'))
    snapshot = robot.vision_snapshot()['box']
    assert snapshot['ready'] and snapshot['rim_z_m'] == .059


def test_web_accepts_holding_robot_for_new_user_task(monkeypatch):
    web,fake = web_fixture(monkeypatch)
    fake.state=lambda:{'server_ready':True,'busy':False,'holding':True,'prepared':True,
                       'initialized':True,'at_observation':True,'can_accept':True}
    async def run():
        assert (await web.handle_text('抓红色'))['ok']
        await web._job
        assert fake.calls == ['red']
    asyncio.run(run())


def test_shorter_lift_requires_complete_plan_and_does_not_open_gripper(robot,monkeypatch):
    heights=[]
    def vertical(pose,*args):
        heights.append(pose.position.z-.08)
        return (False,.8182) if len(heights)==1 else (True,1.)
    monkeypatch.setattr(robot,'vertical_move',vertical)
    monkeypatch.setattr(robot,'call_gripper',lambda *_:pytest.fail('lift cannot change the gripper'))
    assert robot.lift_with_reachable_height((.26,.34,.08),[0.,1.1,0.]) == (True,1.)
    assert heights == pytest.approx([.10,.08])


def test_lift_execution_failure_does_not_retry_from_stale_start(robot,monkeypatch):
    move=Mock(return_value=(False,1.))
    monkeypatch.setattr(robot,'vertical_move',move)
    assert robot.lift_with_reachable_height((.26,.34,.08),[0.,1.1,0.]) == (False,1.)
    move.assert_called_once()


def test_shorter_stack_retreat_requires_full_path_before_moving(robot,monkeypatch):
    heights=[]
    def vertical(pose,*args):
        heights.append(pose.position.z-.08)
        return (False,2/3) if len(heights)==1 else (True,1.)
    monkeypatch.setattr(robot,'vertical_move',vertical)
    assert robot.retreat_with_reachable_height((.28,.32,.08),[0.,1.1,0.]) == (True,1.)
    assert heights == pytest.approx([.05,.04])


def test_stack_retreat_execution_failure_never_retries(robot,monkeypatch):
    move=Mock(return_value=(False,1.))
    monkeypatch.setattr(robot,'vertical_move',move)
    assert robot.retreat_with_reachable_height((.28,.32,.08),[0.,1.1,0.]) == (False,1.)
    move.assert_called_once()


def test_stack_retreat_stops_at_two_cm_and_respects_cancellation(robot,monkeypatch):
    move=Mock(return_value=(False,.5))
    monkeypatch.setattr(robot,'vertical_move',move)
    assert not robot.retreat_with_reachable_height((.28,.32,.08),[0.,1.1,0.])[0]
    assert [c.args[0].position.z-.08 for c in move.call_args_list] == pytest.approx([.05,.04,.03,.02])
    robot.request_cancel();move.reset_mock()
    assert not robot.retreat_with_reachable_height((.28,.32,.08),[0.,1.1,0.])[0]
    move.assert_not_called()


def test_lift_stops_at_minimum_height_and_respects_cancellation(robot,monkeypatch):
    move=Mock(return_value=(False,.5))
    monkeypatch.setattr(robot,'vertical_move',move)
    assert not robot.lift_with_reachable_height((.26,.34,.08),[0.,1.1,0.])[0]
    assert [c.args[0].position.z-.08 for c in move.call_args_list] == pytest.approx([.1,.08,.06])
    robot.request_cancel()
    move.reset_mock()
    assert not robot.lift_with_reachable_height((.26,.34,.08),[0.,1.1,0.])[0]
    move.assert_not_called()


def test_failed_home_does_not_clear_existing_preparation(robot,monkeypatch):
    robot._prepared=True
    monkeypatch.setattr(robot,'count_publishers',lambda _:1)
    monkeypatch.setattr(robot,'move_joints',lambda *args:False)
    assert not robot.home_callback(Trigger.Request(),Trigger.Response()).success
    assert robot._prepared


def test_restart_handoff_restores_collision_scene_without_motion(robot,monkeypatch,tmp_path):
    from rclpy.task import Future
    from moveit_msgs.srv import ApplyPlanningScene
    path=tmp_path/'resume.json'
    path.write_text(json.dumps({'expires_at':time.time()+60,'base_frame':robot.base_frame,
        'holding':True,'initialized':True,'scene':{'is_diff':True,'robot_state':{'is_diff':True,
        'attached_collision_objects':[{'link_name':robot.tcp_frame,'object':{'id':'object','operation':'\x00'}}]}}}))
    future=Future()
    apply=Mock();apply.service_is_ready.return_value=True;apply.call_async.return_value=future
    monkeypatch.setattr(robot,'apply_scene_cli',apply)
    monkeypatch.setattr(robot,'move_joints',lambda *a:pytest.fail('handoff cannot move'))
    robot._load_restart_state(str(path))
    assert robot._restore_pending and robot._holding and not path.exists()
    assert robot.goal_callback(goal())==GoalResponse.REJECT
    robot._restore_scene_tick()
    saved_scene=apply.call_async.call_args.args[0].scene
    assert saved_scene.robot_state.attached_collision_objects[0].object.id=='object'
    future.set_result(ApplyPlanningScene.Response(success=True))
    robot._restore_scene_tick()
    assert not robot._restore_pending and robot._holding and robot._prepared
    assert robot.goal_callback(goal())==GoalResponse.ACCEPT


def split_flow_hardware(robot,monkeypatch):
    """Simulate actuators while retaining the actual pick/place control flow."""
    from x5a_pick_place.pick_place_node import pose_to_xyz
    white=PoseStamped();white.header.frame_id=robot.base_frame
    white.pose.position.x=.24;white.pose.position.y=.27;white.pose.position.z=.024
    white.pose.orientation.w=1.
    robot.cube_poses['white']=white;robot.cube_pose_arrivals['white']=time.monotonic();robot.cube_stable['white']=True
    robot.box_pose=None;robot._prepared=True
    calls=[];stages=[];tcp=[.24,.27,.104]
    monkeypatch.setattr(robot,'clear_planning_scene_objects',lambda:None)
    monkeypatch.setattr(robot,'log_calibration',lambda:None)
    monkeypatch.setattr(robot,'setup_scene',lambda:None)
    monkeypatch.setattr(robot,'setup_box_walls',lambda *a:calls.append(('box_walls',a)))
    monkeypatch.setattr(PickPlaceNode,'call_gripper',lambda self,opening:calls.append(('gripper',opening)) or True)
    monkeypatch.setattr(robot,'move_pregrasp_candidates',lambda x,y,z,oris,h:oris[0])
    def vertical(pose,stage,*args):
        robot.joint_state_arrival=time.monotonic()
        tcp[:]=pose_to_xyz(pose);calls.append(('vertical',stage));return True,1.
    def attach(pose):
        robot._held_object_offset=(0.,0.,-.04);calls.append(('attach',pose));return True
    monkeypatch.setattr(robot,'vertical_move',vertical)
    monkeypatch.setattr(robot,'current_tcp',lambda:tuple(tcp))
    monkeypatch.setattr(robot,'attach_object',attach)
    monkeypatch.setattr(robot,'move_joints',lambda *a:calls.append(('joint',a[1])) or True)
    monkeypatch.setattr(robot,'move_pose_candidates',lambda *a,**kw:calls.append(('place_pose',tuple(a[:3]))) or list(a[3][0]))
    def stack_pre_place(center,orientations):
        from x5a_pick_place.stacking import tcp_for_object_center
        xyz=tcp_for_object_center(center,robot._held_object_offset,orientations[0])
        calls.append(('place_pose',tuple(xyz)))
        return xyz,orientations[0]
    monkeypatch.setattr(robot,'move_stack_pre_place',stack_pre_place)
    monkeypatch.setattr(robot,'detach_object',lambda *a:calls.append(('detach',a)))
    monkeypatch.setattr(robot,'_wait_observation',lambda **kw:True)
    robot.set_stage_callback(stages.append)
    return calls,stages


def known_held_white(robot):
    robot._holding=True;robot._held_color='white';robot._held_grasp_rpy=[0.,1.1,0.]
    robot._held_cube_yaw=0.;robot._held_source_top=(.24,.27,.024);robot._held_object_offset=(0.,0.,-.04)
    robot._prepared=True


def test_pick_only_does_not_need_box_and_keeps_gripper_closed_at_observation(robot,monkeypatch):
    calls,stages=split_flow_hardware(robot,monkeypatch)
    g=goal('white');g.destination_color='hold'
    assert robot.goal_callback(g)==GoalResponse.ACCEPT
    h=handle(g)
    result=robot.execute_callback(h)
    assert result.success and robot.placement_ready() and robot._held_color=='white'
    assert [c for c in calls if c[0]=='gripper']==[('gripper',True),('gripper',False)]
    assert ('joint','RETURN_WITH_HELD_OBJECT') in calls
    assert not any(c[0] in ('place_pose','detach','box_walls') for c in calls)
    assert 'WAITING_PLACE' in stages and robot._stage=='WAITING_PLACE'
    assert all(x not in stages for x in ('MOVE_PRE_PLACE','DESCENDING','RELEASING'))
    state=json.loads(robot.state_callback(Trigger.Request(),Trigger.Response()).message)
    assert state['prepared'] and state['holding_color']=='white' and state['placement_ready']
    assert state['held_grasp']['offset']==[0.,0.,-.04]


def test_internal_clearing_pick_stays_lifted_without_returning_or_opening(robot,monkeypatch):
    calls,stages=split_flow_hardware(robot,monkeypatch)
    ok,message=robot.execute_pick_place('white',pick_only=True,hold_at_lift=True)
    assert ok and robot.placement_ready() and robot._holding
    assert ('joint','RETURN_WITH_HELD_OBJECT') not in calls
    assert [c for c in calls if c[0]=='gripper']==[('gripper',True),('gripper',False)]
    assert 'HOLDING_ABOVE_SOURCE' in stages and 'WAITING_PLACE' not in stages
    assert not any(c[0] in ('place_pose','detach','box_walls') for c in calls)


def test_stack_memory_ignores_stable_measurements_while_camera_is_moving(robot,monkeypatch):
    from std_msgs.msg import Bool
    pose=PoseStamped();pose.header.stamp=robot.get_clock().now().to_msg()
    pose.pose.position.x=.26;pose.pose.position.y=.36;pose.pose.position.z=.021
    robot.scene_memory.record_stack('white','orange',(.30,.30,.041),.03,.002,robot.base_frame,'current',1)
    robot.cube_poses['white']=pose;robot.cube_pose_arrivals['white']=time.monotonic()
    monkeypatch.setattr(robot,'at_observation',lambda:True)
    robot._busy=True
    robot._stage='MOVE_INITIAL_OBSERVATION'
    for _ in range(5):robot._cube_stable_cb('white',Bool(data=True))
    assert robot.scene_memory.has_lower('orange')
    robot._busy=False
    monkeypatch.setattr(robot,'at_observation',lambda:False)
    for _ in range(5):robot._cube_stable_cb('white',Bool(data=True))
    assert robot.scene_memory.has_lower('orange')


@pytest.mark.parametrize('stage,constrained',[('INITIAL_OBSERVATION',True),('RETURN_HOME',False)])
def test_only_initial_observation_adds_bounded_joint_path_constraints(robot,monkeypatch,stage,constrained):
    requests=[]
    result=SimpleNamespace(error_code=SimpleNamespace(val=1))
    done=SimpleNamespace(result=lambda:SimpleNamespace(result=result))
    handle=SimpleNamespace(accepted=True,get_result_async=lambda:done)
    def send(goal):
        requests.append(goal)
        return SimpleNamespace(result=lambda:handle)
    monkeypatch.setattr(robot.move_ac,'send_goal_async',send)
    monkeypatch.setattr(robot,'wait_future',lambda *a:True)
    monkeypatch.setattr(robot,'_update_last_joints_from_result',lambda *a:None)
    robot.plan_only=True;robot.q=[0.]*6;robot.last_joints=None
    assert robot.move_joints(robot.ready_joints,stage,.82,.43)
    assert requests[0].planning_options.plan_only
    assert requests[0].request.start_state.is_diff
    constraints=requests[0].request.path_constraints.joint_constraints
    assert bool(constraints)==constrained
    if constrained:
        assert len(constraints)==6
        for c,target in zip(constraints,robot.ready_joints):
            low,high=c.position-c.tolerance_below,c.position+c.tolerance_above
            assert low <= 0 <= high and low <= target <= high
            assert low >= min(0,target)-.150001 and high <= max(0,target)+.150001


def test_direct_lift_flag_is_rejected_for_a_complete_pick_place(robot,monkeypatch):
    calls,stages=split_flow_hardware(robot,monkeypatch)
    ok,message=robot.execute_pick_place('white',hold_at_lift=True)
    assert not ok and not calls


def test_stack_memory_is_recorded_at_release_even_if_retreat_later_fails(robot,monkeypatch):
    calls,stages=split_flow_hardware(robot,monkeypatch)
    known_held_white(robot)
    robot._placement_destination='red'
    original=robot.vertical_move
    def vertical(pose,stage,*args):
        if stage=='RETREAT':return False,0.
        return original(pose,stage,*args)
    monkeypatch.setattr(robot,'vertical_move',vertical)
    ok,message=robot.execute_place_motion((.30,.30,.081),[[0.,1.1,0.]],(.30,.30,.041))
    assert not ok
    memory=robot.scene_memory.snapshot()
    assert len(memory)==1 and memory[0]['upper']=='white' and memory[0]['lower']=='red'
    assert memory[0]['lower_top']==pytest.approx([.30,.30,.024])


def test_failed_release_does_not_record_a_stack(robot,monkeypatch):
    calls,stages=split_flow_hardware(robot,monkeypatch)
    known_held_white(robot);robot._placement_destination='red'
    monkeypatch.setattr(PickPlaceNode,'call_gripper',lambda self,opening:False)
    assert not robot.execute_place_motion((.30,.30,.081),[[0.,1.1,0.]],(.30,.30,.041))[0]
    assert not robot.scene_memory.snapshot()


def test_actual_pick_removes_old_stack_relation(robot,monkeypatch):
    calls,stages=split_flow_hardware(robot,monkeypatch)
    robot.scene_memory.record_stack('white','red',(.24,.27,.041),.03,.002,robot.base_frame,'current',1)
    assert robot.execute_pick_place('white',pick_only=True,hold_at_lift=True)[0]
    assert not robot.scene_memory.snapshot()


def test_complete_stack_command_records_correct_source_and_destination(robot,monkeypatch):
    calls,stages=split_flow_hardware(robot,monkeypatch)
    red=PoseStamped();red.header.frame_id=robot.base_frame
    red.pose.position.x=.30;red.pose.position.y=.30;red.pose.position.z=.024
    red.pose.orientation.w=1.
    robot.cube_poses['red']=red;robot.cube_pose_arrivals['red']=time.monotonic();robot.cube_stable['red']=True
    assert robot.execute_pick_place('white',destination_color='red')[0]
    record=robot.scene_memory.snapshot()[0]
    assert record['upper']=='white' and record['lower']=='red'
    assert record['lower_top']==pytest.approx([.30,.30,.024])
    assert robot._placement_destination is None


def test_followup_placement_uses_fresh_destination_without_regrasp_or_source_vision(robot,monkeypatch):
    calls,stages=split_flow_hardware(robot,monkeypatch)
    assert robot.execute_pick_place('white',pick_only=True)[0]
    old_source=robot._held_source_top
    robot.cube_poses['white']=None;robot.cube_stable['white']=False
    red=PoseStamped();red.pose.position.x=old_source[0];red.pose.position.y=old_source[1];red.pose.position.z=.024;red.pose.orientation.w=1.
    robot.cube_poses['red']=red;robot.cube_stable['red']=True;robot.cube_pose_arrivals['red']=time.monotonic()
    calls.clear();stages.clear()
    result=robot.place_callback(Trigger.Request(),Trigger.Response(),'red')
    assert result.success and not robot._holding and not robot.placement_ready()
    assert [c for c in calls if c[0]=='gripper']==[('gripper',True)]
    assert not any(c[0]=='attach' for c in calls)
    assert 'GRASPING' not in stages and 'APPROACH' not in stages and 'RELEASING' in stages
    assert robot._stage=='READY'
    assert any(c[0]=='detach' for c in calls)


@pytest.mark.parametrize('reason',['stale','cancel'])
def test_failed_placement_keeps_hold_for_retry(robot,monkeypatch,reason):
    known_held_white(robot);robot.vision_wait_timeout=.001
    red=PoseStamped();red.pose.position.x=.3;red.pose.position.y=.3;red.pose.position.z=.024
    robot.cube_poses['red']=red;robot.cube_stable['red']=True
    robot.cube_pose_arrivals['red']=time.monotonic()-robot.vision_max_age-1
    if reason=='cancel':robot.request_cancel()
    monkeypatch.setattr(robot,'call_gripper',lambda *_:pytest.fail('cannot release before target is valid'))
    before=robot.held_grasp_snapshot()
    ok,message=robot.execute_held_placement('red')
    assert not ok and '夹持状态保留' in message and robot.held_grasp_snapshot()==before


def test_placement_needs_known_cube_but_does_not_block_other_user_actions(robot,monkeypatch):
    robot._holding=True;robot._prepared=True
    monkeypatch.setattr(robot,'move_initial_observation',lambda **kw:pytest.fail('unknown hold must not move'))
    result=robot.place_callback(Trigger.Request(),Trigger.Response(),'red')
    assert not result.success and '先抓取' in result.message
    assert robot.goal_callback(goal('white'))==GoalResponse.ACCEPT


def test_pick_return_failure_retains_cube_and_offset(robot,monkeypatch):
    calls,stages=split_flow_hardware(robot,monkeypatch)
    monkeypatch.setattr(robot,'move_joints',lambda *a:False)
    ok,message=robot.execute_pick_place('white',pick_only=True)
    assert not ok and '返回观察位失败' in message and robot.placement_ready()
    assert [c for c in calls if c[0]=='gripper']==[('gripper',True),('gripper',False)]


def test_held_placement_rejects_same_color_and_busy(robot,monkeypatch):
    known_held_white(robot)
    monkeypatch.setattr(robot,'execute_place_motion',lambda *a:pytest.fail('invalid place must not move'))
    assert not robot.place_callback(Trigger.Request(),Trigger.Response(),'white').success
    robot._busy=True
    assert not robot.place_callback(Trigger.Request(),Trigger.Response(),'red').success
    assert robot.placement_ready()


def test_held_metadata_restores_without_refinding_source(robot,monkeypatch,tmp_path):
    from rclpy.task import Future
    from moveit_msgs.srv import ApplyPlanningScene
    known_held_white(robot);grasp=robot.held_grasp_snapshot();robot._forget_held_grasp()
    path=tmp_path/'held_resume.json';path.write_text(json.dumps({'expires_at':time.time()+60,'base_frame':robot.base_frame,
        'holding':True,'initialized':True,'held_grasp':grasp,'scene':{'is_diff':True,'robot_state':{'is_diff':True,
        'attached_collision_objects':[{'link_name':robot.tcp_frame,'object':{'id':'object','operation':'\x00'}}]}}}))
    future=Future();apply=Mock();apply.service_is_ready.return_value=True;apply.call_async.return_value=future
    monkeypatch.setattr(robot,'apply_scene_cli',apply)
    robot._load_restart_state(str(path));robot._restore_scene_tick()
    future.set_result(ApplyPlanningScene.Response(success=True));robot._restore_scene_tick()
    assert robot.held_grasp_snapshot()==grasp and robot._stage=='WAITING_PLACE'


def test_web_split_then_complete_sentences_use_distinct_routes(monkeypatch):
    web,fake=web_fixture(monkeypatch)
    async def run():
        assert (await web.handle_text('准备'))['ok'];await web._job
        assert not (await web.handle_text('放到红色方块上'))['ok']
        assert (await web.handle_text('抓白色方块'))['ok'];await web._job
        assert web.state['stage']=='WAITING_PLACE' and '白块' in web.state['stage_zh']
        assert (await web.handle_text('放到红色方块上'))['ok'];await web._job
        assert not fake.held_color
        assert (await web.handle_text('把白色方块放到红色方块上'))['ok'];await web._job
        assert fake.calls==['prepare','white',('place','red'),('white','red')]
    asyncio.run(run())


def test_real_bridge_dry_run_tracks_pick_place_and_release():
    from x5a_web_agent.wrist_bridge import WristBridge
    b=WristBridge(dry_run=True)
    assert b.prepare()['success']
    assert b.send_pick('white')['success'] and b.state()['holding_color']=='white'
    assert not b.place('white')['success'] and b.state()['holding']
    assert b.place('red')['success'] and not b.state()['holding']
    assert b.send_pick_place('white','red')['success'] and not b.state()['holding']
    assert b.send_pick('white')['success'] and b.release()['success'] and not b.state()['holding']


def test_http_split_and_complete_flow_on_real_dry_bridge(monkeypatch):
    from fastapi.testclient import TestClient
    from x5a_web_agent.wrist_bridge import WristBridge
    web,_=web_fixture(monkeypatch)
    monkeypatch.setattr(web,'bridge',WristBridge(dry_run=True))
    # Exercise HTTP serialization and background jobs without ROS or ASR startup.
    monkeypatch.setattr(web.app.router,'on_startup',[])
    monkeypatch.setattr(web.app.router,'on_shutdown',[])
    with TestClient(web.app) as client:
        def command(text):
            response=client.post('/api/command',json={'text':text});assert response.status_code==200
            payload=response.json();assert payload['ok'],payload
            for _ in range(40):
                state=client.get('/api/state').json()
                if not state['busy']:return payload,state
                time.sleep(.01)
            pytest.fail('dry bridge job did not finish')
        _,state=command('准备');assert state['prepared']
        payload,state=command('抓白色方块')
        assert payload['parsed']['action']=='pick' and state['holding_color']=='white'
        payload,state=command('放到红色方块上')
        assert payload['parsed']['action']=='place' and not state['holding']
        payload,state=command('把白色方块放到红色方块上')
        assert payload['parsed']['action']=='pick_place' and not state['holding']
        payload,state=command('抓白色方块');assert state['holding']
        payload,state=command('放进盒子里');assert not state['holding']


def test_table_collision_shift_does_not_change_measured_geometry(robot, monkeypatch):
    from moveit_msgs.msg import CollisionObject
    calls=[]
    monkeypatch.setattr(PickPlaceNode,'publish_box',lambda self,name,xyz,size,**kw:calls.append((name,list(xyz),kw)))
    measured=robot.table_z
    for name in ('table','table_mirror'):
        xyz=[robot.table_x,robot.table_y,measured]
        robot.publish_box(name,xyz,[.52,.72,.04])
        assert xyz[2]==measured
        assert calls[-1][1][2]==pytest.approx(measured-.005)
        robot.publish_box(name,xyz,[.52,.72,.04],operation=CollisionObject.REMOVE)
        assert calls[-1][1][2]==measured
    assert robot.table_z==measured


def test_observation_accepts_small_controller_residual_without_changing_joint_limits(robot):
    robot.q=list(robot.ready_joints)
    robot.q[0]+=.045
    assert robot.at_observation()
    robot.q[0]+=.04
    assert not robot.at_observation()
    robot.q[3]=-1.31
    assert robot.servo_readiness_fault()


def test_held_stack_entry_offers_four_centered_alignments(robot,monkeypatch):
    from x5a_pick_place.stacking import tcp_for_object_center
    known_held_white(robot)
    monkeypatch.setattr(robot,'wait_for_placement_target',lambda destination:((.30,.30,.024),2.276))
    monkeypatch.setattr(robot,'publish_box',lambda *args,**kw:None)
    calls=[]
    def place(xyz,orientations,center):
        assert len(orientations)==4
        assert tuple(center)==pytest.approx((.30,.30,.041))
        assert tuple(xyz)==pytest.approx(tcp_for_object_center(center,robot._held_object_offset,orientations[0]))
        calls.append(orientations)
        return True,'planned'
    monkeypatch.setattr(robot,'execute_place_motion',place)
    assert robot.execute_held_placement('red')[0] and len(calls)==1


def test_descent_uses_selected_orientation_tcp_instead_of_first_candidate(robot,monkeypatch):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from x5a_pick_place.pick_place_node import pose_to_xyz
    from x5a_pick_place.stacking import tcp_for_object_center
    calls,stages=split_flow_hardware(robot,monkeypatch)
    known_held_white(robot)
    robot._held_object_offset=(.03,.01,-.02)
    held_offset=robot._held_object_offset
    center=(.30,.30,.041);selected=[0.,1.1,1.57]
    xyz=tcp_for_object_center(center,robot._held_object_offset,selected)
    monkeypatch.setattr(robot,'move_stack_pre_place',lambda *args:(xyz,selected))
    poses=[];original=robot.vertical_move
    def vertical(pose,stage,*args):
        if stage=='DESCEND':poses.append(pose)
        return original(pose,stage,*args)
    monkeypatch.setattr(robot,'vertical_move',vertical)
    assert robot.execute_place_motion((.4,.4,.1),[[0.,1.1,0.],selected],center)[0]
    pose=poses[0];q=pose.orientation
    np.testing.assert_allclose(pose_to_xyz(pose),xyz,atol=1e-12)
    actual=np.array(pose_to_xyz(pose))+Rotation.from_quat([q.x,q.y,q.z,q.w]).apply(held_offset)
    np.testing.assert_allclose(actual,center,atol=1e-12)


def test_no_valid_stack_orientation_keeps_holding_and_never_releases(robot,monkeypatch):
    known_held_white(robot)
    monkeypatch.setattr(robot,'move_stack_pre_place',lambda *args:None)
    monkeypatch.setattr(robot,'vertical_move',lambda *args:pytest.fail('must not descend'))
    monkeypatch.setattr(robot,'call_gripper',lambda *args:pytest.fail('must not release'))
    assert not robot.execute_place_motion((.30,.30,.081),[[0.,1.1,0.]],(.30,.30,.041))[0]
    assert robot._holding and robot.placement_ready()
