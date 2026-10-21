"""Depth/geometry and multi-step failures, with no robot commands or ROS graph."""
import time
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from x5a_vision.clearance import buffer_surface, blockers_from_sides, read_buffer, base_cloud, buffer_candidates, BufferSurfaceIndex
from x5a_task_server.clearing import ClearingMixin


def table_cloud(xy=(.28, .30)):
    x, y = np.meshgrid(np.linspace(-.039, .039, 31), np.linspace(-.039, .039, 31))
    return np.column_stack((x.ravel()+xy[0], y.ravel()+xy[1], np.full(x.size, -.006)))


def test_empty_buffer_requires_full_observed_table_and_rejects_clutter():
    cloud = table_cloud()
    assert buffer_surface(cloud, (.28, .30), -.006)['empty']
    assert not buffer_surface(cloud[:300], (.28, .30), -.006)['empty']
    assert not buffer_surface(np.empty((0, 3)), (.28, .30), -.006)['empty']
    cloud[:10, 2] = .018
    assert not buffer_surface(cloud, (.28, .30), -.006)['empty']


def test_depth_holes_and_lower_surface_are_not_treated_as_free_table():
    cloud = table_cloud()
    cloud[cloud[:, 0] < .28, 2] = -.025
    assert not buffer_surface(cloud, (.28, .30), -.006)['empty']


def test_blocker_requires_current_lower_side_under_unique_second_layer_top():
    points = np.column_stack((np.full(80, .345), np.linspace(.29, .31, 80), np.linspace(.0, .02, 80)))
    colors = np.full(80, 'white')
    assert blockers_from_sides(points, colors, {'orange': np.array([.36,.30,.054])}, ['white'], -.006) == {'white':'orange'}
    assert not blockers_from_sides(points, colors, {'orange': np.array([.36,.30,.024])}, ['white'], -.006)
    assert not blockers_from_sides(points+.10, colors, {'orange': np.array([.36,.30,.054])}, ['white'], -.006)
    tops = {'orange': np.array([.36,.30,.054]), 'red': np.array([.36,.30,.054])}
    assert not blockers_from_sides(points, colors, tops, ['white'], -.006)
    assert not blockers_from_sides(points[:3], colors[:3], tops, ['white'], -.006)


def test_disabled_and_wrong_side_buffers_cannot_be_used(tmp_path):
    path = tmp_path/'buffer.yaml'
    path.write_text(yaml.safe_dump({'enabled':False}))
    with pytest.raises(ValueError): read_buffer(path, 'base', 'current')
    path.write_text(yaml.safe_dump({'enabled':True,'base_frame':'base','observation_side':'current','xy_m':[.28,.30]}))
    assert read_buffer(path, 'base', 'current') == (.28, .30)
    with pytest.raises(ValueError): read_buffer(path, 'base', 'mirror')
    with pytest.raises(ValueError): read_buffer(path, 'other', 'current')


def test_cloud_filters_invalid_depth_instead_of_filling_it():
    points, xs, ys = base_cloud(np.array([[0.,.3],[np.nan,.4]]), [100,0,0,0,100,0,0,0,1], np.eye(4), .2,.8, stride=1)
    assert len(points) == 2 and list(xs) == [1,1] and list(ys) == [0,1]


class Session(ClearingMixin):
    def __init__(self):
        self.allowed_colors = ('red','white','orange')
        self.plan_only = self._holding = False
        self._held_color = 'orange'
        self._held_grasp_rpy = [0.,1.1,0.]
        self._held_object_offset = [.01,0.,-.05]
        self.obj_sx = self.obj_sy = self.obj_sz = .03
        self.table_z, self.table_sz, self.stack_clearance = -.026, .04, .002
        self.vision_wait_timeout = .01
        self.cube_pose_arrivals = {'white':0.0}
        self.cube_poses = {}
        self.calls, self.canceled = [], False
        self.pick_ok = self.place_ok = self.target_visible = True
        self.initial_buffer_fault = self.late_buffer_fault = ''
    def _select_buffer(self, blocker, source_top, held=False):
        return (None, self.late_buffer_fault) if held and self.late_buffer_fault else ((None, self.initial_buffer_fault) if self.initial_buffer_fault else ((.28,.30), ''))
    def _check_buffer(self, *args): return self.initial_buffer_fault
    def _await_buffer(self, *args): return self.late_buffer_fault
    def _direct_buffer_after_lift(self, *args): return self.late_buffer_fault
    def _clearing_details(self, after=0.0): return {'blocked_by':{'white':'orange'}, 'stamp_ns':100}
    def _cancelled(self): return self.canceled
    def servo_readiness_fault(self): return ''
    def report_stage(self, stage, **kwargs): self.calls.append(stage)
    def _wait_observation(self): self.calls.append('observation'); return True
    def placement_ready(self): return self._holding
    def vision_snapshot(self):
        return {'orange':{'ready':True, 'xyz':[.36,.30,.054]},
                'white':{'ready':self.target_visible and bool(self.cube_poses), 'status':'no_top_face', 'xyz':[.36,.30,.024]}}
    def execute_pick_place(self, color, **kwargs):
        self.calls.append(('pick',color,kwargs))
        self._holding = self.pick_ok
        if self.pick_ok and hasattr(self,'scene_memory'):
            self.scene_memory.moved(color)
        return self.pick_ok, 'pick failed'
    def execute_place_motion(self, *args):
        self.calls.append(('place',args))
        if self.place_ok:
            self._holding = False
            self.cube_pose_arrivals['white'] = time.monotonic()+1
            self.cube_poses['white'] = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=0,nanosec=100)))
        return self.place_ok, 'place failed'


def test_clear_waits_for_new_target_and_uses_measured_held_offset():
    robot = Session()
    assert robot.clear_target_if_needed('white')[0]
    assert ('pick','orange',{'pick_only':True,'hold_at_lift':True}) in robot.calls
    assert robot.calls.index('observation') > next(i for i,c in enumerate(robot.calls) if isinstance(c,tuple) and c[0]=='place')
    placement = next(c for c in robot.calls if isinstance(c,tuple) and c[0]=='place')
    assert placement[1][2] == pytest.approx((.28,.30,.011))
    assert not np.allclose(placement[1][0], placement[1][2])
    assert robot.calls[-1] == 'REOBSERVE_UNCOVERED'


@pytest.mark.parametrize('field,value', [('initial_buffer_fault','occupied'),('canceled',True),('plan_only',True)])
def test_reject_before_moving_if_buffer_cancel_or_plan_only(field,value):
    robot = Session(); setattr(robot,field,value)
    assert not robot.clear_target_if_needed('white')[0]
    assert not any(isinstance(c,tuple) and c[0]=='pick' for c in robot.calls)


def test_late_buffer_occupation_retains_upper_cube_and_blocks_target():
    robot = Session(); robot.late_buffer_fault = 'occupied'
    assert not robot.clear_target_if_needed('white')[0]
    assert robot._holding
    assert not any(isinstance(c,tuple) and c[0]=='place' for c in robot.calls)


def test_failed_place_or_target_reobservation_stops_sequence():
    robot = Session(); robot.place_ok = False
    assert not robot.clear_target_if_needed('white')[0] and robot._holding
    robot = Session(); robot.target_visible = False
    ok, message = robot.clear_target_if_needed('white')
    assert not ok and '目标未获得新的稳定顶面定位' in message
    assert not robot._holding


BOUNDS = {'x_min':0.,'x_max':.5,'y_min':0.,'y_max':.6,'r_max':.65}


def test_index_never_accepts_missing_depth_or_occupied_boundary_patch():
    xy = (.28, .30)
    cloud = table_cloud(xy)
    assert BufferSurfaceIndex(cloud, -.006, BOUNDS).check(xy)['empty']
    assert not BufferSurfaceIndex(cloud[:300], -.006, BOUNDS).check(xy)['empty']
    assert not BufferSurfaceIndex(np.empty((0, 3)), -.006, BOUNDS).check(xy)['empty']
    clutter = np.array([[xy[0]+.0399, xy[1], .03]]*3)
    assert not BufferSurfaceIndex(np.vstack((cloud, clutter)), -.006, BOUNDS).check(xy)['empty']
    # Points just beyond a coverage cell's edge cannot certify that cell.
    edge_points = cloud.copy()
    edge_points[:, 0] = xy[0]+.0399
    assert not BufferSurfaceIndex(edge_points, -.006, BOUNDS).check(xy)['empty']


def test_index_accepted_patches_also_pass_exact_point_cloud_check():
    rng = np.random.default_rng(47)
    points = rng.uniform([.05, .05, -.007], [.45, .55, -.005], (60000, 3))
    clutter = rng.uniform([.23, .23, .02], [.28, .28, .04], (500, 3))
    points = np.vstack((points, clutter))
    index = BufferSurfaceIndex(points, -.006, BOUNDS)
    accepted = 0
    for xy in rng.uniform([.10, .10], [.40, .50], (80, 2)):
        if index.check(xy)['empty']:
            accepted += 1
            assert buffer_surface(points, xy, -.006)['empty']
    assert accepted > 30


def test_expanded_search_finds_free_patch_beyond_original_three_rings():
    source = (.30, .30)
    xy = (source[0]-.05-.03*np.sqrt(2)-.11, source[1])
    stats = {}
    candidates = buffer_candidates(table_cloud(xy), source, .03, -.006, BOUNDS, diagnostics=stats)
    assert candidates and candidates[0]['xy_m'] == pytest.approx(xy)
    assert stats['tested'] == 80 and stats['available'] == len(candidates)
    assert sum(stats[k] for k in ('outside_workspace', 'occupied', 'insufficient_depth', 'available')) == 80


@pytest.mark.parametrize('source', [(.20,.25),(.31,.36),(.39,.29)])
def test_candidates_follow_current_source_instead_of_previous_fixed_point(source):
    radius = .05+.03*np.sqrt(2)
    xy = (source[0]-radius, source[1])
    candidates = buffer_candidates(table_cloud(xy), source, .03, -.006, BOUNDS)
    assert candidates and candidates[0]['xy_m'] == pytest.approx(xy)
    assert np.linalg.norm(np.array(candidates[0]['xy_m'])-source) > .09


def test_candidates_skip_occupied_and_outside_visible_regions():
    source = (.30,.30)
    radius = .05+.03*np.sqrt(2)
    left = table_cloud((source[0]-radius,source[1]))
    left[:20,2] = .03
    upper = table_cloud((source[0],source[1]+radius))
    candidates = buffer_candidates(np.vstack((left,upper)), source,.03,-.006,BOUNDS)
    assert candidates and candidates[0]['xy_m'] == pytest.approx((source[0],source[1]+radius))
    assert not buffer_candidates(left,source,.03,-.006,BOUNDS)
    assert not buffer_candidates(np.empty((0,3)),source,.03,-.006,BOUNDS)


def test_dynamic_selection_skips_unreachable_point_and_reserves_a_new_token(tmp_path):
    path=tmp_path/'buffer.yaml';path.write_text('enabled: true\nmode: auto\n')
    robot=Session();robot.buffer_file=str(path);robot._buffer_selection=None
    robot._inside_buffer_table=lambda xy: True
    data={'buffer_candidates':{'orange':[{'xy_m':[.20,.30]},{'xy_m':[.26,.30]}]}}
    robot._clearing_details=lambda *a: data
    robot._buffer_reachable=lambda xy,*a: xy[0]>.20
    selected=[]
    def request(xy): selected.append(xy);return time.monotonic()
    robot._request_buffer=request
    robot._await_buffer=lambda *a: ''
    robot._freeze_buffer_precheck=lambda *a: True
    xy,reason=ClearingMixin._select_buffer(robot,'orange',(.36,.30,.054))
    assert xy==(.26,.30) and not reason and selected==[(.20,.30),(.26,.30)]
    assert robot._buffer_diagnostics['unreachable'] == 1
    assert robot._buffer_diagnostics['selected_xy_m'] == [.26, .30]
    data['buffer_candidates']['orange']=[{'xy_m':[.42,.35]}]
    xy,reason=ClearingMixin._select_buffer(robot,'orange',(.36,.30,.054))
    assert xy==(.42,.35) and robot._buffer_options==[(.42,.35)]


def test_after_pick_tries_another_candidate_if_first_becomes_occupied(tmp_path):
    robot=Session();robot._buffer_options=[(.20,.30),(.26,.30)]
    robot._buffer_selection={'xy_m':[.20,.30]}
    robot._inside_buffer_table=lambda xy: True
    robot._buffer_reachable=lambda *a: True
    robot._request_buffer=lambda xy: xy
    robot._await_buffer=lambda xy,after: 'occupied' if xy[0]==.20 else ''
    robot._freeze_buffer_precheck=lambda *a: True
    xy,reason=ClearingMixin._select_buffer(robot,'orange',(.36,.30,.054),held=True)
    assert xy==(.26,.30) and not reason


def test_selected_buffer_checks_token_and_side_not_just_same_coordinates():
    robot=Session();robot.base_frame='base';robot.observation_side='current'
    robot._buffer_selection={'token':'new'}
    data={'selected_buffer':{'xy_m':[.28,.30],'base_frame':'base','observation_side':'current','token':'old','empty':True}}
    robot._clearing_details=lambda *a: data
    assert ClearingMixin._check_buffer(robot,(.28,.30))
    data['selected_buffer']['token']='new'
    assert not ClearingMixin._check_buffer(robot,(.28,.30))
    data['selected_buffer']['observation_side']='mirror'
    assert ClearingMixin._check_buffer(robot,(.28,.30))


def test_repeated_same_frame_cannot_satisfy_three_fresh_frames():
    robot=Session();robot.vision_wait_timeout=.06
    robot._check_buffer=lambda *a: ''
    robot._clearing_details=lambda *a: {'stamp_ns':1}
    assert ClearingMixin._await_buffer(robot,(.28,.30),0.0)
    robot.vision_wait_timeout=.3
    stamps=iter(range(10))
    robot._clearing_details=lambda *a: {'stamp_ns':next(stamps)}
    assert not ClearingMixin._await_buffer(robot,(.28,.30),0.0)


def test_reachability_uses_collision_aware_ik_and_rejects_joint_limits():
    robot=Session();robot.pre_place_h=.06
    robot.pose_rpy=lambda *a: a
    calls=[]
    robot.ik_joints_for_pose=lambda pose,**kw: calls.append((pose,kw)) or [0.]*6
    robot.servo_bounds_fault=lambda joints: ''
    assert ClearingMixin._buffer_reachable(robot,(.28,.30),(.36,.30,.054),'orange',held=True)
    assert len(calls)==2 and all(kw=={'collision_aware_only':True} for pose,kw in calls)
    robot.servo_bounds_fault=lambda joints: 'limit'
    assert not ClearingMixin._buffer_reachable(robot,(.28,.30),(.36,.30,.054),'orange',held=True)


def direct_session():
    robot=Session();robot.base_frame='base';robot.observation_side='current'
    robot.buffer_precheck_ttl=60.
    robot._buffer_selection={'token':'task'}
    robot._buffer_precheck={'token':'task','xy_m':[.28,.30],'base_frame':'base',
        'observation_side':'current','stamp_ns':100,'verified_at':time.monotonic(),'invalid_reason':''}
    robot._buffer_reachable=lambda *a,**kw: True
    return robot


def test_direct_transfer_uses_frozen_proof_and_actual_offset_without_camera_return():
    robot=direct_session()
    robot._wait_observation=lambda: pytest.fail('direct transfer must not return for another observation')
    robot._select_buffer=lambda *a,**kw: pytest.fail('direct transfer must not select again')
    assert not ClearingMixin._direct_buffer_after_lift(robot,(.28,.30),(.36,.30,.054),'orange')


@pytest.mark.parametrize('case', ['expired','occupied','side','token','canceled','unreachable'])
def test_direct_transfer_rejects_invalid_proof_and_keeps_gripper(case):
    robot=direct_session();robot._holding=True
    if case=='expired': robot._buffer_precheck['verified_at']-=61
    if case=='occupied': robot._buffer_precheck['invalid_reason']='occupied'
    if case=='side': robot.observation_side='mirror'
    if case=='token': robot._buffer_selection['token']='other'
    if case=='canceled': robot.canceled=True
    if case=='unreachable': robot._buffer_reachable=lambda *a,**kw: False
    assert ClearingMixin._direct_buffer_after_lift(robot,(.28,.30),(.36,.30,.054),'orange')
    assert robot._holding


def test_observed_occupation_invalidates_proof_but_camera_occlusion_does_not():
    robot=direct_session()
    data={'clearing':{'stamp_ns':101,'selected_buffer':{'token':'task','empty':False,'reason':'暂放位未完整可见或深度不足'}}}
    robot._record_buffer_change(data)
    assert not robot._buffer_precheck['invalid_reason']
    data['clearing']['selected_buffer']['reason']='暂放位或夹爪周围有物体占用'
    robot._record_buffer_change(data)
    assert robot._buffer_precheck['invalid_reason']


def test_vision_confirmation_and_ik_precheck_actually_run_concurrently(tmp_path):
    path=tmp_path/'buffer.yaml';path.write_text('enabled: true\nmode: auto\n')
    robot=Session();robot.buffer_file=str(path)
    robot._clearing_details=lambda *a: {'buffer_candidates':{'orange':[{'xy_m':[.26,.30]}]}}
    robot._inside_buffer_table=lambda *a: True
    robot._request_buffer=lambda *a: 0.
    robot._freeze_buffer_precheck=lambda *a: True
    ik_started, vision_done = threading.Event(), threading.Event()
    def ik(*args):
        ik_started.set()
        assert vision_done.wait(1)
        return True
    def vision(*args):
        assert ik_started.wait(1)
        vision_done.set()
        return ''
    robot._buffer_reachable=ik;robot._await_buffer=vision
    xy,reason=ClearingMixin._select_buffer(robot,'orange',(.36,.30,.054))
    assert xy==(.26,.30) and not reason


def test_freeze_rechecks_latest_frame_and_catches_occupation_during_handoff():
    robot=direct_session()
    robot._buffer_selection.update(xy_m=[.28,.30],base_frame='base',observation_side='current')
    clear={'stamp_ns':101,'selected_buffer':{'token':'task','xy_m':[.28,.30],'empty':True}}
    robot._clearing_details=lambda *a: clear
    robot._detection_details={'clearing':clear}
    assert robot._freeze_buffer_precheck((.28,.30))
    robot._detection_details={'clearing':{'stamp_ns':102,'selected_buffer':{'token':'task','empty':False,'reason':'有物体占用'}}}
    assert not robot._freeze_buffer_precheck((.28,.30))
    clear['selected_buffer']['empty']=False
    assert not robot._freeze_buffer_precheck((.28,.30))


class TripleSession(Session):
    def __init__(self):
        super().__init__()
        self.tower=['orange','white','red']
        self.removed=[]
        self._held_color=None
        self.fail_second_place=False
        self.fresh_next=True
        self.next_shift=0.
        self.cancel_after_first=False
        self.cube_poses={}
        self.cube_pose_arrivals={}
        self._expose()
    def _expose(self):
        color=self.tower[-1]
        self.cube_poses[color]=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=0,nanosec=100+len(self.removed))))
        self.cube_pose_arrivals[color]=time.monotonic()+1 if self.fresh_next else 0.
    def _clearing_details(self,after=0.):
        return {'blocked_by':{color:self.tower[-1] for color in self.tower[:-1]},'stamp_ns':100+len(self.removed)}
    def vision_snapshot(self):
        result={color:{'ready':False,'status':'no_top_face','xyz':None} for color in self.allowed_colors}
        top=self.tower[-1]
        result[top]={'ready':True,'status':'stable','xyz':[.36+self.next_shift*(bool(self.removed)),.30,.024+.032*(len(self.tower)-1)]}
        for index,color in enumerate(self.removed):
            result[color]={'ready':True,'status':'stable','xyz':[.20,.30+index*.10,.026]}
        return result
    def execute_pick_place(self,color,**kwargs):
        self.calls.append(('pick',color,kwargs))
        assert color==self.tower[-1]
        self._holding=True;self._held_color=color
        if hasattr(self,'scene_memory'): self.scene_memory.moved(color)
        return True,''
    def execute_place_motion(self,*args):
        self.calls.append(('place',args))
        if self.fail_second_place and self.removed:return False,'规划失败'
        self.removed.append(self.tower.pop());self._holding=False
        self._expose()
        if self.cancel_after_first:self.canceled=True
        return True,''


def test_clear_three_layer_bottom_removes_top_then_middle_with_new_observations():
    robot=TripleSession()
    assert robot.clear_target_if_needed('orange')[0]
    assert [c[1] for c in robot.calls if isinstance(c,tuple) and c[0]=='pick']==['red','white']
    assert robot.tower==['orange'] and not robot._holding
    assert robot.calls.count('REOBSERVE_UNCOVERED')==2
    assert robot.calls.count('observation')==2


def test_clear_middle_of_three_layer_tower_leaves_middle_on_its_support():
    robot=TripleSession()
    assert robot.clear_target_if_needed('white')[0]
    assert robot.removed==['red'] and robot.tower==['orange','white']


@pytest.mark.parametrize('case',['stale_next','shifted_next','cancel','second_place_failure'])
def test_three_layer_clearing_stops_on_changed_evidence_or_failure(case):
    robot=TripleSession()
    if case=='stale_next':robot.fresh_next=False
    if case=='shifted_next':robot.next_shift=.04
    if case=='cancel':robot.cancel_after_first=True
    if case=='second_place_failure':robot.fail_second_place=True
    assert not robot.clear_target_if_needed('orange')[0]
    picks=[c[1] for c in robot.calls if isinstance(c,tuple) and c[0]=='pick']
    assert picks==(['red','white'] if case=='second_place_failure' else ['red'])
    assert robot.removed==['red']
    assert robot._holding == (case=='second_place_failure')


def test_three_layer_clearing_can_use_hidden_lower_memory_without_side_colors():
    from x5a_task_server.scene_memory import SceneMemory
    robot=TripleSession();robot.base_frame='base';robot.observation_side='current'
    robot.scene_memory=SceneMemory()
    robot.scene_memory.record_stack('white','orange',(.36,.30,.041),.03,.002,'base','current',1)
    robot.scene_memory.record_stack('red','white',(.36,.30,.073),.03,.002,'base','current',2)
    robot._clearing_details=lambda after=0.:{'blocked_by':{},'stamp_ns':100+len(robot.removed)}
    assert robot.clear_target_if_needed('orange')[0]
    assert robot.removed==['red','white'] and not robot.scene_memory.snapshot()


def test_live_side_depth_resolves_third_layer_blocker_without_guessing_middle_pose():
    bottom=np.column_stack((np.full(80,.345),np.linspace(.29,.31,80),np.linspace(.0,.02,80)))
    middle=bottom.copy();middle[:,2]+=.032
    points=np.vstack([bottom,middle]);colors=np.array(['orange']*80+['white']*80)
    tops={'red':np.array([.36,.30,.088])}
    assert blockers_from_sides(points,colors,tops,['orange','white'],-.006)=={'orange':'red','white':'red'}
    assert not blockers_from_sides(points,colors,tops,['orange','white'],-.006,max_layers=2)


def test_bottom_restack_can_clear_a_destination_in_the_same_tower():
    from x5a_task_server.wrist_task_server import WristTaskServer
    from x5a_task_server.scene_memory import SceneMemory
    robot=TripleSession();robot.base_frame='base';robot.observation_side='current'
    robot.scene_memory=SceneMemory()
    robot.scene_memory.record_stack('white','orange',(.36,.30,.041),.03,.002,'base','current',1)
    robot.scene_memory.record_stack('red','white',(.36,.30,.073),.03,.002,'base','current',2)
    robot.stack_destination_fault=lambda destination,source=None:WristTaskServer.stack_destination_fault(robot,destination,source)
    assert WristTaskServer.execute_color_step(robot,'orange','white')[0]
    assert [c[1] for c in robot.calls if isinstance(c,tuple) and c[0]=='pick']==['red','white','orange']
    assert robot.calls[-1][2]=={'destination_color':'white'}


def test_visible_bottom_fragment_cannot_skip_the_still_covered_middle_layer():
    from x5a_task_server.scene_memory import SceneMemory
    robot=TripleSession();robot.base_frame='base';robot.observation_side='current'
    robot.scene_memory=SceneMemory()
    robot.scene_memory.record_stack('white','orange',(.36,.30,.041),.03,.002,'base','current',1)
    robot.scene_memory.record_stack('red','white',(.36,.30,.073),.03,.002,'base','current',2)
    original=robot.vision_snapshot
    def partial_bottom():
        result=original()
        result['orange']={'ready':True,'status':'stable','xyz':[.36,.30,.024]}
        robot.cube_pose_arrivals['orange']=time.monotonic()+1
        robot.cube_poses['orange']=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=0,nanosec=100+len(robot.removed))))
        return result
    robot.vision_snapshot=partial_bottom
    assert robot.clear_target_if_needed('orange')[0]
    assert robot.removed==['red','white']
