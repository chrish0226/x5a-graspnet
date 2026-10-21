"""Color-only wrist tasks: prepare first, then one complete pick/place cycle."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from moveit_msgs.msg import PlanningScene, CollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from rosidl_runtime_py.set_message import set_message_fields
from x5a_pick_place.pick_place_node import ALLOWED_COLORS, PickPlaceNode, wrap_pi, pose_to_xyz, quat_to_yaw
from x5a_pick_place.stacking import held_stack_center, aligned_stack_candidates, tcp_for_object_center
from x5a_task_interfaces.action import X5aPickPlace
from x5a_task_interfaces.action import ColorSequence
from x5a_pick_place.color_sequence import color_step_pairs
from x5a_task_server.observation_profiles import mirrored_seed, crosses_home_plane
from x5a_task_server.clearing import ClearingMixin
from x5a_task_server.scene_memory import SceneMemory
from x5a_vision.clearance import DEFAULT_BUFFER_FILE


class WristTaskServer(ClearingMixin, PickPlaceNode):
    def __init__(self):
        super().__init__(node_name='wrist_task_server', tf_spin_thread=False)
        self.scene_memory = SceneMemory()
        self._placement_destination = None
        self._busy_lock = threading.Lock()
        self._busy = False
        self._prepared = False
        self._holding = False
        self._forget_held_grasp()
        self._stage = 'IDLE'
        self._sequence_step = self._sequence_total = 0
        self._last_moved_color = self._last_support_color = None
        self._restore_pending = False
        self._restore_future = None
        self._image_stamp = 0
        self._image_arrival = 0.0
        self._detection_details = {}
        self._details_arrival = 0.0
        self.declare_parameter('clearing.buffer_file', DEFAULT_BUFFER_FILE)
        self.buffer_file = str(self.get_parameter('clearing.buffer_file').value)
        self._buffer_selection = None
        self._buffer_options = []
        self._buffer_precheck = None
        self.declare_parameter('clearing.precheck_ttl_s', 60.0)
        self.buffer_precheck_ttl = float(self.get_parameter('clearing.precheck_ttl_s').value)
        if not math.isfinite(self.buffer_precheck_ttl) or not 5.0 <= self.buffer_precheck_ttl <= 120.0:
            raise ValueError('clearing.precheck_ttl_s must be 5–120 seconds')
        self.buffer_request_pub = self.create_publisher(String, '/wrist_pick/buffer_request',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._motor_health = None
        self._motor_health_arrival = 0.0
        self.declare_parameter('wrist.scene_objects_collision_enabled', True)
        self.scene_objects_collision_enabled = bool(self.get_parameter('wrist.scene_objects_collision_enabled').value)
        self.declare_parameter('wrist.table_collision_offset_m', 0.0)
        self.declare_parameter('wrist.observation_tolerance_rad', .03)
        self.table_collision_offset = float(self.get_parameter('wrist.table_collision_offset_m').value)
        self.observation_tolerance = float(self.get_parameter('wrist.observation_tolerance_rad').value)
        if not math.isfinite(self.table_collision_offset) or not -.005 <= self.table_collision_offset <= 0.0:
            raise ValueError('table collision offset must be within -5..0 mm')
        if not math.isfinite(self.observation_tolerance) or not .01 <= self.observation_tolerance <= .08:
            raise ValueError('observation tolerance must be within 0.01..0.08 rad')
        self.declare_parameter('task.action_name', '/wrist_pick/pick_place')
        self.declare_parameter('wrist.prepare_timeout_s', 8.0)
        self.declare_parameter('wrist.home_joints', [0.0]*6)
        self.home_joints = list(self.get_parameter('wrist.home_joints').value)
        if len(self.home_joints) != 6 or not all(math.isfinite(v) for v in self.home_joints):
            raise ValueError('wrist.home_joints requires six finite joint angles')
        self.current_ready_joints = list(self.ready_joints)
        self.current_pre_grasp_joints = list(self.pre_grasp_ready_joints)
        self.current_table_y = self.table_y
        self.declare_parameter('wrist.mirror_ready_joints', mirrored_seed(self.ready_joints, self.home_joints))
        self.mirror_ready_joints = list(self.get_parameter('wrist.mirror_ready_joints').value)
        if len(self.mirror_ready_joints) != 6 or self.servo_bounds_fault(self.mirror_ready_joints):
            raise ValueError('wrist.mirror_ready_joints must satisfy active servo limits')
        self.observation_side = 'current'
        self._prepare_timeout = float(self.get_parameter('wrist.prepare_timeout_s').value)
        self._callbacks = ReentrantCallbackGroup()
        self._arm_recover = self.create_client(Trigger, '/arm/recover',
                                               callback_group=self._callbacks)
        self.create_subscription(String, '/arm/health', self._on_motor_health,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=self._callbacks)
        self.stage_pub = self.create_publisher(String, '/wrist_pick/task_stage', 10)
        self.busy_pub = self.create_publisher(Bool, '/wrist_pick/task_busy', 10)
        self.prepared_pub = self.create_publisher(Bool, '/wrist_pick/prepared', 10)
        self.observation_side_pub = self.create_publisher(String, '/wrist_pick/observation_side',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Image, '/wrist_camera/color/image_raw', self._on_image,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
                                 callback_group=self._callbacks)
        self.create_subscription(String, '/x5a_vision/detection_details', self._on_detection_details,
                                 10, callback_group=self._callbacks)
        self.set_stage_callback(self._on_stage)
        self._action = ActionServer(
            self, X5aPickPlace, str(self.get_parameter('task.action_name').value),
            execute_callback=self.execute_callback, goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback, callback_group=self._callbacks)
        self._sequence_action = ActionServer(
            self, ColorSequence, '/wrist_pick/color_sequence',
            execute_callback=self.sequence_execute_callback, goal_callback=self.sequence_goal_callback,
            cancel_callback=self.cancel_callback, callback_group=self._callbacks)
        for name, callback in [('prepare', self.prepare_callback), ('prepare_current', self.prepare_current_callback),
                               ('prepare_mirror', self.prepare_mirror_callback), ('go_home', self.home_callback), ('state', self.state_callback),
                               ('cancel', self.cancel_service), ('release', self.release_callback)]:
            self.create_service(Trigger, '/wrist_pick/' + name, callback,
                                callback_group=self._callbacks)
        for destination in ('box', *ALLOWED_COLORS):
            self.create_service(Trigger, '/wrist_pick/place_' + destination,
                                lambda req, resp, d=destination: self.place_callback(req, resp, d),
                                callback_group=self._callbacks)
        for color in ALLOWED_COLORS:
            self.create_service(Trigger, '/wrist_pick/check_buffer_' + color,
                lambda req, resp, c=color: self.check_buffer_callback(req, resp, c),
                callback_group=self._callbacks)
        self.create_timer(0.5, self._publish_status)
        self._load_restart_state(os.environ.get('X5A_WRIST_RESTORE_FILE', ''))
        self._publish_status()
        self.get_logger().info('WRIST READY: say 准备, then 抓红色 / 抓白色 / 抓橙色')

    def check_buffer_callback(self, _request, response, color):
        """Inspect the current scene and solve IK without issuing robot motion."""
        if not self._reserve():
            response.success, response.message = False, '任务忙，请稍后检查'
            return response
        previous_stage = self._stage
        try:
            if not self.at_observation() or self._holding:
                raise ValueError('只读选点检查需要在观察位，且夹爪中没有方块')
            blocker = self._resolve_blocker(color)
            if blocker not in ALLOWED_COLORS:
                raise ValueError('当前尚未确认目标的上层遮挡关系')
            item = self.vision_snapshot()[blocker]
            if not item['ready']:
                raise ValueError('上层方块尚未稳定定位')
            xy, reason = self._select_buffer(blocker, tuple(item['xyz']))
            response.success = xy is not None
            response.message = json.dumps({'target_color':color, 'blocker':blocker,
                                           'xy_m':xy, 'reason':reason, 'motion_sent':False}, ensure_ascii=False)
        except Exception as exc:
            response.success, response.message = False, str(exc)
        finally:
            self._finish(True, stage=previous_stage)
        return response

    def _load_restart_state(self, filename):
        """Consume an expiring, local handoff; restoring a scene never moves the arm."""
        if not filename or not Path(filename).exists():
            return
        saved = json.loads(Path(filename).read_text())
        if saved['expires_at'] < time.time() or saved['base_frame'] != self.base_frame:
            raise ValueError('expired or incompatible wrist restart state')
        # Humble's dictionary converter represents the octet operation as a
        # one-character string. Its setter needs a one-byte sequence instead.
        objects = list(saved['scene'].get('world', {}).get('collision_objects', []))
        objects += [a['object'] for a in saved['scene'].get('robot_state', {}).get('attached_collision_objects', [])]
        for obj in objects:
            operation = obj.get('operation')
            if isinstance(operation, str) and len(operation) == 1:
                obj['operation'] = [ord(operation)]
            elif isinstance(operation, int):
                obj['operation'] = [operation]
        self._restore_scene = PlanningScene()
        set_message_fields(self._restore_scene, saved['scene'])
        Path(filename).unlink()
        self._holding = bool(saved['holding'])
        grasp = saved.get('held_grasp')
        if self._holding and grasp:
            self._held_color = grasp['color']
            self._held_grasp_rpy = grasp['grasp_rpy']
            self._held_cube_yaw = grasp['cube_yaw']
            self._held_source_top = grasp['source_top']
            self._held_object_offset = grasp['offset']
            if not self.placement_ready():
                raise ValueError('invalid held grasp in restart state')
        self.select_observation_side(saved.get('observation_side', 'current'))
        self._prepared = bool(saved['initialized'])
        self._restore_pending = True
        self._stage = 'RECOVERING_STATE'
        self._restore_timer = self.create_timer(.25, self._restore_scene_tick)

    def _restore_scene_tick(self):
        if self._restore_future is None:
            if self.apply_scene_cli.service_is_ready():
                request = ApplyPlanningScene.Request(scene=self._restore_scene)
                self._restore_future = self.apply_scene_cli.call_async(request)
            return
        if not self._restore_future.done():
            return
        response = self._restore_future.result()
        if response is None or not response.success:
            self._restore_future = None
            return
        with self._busy_lock:
            self._restore_pending = False
            self._stage = (('WAITING_PLACE' if self.placement_ready() else 'READY')
                           if self._prepared and self.at_observation() else 'IDLE')
        self._restore_timer.cancel()
        self._publish_status()

    def _on_image(self, msg):
        self._image_stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self._image_arrival = time.monotonic()

    def _on_detection_details(self, msg):
        try:
            details = json.loads(msg.data)
            if isinstance(details, dict):
                self._detection_details = details
                self._details_arrival = time.monotonic()
                self._record_buffer_change(details)
        except (ValueError, TypeError):
            pass

    def _cube_stable_cb(self, color, msg):
        super()._cube_stable_cb(color, msg)
        if (not msg.data or not hasattr(self, 'scene_memory')
                or (self._busy and self._stage not in ('READY','WAITING_VISION','SELECTING_BUFFER','REOBSERVE_UNCOVERED'))
                or not self.at_observation()):
            return
        pose = self.cube_poses.get(color)
        if pose is None or time.monotonic()-self.cube_pose_arrivals.get(color, 0.) > self.vision_max_age:
            return
        stamp = pose.header.stamp.sec*1_000_000_000+pose.header.stamp.nanosec
        for entry in self.scene_memory.snapshot():
            if color in (entry['upper'], entry['lower']):
                self.scene_memory.observe(color, pose_to_xyz(pose.pose), self.base_frame,
                    self.observation_side, stamp, self.vision_snapshot().get(entry['upper']),
                    scene=self.vision_snapshot())
                break

    def on_object_picked(self, color):
        self.scene_memory.moved(color)

    def on_object_placed(self, color, center, destination):
        if color not in ALLOWED_COLORS:
            return
        self._last_moved_color = color
        self._last_support_color = destination if destination in ALLOWED_COLORS else None
        self.scene_memory.moved(color)
        if destination in ALLOWED_COLORS:
            self.scene_memory.record_stack(color, destination, center, self.obj_sz, self.stack_clearance,
                self.base_frame, self.observation_side, self.get_clock().now().nanoseconds)
            self.log_stage('SCENE_MEMORY_STACK', upper=color, lower=destination, center=list(center))

    def vision_snapshot(self):
        snapshot = super().vision_snapshot()
        if time.monotonic()-self._details_arrival > 5.0:
            return snapshot
        labels = {'no_top_face': '未识别到完整顶面；若被压住，请先移走上层方块',
                  'missing': '当前未检测到；摆回可见位置后可再次抓取',
                  'out_of_workspace': '当前方块超出抓取范围',
                  'height_out_of_range': '当前方块高度超出三层范围'}
        for color in ALLOWED_COLORS:
            reason = self._detection_details.get(color)
            if not snapshot[color]['ready'] and reason in labels:
                snapshot[color].update(status=reason, status_zh=labels[reason])
                # Old coordinates are retained only as history, never shown as
                # the live location of a block that has been moved or covered.
                snapshot[color]['last_xyz'] = snapshot[color]['xyz']
                snapshot[color]['xyz'] = None
        box_details = self._detection_details.get('box')
        if isinstance(box_details, dict):
            for key in ('rim_z_m', 'tilt_deg', 'plane_residual_m'):
                value = box_details.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    snapshot['box'][key] = value
            if not snapshot['box']['ready'] and box_details.get('status') == 'no_valid_depth':
                snapshot['box'].update(status='no_valid_depth', status_zh='未获得有效盒沿深度')
                snapshot['box']['last_xyz'] = snapshot['box']['xyz']
                snapshot['box']['xyz'] = None
        return snapshot

    def at_observation(self):
        return (not self.servo_readiness_fault() and len(self.ready_joints) == 6
                and all(abs(wrap_pi(a - b)) <= self.observation_tolerance for a, b in zip(self.q, self.ready_joints)))

    def _reserve(self):
        with self._busy_lock:
            if self._busy or self._restore_pending:
                return False
            self._busy = True
            self._cancel_requested = False
        self._publish_status()
        return True

    def _finish(self, ok, prepared=None, stage=None):
        # Hold ownership until the callback finishes, including terminal stages.
        with self._busy_lock:
            if prepared is not None:
                self._prepared = bool(prepared and not self.plan_only)
            ready = self._prepared and self.at_observation()
            self._stage = stage if ok and stage else (('WAITING_PLACE' if self.placement_ready() else 'READY')
                                                      if ok and ready else ('SUCCESS' if ok else 'FAILED'))
            self._busy = False
        self._publish_status()

    def _on_stage(self, stage):
        # PickPlaceNode reports SUCCESS before its callback returns. Do not
        # release the operation lock from a status notification.
        self._stage = stage
        self._publish_status()

    def _publish_status(self):
        with self._busy_lock:
            busy, prepared, stage = self._busy or self._restore_pending, self._prepared, self._stage
        ready = prepared and not busy and self.at_observation()
        self.stage_pub.publish(String(data=stage))
        self.busy_pub.publish(Bool(data=busy))
        self.prepared_pub.publish(Bool(data=ready))
        self.observation_side_pub.publish(String(data=self.observation_side))

    def state_callback(self, _request, response):
        with self._busy_lock:
            busy = self._busy or self._restore_pending
            prepared = self._prepared
        response.success = True
        response.message = json.dumps({
            'busy': busy, 'prepared': prepared and not busy and self.at_observation(),
            'holding': self._holding, 'stage': self._stage, 'plan_only': self.plan_only,
            'holding_color': self._held_color if self._holding else None,
            'last_moved_color': self._last_moved_color,
            'last_support_color': self._last_support_color,
            'sequence_step': self._sequence_step,
            'sequence_total': self._sequence_total,
            'holding_label': {'red':'红块','white':'白块','orange':'橙块'}.get(self._held_color) if self._holding else None,
            'placement_ready': self.placement_ready(),
            'held_grasp': self.held_grasp_snapshot(),
            'initialized': prepared,
            'at_observation': self.at_observation(),
            'can_accept': prepared and not busy and not self.servo_readiness_fault(),
            'observation_side': self.observation_side,
            'observation_label': '镜像侧' if self.observation_side == 'mirror' else '当前侧',
            'arm_fault': self.servo_readiness_fault(),
            'joint_feedback_advisory': getattr(self, '_servo_feedback_advisory', ''),
            'arm_health': self._motor_health,
            'vision': self.vision_snapshot(),
            'clearing': self._clearing_details(),
            'buffer_diagnostics': getattr(self, '_buffer_diagnostics', {}),
            'scene_memory': self.scene_memory.snapshot(),
        }, ensure_ascii=False)
        return response

    def _wait_observation(self, require_image=True, target=None):
        """Verify measured pose and a fresh exposure; no colored target required."""
        deadline = time.monotonic() + self._prepare_timeout
        anchor = None
        settled = time.monotonic()
        exposure_after = self.get_clock().now().nanoseconds
        while time.monotonic() < deadline:
            if self._cancelled():
                return False
            now = time.monotonic()
            at_target = (self.at_observation() if target is None else
                         not self.servo_readiness_fault() and
                         all(abs(wrap_pi(a-b)) <= self.observation_tolerance for a, b in zip(self.q, target)))
            if not at_target:
                anchor = None
                settled = now
                exposure_after = self.get_clock().now().nanoseconds
                time.sleep(0.02)
                continue
            q = list(self.q)
            moving = self.dq is not None and any(not math.isfinite(v) or abs(v) > 0.02 for v in self.dq)
            if anchor is None or moving or any(abs(wrap_pi(a-b)) > 0.003 for a, b in zip(q, anchor)):
                anchor = q
                settled = now
                exposure_after = self.get_clock().now().nanoseconds
            age = (self.get_clock().now().nanoseconds - self._image_stamp) * 1e-9
            image_ok = (self._image_stamp > exposure_after and now - self._image_arrival <= 0.5
                        and -0.05 <= age <= 0.5)
            if now - settled >= self.reobserve_settle and (not require_image or image_ok):
                return True
            time.sleep(0.02)
        return False

    def home_callback(self, _request, response):
        if not self._reserve():
            response.success, response.message = False, '机械臂正在执行任务，请稍后'
            return response
        ok = False
        previous_prepared = self._prepared
        try:
            if not self.wait_ready() or self.servo_readiness_fault():
                response.message = '机械臂反馈或规划服务未就绪'
            elif self.count_publishers('/arm_cmd') != 1 and not self.plan_only:
                response.message = '存在重复或缺失的机械臂控制来源，请关闭手动控制窗口'
            else:
                self.last_joints = None
                self.publish_box('table', [self.table_x, self.table_y, self.table_z],
                                 [self.table_sx, self.table_sy, self.table_sz])
                self.report_stage('GO_HOME')
                if self._cancelled():
                    response.message = '归位已取消'
                elif not self.move_joints(self.home_joints, 'GO_HOME', self.transit_vel, self.transit_acc):
                    response.message = '返回原始收起姿态的规划或执行失败'
                elif self.plan_only:
                    ok, response.message = True, '原始收起姿态规划通过；仅规划模式，机械臂未移动'
                elif not self._wait_observation(require_image=False, target=self.home_joints):
                    response.message = '未确认机械臂在原始收起姿态停稳'
                else:
                    ok, response.message = True, '已返回原始收起姿态；下次抓取请先说“准备”'
            response.success = ok
        except Exception as exc:
            response.success, response.message = False, f'归位失败：{exc}'
        finally:
            self._finish(ok, prepared=False if ok else previous_prepared, stage='HOME')
        return response

    def prepare_callback(self, _request, response):
        return self.prepare_observation(_request, response, self.observation_side)

    def prepare_current_callback(self, _request, response):
        return self.prepare_observation(_request, response, 'current')

    def prepare_mirror_callback(self, _request, response):
        return self.prepare_observation(_request, response, 'mirror')

    def select_observation_side(self, side):
        if side not in ('current', 'mirror'):
            raise ValueError('invalid observation side')
        changed = side != self.observation_side
        self.observation_side = side
        self.ready_joints = list(self.mirror_ready_joints if side == 'mirror' else self.current_ready_joints)
        self.pre_grasp_ready_joints = (mirrored_seed(self.current_pre_grasp_joints, self.home_joints)
                                      if side == 'mirror' else list(self.current_pre_grasp_joints))
        self.table_y = -self.current_table_y if side == 'mirror' else self.current_table_y
        if changed:
            self.cube_poses = {color: None for color in self.allowed_colors}
            self.cube_pose_arrivals = {color: 0.0 for color in self.allowed_colors}
            self.cube_stable = {color: False for color in self.allowed_colors}
            self.vision_pose = self.box_pose = None
            self.vision_stable = self.box_stable = False
            self._session_box_xyz = None
            self._detection_details = {}
        self.observation_side_pub.publish(String(data=side))

    def move_initial_observation(self, observe_target=True):
        # Retain a table obstacle on each side while planning the side change.
        self.publish_box('table_mirror', [self.table_x, -self.table_y, self.table_z],
                         [self.table_sx, self.table_sy, self.table_sz])
        if crosses_home_plane(self.q, self.ready_joints, self.home_joints):
            self.publish_box('table', [self.table_x, self.table_y, self.table_z],
                             [self.table_sx, self.table_sy, self.table_sz])
            if not self.move_joints(self.home_joints, 'OBSERVATION_SIDE_HOME', self.transit_vel, self.transit_acc):
                return False
        return super().move_initial_observation(observe_target=observe_target)

    def prepare_observation(self, _request, response, side):
        if not self._reserve():
            response.success, response.message = False, '机械臂正在执行任务，请稍后'
            return response
        self._prepared = False
        ok = False
        try:
            self.select_observation_side(side)
            recovery_fault = self.recover_arm_for_prepare()
            if recovery_fault:
                response.message = recovery_fault
            elif not self.wait_ready() or self.servo_readiness_fault():
                response.message = '机械臂反馈或规划服务未就绪'
            elif not self.handeye_tf_ok():
                response.message = '腕部相机坐标变换未就绪'
            elif not self.move_initial_observation(observe_target=False):
                response.message = '移动到观察位失败，请检查规划与控制日志'
            elif self.plan_only:
                ok, response.message = True, '观察位规划通过；仅规划模式，机械臂未移动'
            elif not self._wait_observation():
                response.message = '未确认观察位停稳及新相机画面，请重新准备'
            else:
                label = '镜像侧' if self.observation_side == 'mirror' else '当前侧'
                ok, response.message = True, f'已到{label}观察位，可以说“抓红色”“抓白色”或“抓橙色”'
            response.success = ok
        except Exception as exc:
            self.get_logger().error(f'prepare failed: {exc}')
            response.success, response.message = False, f'准备失败：{exc}'
        finally:
            self._finish(ok, prepared=ok)
        return response

    def recover_arm_for_prepare(self):
        # A user-requested prepare is explicit recovery; normal pick goals never
        # clear motor protection or resume a pre-outage trajectory.
        if self.plan_only or not self._arm_recover.service_is_ready():
            return ''  # Original vendor driver has no recovery service.
        future = self._arm_recover.call_async(Trigger.Request())
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            return '保护复位服务未响应，请检查驱动'
        reply = future.result()
        if not reply.success:
            return '保护尚未复位：' + reply.message
        deadline = time.monotonic() + 1.0
        while self._motor_health is not None and not self._motor_health.get('ready', False):
            if time.monotonic() >= deadline:
                return '保护复位后尚未收到新的电机就绪状态'
            time.sleep(0.01)
        return ''

    def _on_motor_health(self, msg):
        try:
            health = json.loads(msg.data)
            if not isinstance(health, dict) or not all(
                isinstance(health.get(k), bool) for k in ('online', 'protected', 'ready')
            ):
                raise ValueError('invalid health')
        except (ValueError, TypeError):
            health = {'online': False, 'protected': True, 'ready': False}
        self._motor_health = health
        self._motor_health_arrival = time.monotonic()

    def servo_readiness_fault(self):
        fault = super().servo_readiness_fault()
        if fault:
            return fault
        health = getattr(self, '_motor_health', None)
        if health is None:
            return ''  # Compatibility with the original vendor driver.
        if time.monotonic() - self._motor_health_arrival > .5:
            return '电机反馈健康状态已过期'
        if not health.get('online', False):
            return '电机反馈已中断'
        if health.get('protected', True):
            return '电机保护尚未复位，请先取消任务再说准备'
        if not health.get('ready', False):
            return '电机反馈正在恢复'
        return ''

    def goal_callback(self, request):
        color = str(request.target_color or '').strip().lower()
        destination = str(request.destination_color or 'box').strip().lower()
        colors = list(request.target_colors)
        invalid_batch = bool(colors) and (not 2 <= len(colors) <= 3 or len(set(colors)) != len(colors)
                          or any(c not in ALLOWED_COLORS for c in colors)
                          or color != colors[0] or destination != 'box')
        with self._busy_lock:
            if (invalid_batch or color not in ALLOWED_COLORS or request.skip_return_home or request.reuse_frozen_box
                    or destination not in ('box', 'hold', *ALLOWED_COLORS) or destination == color
                    or self._busy or self._restore_pending or not self._prepared or self.servo_readiness_fault()):
                self.get_logger().warn('rejecting grasp: requires initial 准备 and fresh feedback')
                return GoalResponse.REJECT
            self._busy = True
            self._cancel_requested = False
        self._publish_status()
        return GoalResponse.ACCEPT

    def execute_callback(self, handle):
        result = X5aPickPlace.Result()
        ok = False
        prepared = False
        completed = []
        try:
            # Recheck after acceptance: manual movement/disconnected feedback
            # must not turn a stale READY latch into a grasp.
            if handle.is_cancel_requested or self._cancelled():
                message = '任务已取消'
            elif self.servo_readiness_fault():
                message = '机械臂反馈不可用，请先恢复反馈'
            else:
                observation_ok = True
                if not self.at_observation():
                    observation_ok = self.move_initial_observation(observe_target=False) and self._wait_observation()
                if not observation_ok:
                    message = '自动返回观察位失败，请检查机械臂状态后重试'
                else:
                    destination = str(handle.request.destination_color or 'box').strip().lower()
                    colors = list(handle.request.target_colors) or [str(handle.request.target_color)]
                    completed = []
                    names = {'red': '红块', 'white': '白块', 'orange': '橙块'}
                    for index, color in enumerate(colors):
                        if handle.is_cancel_requested or self._cancelled():
                            ok, message = False, '任务已取消'
                            break
                        if len(colors) > 1:
                            feedback = X5aPickPlace.Feedback()
                            feedback.stage = f'队列 {index+1}/{len(colors)}：{names[color]}放进盒子'
                            handle.publish_feedback(feedback)
                            self.report_stage(feedback.stage)
                        ok, message = self.execute_color_step(color, destination)
                        if ok:
                            completed.append(names[color])
                            if not self.plan_only and not self._wait_observation(require_image=False):
                                ok, message = False, '已放置，但未确认返回观察位；下一条抓取指令会先尝试返回观察位'
                        if not ok:
                            if len(colors) > 1:
                                done = '、'.join(completed) or '无'
                                message = f'队列停止于第 {index+1}/{len(colors)} 项（{names[color]}）；已放置：{done}。{message}'
                            break
                    if ok and len(colors) > 1:
                        message = '已依次把' + '、'.join(completed) + '放进盒子'
            if handle.is_cancel_requested or self._cancelled():
                ok, prepared = False, False
                message = '任务已取消' + ('；已放置：' + '、'.join(completed) if completed else '')
                handle.canceled()
            elif ok:
                handle.succeed()
            else:
                handle.abort()
            result.success, result.message = ok, message
        except Exception as exc:
            ok = False
            result.success, result.message = False, f'抓放失败：{exc}'
            handle.abort()
        finally:
            self.log_stage('TASK_RESULT', success=bool(result.success), message=result.message)
            self._finish(ok)
        return result

    def stack_destination_fault(self, destination, source=None):
        if destination not in ALLOWED_COLORS:
            return ''
        blocker = self._resolve_blocker(destination)
        if source is not None and blocker is not None and blocker == self._resolve_blocker(source):
            return ''  # Clearing the requested source also uncovers this support.
        if ((blocker is not None and blocker != source)
                or (self.scene_memory.has_lower(destination) and blocker is None)):
            return '放置目标上已有方块或支撑关系尚未确认，请先移开上层方块'
        return ''

    def execute_color_step(self, color, destination):
        """An explicit name of the held cube means place, never regrasp it."""
        if self._holding:
            if not self.placement_ready():
                return False, '持物颜色或夹持几何未确认，请先检查当前夹爪状态'
            if color != self._held_color:
                return False, '当前夹着另一颜色，请先放置当前方块'
            if destination == 'hold':
                return True, '当前已夹持指定方块，等待放置指令'
            return self.execute_held_placement(destination)
        fault = self.stack_destination_fault(destination, source=color)
        if fault:
            return False, fault
        kwargs = ({'pick_only': True} if destination == 'hold' else
                  {'destination_color': destination} if destination != 'box' else {})
        ok, message = self.clear_target_if_needed(color)
        if not ok:
            return False, message
        fault = self.stack_destination_fault(destination, source=color)
        return (False, fault) if fault else self.execute_pick_place(color, **kwargs)

    @staticmethod
    def sequence_tasks(request):
        if len(request.source_colors) != len(request.destination_colors):
            raise ValueError('连续指令的来源和目标数量不一致')
        tasks = [{'action': 'pick_place', 'target_color': source, 'destination_color': destination}
                 for source, destination in zip(request.source_colors, request.destination_colors)]
        if len(tasks) < 2:
            raise ValueError('连续指令至少需要两步')
        return tasks

    def sequence_goal_callback(self, request):
        try:
            color_step_pairs(self.sequence_tasks(request), self._held_color if self._holding else None)
        except ValueError as exc:
            self.get_logger().warn('rejecting colour sequence: '+str(exc))
            return GoalResponse.REJECT
        with self._busy_lock:
            if (self._busy or self._restore_pending or not self._prepared or self.servo_readiness_fault()
                    or (self._holding and not self.placement_ready())):
                return GoalResponse.REJECT
            self._busy, self._cancel_requested = True, False
        self._publish_status()
        return GoalResponse.ACCEPT

    def sequence_execute_callback(self, handle):
        result = ColorSequence.Result()
        ok, message = False, '连续指令未执行'
        try:
            pairs = color_step_pairs(self.sequence_tasks(handle.request), self._held_color if self._holding else None)
            self._sequence_total = len(pairs)
            names = {'red': '红块', 'white': '白块', 'orange': '橙块', 'box': '盒子', 'hold': '夹持等待'}
            for index, (source, destination) in enumerate(pairs, 1):
                self._sequence_step = index
                if handle.is_cancel_requested or self._cancelled():
                    ok, message = False, '任务已取消'
                    break
                if self.servo_readiness_fault():
                    ok, message = False, self.servo_readiness_fault()
                    break
                feedback = ColorSequence.Feedback(step_index=index, step_total=len(pairs),
                    stage=f'第 {index}/{len(pairs)} 步：{names[source]} → {names[destination]}')
                handle.publish_feedback(feedback)
                self.report_stage(feedback.stage)
                if not self.at_observation() and not (self.move_initial_observation(observe_target=False) and self._wait_observation()):
                    ok, message = False, '返回观察位失败'
                    break
                ok, message = self.execute_color_step(source, destination)
                if not ok:
                    break
                result.completed_steps = index
                if not self.plan_only and not self._wait_observation(require_image=index < len(pairs)):
                    ok, message = False, '本步已完成，但未确认观察位或新画面'
                    break
            if handle.is_cancel_requested or self._cancelled():
                ok, message = False, '任务已取消'
                handle.canceled()
            elif ok:
                message = f'连续 {len(pairs)} 步全部完成'
                handle.succeed()
            else:
                handle.abort()
            if not ok:
                message = f'队列停止于第 {self._sequence_step}/{len(pairs)} 步；已完成 {result.completed_steps} 步。{message}'
            result.success, result.message = ok, message
        except Exception as exc:
            ok = False
            result.success, result.message = False, f'连续指令失败：{exc}'
            handle.abort()
        finally:
            self.log_stage('SEQUENCE_RESULT', success=bool(result.success), completed=result.completed_steps, message=result.message)
            self._sequence_step = self._sequence_total = 0
            self._finish(ok)
        return result

    def _forget_held_grasp(self):
        self._held_color = self._held_grasp_rpy = self._held_cube_yaw = self._held_source_top = None
        self._held_object_offset = None

    def placement_ready(self):
        vectors = (self._held_grasp_rpy, self._held_source_top, self._held_object_offset)
        return bool(self._holding and self._held_color in ALLOWED_COLORS
                    and all(v is not None and len(v) == 3 and all(math.isfinite(x) for x in v) for v in vectors)
                    and isinstance(self._held_cube_yaw, (int, float)) and math.isfinite(self._held_cube_yaw))

    def held_grasp_snapshot(self):
        if not self.placement_ready():
            return None
        return {'color':self._held_color, 'grasp_rpy':list(self._held_grasp_rpy),
                'cube_yaw':self._held_cube_yaw, 'source_top':list(self._held_source_top),
                'offset':list(self._held_object_offset)}

    def wait_for_placement_target(self, destination):
        deadline = time.monotonic() + self.vision_wait_timeout
        while time.monotonic() < deadline:
            if self._cancelled():
                return None
            item = self.vision_snapshot()[destination]
            if item['ready']:
                pose = self.box_pose if destination == 'box' else self.cube_poses[destination]
                return tuple(item['xyz']), quat_to_yaw(pose.pose.orientation)
            time.sleep(.05)
        return None

    def execute_held_placement(self, destination):
        if not self.placement_ready():
            return False, '当前没有已定位夹持的方块，请先说“抓白色方块”等抓取指令'
        if destination == self._held_color:
            return False, '放置目标必须是盒子或另一种颜色的方块'
        fault = self.stack_destination_fault(destination)
        if fault:
            return False, fault+'；夹持状态保留'
        self.report_stage('WAITING_VISION')
        target = self.wait_for_placement_target(destination)
        if target is None:
            label = {'box':'盒子','red':'红块','white':'白块','orange':'橙块'}[destination]
            reason = '任务已取消' if self._cancelled() else f'{label}{self.vision_snapshot()[destination]["status_zh"]}'
            return False, reason + '；夹持状态保留，可以稍后重试放置'
        xyz, yaw = target
        center = None
        if destination == 'box':
            self.box_yaw = yaw if self.box_use_pose_yaw else self.box_yaw
            self.setup_box_walls(xyz[0], xyz[1])
            place_xyz = (xyz[0], xyz[1], self.place_z)
            orientations = self.place_orientation_candidates(xyz[0], xyz[1])
        else:
            center = held_stack_center(xyz, (self.obj_sx,self.obj_sy,self.obj_sz),
                                       self.table_z + .5*self.table_sz, self.stack_clearance)
            orientations = aligned_stack_candidates(self._held_grasp_rpy, self._held_cube_yaw, yaw)
            place_xyz = tcp_for_object_center(center, self._held_object_offset, orientations[0])
            self.publish_box('cube_obstacle_' + destination,
                             [xyz[0],xyz[1],xyz[2]-.5*self.obj_sz],
                             [self.obj_sx,self.obj_sy,self.obj_sz], yaw=yaw)
        previous_destination = self._placement_destination
        self._placement_destination = destination if destination in ALLOWED_COLORS else None
        try:
            return self.execute_place_motion(place_xyz, orientations, center)
        finally:
            self._placement_destination = previous_destination

    def place_callback(self, _request, response, destination):
        if not self._reserve():
            response.success, response.message = False, '机械臂正在执行任务，请稍后'
            return response
        ok = False
        try:
            if not self.placement_ready():
                response.message = '当前没有已定位夹持的方块，请先抓取方块'
            elif destination == self._held_color:
                response.message = '放置目标必须是盒子或另一种颜色的方块'
            elif not self._prepared:
                response.message = '请先说“准备”，回到观察位后再放置'
            elif self.servo_readiness_fault() or not self.wait_ready():
                response.message = self.servo_readiness_fault() or '机械臂反馈或规划服务未就绪'
            elif not self.at_observation() and not (self.move_initial_observation(observe_target=False) and self._wait_observation()):
                response.message = '返回观察位失败；夹持状态保留'
            else:
                ok, response.message = self.execute_held_placement(destination)
                if ok and not self.plan_only and not self._wait_observation(require_image=False):
                    ok, response.message = False, '已放置，但未确认返回观察位'
            response.success = ok
        except Exception as exc:
            response.success, response.message = False, f'放置失败：{exc}'
        finally:
            self._finish(ok)
        return response

    def publish_box(self, name, xyz, size, frame=None, operation=CollisionObject.ADD, yaw=0.0):
        if name in ('table', 'table_mirror') and operation == CollisionObject.ADD:
            xyz = [xyz[0], xyz[1], xyz[2] + self.table_collision_offset]
        environment_object = (name in ('object', 'source_support')
                              or name.startswith(('cube_obstacle_', 'box_wall_')))
        if environment_object and not self.scene_objects_collision_enabled and operation == CollisionObject.ADD:
            return
        return super().publish_box(name, xyz, size, frame=frame, operation=operation, yaw=yaw)

    def set_preflight_target_present(self, present):
        # With environment models disabled, neither temporary removal nor
        # restoration is needed. MoveIt rejects removing a nonexistent object,
        # which otherwise aborts the approach before any IK is attempted.
        if not self.scene_objects_collision_enabled:
            return True
        return super().set_preflight_target_present(present)

    def execute_moveit_trajectory(self, trajectory, stage):
        # A rejected/aborted execution can have moved part of the path. Unwind
        # all pose/offset candidates instead of treating it as an IK failure.
        ok = super().execute_moveit_trajectory(trajectory, stage)
        if not ok:
            fault = self.servo_readiness_fault() or '轨迹执行失败或超时'
            raise RuntimeError(f'{stage}执行中断：{fault}；本次任务已停止')
        return True

    def call_gripper(self, opening):
        # Retain possible holding as information after a close timeout;
        # it does not gate prepare, home or the next user-requested task.
        if not opening and not self.plan_only:
            self._holding = True
        ok = super().call_gripper(opening)
        if ok and opening:
            self._holding = False
            self._forget_held_grasp()
        return ok

    def reset_task_state(self, color):
        cancelled = self._cancel_requested
        super().reset_task_state(color, preserve_live_vision=True)
        if cancelled:
            self._cancel_requested = True

    def cancel_callback(self, _handle):
        self.request_cancel()
        return CancelResponse.ACCEPT

    def cancel_service(self, _request, response):
        with self._busy_lock:
            busy = self._busy
            if busy:
                self.request_cancel()
        response.success = busy
        response.message = '已请求取消，将在当前动作结束后停止' if busy else '当前没有运行中的任务'
        return response

    def release_callback(self, _request, response):
        if not self._reserve():
            response.success, response.message = False, '机械臂正在执行任务，请稍后'
            return response
        ok = False
        try:
            ok = not self.servo_readiness_fault() and self.call_gripper(True)
            if ok:
                from moveit_msgs.msg import AttachedCollisionObject
                attached = AttachedCollisionObject()
                attached.object.id = 'object'
                attached.object.operation = CollisionObject.REMOVE
                scene = PlanningScene()
                scene.is_diff = scene.robot_state.is_diff = True
                scene.robot_state.attached_collision_objects = [attached]
                self.scene_pub.publish(scene)
            response.success = ok
            response.message = '夹爪已松开，可以继续发抓取指令' if ok and self._prepared else ('夹爪已松开，请先说“准备”' if ok else '松开夹爪失败')
        except Exception as exc:
            response.success, response.message = False, str(exc)
        finally:
            self._finish(ok)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = WristTaskServer()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
