"""ROS front door for wrist color tasks; never publishes robot commands."""
import json
import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from x5a_task_interfaces.action import X5aPickPlace, ColorSequence
from action_msgs.msg import GoalStatus
from x5a_pick_place.color_sequence import color_step_pairs


class WristBridge:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self._prepared = False
        self._held_color = None
        self._last_moved_color = self._last_support_color = None
        self._server_busy = False
        self._lock = threading.Lock()
        self._active = False
        self._pending = None
        self._goal = None
        self._stage_cb = None
        self._node = None
        self._owned = False
        self._cancel_event = threading.Event()

    def start(self):
        if not rclpy.ok():
            rclpy.init()
            self._owned = True
        self._node = Node('wrist_voice_agent')
        cb = ReentrantCallbackGroup()
        self._client = ActionClient(self._node, X5aPickPlace, '/wrist_pick/pick_place', callback_group=cb)
        self._sequence_client = ActionClient(self._node, ColorSequence, '/wrist_pick/color_sequence', callback_group=cb)
        self._services = {name: self._node.create_client(Trigger, '/wrist_pick/' + name, callback_group=cb)
                          for name in ('prepare', 'prepare_current', 'prepare_mirror', 'go_home', 'state', 'cancel', 'release',
                                       'place_box', 'place_red', 'place_white', 'place_orange')}
        self._node.create_subscription(String, '/wrist_pick/task_stage', self._on_stage, 10)
        self._node.create_subscription(Bool, '/wrist_pick/task_busy', self._on_busy, 10)
        self._node.create_subscription(Bool, '/wrist_pick/prepared', self._on_prepared, 10)
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def set_stage_callback(self, callback):
        self._stage_cb = callback

    def _on_stage(self, msg):
        if self._stage_cb:
            self._stage_cb(msg.data)

    def _on_busy(self, msg):
        self._server_busy = bool(msg.data)

    def _on_prepared(self, msg):
        self._prepared = bool(msg.data)

    @property
    def busy(self):
        with self._lock:
            return self._active or self._server_busy or (self._pending is not None and not self._pending.done())

    def server_ready(self, timeout=0.2):
        return self.dry_run or (self._node is not None and self._client.wait_for_server(timeout_sec=timeout)
                                and self._services['prepare'].wait_for_service(timeout_sec=timeout))

    @staticmethod
    def _wait(future, timeout):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            raise TimeoutError('响应超时；任务可能仍在执行，请查看状态或请求取消')
        return future.result()

    def _call(self, name, timeout):
        client = self._services[name]
        if not client.wait_for_service(timeout_sec=1.0):
            return {'success': False, 'message': '腕部服务未就绪：' + name}
        future = client.call_async(Trigger.Request())
        if name not in ('state', 'cancel'):
            self._pending = future
        response = self._wait(future, timeout)
        return {'success': bool(response.success), 'message': response.message}

    def state(self):
        if self.dry_run:
            return {'server_ready': True, 'prepared': self._prepared, 'initialized':self._prepared,
                    'busy': self.busy, 'holding': self._held_color is not None,
                    'holding_color':self._held_color, 'placement_ready':self._held_color is not None,
                    'last_moved_color':self._last_moved_color, 'last_support_color':self._last_support_color}
        try:
            reply = self._call('state', 1.0)
            if not reply['success']:
                raise RuntimeError(reply['message'])
            state = json.loads(reply['message'])
            state['server_ready'] = self.server_ready(0.05)
            state['busy'] = bool(state.get('busy') or self.busy)
            return state
        except Exception:
            return {'server_ready': False, 'prepared': False, 'busy': self.busy, 'holding': None}

    def _operate(self, operation):
        with self._lock:
            if self._active or self._server_busy or (self._pending is not None and not self._pending.done()):
                return {'success': False, 'message': '机器人正在执行上一任务，请稍后'}
            self._active = True
        try:
            return operation()
        except Exception as exc:
            return {'success': False, 'message': str(exc)}
        finally:
            with self._lock:
                self._active = False

    def prepare(self, observation_side=''):
        if observation_side not in ('', 'current', 'mirror'):
            return {'success': False, 'message': '观察方向无效'}
        def run():
            if self.dry_run:
                self._prepared = True
                return {'success': True, 'message': '演练：准备指令已解析，未发送运动指令'}
            service = 'prepare_' + observation_side if observation_side else 'prepare'
            return self._call(service, 180.0)
        return self._operate(run)

    def release(self):
        def run():
            if self.dry_run:
                self._held_color = None
                return {'success': True, 'message': '演练：松开夹爪，未发送运动指令'}
            return self._call('release', 30.0)
        return self._operate(run)

    def go_home(self):
        def run():
            if self.dry_run:
                self._prepared = False
                return {'success': True, 'message': '演练：返回原始收起姿态，未发送运动指令'}
            return self._call('go_home', 180.0)
        return self._operate(run)

    def send_pick_place(self, color, destination_color='box', target_colors=None):
        colors = list(target_colors or [])
        if colors and (not 2 <= len(colors) <= 3 or len(set(colors)) != len(colors)
                       or any(c not in ('red', 'white', 'orange') for c in colors)
                       or destination_color != 'box' or color != colors[0]):
            return {'success': False, 'message': '队列只支持两到三种不同颜色，依次放进盒子'}
        if color not in ('red', 'white', 'orange'):
            return {'success': False, 'message': '只支持红、白、橙方块'}
        if destination_color not in ('box', 'hold', 'red', 'white', 'orange') or destination_color == color:
            return {'success': False, 'message': '放置目标必须是盒子或另一种颜色的方块'}
        def run():
            if self.dry_run:
                return self._dry_color_step(color, destination_color)
            if not self._client.wait_for_server(timeout_sec=1.0):
                return {'success': False, 'message': '腕部抓放服务未就绪'}
            goal = X5aPickPlace.Goal()
            goal.target_color = color
            goal.destination_color = destination_color
            goal.target_colors = colors
            goal.skip_return_home = False
            goal.reuse_frozen_box = False
            future = self._client.send_goal_async(goal)
            self._pending = future
            try:
                self._goal = self._wait(future, 10.0)
            except TimeoutError:
                # A late acceptance must not start an unattended new task.
                future.add_done_callback(lambda f: f.result().cancel_goal_async()
                                         if f.result() and f.result().accepted else None)
                raise
            if not self._goal or not self._goal.accepted:
                return {'success': False, 'message': '任务被拒绝：需完成首次准备，并确认机械臂反馈可用'}
            self._pending = self._goal.get_result_async()
            # A covered target adds one complete pick/table-place cycle.
            try:
                wrapped = self._wait(self._pending, 720.0 * max(1, len(colors)))
            except TimeoutError:
                self._goal.cancel_goal_async()
                raise
            self._goal = None
            return {'success': bool(wrapped.result.success), 'message': wrapped.result.message}
        return self._operate(run)

    def send_sequence(self, colors):
        # One ROS goal holds robot ownership across every item and cancellation.
        return self.send_pick_place(colors[0], target_colors=colors)

    def _dry_color_step(self, color, destination):
        if not self._prepared:
            return {'success': False, 'message': '请先说“准备”'}
        if self._held_color and self._held_color != color:
            return {'success': False, 'message': '当前夹着另一颜色，请先放置当前方块'}
        self._held_color = color if destination == 'hold' else None
        if destination != 'hold':
            self._last_moved_color = color
            self._last_support_color = destination if destination in ('red', 'white', 'orange') else None
        return {'success': True, 'message': '演练：颜色抓放，未发送运动指令'}

    def send_task_sequence(self, tasks, cancel_event=None):
        # The web owner creates this before scheduling its worker: cancellation
        # during startup must not be erased by a late worker.
        cancelled = cancel_event if cancel_event is not None else threading.Event()
        def run():
            self._cancel_event = cancelled
            if cancelled.is_set():
                return {'success': False, 'message': '连续任务已取消', 'completed_steps': 0}
            snapshot = self.state()
            if snapshot.get('holding') and not snapshot.get('holding_color'):
                return {'success': False, 'message': '持物颜色未确认，未开始连续任务'}
            pairs = color_step_pairs(tasks, snapshot.get('holding_color') if snapshot.get('holding') else None)
            if len(pairs) < 2:
                return {'success': False, 'message': '连续指令至少需要两步'}
            if cancelled.is_set():
                return {'success': False, 'message': '连续任务已取消', 'completed_steps': 0}
            if self.dry_run:
                completed = 0
                for source, destination in pairs:
                    if cancelled.is_set():
                        return {'success': False, 'message': '连续任务已取消', 'completed_steps': completed}
                    result = self._dry_color_step(source, destination)
                    if not result['success']:
                        return {**result, 'completed_steps': completed}
                    completed += 1
                return {'success': True, 'message': f'演练：连续 {completed} 步完成，未发送运动指令', 'completed_steps': completed}
            if not self._sequence_client.wait_for_server(timeout_sec=1.):
                return {'success': False, 'message': '连续抓放服务未就绪，请重新启动本工作区以加载新接口'}
            if cancelled.is_set():
                return {'success': False, 'message': '连续任务已取消', 'completed_steps': 0}
            goal = ColorSequence.Goal(source_colors=[s for s, d in pairs], destination_colors=[d for s, d in pairs])
            future = self._sequence_client.send_goal_async(goal)
            self._pending = future
            try:
                self._goal = self._wait(future, 10.)
            except TimeoutError:
                future.add_done_callback(lambda f: f.result().cancel_goal_async() if f.result() and f.result().accepted else None)
                raise
            if not self._goal or not self._goal.accepted:
                return {'success': False, 'message': '连续任务被拒绝：请确认准备、持物和机械臂反馈状态'}
            if cancelled.is_set():
                self._goal.cancel_goal_async()
            self._pending = self._goal.get_result_async()
            try:
                wrapped = self._wait(self._pending, 720.*len(pairs))
            except TimeoutError:
                self._goal.cancel_goal_async()
                raise
            self._goal = None
            return {'success': wrapped.status == GoalStatus.STATUS_SUCCEEDED and bool(wrapped.result.success),
                    'message': wrapped.result.message, 'completed_steps': wrapped.result.completed_steps}
        return self._operate(run)

    def send_pick(self, color):
        return self.send_pick_place(color, destination_color='hold')

    def place(self, destination_color='box'):
        if destination_color not in ('box','red','white','orange'):
            return {'success':False,'message':'放置目标无效'}
        def run():
            if self.dry_run:
                if not self._held_color:
                    return {'success':False,'message':'当前没有夹持方块，请先抓取'}
                if self._held_color == destination_color:
                    return {'success':False,'message':'放置目标必须是另一种颜色'}
                return self._dry_color_step(self._held_color, destination_color)
            return self._call('place_' + destination_color, 240.0)
        return self._operate(run)

    def cancel(self):
        self._cancel_event.set()
        if self.dry_run:
            self._prepared = False
            return {'success': True, 'message': '演练：取消指令已解析'}
        if self._goal is not None:
            self._goal.cancel_goal_async()
        try:
            return self._call('cancel', 3.0)
        except Exception as exc:
            return {'success': False, 'message': str(exc)}

    def shutdown(self):
        if self._node is not None:
            self._executor.shutdown(timeout_sec=3.0)
            self._thread.join(timeout=3.0)
            self._node.destroy_node()
        if self._owned:
            rclpy.try_shutdown()
