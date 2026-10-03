"""Execute the latest remotely planned grasp/place on the real arm.

Subscribes to /display_planned_path (reliable, so the last plan from
x5a_remote_grasp is always received) and runs the MoveIt trajectories through
the EXISTING proven control chain:
    /x5a_arm_controller/follow_joint_trajectory  (via official adapter)
    /x5a_gripper_controller/gripper_cmd          (GripperCommand action)
This node never publishes /arm_cmd itself.

Stage names come from the segment-count table (the planner in x5a_remote_grasp
builds 3/4/6-segment plans), and only plans carrying the planner's model_id
marker are accepted: move_group also publishes its own plans on the same topic,
and those must never be executed.

Services (all Trigger, all blocking = the response carries the real outcome):
    /x5a_remote_grasp/execute        run the latest plan
    /x5a_remote_grasp/release        open the gripper in place (drop the object)
    /x5a_remote_grasp/go_home        joint-interpolate to the all-zero pose
    /x5a_remote_grasp/go_ready       joint-interpolate to poses.ready_joints
    /x5a_remote_grasp/executor_state report {enabled, running, holding, ...}

The response message is a JSON string so callers (web_agent) can read the real
stage plus whether the gripper is now holding an object.

Safety gates:
  * execution is disabled unless launched with execution.enabled:=true
  * a plan only executes on an explicit /x5a_remote_grasp/execute call
  * only plans from x5a_remote_grasp (model_id marker) are accepted
  * refuses to run when /arm_cmd publisher count != 1
  * plan must be fresher than execution.plan_max_age_s
  * a place-only plan never opens the gripper at the start (it is already holding)
"""
import json
import threading
import time
import numpy as np
from x5a_remote_grasp.place_geometry import object_in_tool
from functools import wraps
from typing import Dict, List, Tuple

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from x5a_remote_grasp.moveit_plan import DISPLAY_MODEL_ID
from x5a_remote_grasp.grasp_quality import held_from_gap, trajectory_fingerprint

# 段名由规划器的 poses 列表决定：抓取=4、抓取+放置=6、纯放置=3。
STAGE_TABLE = {
    3: ("TRANSPORT", "PLACE", "RETREAT"),
    4: ("PREGRASP", "GRASP", "LIFT", "RETREAT"),
    6: ("PREGRASP", "GRASP", "LIFT", "TRANSPORT", "PLACE", "RETREAT"),
}
GRASP_STAGES = frozenset({"PREGRASP", "GRASP"})


def guarded_service(callback):
    """服务回调里的任何异常都不该弄死执行器。

    回调异常会从 spin 里冒出来直接终止进程（曾因 time_from_start 类型写错崩过），
    而执行器一死，后续每条指令都只能报「服务未就绪」，用户完全看不出原因。
    这里统一兜住：记日志 + 返回失败响应，进程继续活着。
    """

    @wraps(callback)
    def wrapper(self, request, response):
        try:
            return callback(self, request, response)
        except Exception as exc:  # noqa: BLE001 - 保命优先
            self.get_logger().error(f"{callback.__name__} 异常: {exc!r}")
            response.success = False
            response.message = self._json({
                "stage": "FAILED",
                "message": f"执行器内部异常: {exc}",
                "holding": self.holding,
            })
            return response

    return wrapper


class RemoteGraspExecutor(Node):
    def __init__(self) -> None:
        super().__init__("x5a_remote_grasp_executor")
        self.declare_parameter("execution.enabled", False)
        self.declare_parameter("execution.plan_max_age_s", 30.0)
        self.declare_parameter('execution.grasp_position_tolerance_m', 0.007)
        self.declare_parameter('execution.grasp_angle_tolerance_rad', 0.12)
        self.declare_parameter('execution.place_position_tolerance_m', 0.003)
        self.declare_parameter('execution.place_angle_tolerance_rad', 0.05)
        self.declare_parameter('execution.min_held_gap_m', 0.004)
        self.declare_parameter("execution.open_width_m", 0.044)
        self.declare_parameter("execution.close_width_m", 0.0)
        self.declare_parameter("execution.trajectory_timeout_s", 40.0)
        # GripperCommand.position = 单指行程（URDF joint7/joint8, 0..0.044m）。
        # 与 x5a_pick_place 一致：开=0.044，合=0.0（靠堵转夹住物体）。
        self.declare_parameter("gripper.position_ratio", 1.0)
        self.declare_parameter("gripper.max_position_m", 0.044)
        self.declare_parameter("poses.ready_joints", [0.72, 0.85, 0.95, -0.55, 0.0, 0.0])
        self.declare_parameter("controllers.arm_fjt", "/x5a_arm_controller/follow_joint_trajectory")
        self.declare_parameter("controllers.gripper", "/x5a_gripper_controller/gripper_cmd")
        self.declare_parameter("topics.display_path", "/display_planned_path")
        self.enabled = bool(self.get_parameter("execution.enabled").value)
        self.plan_max_age = float(self.get_parameter("execution.plan_max_age_s").value)
        self.grasp_position_tolerance = float(self.get_parameter('execution.grasp_position_tolerance_m').value)
        self.grasp_angle_tolerance = float(self.get_parameter('execution.grasp_angle_tolerance_rad').value)
        self.place_position_tolerance = float(self.get_parameter('execution.place_position_tolerance_m').value)
        self.place_angle_tolerance = float(self.get_parameter('execution.place_angle_tolerance_rad').value)
        self.min_held_gap = float(self.get_parameter('execution.min_held_gap_m').value)
        self.open_width = float(self.get_parameter("execution.open_width_m").value)
        self.close_width = float(self.get_parameter("execution.close_width_m").value)
        self.traj_timeout = float(self.get_parameter("execution.trajectory_timeout_s").value)
        # GripperCommand.position 是单指行程 (0..0.044m，URDF joint7/joint8 limit)，
        # 不是总开口宽度——换算比例 1.0，超限会被 adapter 拒绝。
        self.gripper_ratio = float(self.get_parameter("gripper.position_ratio").value)
        self.gripper_max_pos = float(self.get_parameter("gripper.max_position_m").value)
        self.ready_joints = [float(v) for v in self.get_parameter("poses.ready_joints").value]
        fjt_name = str(self.get_parameter("controllers.arm_fjt").value)
        gripper_name = str(self.get_parameter("controllers.gripper").value)
        display_topic = str(self.get_parameter("topics.display_path").value)

        cb = ReentrantCallbackGroup()
        self.latest_plan = None
        self.latest_plan_time = 0.0
        self.latest_joints = None
        self.joints_monotonic = 0.0
        self.holding_state = 'unknown'
        self.validated_tool = None
        self.validated_place_tool = None
        self.held_geometry = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.running = False
        # 夹爪里是否有物体：抓取计划合爪后置位，放置/放下后清零。
        # agent 的会话状态以此为准（重启 agent 后可用 executor_state 重新对齐）。
        self.holding = False
        self._done = threading.Event()
        self._outcome: Dict[str, object] = {}
        plan_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            DisplayTrajectory, display_topic, self._on_plan, plan_qos, callback_group=cb
        )
        self.fjt_client = ActionClient(self, FollowJointTrajectory, fjt_name)
        self.gripper_client = ActionClient(self, GripperCommand, gripper_name)
        self.validation_client = self.create_client(Trigger, '/x5a_remote_grasp/validate_latest', callback_group=cb)
        self.execute_srv = self.create_service(
            Trigger, "/x5a_remote_grasp/execute", self._on_execute, callback_group=cb
        )
        self.release_srv = self.create_service(
            Trigger, "/x5a_remote_grasp/release", self._on_release, callback_group=cb
        )
        self.home_srv = self.create_service(
            Trigger, "/x5a_remote_grasp/go_home", self._on_go_home, callback_group=cb
        )
        self.ready_srv = self.create_service(
            Trigger, "/x5a_remote_grasp/go_ready", self._on_go_ready, callback_group=cb
        )
        self.state_srv = self.create_service(
            Trigger, "/x5a_remote_grasp/executor_state", self._on_state, callback_group=cb
        )
        self.create_subscription(
            JointState, "/joint_states", self._on_joints, 10, callback_group=cb
        )
        self.get_logger().info(
            f"executor ready enabled={self.enabled} fjt={fjt_name} gripper={gripper_name}"
        )

    # ---------- plumbing ----------
    @staticmethod
    def _json(payload: Dict[str, object]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _duration_message(seconds: float) -> DurationMsg:
        """builtin_interfaces/Duration —— 消息字段只认这个类型，不能用
        rclpy.duration.Duration（赋值断言失败会让执行器整个崩掉）。"""
        message = DurationMsg()
        seconds = max(0.0, float(seconds))
        message.sec = int(seconds)
        message.nanosec = int((seconds - int(seconds)) * 1.0e9)
        return message

    def _on_plan(self, msg: DisplayTrajectory) -> None:
        try:
            self._on_plan_impl(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"_on_plan 异常: {exc!r}")

    def _on_plan_impl(self, msg: DisplayTrajectory) -> None:
        if str(getattr(msg, "model_id", "") or "") != DISPLAY_MODEL_ID:
            # move_group 也会往同一 topic 发它自己的规划（通常 1 段），必须忽略
            self.get_logger().info(
                f"ignoring plan from model_id={msg.model_id!r} "
                f"({len(msg.trajectory)} segment(s)); not from x5a_remote_grasp"
            )
            return
        self.latest_plan = msg
        self.latest_plan_time = time.monotonic()
        self.get_logger().info(
            f"received plan with {len(msg.trajectory)} trajectory segment(s): "
            + ",".join(self._stage_names(msg))
        )

    def _on_joints(self, msg) -> None:
        self.latest_joints = (list(msg.name), list(msg.position))
        self.joints_monotonic = time.monotonic()

    def _holding_observation(self):
        if self.latest_joints is None:
            return 'unknown'
        names, positions = self.latest_joints
        try:
            gap = float(positions[names.index('joint7')]+positions[names.index('joint8')])
        except (ValueError, IndexError):
            return 'unknown'
        return held_from_gap(gap, time.monotonic()-self.joints_monotonic < 0.5, self.min_held_gap)

    def _verify_grasp_pose(self):
        if self.validated_tool is None:
            raise RuntimeError('抓取缺少经验证的 TCP 目标')
        position, angle = self.grasp_position_tolerance, self.grasp_angle_tolerance
        if self.validated_place_tool is not None:
            # A combined plan predicts the attachment before closure. Bound
            # that prediction as tightly as the subsequent release pose.
            position = min(position, self.place_position_tolerance)
            angle = min(angle, self.place_angle_tolerance)
        return self._verify_tool_pose(self.validated_tool, position, angle, '抓取')

    def _verify_place_pose(self):
        if self.validated_place_tool is None:
            raise RuntimeError('放置缺少经验证的物体底面/TCP 目标，未松爪')
        return self._verify_tool_pose(self.validated_place_tool, self.place_position_tolerance,
                                      self.place_angle_tolerance, '放置')

    def _verify_tool_pose(self, expected, position_tolerance, angle_tolerance, label):
        from x5a_remote_grasp.geometry import transform_from_xyz_quat
        target = np.asarray(expected)
        deadline = time.monotonic()+1.0
        while True:
            actual = self.tf_buffer.lookup_transform('base_link', 'tool0', Time())
            stamp = actual.header.stamp
            age = (self.get_clock().now().nanoseconds/1e9)-(stamp.sec+stamp.nanosec/1e9)
            if not -0.1 <= age <= 0.5:
                raise RuntimeError('实际 TCP 状态已过期')
            p, q = actual.transform.translation, actual.transform.rotation
            matrix = transform_from_xyz_quat([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
            position = float(np.linalg.norm(matrix[:3, 3]-target[:3, 3]))
            angle = float(np.arccos(np.clip((np.trace(matrix[:3, :3].T@target[:3, :3])-1)/2, -1, 1)))
            if position <= position_tolerance and angle <= angle_tolerance:
                return matrix
            if time.monotonic() >= deadline:
                action = '松爪' if label == '放置' else '合爪'
                raise RuntimeError(f'实际{label}点误差 {position*1000:.1f}mm/{np.degrees(angle):.1f}度，未{action}')
            time.sleep(0.1)

    @staticmethod
    def _stage_names(plan: DisplayTrajectory) -> List[str]:
        """段名由段数与规划器的 poses 列表对应关系给出（见 STAGE_TABLE）。"""
        count = len(plan.trajectory)
        names = STAGE_TABLE.get(count)
        if names is None:
            return [f"SEG{index}" for index in range(count)]
        return list(names)

    def _block_until_done(self, worker: threading.Thread, timeout: float, response):
        """阻塞到 worker 线程结束，把真实结果写回服务响应。"""
        self._done.clear()
        self._outcome = {
            "ok": False, "stage": "FAILED", "message": "未开始", "holding": self.holding
        }
        self.running = True
        worker.start()
        if not self._done.wait(timeout):
            response.success = False
            response.message = self._json({
                "stage": "TIMEOUT",
                "message": f"{timeout:.0f}s 内没有执行完",
                "holding": self.holding,
            })
            return response
        outcome = dict(self._outcome)
        response.success = bool(outcome.get("ok"))
        response.message = self._json(outcome)
        return response

    # ---------- services ----------
    @guarded_service
    def _on_execute(self, request, response):
        if not self.enabled:
            response.success = False
            response.message = self._json({
                "stage": "FAILED", "message":
                "执行未启用：请以 execution.enabled:=true 启动执行器", "holding": self.holding,
            })
            return response
        if self.running:
            response.success = False
            response.message = self._json({
                "stage": "BUSY", "message": "执行器忙碌", "holding": self.holding,
            })
            return response
        if self.latest_plan is None or (
            time.monotonic() - self.latest_plan_time > self.plan_max_age
        ):
            response.success = False
            response.message = self._json({
                "stage": "FAILED",
                "message": f"没有新鲜的规划（{self.plan_max_age:.0f} 秒内有效）",
                "holding": self.holding,
            })
            return response
        publisher_count = self.count_publishers("/arm_cmd")
        if publisher_count != 1:
            response.success = False
            response.message = self._json({
                "stage": "FAILED",
                "message": f"/arm_cmd publisher 数量为 {publisher_count}（应为 1），拒绝执行",
                "holding": self.holding,
            })
            return response
        stages = self._stage_names(self.latest_plan)
        if len(self.latest_plan.trajectory) not in STAGE_TABLE:
            response.success = False
            response.message = self._json({
                "stage": "FAILED",
                "message": f"规划段数 {len(self.latest_plan.trajectory)} 不被支持"
                           f"（应为 3/4/6），拒绝执行",
                "holding": self.holding,
            })
            return response
        timeout = self.traj_timeout * max(1, len(stages)) + 30.0
        plan = self.latest_plan
        # Run preflight inside the worker (running is set first), so another
        # execute/release cannot interleave with the perception check.
        return self._block_until_done(
            threading.Thread(target=self._run, args=(plan, stages), daemon=True),
            timeout,
            response,
        )

    @guarded_service
    def _on_release(self, request, response):
        """原地张开夹爪（"放下"）：不动手臂，只释放物体。"""
        if not self.enabled:
            response.success = False
            response.message = self._json({
                "stage": "FAILED", "message": "执行未启用", "holding": self.holding,
            })
            return response
        if self.running:
            response.success = False
            response.message = self._json({
                "stage": "BUSY", "message": "执行器忙碌", "holding": self.holding,
            })
            return response
        ok = self._gripper(self.open_width, 3.0)
        if ok:
            self.holding = False
            self.holding_state = 'empty'
            self.held_geometry = None
            self.get_logger().info("released: gripper opened in place")
        response.success = bool(ok)
        response.message = self._json({
            "stage": "SUCCESS" if ok else "FAILED",
            "message": "已张开夹爪放下物体" if ok else "张开夹爪失败",
            "holding": self.holding,
        })
        return response

    @guarded_service
    def _on_go_home(self, request, response):
        """恢复 Home（全零关节位）。直线关节插值、低速——不走 MoveIt。
        手里有物体时也照常执行（用户选择），但不动夹爪、会打 warning。"""
        if self.holding:
            self.get_logger().warn("going home while the gripper is still holding an object")
        return self._start_joint_move(
            [0.0] * 6, "HOME", "开始返回 Home（约 5 秒，低速直线插值到全零关节位）",
            "HOME SUCCESS: 已回到全零关节位", response,
        )

    @guarded_service
    def _on_go_ready(self, request, response):
        """到预抓取就绪姿态（poses.ready_joints，与 HSV 链路一致）。"""
        if self.holding:
            self.get_logger().warn("going to ready pose while still holding an object")
        return self._start_joint_move(
            list(self.ready_joints), "READY", "开始到预抓取就绪姿态（约 5 秒，低速关节插值）",
            "READY SUCCESS: 已到预抓取就绪姿态", response,
        )

    @guarded_service
    def _on_state(self, request, response):
        plan_age = (
            None if self.latest_plan is None
            else round(time.monotonic() - self.latest_plan_time, 2)
        )
        response.success = True
        response.message = self._json({
            "enabled": self.enabled,
            "running": self.running,
            "holding": self.holding,
            "holding_state": self.holding_state,
            "held_geometry": self.held_geometry if self.holding else None,
            "plan_age_s": plan_age,
            "plan_stages": [] if self.latest_plan is None else self._stage_names(self.latest_plan),
        })
        return response

    def _start_joint_move(self, target: List[float], label: str, start_msg: str,
                          ok_msg: str, response):
        if self.running:
            response.success = False
            response.message = self._json({
                "stage": "BUSY", "message": "执行器忙碌", "holding": self.holding,
            })
            return response
        if self.latest_joints is None:
            response.success = False
            response.message = self._json({
                "stage": "FAILED", "message": "尚未收到 /joint_states，无法确定当前关节位置",
                "holding": self.holding,
            })
            return response
        arm_names = [f"joint{i}" for i in range(1, 7)]
        index = {n: i for i, n in enumerate(self.latest_joints[0])}
        if not all(n in index for n in arm_names):
            response.success = False
            response.message = self._json({
                "stage": "FAILED", "message": "/joint_states 中缺少 joint1..joint6",
                "holding": self.holding,
            })
            return response
        current = [self.latest_joints[1][index[n]] for n in arm_names]
        trajectory = JointTrajectory()
        trajectory.joint_names = arm_names
        steps = 20
        for step in range(1, steps + 1):
            ratio = step / steps
            point = JointTrajectoryPoint()
            point.positions = [c + (t - c) * ratio for c, t in zip(current, target)]
            # time_from_start 必须是 builtin_interfaces/Duration；早先误用了
            # rclpy.duration.Duration，赋值断言失败会把整个执行器进程带崩。
            point.time_from_start = self._duration_message(5.0 * ratio)
            trajectory.points.append(point)
        worker = threading.Thread(
            target=self._run_joints, args=(trajectory, label, start_msg, ok_msg), daemon=True
        )
        return self._block_until_done(worker, 30.0, response)

    def _run_joints(self, trajectory, label: str, start_msg: str, ok_msg: str) -> None:
        outcome: Dict[str, object] = {"ok": False, "stage": "FAILED", "message": ""}
        try:
            ok, code = self._follow(trajectory)
            if ok:
                outcome.update(ok=True, stage="SUCCESS", message=ok_msg)
                self.get_logger().info(f"{label} SUCCESS: {ok_msg}")
            else:
                outcome.update(message=f"{label} 段轨迹失败 (错误码 {code})")
                self.get_logger().error(f"{label} FAILED (code {code})")
        except Exception as exc:
            outcome.update(message=str(exc))
            self.get_logger().error(f"{label} FAILED: {exc}")
        finally:
            outcome["holding"] = self.holding
            outcome["start"] = start_msg
            self._outcome = outcome
            self.running = False
            self._done.set()

    # ---------- plan execution ----------
    def _run(self, plan: DisplayTrajectory, stages: List[str]) -> None:
        outcome: Dict[str, object] = {"ok": False, "stage": "FAILED", "message": ""}
        try:
            if not self.validation_client.wait_for_service(timeout_sec=2.0):
                raise RuntimeError('执行前感知复核服务未就绪')
            future = self.validation_client.call_async(Trigger.Request())
            if not self._wait_future(future, 35.0):
                raise RuntimeError('执行前目标复核超时')
            validation = future.result()
            if validation is None or not validation.success:
                raise RuntimeError('执行前复核失败: '+('无响应' if validation is None else validation.message))
            checked = json.loads(validation.message)
            if checked.get('plan_fingerprint') != trajectory_fingerprint(plan.trajectory):
                raise RuntimeError('规划已经更换，请重新执行最新规划')
            self.validated_tool = checked.get('tool')
            self.validated_place_tool = checked.get('place_tool')
            self.latest_plan = None  # consumed plans cannot accidentally rerun
            grasp_stages = [name for name in stages if name in GRASP_STAGES]
            if 'PLACE' in stages and self.validated_place_tool is None:
                raise RuntimeError('放置计划缺少经验证的释放 TCP 目标，请重新规划')
            if grasp_stages:
                # Validate the attachment contract before any gripper/arm goal.
                object_in_tool(checked.get('object_cloud_base'), self.validated_tool)
                if self.holding:
                    raise RuntimeError('仍在持物，拒绝执行新的抓取计划')
                if not self._gripper(self.open_width, 2.0):
                    raise RuntimeError("打开夹爪失败")
                self.holding = False
                self.holding_state = 'empty'
                self.held_geometry = None
            elif not self.holding:
                raise RuntimeError('未确认持物，拒绝执行纯放置')
            for index, trajectory in enumerate(plan.trajectory):
                stage = stages[index]
                self.get_logger().info(f"executing {stage} ({index + 1}/{len(stages)})")
                ok, code = self._follow(trajectory.joint_trajectory)
                if not ok:
                    raise RuntimeError(f"{stage} 段轨迹执行失败 (错误码 {code})")
                if stage == "GRASP":
                    actual_grasp = self._verify_grasp_pose()
                    if not self._gripper(self.close_width, 3.0):
                        raise RuntimeError("闭合夹爪失败")
                    self.holding_state = self._holding_observation()
                    self.holding = self.holding_state != 'empty'
                    if self.holding_state != 'held':
                        raise RuntimeError('夹爪闭合后未确认持物: '+self.holding_state)
                    self.held_geometry = {
                        'cloud_tool': object_in_tool(checked['object_cloud_base'], actual_grasp).tolist(),
                        'grasp_rotation': actual_grasp[:3, :3].tolist(),
                        'grasp_fingerprint': checked['plan_fingerprint'],
                    }
                    self.get_logger().info('gripper contact confirmed by fresh finger gap')
                if stage == 'LIFT' and grasp_stages:
                    self.holding_state = self._holding_observation()
                    self.holding = self.holding_state != 'empty'
                    if self.holding_state != 'held':
                        raise RuntimeError('提起后未确认持物，可能滑落: '+self.holding_state)
                if stage == "PLACE":
                    self._verify_place_pose()
                    if not self._gripper(self.open_width, 3.0):
                        raise RuntimeError("放置后张开夹爪失败")
                    self.holding = False
                    self.holding_state = 'empty'
                    self.held_geometry = None
                    self.get_logger().info("release TCP verified, gripper opened; placement not visually confirmed")
            placed = "PLACE" in stages
            message = "放置轨迹执行完成，夹爪已松开" if placed else "抓取完成，物体保持夹持"
            outcome.update(ok=True, stage="SUCCESS", message=message)
            self.get_logger().info(f"EXECUTION SUCCESS: {message}")
        except Exception as exc:
            outcome.update(ok=False, stage="FAILED", message=str(exc))
            self.get_logger().error(f"EXECUTION FAILED: {exc}")
        finally:
            outcome["holding"] = self.holding
            outcome['holding_state'] = self.holding_state
            self._outcome = outcome
            self.running = False
            self._done.set()

    def _wait_future(self, future, timeout: float) -> bool:
        """Block the worker thread on a future served by the main executor."""
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        return done.wait(timeout)

    def _follow(self, joint_trajectory) -> Tuple[bool, int]:
        """执行一段轨迹；任何跟踪失败都不是有效触底信号。"""
        if not self.fjt_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("FollowJointTrajectory 服务未就绪")
            return False, -1
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory
        future = self.fjt_client.send_goal_async(goal)
        if not self._wait_future(future, 5.0):
            self.get_logger().error("发送轨迹目标超时")
            return False, -1
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("轨迹目标被拒绝")
            return False, -1
        result_future = handle.get_result_async()
        if not self._wait_future(result_future, self.traj_timeout):
            self.get_logger().error("轨迹执行超时")
            handle.cancel_goal_async()
            return False, -1
        code = int(result_future.result().result.error_code)
        if code != 0:
            self.get_logger().error(f"轨迹执行返回错误码 {code}")
            return False, code
        return True, 0

    def _gripper(self, width: float, timeout: float) -> bool:
        if not self.gripper_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("GripperCommand 服务未就绪")
            return False
        goal = GripperCommand.Goal()
        goal.command.position = min(
            self.gripper_max_pos, max(0.0, float(width) * self.gripper_ratio)
        )
        goal.command.max_effort = 0.0
        future = self.gripper_client.send_goal_async(goal)
        if not self._wait_future(future, 5.0):
            self.get_logger().error("发送夹爪目标超时")
            return False
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("夹爪目标被拒绝")
            return False
        result_future = handle.get_result_async()
        if not self._wait_future(result_future, timeout + 5.0):
            self.get_logger().error("夹爪动作超时")
            return False
        result = result_future.result().result
        reached = bool(getattr(result, "reached_goal", True))
        stalled = bool(getattr(result, "stalled", False))
        return reached or stalled


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RemoteGraspExecutor()
    # MultiThreadedExecutor 是必须的：/execute 会阻塞等动作序列结束，而
    # FollowJointTrajectory / GripperCommand 的回调仍需由 spin 线程服务，
    # 单线程 spin 下会死锁。
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
