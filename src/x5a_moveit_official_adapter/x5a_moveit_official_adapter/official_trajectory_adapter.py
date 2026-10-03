#!/usr/bin/env python3
"""Expose standard MoveIt interfaces over the official ARX X5 ROS 2 API.

Hardware-facing topics are strictly the vendor interfaces:
  /arm_status (arx5_arm_msg/RobotStatus)
  /arm_cmd    (arx5_arm_msg/RobotCmd)
"""
from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Sequence, Tuple

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from arx5_arm_msg.msg import RobotCmd, RobotStatus


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
STATE_JOINTS = ARM_JOINTS + ["joint7", "joint8"]


def point_time(point: JointTrajectoryPoint) -> float:
    return float(point.time_from_start.sec) + 1e-9 * float(point.time_from_start.nanosec)


def now_ns() -> int:
    # CLOCK_MONOTONIC, same domain as std::chrono::steady_clock in the C++
    # MTC task server, so timestamps correlate across processes.
    return time.monotonic_ns()


class OfficialTrajectoryAdapter(Node):
    def __init__(self) -> None:
        super().__init__("x5a_official_trajectory_adapter")
        self.cb = ReentrantCallbackGroup()
        self._declare_parameters()
        self._load_parameters()

        self.lock = threading.Lock()
        self.q: Optional[List[float]] = None
        self.dq: List[float] = [0.0] * 6
        self.effort: List[float] = [0.0] * 6
        self.end_pos: List[float] = [0.0] * 6
        self.gripper = 0.0
        self.last_status_time = 0.0
        self.state_started_time = 0.0
        self.last_state_jump_time = time.monotonic()
        self.state_fault = ""
        self.goal_active = False

        latest_reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latest_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.js_pub = self.create_publisher(
            JointState, self.joint_state_topic, latest_reliable
        )
        self.cmd_pub = self.create_publisher(
            RobotCmd, self.command_topic, latest_reliable
        )
        self.create_subscription(
            RobotStatus,
            self.status_topic,
            self._status_cb,
            latest_best_effort,
            callback_group=self.cb,
        )
        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.action_name,
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=self._cancel,
            callback_group=self.cb,
        )
        self.gripper_action_server = ActionServer(
            self,
            GripperCommand,
            self.gripper_action_name,
            execute_callback=self._execute_gripper,
            goal_callback=self._gripper_goal,
            cancel_callback=self._cancel,
            callback_group=self.cb,
        )
        self.get_logger().info(
            f"MoveIt adapter ready: {self.action_name} -> official {self.command_topic}; "
            f"gripper={self.gripper_action_name}; official {self.status_topic} "
            f"-> {self.joint_state_topic}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("official_status_topic", "/arm_status")
        self.declare_parameter("official_command_topic", "/arm_cmd")
        self.declare_parameter("action_name", "/x5a_arm_controller/follow_joint_trajectory")
        self.declare_parameter(
            "gripper_action_name", "/x5a_gripper_controller/gripper_cmd"
        )
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("position_mode", 5)
        self.declare_parameter("command_rate_hz", 20.0)
        self.declare_parameter("joint_state_rate_hz", 30.0)
        self.declare_parameter("status_timeout", 0.25)
        self.declare_parameter("state_settle_sec", 2.0)
        self.declare_parameter("max_status_jump_rad", 0.08)
        self.declare_parameter("max_status_speed_rad_s", 1.5)
        self.declare_parameter("start_tolerance_rad", 0.15)
        self.declare_parameter("goal_tolerance_rad", 0.01)
        self.declare_parameter("goal_time_tolerance_sec", 2.0)
        self.declare_parameter("tracking_grace_sec", 0.75)
        self.declare_parameter("tracking_tolerance_rad", 0.15)
        self.declare_parameter("response_command_delta_rad", 0.03)
        self.declare_parameter("response_actual_delta_rad", 0.004)
        self.declare_parameter("response_timeout_sec", 0.75)
        self.declare_parameter("max_joint_speed_rad_s", 1.0)
        self.declare_parameter("gripper_closed_readout", 0.0)
        self.declare_parameter("gripper_open_readout", 5.0)
        self.declare_parameter("gripper_finger_max_m", 0.044)
        self.declare_parameter("gripper_command_duration_sec", 0.8)
        self.declare_parameter("gripper_goal_tolerance_m", 0.004)
        self.declare_parameter("gripper_open_threshold_m", 0.03)

    def _load_parameters(self) -> None:
        g = lambda name: self.get_parameter(name).value
        self.status_topic = str(g("official_status_topic"))
        self.command_topic = str(g("official_command_topic"))
        self.action_name = str(g("action_name"))
        self.gripper_action_name = str(g("gripper_action_name"))
        self.joint_state_topic = str(g("joint_state_topic"))
        self.position_mode = int(g("position_mode"))
        self.rate_hz = float(g("command_rate_hz"))
        self.state_rate_hz = float(g("joint_state_rate_hz"))
        self.state_period = 1.0 / max(1.0, self.state_rate_hz)
        self.status_timeout = float(g("status_timeout"))
        self.state_settle = float(g("state_settle_sec"))
        self.max_status_jump = float(g("max_status_jump_rad"))
        self.max_status_speed = float(g("max_status_speed_rad_s"))
        self.start_tolerance = float(g("start_tolerance_rad"))
        self.goal_tolerance = float(g("goal_tolerance_rad"))
        self.goal_time_tolerance = float(g("goal_time_tolerance_sec"))
        self.tracking_grace = float(g("tracking_grace_sec"))
        self.tracking_tolerance = float(g("tracking_tolerance_rad"))
        self.response_command_delta = float(g("response_command_delta_rad"))
        self.response_actual_delta = float(g("response_actual_delta_rad"))
        self.response_timeout = float(g("response_timeout_sec"))
        self.max_speed = float(g("max_joint_speed_rad_s"))
        self.gripper_closed = float(g("gripper_closed_readout"))
        self.gripper_open = float(g("gripper_open_readout"))
        self.finger_max = float(g("gripper_finger_max_m"))
        self.gripper_duration = float(g("gripper_command_duration_sec"))
        self.gripper_tolerance = float(g("gripper_goal_tolerance_m"))
        self.gripper_open_threshold = float(g("gripper_open_threshold_m"))

    def finger_position(self, readout: float) -> float:
        span = self.gripper_open - self.gripper_closed
        if abs(span) < 1e-9:
            return 0.0
        fraction = (readout - self.gripper_closed) / span
        return self.finger_max * max(0.0, min(1.0, fraction))

    def gripper_readout(self, finger_position: float) -> float:
        if self.finger_max <= 1e-9:
            return self.gripper_closed
        fraction = max(0.0, min(1.0, finger_position / self.finger_max))
        return self.gripper_closed + fraction * (
            self.gripper_open - self.gripper_closed
        )

    @staticmethod
    def _fmt_joints(values: Optional[Sequence[float]]) -> str:
        if values is None:
            return "[]"
        return "[" + ",".join(f"{float(v):.4f}" for v in values) + "]"

    def _arm_abort(
        self,
        goal_handle,
        result: FollowJointTrajectory.Result,
        code: int,
        reason: str,
        elapsed: Optional[float] = None,
        desired: Optional[Sequence[float]] = None,
        actual: Optional[Sequence[float]] = None,
    ) -> FollowJointTrajectory.Result:
        parts = [f"[ARM] ARM_ABORT t_ns={now_ns()} reason={reason}"]
        if elapsed is not None:
            parts.append(f"elapsed={elapsed:.3f}")
        if desired is not None:
            parts.append(f"desired={self._fmt_joints(desired)}")
        if actual is not None:
            parts.append(f"actual={self._fmt_joints(actual)}")
        if (
            desired is not None
            and actual is not None
            and len(desired) == len(actual) == 6
        ):
            error_per_joint = [round(a - d, 4) for a, d in zip(actual, desired)]
            max_error = max(abs(e) for e in error_per_joint)
            parts.append(f"error_per_joint={error_per_joint}")
            parts.append(f"max_error={max_error:.4f}")
        self.get_logger().error(" ".join(parts))
        result.error_code = code
        result.error_string = reason
        goal_handle.abort()
        return result

    def _status_cb(self, msg: RobotStatus) -> None:
        if len(msg.joint_pos) < 6:
            return
        received_at = time.monotonic()
        with self.lock:
            # The vendor publishes from a 1 ms timer.  MoveIt only needs a
            # fresh latest sample; processing every vendor message saturated
            # a CPU core and could starve the driver's CAN receive thread.
            if (
                self.last_status_time > 0.0
                and received_at - self.last_status_time < self.state_period
            ):
                return
            new_q = [float(msg.joint_pos[i]) for i in range(6)]
            if self.q is None:
                self.state_started_time = received_at
                self.last_state_jump_time = received_at
            else:
                status_gap = received_at - self.last_status_time
                jump = max(abs(a - b) for a, b in zip(new_q, self.q))
                # A fixed per-message jump threshold falsely reports a state
                # fault after a short scheduler/CAN gap.  Scale the allowed
                # delta by elapsed time while retaining the original 0.08 rad
                # noise floor at the normal 30 Hz publication rate.
                allowed_jump = max(
                    self.max_status_jump,
                    self.max_status_speed * max(0.0, status_gap),
                )
                if status_gap > self.status_timeout:
                    # A real feedback outage is allowed to recover, but the
                    # arm must publish stable state for state_settle seconds
                    # before another command is accepted.
                    self.last_state_jump_time = received_at
                    self.state_fault = (
                        f"official state recovering after {status_gap:.3f}s gap"
                    )
                elif jump > allowed_jump:
                    self.last_state_jump_time = received_at
                    self.state_fault = (
                        f"official state jumped by {jump:.3f} rad "
                        f"(allowed {allowed_jump:.3f})"
                    )
                elif (
                    self.state_fault
                    and received_at - self.last_state_jump_time >= self.state_settle
                ):
                    self.state_fault = ""
            self.q = new_q
            self.dq = [float(msg.joint_vel[i]) for i in range(6)] if len(msg.joint_vel) >= 6 else [0.0] * 6
            self.effort = [float(msg.joint_cur[i]) for i in range(6)] if len(msg.joint_cur) >= 6 else [0.0] * 6
            self.end_pos = [float(msg.end_pos[i]) for i in range(6)]
            self.gripper = float(msg.joint_pos[6]) if len(msg.joint_pos) > 6 else self.gripper
            self.last_status_time = received_at
            q = list(self.q); dq = list(self.dq); effort = list(self.effort)
            finger = self.finger_position(self.gripper)
        js = JointState()
        js.header = msg.header
        js.name = list(STATE_JOINTS)
        js.position = q + [finger, finger]
        js.velocity = dq + [0.0, 0.0]
        js.effort = effort + [0.0, 0.0]
        self.js_pub.publish(js)

    def snapshot(self) -> Tuple[Optional[List[float]], List[float], float, float]:
        with self.lock:
            q = None if self.q is None else list(self.q)
            return q, list(self.end_pos), float(self.gripper), float(self.last_status_time)

    def readiness_fault(self) -> str:
        now = time.monotonic()
        with self.lock:
            if self.q is None:
                return "no official RobotStatus"
            if now - self.last_status_time > self.status_timeout:
                return "official RobotStatus is stale"
            if now - self.state_started_time < self.state_settle:
                return "official state has not settled after startup"
            if self.state_fault or now - self.last_state_jump_time < self.state_settle:
                return self.state_fault or "official state has not settled after a jump"
        return ""

    def _goal(self, request: FollowJointTrajectory.Goal) -> GoalResponse:
        traj = request.trajectory
        if self.goal_active:
            self.get_logger().error("rejecting trajectory: another goal is active")
            return GoalResponse.REJECT
        if list(traj.joint_names) != ARM_JOINTS or not traj.points:
            self.get_logger().error("rejecting trajectory: expected joint1..joint6 in canonical order")
            return GoalResponse.REJECT
        previous = -1.0
        for p in traj.points:
            t = point_time(p)
            if len(p.positions) != 6 or t < previous or not all(math.isfinite(v) for v in p.positions):
                self.get_logger().error("rejecting malformed trajectory")
                return GoalResponse.REJECT
            previous = t
        fault = self.readiness_fault()
        if fault:
            self.get_logger().error(f"rejecting trajectory: {fault}")
            return GoalResponse.REJECT
        # This adapter must be the only writer to the official hardware command topic.
        if self.count_publishers(self.command_topic) != 1:
            self.get_logger().error(
                f"rejecting trajectory: {self.command_topic} has "
                f"{self.count_publishers(self.command_topic)} publishers (expected adapter only)"
            )
            return GoalResponse.REJECT
        self.goal_active = True
        start_joint, _, _, _ = self.snapshot()
        self.get_logger().info(
            f"[ARM] ARM_GOAL_ACCEPTED t_ns={now_ns()} "
            f"trajectory_points={len(traj.points)} "
            f"trajectory_duration={point_time(traj.points[-1]):.3f} "
            f"start_joint={self._fmt_joints(start_joint)} "
            f"goal_joint={self._fmt_joints(traj.points[-1].positions)}"
        )
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _gripper_goal(self, request: GripperCommand.Goal) -> GoalResponse:
        position = float(request.command.position)
        if self.goal_active:
            self.get_logger().error("rejecting gripper command: another command is active")
            return GoalResponse.REJECT
        if not math.isfinite(position) or position < -1e-6 or position > self.finger_max + 1e-6:
            self.get_logger().error(
                f"rejecting gripper position {position:.4f} m; expected 0..{self.finger_max:.4f} m"
            )
            return GoalResponse.REJECT
        fault = self.readiness_fault()
        if fault:
            self.get_logger().error(f"rejecting gripper command: {fault}")
            return GoalResponse.REJECT
        if self.count_publishers(self.command_topic) != 1:
            self.get_logger().error(
                f"rejecting gripper command: {self.command_topic} has "
                f"{self.count_publishers(self.command_topic)} publishers"
            )
            return GoalResponse.REJECT
        self.goal_active = True
        _, _, actual_readout, _ = self.snapshot()
        self.get_logger().info(
            f"[GRIPPER] GRIPPER_GOAL_ACCEPTED t_ns={now_ns()} "
            f"target_m={position:.4f} target_vendor={self.gripper_readout(position):.4f} "
            f"actual_before={self.finger_position(actual_readout):.4f}"
        )
        return GoalResponse.ACCEPT

    def _publish_command(self, joints: Sequence[float], end_pos: Sequence[float], gripper: float) -> None:
        msg = RobotCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(6):
            msg.joint_pos[i] = float(joints[i])
            msg.end_pos[i] = float(end_pos[i])
        msg.gripper = float(gripper)
        msg.mode = self.position_mode
        self.cmd_pub.publish(msg)

    def _execute_gripper(self, goal_handle):
        result = GripperCommand.Result()
        target_m = max(
            0.0, min(self.finger_max, float(goal_handle.request.command.position))
        )
        target_readout = self.gripper_readout(target_m)
        try:
            _, _, actual_readout, _ = self.snapshot()
            actual_before = self.finger_position(actual_readout)
            self.get_logger().info(
                f"[GRIPPER] GRIPPER_EXEC_START t_ns={now_ns()} "
                f"target_m={target_m:.4f} actual_before={actual_before:.4f}"
            )
            deadline = time.monotonic() + self.gripper_duration
            period = 1.0 / max(5.0, self.rate_hz)
            while time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return result
                q, end_pos, actual_readout, stamp = self.snapshot()
                if q is None or time.monotonic() - stamp > self.status_timeout:
                    self.get_logger().error(
                        f"[GRIPPER] GRIPPER_ABORT t_ns={now_ns()} "
                        f"reason=official RobotStatus lost target_m={target_m:.4f}"
                    )
                    goal_handle.abort()
                    return result
                if self.count_publishers(self.command_topic) != 1:
                    self.get_logger().error(
                        f"[GRIPPER] GRIPPER_ABORT t_ns={now_ns()} "
                        f"reason=another /arm_cmd publisher appeared target_m={target_m:.4f}"
                    )
                    goal_handle.abort()
                    return result
                self._publish_command(q, end_pos, target_readout)
                actual_m = self.finger_position(actual_readout)
                feedback = GripperCommand.Feedback()
                feedback.position = actual_m
                feedback.effort = 0.0
                feedback.stalled = False
                feedback.reached_goal = abs(actual_m - target_m) <= self.gripper_tolerance
                goal_handle.publish_feedback(feedback)
                time.sleep(period)

            _, _, actual_readout, _ = self.snapshot()
            result.position = self.finger_position(actual_readout)
            result.effort = 0.0
            result.stalled = False
            actual_m = result.position
            # The vendor API exposes a position target but no independent
            # grasp-complete flag. An OPEN target must actually be reached
            # (a stuck ~0 m reading must never be reported to MoveIt as
            # success); a CLOSE target is lenient because a real grasp blocks
            # the fingers before reaching 0 m.
            if (
                target_m > self.gripper_open_threshold
                and abs(actual_m - target_m) > self.gripper_tolerance
            ):
                reason = (
                    f"open target {target_m:.4f} m not reached; "
                    f"actual {actual_m:.4f} m"
                )
                self.get_logger().error(
                    f"[GRIPPER] GRIPPER_ABORT t_ns={now_ns()} reason={reason} "
                    f"target_m={target_m:.4f} actual_after={actual_m:.4f}"
                )
                result.reached_goal = False
                goal_handle.abort()
                return result
            result.reached_goal = abs(actual_m-target_m) <= self.gripper_tolerance
            result.stalled = not result.reached_goal and target_m <= self.gripper_open_threshold
            goal_handle.succeed()
            self.get_logger().info(
                f"[GRIPPER] GRIPPER_SUCCESS t_ns={now_ns()} "
                f"target_m={target_m:.4f} actual_after={actual_m:.4f}"
            )
            return result
        finally:
            self.goal_active = False

    @staticmethod
    def _desired_at(points: List[Tuple[float, List[float]]], elapsed: float) -> List[float]:
        if elapsed <= points[0][0]:
            return list(points[0][1])
        for index in range(len(points) - 1):
            t0, q0 = points[index]; t1, q1 = points[index + 1]
            if elapsed <= t1:
                alpha = 1.0 if t1 <= t0 else (elapsed - t0) / (t1 - t0)
                return [a + (b - a) * alpha for a, b in zip(q0, q1)]
        return list(points[-1][1])

    def _feedback(self, goal_handle, desired: Sequence[float], actual: Sequence[float]) -> None:
        fb = FollowJointTrajectory.Feedback()
        fb.header.stamp = self.get_clock().now().to_msg()
        fb.joint_names = list(ARM_JOINTS)
        fb.desired.positions = [float(v) for v in desired]
        fb.actual.positions = [float(v) for v in actual]
        fb.error.positions = [float(a - d) for a, d in zip(actual, desired)]
        goal_handle.publish_feedback(fb)

    def _execute(self, goal_handle):
        result = FollowJointTrajectory.Result()
        try:
            q0, end_pos, gripper, _ = self.snapshot()
            if q0 is None:
                return self._arm_abort(
                    goal_handle, result, FollowJointTrajectory.Result.INVALID_GOAL,
                    "no official RobotStatus",
                )
            raw = [(point_time(p), [float(v) for v in p.positions]) for p in goal_handle.request.trajectory.points]
            if raw[0][0] > 1e-6:
                points = [(0.0, list(q0))] + raw
            else:
                points = raw
            start_error = max(abs(a - b) for a, b in zip(q0, points[0][1]))
            if start_error > self.start_tolerance:
                return self._arm_abort(
                    goal_handle, result, FollowJointTrajectory.Result.INVALID_GOAL,
                    f"start error {start_error:.3f} rad",
                    elapsed=0.0, desired=points[0][1], actual=q0,
                )
            for (ta, qa), (tb, qb) in zip(points, points[1:]):
                if tb > ta:
                    speed = max(abs(b - a) / (tb - ta) for a, b in zip(qa, qb))
                    if speed > self.max_speed:
                        return self._arm_abort(
                            goal_handle, result, FollowJointTrajectory.Result.INVALID_GOAL,
                            f"trajectory speed {speed:.3f} exceeds {self.max_speed:.3f} rad/s",
                        )

            duration = points[-1][0]
            period = 1.0 / max(5.0, self.rate_hz)
            started = time.monotonic()
            self.get_logger().info(
                f"[ARM] ARM_EXEC_START t_ns={now_ns()} "
                f"trajectory_points={len(points)} trajectory_duration={duration:.3f}"
            )
            command_count = 0
            response_started: List[Optional[float]] = [None] * 6
            while True:
                if goal_handle.is_cancel_requested:
                    q, end_pos, gripper, _ = self.snapshot()
                    if q is not None: self._publish_command(q, end_pos, gripper)
                    goal_handle.canceled(); return result
                elapsed = min(time.monotonic() - started, duration)
                desired = self._desired_at(points, elapsed)
                actual, latest_end, latest_gripper, stamp = self.snapshot()
                if actual is None or time.monotonic() - stamp > self.status_timeout:
                    return self._arm_abort(
                        goal_handle, result, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                        "official RobotStatus lost during execution",
                        elapsed=elapsed, desired=desired,
                    )
                if self.count_publishers(self.command_topic) != 1:
                    return self._arm_abort(
                        goal_handle, result, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                        "another /arm_cmd publisher appeared",
                        elapsed=elapsed, desired=desired, actual=actual,
                    )
                fault = self.readiness_fault()
                if fault:
                    return self._arm_abort(
                        goal_handle, result, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                        fault,
                        elapsed=elapsed, desired=desired, actual=actual,
                    )
                if elapsed >= self.tracking_grace:
                    tracking_errors = [abs(a - d) for a, d in zip(actual, desired)]
                    if max(tracking_errors) > self.tracking_tolerance:
                        self._publish_command(actual, latest_end, gripper)
                        return self._arm_abort(
                            goal_handle, result, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                            f"tracking error {max(tracking_errors):.3f} rad",
                            elapsed=elapsed, desired=desired, actual=actual,
                        )
                    response_now = time.monotonic()
                    for index, (commanded, moved) in enumerate(
                        zip(
                            (abs(d - s) for d, s in zip(desired, q0)),
                            (abs(a - s) for a, s in zip(actual, q0)),
                        )
                    ):
                        if commanded < self.response_command_delta:
                            response_started[index] = None
                            continue
                        if response_started[index] is None:
                            response_started[index] = response_now
                            continue
                        if (
                            response_now - response_started[index]
                            >= self.response_timeout
                            and moved < self.response_actual_delta
                        ):
                            self._publish_command(actual, latest_end, gripper)
                            return self._arm_abort(
                                goal_handle, result, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                                f"joint{index + 1} did not respond to position commands",
                                elapsed=elapsed, desired=desired, actual=actual,
                            )
                self._publish_command(desired, latest_end, gripper)
                command_count += 1
                self._feedback(goal_handle, desired, actual)
                if elapsed >= duration:
                    break
                time.sleep(period)

            deadline = time.monotonic() + self.goal_time_tolerance
            final = points[-1][1]
            error = float("inf")
            actual = None
            while time.monotonic() < deadline:
                actual, latest_end, _, stamp = self.snapshot()
                if actual is None or time.monotonic() - stamp > self.status_timeout:
                    break
                self._publish_command(final, latest_end, gripper)
                command_count += 1
                error = max(abs(a - b) for a, b in zip(actual, final))
                self._feedback(goal_handle, final, actual)
                if error <= self.goal_tolerance:
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    goal_handle.succeed()
                    self.get_logger().info(
                        f"[ARM] ARM_SUCCESS t_ns={now_ns()} "
                        f"elapsed={time.monotonic() - started:.3f} "
                        f"max_error={error:.4f} commands={command_count}"
                    )
                    return result
                time.sleep(period)
            if actual is None:
                return self._arm_abort(
                    goal_handle, result, FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                    "official RobotStatus lost while settling on goal",
                    elapsed=time.monotonic() - started, desired=final,
                )
            return self._arm_abort(
                goal_handle, result, FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                f"goal error {error:.3f} rad",
                elapsed=time.monotonic() - started, desired=final, actual=actual,
            )
        finally:
            self.goal_active = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OfficialTrajectoryAdapter()
    # Two threads are sufficient: one action execution thread and one latest
    # status callback thread. More threads only increase contention in Python.
    executor = MultiThreadedExecutor(num_threads=2)
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
