#!/usr/bin/env python3
"""Pose-driven pick and place for ARX X5A using MoveIt 2 planning.

Uses:
- MoveGroup action for planning only
- compute_cartesian_path service for vertical approach/lift/descend/retreat
- standard MoveIt ExecuteTrajectory and GripperCommand actions
- /joint_states from the single official ARX hardware adapter
- PlanningScene topics for table/object attach/detach
"""
from __future__ import annotations

import math
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from control_msgs.action import GripperCommand
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PlanningScene,
    PositionConstraint,
    RobotState,
    RobotTrajectory,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from action_msgs.msg import GoalStatus
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from x5a_handeye.x5a_fk import fk_base_tool0
from x5a_pick_place.reobserve import ReobserveWindow, bounded_correction
from x5a_pick_place.retiming import scale_cartesian_timing
from x5a_pick_place.stacking import stack_geometry, tcp_for_object_center, aligned_stack_rpy


ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]
ALLOWED_COLORS = ("red", "white", "orange")
ALLOWED_SELECTORS = ("nearest", "farthest")
ALLOWED_TARGETS = ALLOWED_COLORS + ALLOWED_SELECTORS


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def pose_xyz_quat(
    x: float, y: float, z: float, quat: Quaternion
) -> Pose:
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = float(z)
    p.orientation = quat
    return p


def pose_to_xyz(pose: Pose) -> Tuple[float, float, float]:
    return (pose.position.x, pose.position.y, pose.position.z)


def quat_to_yaw(q: Quaternion) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def gripper_result_ok(opening, result, status, min_held_gap=0.0):
    if status != GoalStatus.STATUS_SUCCEEDED or not math.isfinite(result.position):
        return False
    if opening:
        return bool(result.reached_goal)
    return bool(result.reached_goal or result.stalled) and 2.0 * result.position >= min_held_gap


def unique_rpy(
    candidates: Sequence[Sequence[float]], cap: int = 12
) -> List[List[float]]:
    out: List[List[float]] = []
    for raw in candidates:
        if len(raw) < 3:
            continue
        rpy = [wrap_pi(float(raw[0])), wrap_pi(float(raw[1])), wrap_pi(float(raw[2]))]
        if any(
            all(abs(wrap_pi(a - b)) < 0.05 for a, b in zip(rpy, existing))
            for existing in out
        ):
            continue
        out.append(rpy)
        if len(out) >= cap:
            break
    return out


class PickPlaceNode(Node):
    def __init__(self, node_name: str = "x5a_pick_place", tf_spin_thread: bool = True) -> None:
        super().__init__(node_name)
        self.cb = ReentrantCallbackGroup()
        self._declare_params()
        self._load_params()

        self.q: Optional[List[float]] = None
        self.dq: Optional[List[float]] = None
        self.gripper_position = 0.0
        self.joint_state_arrival = 0.0
        self.create_subscription(
            JointState, self.joint_state_topic, self._joint_state_cb, 10,
            callback_group=self.cb
        )
        self.allowed_colors = ALLOWED_COLORS
        self.cube_poses: Dict[str, Optional[PoseStamped]] = {
            color: None for color in self.allowed_colors
        }
        self.cube_pose_arrivals: Dict[str, float] = {
            color: 0.0 for color in self.allowed_colors
        }
        self.cube_stable: Dict[str, bool] = {
            color: False for color in self.allowed_colors
        }
        for color in self.allowed_colors:
            self.create_subscription(
                PoseStamped,
                f"/x5a_vision/{color}_cube_pose",
                self._make_cube_pose_cb(color),
                10,
                callback_group=self.cb,
            )
            self.create_subscription(
                Bool,
                f"/x5a_vision/{color}_cube_stable",
                self._make_cube_stable_cb(color),
                10,
                callback_group=self.cb,
            )
        self.create_subscription(
            PoseStamped, self.box_pose_topic, self._box_pose_cb, 10,
            callback_group=self.cb
        )
        self.create_subscription(
            Bool, self.box_stable_topic, self._box_stable_cb, 10,
            callback_group=self.cb
        )
        self.vision_pose: Optional[PoseStamped] = None
        self.vision_pose_arrival = 0.0
        self.vision_stable = False
        self.box_pose: Optional[PoseStamped] = None
        self.box_pose_arrival = 0.0
        self.box_stable = False
        self._stage_callback: Optional[Callable[[str], None]] = None
        self._cancel_requested = False
        self._session_box_xyz: Optional[Tuple[float, float, float]] = None

        self.move_ac = ActionClient(self, MoveGroup, "move_action", callback_group=self.cb)
        self.execute_ac = ActionClient(
            self, ExecuteTrajectory, self.execute_action, callback_group=self.cb
        )
        self.gripper_ac = ActionClient(
            self, GripperCommand, self.gripper_action, callback_group=self.cb
        )
        self.cart_cli = self.create_client(
            GetCartesianPath, "/compute_cartesian_path", callback_group=self.cb
        )
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik", callback_group=self.cb)
        self.apply_scene_cli = self.create_client(
            ApplyPlanningScene, "apply_planning_scene", callback_group=self.cb)

        scene_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", scene_qos)
        self.grasp_quat = rpy_to_quat(*self.grasp_rpy)
        self.transit_quat = rpy_to_quat(*self.transit_rpy)
        self.last_joints = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=tf_spin_thread)

        self.get_logger().info(
            f"pick_place ready plan_only={self.plan_only} vision={self.vision_enabled} "
            f"execution={self.execute_action} gripper={self.gripper_action} "
            f"state={self.joint_state_topic} target_color={self.target_color}"
        )

    def _declare_params(self) -> None:
        self.declare_parameter("plan_only", False)
        self.declare_parameter("planning.group", "arm")
        self.declare_parameter("planning.base_frame", "base_link")
        self.declare_parameter("planning.tcp_frame", "tool0")
        self.declare_parameter("planning.transit_velocity_scaling", 0.45)
        self.declare_parameter("planning.transit_acceleration_scaling", 0.18)
        self.declare_parameter("planning.precision_velocity_scaling", 0.15)
        self.declare_parameter("planning.precision_acceleration_scaling", 0.08)
        self.declare_parameter("planning.lift_retreat_velocity_scaling", 0.20)
        self.declare_parameter("planning.lift_retreat_acceleration_scaling", 0.10)
        self.declare_parameter("planning.planning_time", 5.0)
        self.declare_parameter("planning.max_attempts", 8)
        self.declare_parameter("planning.joint_goal_tolerance_rad", 0.05)
        self.declare_parameter("execution.action_name", "/execute_trajectory")
        self.declare_parameter("execution.timeout", 120.0)
        self.declare_parameter("execution.require_servo_bounds", False)
        self.declare_parameter("execution.min_held_gap_m", 0.0)
        self.declare_parameter("execution.joint_lower", [-3.14, -0.05, -0.1, -1.28, -1.57, -2.0])
        self.declare_parameter("execution.joint_upper", [2.618, 3.5, 3.2, 1.55, 1.57, 2.0])
        self.declare_parameter("state.joint_state_topic", "/joint_states")
        self.declare_parameter("vision.enabled", False)
        self.declare_parameter("vision.camera_frame", "camera_link")
        self.declare_parameter("planning.use_tf_tcp", False)
        self.declare_parameter("motion.require_cartesian", False)
        self.declare_parameter("calibration.result_file", "")
        self.declare_parameter("target_color", "red")
        self.declare_parameter("vision.max_pose_age", 0.5)
        self.declare_parameter('vision.max_box_pose_age', 3.0)
        self.declare_parameter("vision.wait_timeout", 10.0)
        self.declare_parameter("vision.reobserve.enabled", False)
        self.declare_parameter("vision.reobserve.height", 0.18)
        self.declare_parameter("vision.reobserve.timeout", 5.0)
        self.declare_parameter("vision.reobserve.settle_time", 0.5)
        self.declare_parameter("vision.reobserve.max_correction_m", 0.015)
        self.declare_parameter("vision.reobserve.max_std_m", 0.0025)
        self.declare_parameter("vision.reobserve.max_frame_gap_s", 0.2)
        self.declare_parameter("vision.reobserve.backoff_m", 0.0)
        self.declare_parameter("vision.reobserve.verify_tolerance_m", 0.003)
        self.declare_parameter("box.pose_topic", "/x5a_vision/box_pose")
        self.declare_parameter("box.stable_topic", "/x5a_vision/box_stable")
        self.declare_parameter("box.collision.enabled", False)
        self.declare_parameter("box.collision.inner_x", 0.0)
        self.declare_parameter("box.collision.inner_y", 0.0)
        self.declare_parameter("box.collision.height", 0.0)
        self.declare_parameter("box.collision.wall", 0.005)
        self.declare_parameter("box.collision.yaw", 0.0)
        self.declare_parameter("box.collision.use_pose_yaw", False)
        self.declare_parameter("motion.pre_grasp_ready_enabled", True)
        self.declare_parameter("motion.initial_observation_enabled", False)
        self.declare_parameter("planning.precheck_approach", False)
        self.declare_parameter("grasp.pitch_fallbacks", [1.05, 1.0])
        self.declare_parameter("vision.scene_other_cubes", False)
        self.declare_parameter('place.stack_clearance_m', .002)
        self.declare_parameter("grasp_orientation_rpy", [3.14159265, 0.0, 0.0])
        self.declare_parameter("transit_orientation_rpy", [0.0, 1.15, 0.0])
        for ns, keys in {
            "object": ["x", "y", "z", "size_x", "size_y", "size_z"],
            "place": ["x", "y", "z"],
            "table": ["x", "y", "z", "size_x", "size_y", "size_z"],
            "motion": [
                "pre_grasp_height",
                "lift_height",
                "pre_place_height",
                "retreat_height",
                "cartesian_step",
                "jump_threshold",
                "min_cartesian_fraction",
            ],
            "grasp": ["x_offset", "y_offset", "z_offset"],
        }.items():
            for k in keys:
                default = 0.0
                if k.endswith("height"):
                    default = 0.10
                if k == "cartesian_step":
                    default = 0.01
                if k == "min_cartesian_fraction":
                    default = 0.95
                if k.startswith("size"):
                    default = 0.03 if ns == "object" else 0.5
                self.declare_parameter(f"{ns}.{k}", default)
        self.declare_parameter(
            "gripper.action_name", "/x5a_gripper_controller/gripper_cmd"
        )
        self.declare_parameter("gripper.open_position_m", 0.044)
        self.declare_parameter("gripper.close_position_m", 0.0)
        self.declare_parameter("gripper.max_effort", 0.0)
        self.declare_parameter("gripper.timeout", 3.0)
        self.declare_parameter(
            "ready_joints", [-0.2, 0.3, 0.6, -0.3, 0.0, 0.0]
        )
        self.declare_parameter(
            "pre_grasp_ready_joints", [0.0, 0.85, 0.95, -0.55, 0.0, 0.0]
        )

    def _load_params(self) -> None:
        g = self.get_parameter
        self.plan_only = bool(g("plan_only").value)
        self.group = str(g("planning.group").value)
        self.base_frame = str(g("planning.base_frame").value)
        self.tcp_frame = str(g("planning.tcp_frame").value)
        self.transit_vel = float(g("planning.transit_velocity_scaling").value)
        self.transit_acc = float(g("planning.transit_acceleration_scaling").value)
        self.precision_vel = float(g("planning.precision_velocity_scaling").value)
        self.precision_acc = float(g("planning.precision_acceleration_scaling").value)
        self.lift_retreat_vel = float(
            g("planning.lift_retreat_velocity_scaling").value
        )
        self.lift_retreat_acc = float(
            g("planning.lift_retreat_acceleration_scaling").value
        )
        self.planning_time = float(g("planning.planning_time").value)
        self.joint_goal_tolerance = float(g("planning.joint_goal_tolerance_rad").value)
        if not math.isfinite(self.joint_goal_tolerance) or self.joint_goal_tolerance <= 0:
            raise ValueError('joint goal tolerance must be finite and positive')
        self.max_attempts = int(g("planning.max_attempts").value)
        self.execute_action = str(g("execution.action_name").value)
        self.execution_timeout = float(g("execution.timeout").value)
        self.require_servo_bounds = bool(g("execution.require_servo_bounds").value)
        self.min_held_gap = float(g("execution.min_held_gap_m").value)
        self.servo_lower = list(g("execution.joint_lower").value)
        self.servo_upper = list(g("execution.joint_upper").value)
        if self.require_servo_bounds and (len(self.servo_lower) != 6 or len(self.servo_upper) != 6
                or any(not math.isfinite(a) or not math.isfinite(b) or a >= b
                       for a, b in zip(self.servo_lower, self.servo_upper))):
            raise ValueError("servo bounds require six finite ordered intervals")
        self.joint_state_topic = str(g("state.joint_state_topic").value)
        self.vision_enabled = bool(g("vision.enabled").value)
        self.camera_frame = str(g("vision.camera_frame").value)
        self.use_tf_tcp = bool(g("planning.use_tf_tcp").value)
        self.require_cartesian = bool(g("motion.require_cartesian").value)
        self.calibration_result_file = str(g("calibration.result_file").value)
        self.target_color = str(g("target_color").value).strip().lower()
        if self.target_color not in ALLOWED_COLORS:
            raise ValueError(
                f"target_color must be red, white, or orange; got {self.target_color!r}"
            )
        self.vision_pose_topic = f"/x5a_vision/{self.target_color}_cube_pose"
        self.vision_stable_topic = f"/x5a_vision/{self.target_color}_cube_stable"
        self.vision_max_age = float(g("vision.max_pose_age").value)
        self.box_max_age = float(g('vision.max_box_pose_age').value)
        if any(not math.isfinite(v) or not 0 < v <= 5 for v in (self.vision_max_age, self.box_max_age)):
            raise ValueError('vision pose ages must be positive and at most five seconds')
        self.vision_wait_timeout = float(g("vision.wait_timeout").value)
        self.reobserve_enabled = bool(g("vision.reobserve.enabled").value)
        self.reobserve_height = float(g("vision.reobserve.height").value)
        self.reobserve_timeout = float(g("vision.reobserve.timeout").value)
        self.reobserve_settle = float(g("vision.reobserve.settle_time").value)
        self.reobserve_max_delta = float(g("vision.reobserve.max_correction_m").value)
        self.reobserve_max_std = float(g("vision.reobserve.max_std_m").value)
        self.reobserve_max_frame_gap = float(g("vision.reobserve.max_frame_gap_s").value)
        self.reobserve_backoff = float(g("vision.reobserve.backoff_m").value)
        self.reobserve_verify_tolerance = float(g("vision.reobserve.verify_tolerance_m").value)
        if not math.isfinite(self.reobserve_verify_tolerance) or not 0 < self.reobserve_verify_tolerance <= .01:
            raise ValueError('correction verification tolerance must be within (0, 0.01] m')
        if not math.isfinite(self.reobserve_backoff) or not 0 <= self.reobserve_backoff <= .10:
            raise ValueError('observation backoff must be between 0 and 0.10 m')
        if self.reobserve_enabled:
            if not self.vision_enabled:
                raise ValueError("reobserve requires vision.enabled")
            values = [self.reobserve_height, self.reobserve_timeout,
                      self.reobserve_settle, self.reobserve_max_delta, self.reobserve_max_std,
                      self.reobserve_max_frame_gap]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("reobserve parameters must be finite and positive")
        self.box_pose_topic = str(g("box.pose_topic").value)
        self.box_stable_topic = str(g("box.stable_topic").value)
        self.box_collision_enabled = bool(g("box.collision.enabled").value)
        self.box_inner_x = float(g("box.collision.inner_x").value)
        self.box_inner_y = float(g("box.collision.inner_y").value)
        self.box_height = float(g("box.collision.height").value)
        self.box_wall = float(g("box.collision.wall").value)
        self.box_yaw = float(g("box.collision.yaw").value)
        self.box_use_pose_yaw = bool(g("box.collision.use_pose_yaw").value)
        self.pre_grasp_ready_enabled = bool(g("motion.pre_grasp_ready_enabled").value)
        self.initial_observation_enabled = bool(g("motion.initial_observation_enabled").value)
        self.approach_precheck = bool(g("planning.precheck_approach").value)
        self.grasp_pitch_fallbacks = list(g("grasp.pitch_fallbacks").value)
        if any(not math.isfinite(v) or not 0.9 <= v <= 1.45 for v in self.grasp_pitch_fallbacks):
            raise ValueError('grasp pitch fallbacks must be within [0.9, 1.45] rad')
        self.scene_other_cubes = bool(g("vision.scene_other_cubes").value)
        if self.box_collision_enabled and (not math.isfinite(self.box_yaw) or any(
                not math.isfinite(v) or v <= 0 for v in
                [self.box_inner_x, self.box_inner_y, self.box_height, self.box_wall])):
            raise ValueError("box collision geometry requires measured positive dimensions")
        self.grasp_rpy = [float(x) for x in g("grasp_orientation_rpy").value]
        self.transit_rpy = [float(x) for x in g("transit_orientation_rpy").value]
        self.obj_x = float(g("object.x").value)
        self.obj_y = float(g("object.y").value)
        self.obj_z = float(g("object.z").value)
        self.obj_sx = float(g("object.size_x").value)
        self.obj_sy = float(g("object.size_y").value)
        self.obj_sz = float(g("object.size_z").value)
        self.place_x = float(g("place.x").value)
        self.place_y = float(g("place.y").value)
        self.place_z = float(g("place.z").value)
        self.stack_clearance = float(g('place.stack_clearance_m').value)
        if not math.isfinite(self.stack_clearance) or not 0 <= self.stack_clearance <= .005:
            raise ValueError('place.stack_clearance_m must be between 0 and 5 mm')
        self.table_x = float(g("table.x").value)
        self.table_y = float(g("table.y").value)
        self.table_z = float(g("table.z").value)
        self.table_sx = float(g("table.size_x").value)
        self.table_sy = float(g("table.size_y").value)
        self.table_sz = float(g("table.size_z").value)
        self.pre_grasp_h = float(g("motion.pre_grasp_height").value)
        self.declare_parameter('motion.minimum_lift_height', 0.06)
        self.lift_h = float(g("motion.lift_height").value)
        self.minimum_lift_h = float(g('motion.minimum_lift_height').value)
        if not 0.02 <= self.minimum_lift_h <= self.lift_h:
            raise ValueError('minimum lift height must be between 0.02 m and lift_height')
        self.pre_place_h = float(g("motion.pre_place_height").value)
        self.retreat_h = float(g("motion.retreat_height").value)
        self.cart_step = float(g("motion.cartesian_step").value)
        self.jump_threshold = float(g("motion.jump_threshold").value)
        self.min_frac = float(g("motion.min_cartesian_fraction").value)
        self.gx = float(g("grasp.x_offset").value)
        self.gy = float(g("grasp.y_offset").value)
        self.gz = float(g("grasp.z_offset").value)
        self.gripper_action = str(g("gripper.action_name").value)
        self.gripper_open_position = float(g("gripper.open_position_m").value)
        self.gripper_close_position = float(g("gripper.close_position_m").value)
        self.gripper_max_effort = float(g("gripper.max_effort").value)
        self.gripper_timeout = float(g("gripper.timeout").value)
        self.ready_joints = [float(x) for x in g("ready_joints").value]
        self.pre_grasp_ready_joints = [
            float(x) for x in g("pre_grasp_ready_joints").value
        ]

    def _joint_state_cb(self, msg: JointState) -> None:
        index = {name: i for i, name in enumerate(msg.name)}
        if any(name not in index for name in ARM_JOINTS):
            return
        self.q = [float(msg.position[index[name]]) for name in ARM_JOINTS]
        self.joint_state_arrival = time.monotonic()
        self.dq = [
            float(msg.velocity[index[name]])
            if index[name] < len(msg.velocity) else 0.0
            for name in ARM_JOINTS
        ]
        if "joint7" in index and index["joint7"] < len(msg.position):
            self.gripper_position = float(msg.position[index["joint7"]])

    def _make_cube_pose_cb(self, color: str):
        def _cb(msg: PoseStamped) -> None:
            self._cube_pose_cb(color, msg)
        return _cb

    def _make_cube_stable_cb(self, color: str):
        def _cb(msg: Bool) -> None:
            self._cube_stable_cb(color, msg)
        return _cb

    def _cube_pose_cb(self, color: str, msg: PoseStamped) -> None:
        if msg.header.frame_id != self.base_frame:
            return
        self.cube_poses[color] = msg
        self.cube_pose_arrivals[color] = time.monotonic()
        if color == self.target_color:
            self.vision_pose = msg
            self.vision_pose_arrival = self.cube_pose_arrivals[color]

    def _cube_stable_cb(self, color: str, msg: Bool) -> None:
        self.cube_stable[color] = bool(msg.data)
        if color == self.target_color:
            self.vision_stable = bool(msg.data)

    def _vision_pose_cb(self, msg: PoseStamped) -> None:
        self._cube_pose_cb(self.target_color, msg)

    def _vision_stable_cb(self, msg: Bool) -> None:
        self._cube_stable_cb(self.target_color, msg)

    def set_stage_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        self._stage_callback = callback

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _cancelled(self) -> bool:
        return bool(self._cancel_requested)

    def report_stage(self, stage: str, **kwargs) -> None:
        self.log_stage(stage, **kwargs)
        callback = self._stage_callback
        if callback is None:
            return
        try:
            callback(stage)
        except Exception as exc:
            self.get_logger().warn(f"stage callback failed: {exc}")

    def reset_task_state(self, target_color: str, preserve_live_vision=False) -> None:
        """Drop frozen poses from the previous task before a new color is selected."""
        color = target_color.strip().lower()
        self.target_color = color
        self.vision_pose_topic = f"/x5a_vision/{color}_cube_pose"
        self.vision_stable_topic = f"/x5a_vision/{color}_cube_stable"
        if not preserve_live_vision:
            for name in self.allowed_colors:
                self.cube_poses[name] = None
                self.cube_pose_arrivals[name] = 0.0
                self.cube_stable[name] = False
            self.box_pose = None
            self.box_pose_arrival = 0.0
            self.box_stable = False
        self.vision_pose = self.cube_poses.get(color)
        self.vision_pose_arrival = self.cube_pose_arrivals.get(color, 0.0)
        self.vision_stable = self.cube_stable.get(color, False)
        self.last_joints = None
        self._cancel_requested = False
        self.get_logger().info(f"reset task state target_color={color}")

    def clear_planning_scene_objects(self) -> None:
        """Remove leftover attached/world objects so the next task can rebuild the scene."""
        aco = AttachedCollisionObject()
        aco.object.id = "object"
        aco.object.header.frame_id = self.tcp_frame
        aco.object.header.stamp = self.get_clock().now().to_msg()
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [aco]
        self.scene_pub.publish(scene)
        self.publish_box(
            "object",
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
            operation=CollisionObject.REMOVE,
        )
        self.publish_box('source_support', [0., 0., 0.], [.01, .01, .01],
                         operation=CollisionObject.REMOVE)
        if self.box_collision_enabled:
            for i in range(4):
                self.publish_box(f"box_wall_{i}", [0., 0., 0.], [0.01]*3,
                                 operation=CollisionObject.REMOVE)
        time.sleep(0.1)

    def handeye_tf_ok(self) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, Time(), timeout=Duration(seconds=0.5)
            )
            return True
        except Exception as exc:
            self.get_logger().error(f"TF {self.base_frame}->{self.camera_frame} unavailable: {exc}")
            return False

    def _box_pose_cb(self, msg: PoseStamped) -> None:
        if msg.header.frame_id != self.base_frame:
            return
        self.box_pose = msg
        self.box_pose_arrival = time.monotonic()

    def _box_stable_cb(self, msg: Bool) -> None:
        self.box_stable = bool(msg.data)

    def wait_ready(self, timeout: float = 30.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if (
                self.q is not None
                and self.move_ac.wait_for_server(timeout_sec=0.1)
                and self.cart_cli.wait_for_service(timeout_sec=0.1)
                and (
                    self.plan_only
                    or (
                        self.execute_ac.wait_for_server(timeout_sec=0.1)
                        and self.gripper_ac.wait_for_server(timeout_sec=0.1)
                    )
                )
            ):
                return True
            time.sleep(0.05)
        return False

    @staticmethod
    def wait_future(future, timeout: float) -> bool:
        """Wait while the node's dedicated executor thread services callbacks."""
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.done()

    def current_tcp(self) -> Optional[Tuple[float, float, float]]:
        if self.q is None:
            return None
        if self.use_tf_tcp:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, self.tcp_frame, Time(),
                                                      timeout=Duration(seconds=0.3))
                stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                age = self.get_clock().now().nanoseconds * 1e-9 - stamp
                if age < -0.05 or age > 0.5:
                    return None
                t = tf.transform.translation
                return (t.x, t.y, t.z)
            except Exception as exc:
                self.get_logger().error(f"calibrated TCP unavailable: {exc}")
                return None
        t = fk_base_tool0(self.q)[:3, 3]
        return (float(t[0]), float(t[1]), float(t[2]))

    def log_stage(self, stage: str, **kwargs) -> None:
        parts = [f"[{stage}]"]
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        self.get_logger().info(" ".join(parts))

    def execute_moveit_trajectory(self, trajectory: RobotTrajectory, stage: str) -> bool:
        """Execute a planned path through MoveIt's standard execution pipeline."""
        if not trajectory.joint_trajectory.points:
            self.log_stage(stage, execution="FAIL", reason="empty trajectory")
            return False
        if self.require_servo_bounds:
            if self.servo_readiness_fault():
                self.log_stage(stage, execution="FAIL", reason=self.servo_readiness_fault())
                return False
            names = list(trajectory.joint_trajectory.joint_names)
            if set(names) != set(ARM_JOINTS):
                return False
            for point in trajectory.joint_trajectory.points:
                if len(point.positions) != len(names):
                    return False
                values = [point.positions[names.index(name)] for name in ARM_JOINTS]
                if self.servo_bounds_fault(values):
                    self.log_stage(stage, execution="FAIL", reason=self.servo_bounds_fault(values))
                    return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        future = self.execute_ac.send_goal_async(goal)
        if not self.wait_future(future, 10.0):
            self.log_stage(stage, execution="FAIL", reason="goal timeout")
            return False
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.log_stage(stage, execution="FAIL", reason="goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        if not self.wait_future(result_future, self.execution_timeout):
            cancellation = goal_handle.cancel_goal_async()
            self.wait_future(cancellation, 5.0)
            self.log_stage(stage, execution="FAIL", reason="result timeout")
            return False
        result = result_future.result().result
        ok = result.error_code.val == 1
        if ok:
            self.last_joints = list(self.q) if self.q is not None else self.last_joints
        self.log_stage(
            stage,
            execution="PASS" if ok else f"FAIL({result.error_code.val})",
            interface="MoveIt ExecuteTrajectory",
        )
        return ok

    def call_gripper(self, open_gripper: bool) -> bool:
        stage = "OPEN_GRIPPER" if open_gripper else "CLOSE_GRIPPER"
        if self.plan_only:
            self.log_stage(stage, result="SKIP_PLAN_ONLY")
            return True
        if self.require_servo_bounds and self.servo_readiness_fault():
            self.log_stage(stage, result="FAIL", reason=self.servo_readiness_fault())
            return False
        if not self.gripper_ac.wait_for_server(timeout_sec=2.0):
            self.log_stage(stage, result="FAIL", reason="GripperCommand unavailable")
            return False
        target = self.gripper_open_position if open_gripper else self.gripper_close_position
        goal = GripperCommand.Goal()
        goal.command.position = float(target)
        goal.command.max_effort = float(self.gripper_max_effort)
        future = self.gripper_ac.send_goal_async(goal)
        if not self.wait_future(future, 2.0):
            self.log_stage(stage, result="FAIL", reason="goal timeout")
            return False
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.log_stage(stage, result="FAIL", reason="goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        if not self.wait_future(result_future, self.gripper_timeout):
            self.log_stage(stage, result="FAIL", reason="result timeout")
            return False
        response = result_future.result()
        result = response.result
        ok = gripper_result_ok(open_gripper, result, response.status, self.min_held_gap)
        self.log_stage(
            stage,
            result="PASS" if ok else "FAIL",
            target_m=target,
            feedback_m=float(result.position),
            interface="control_msgs/GripperCommand",
        )
        return ok

    def servo_bounds_fault(self, q) -> str:
        if q is None or len(q) != 6:
            return "missing six-joint feedback"
        for i, (v, lo, hi) in enumerate(zip(q, self.servo_lower, self.servo_upper), 1):
            if not math.isfinite(v) or not lo <= v <= hi:
                return f"joint{i}={v:.4f} outside active servo range [{lo:.4f},{hi:.4f}]"
        return ""

    def servo_readiness_fault(self) -> str:
        if time.monotonic() - self.joint_state_arrival > 0.5:
            return "stale joint feedback"
        return self.servo_bounds_fault(self.q)

    def publish_box(
        self,
        name: str,
        xyz: Sequence[float],
        size: Sequence[float],
        frame: Optional[str] = None,
        operation=CollisionObject.ADD,
        yaw: float = 0.0,
    ) -> None:
        co = CollisionObject()
        co.id = name
        co.header.frame_id = frame or self.base_frame
        co.header.stamp = self.get_clock().now().to_msg()
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(size[0]), float(size[1]), float(size[2])]
        pose = Pose()
        pose.orientation = rpy_to_quat(0., 0., yaw)
        pose.position.x, pose.position.y, pose.position.z = map(float, xyz)
        co.primitives = [prim]
        co.primitive_poses = [pose]
        co.operation = operation
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [co]
        self.scene_pub.publish(scene)

    def setup_scene(self) -> None:
        self.publish_box(
            "table",
            [self.table_x, self.table_y, self.table_z],
            [self.table_sx, self.table_sy, self.table_sz],
        )
        self.publish_box(
            "object",
            [self.scene_obj_x, self.scene_obj_y, self.scene_obj_z],
            [self.obj_sx, self.obj_sy, self.obj_sz],
        )
        # A visible upper cube may hide the lower cube completely. Keep its
        # support volume in the scene even when the lower color has no pose.
        table_top = self.table_z + .5*self.table_sz
        support_height = self.scene_obj_z - .5*self.obj_sz - table_top
        if support_height > .5*self.obj_sz:
            yaw = quat_to_yaw(self.vision_pose.pose.orientation) if self.vision_pose is not None else 0.
            self.publish_box('source_support',
                             [self.scene_obj_x, self.scene_obj_y, table_top+.5*support_height],
                             [self.obj_sx, self.obj_sy, support_height], yaw=yaw)
        if self.scene_other_cubes and self.vision_enabled:
            for color in ALLOWED_COLORS:
                key = 'cube_obstacle_' + color
                self.publish_box(key, [0., 0., 0.], [self.obj_sx, self.obj_sy, self.obj_sz],
                                 operation=CollisionObject.REMOVE)
                pose = self.cube_poses.get(color)
                if color == self.target_color or pose is None:
                    continue
                if time.monotonic()-self.cube_pose_arrivals.get(color, 0.) > self.vision_max_age:
                    continue
                x, y, top = pose_to_xyz(pose.pose)
                self.publish_box(key, [x, y, top-self.obj_sz*.5],
                                 [self.obj_sx, self.obj_sy, self.obj_sz],
                                 yaw=quat_to_yaw(pose.pose.orientation))
        time.sleep(0.3)
        self.log_stage("SCENE", table="PASS", object="PASS")

    def setup_box_walls(self, x: float, y: float) -> None:
        if not self.box_collision_enabled:
            return
        w, sx, sy = self.box_wall, self.box_inner_x, self.box_inner_y
        h = self.box_height
        z = self.table_z + self.table_sz * 0.5 + h * 0.5
        walls = [((sx+w)*0.5, 0., w, sy+2*w), (-(sx+w)*0.5, 0., w, sy+2*w),
                 (0., (sy+w)*0.5, sx, w), (0., -(sy+w)*0.5, sx, w)]
        for i, (dx, dy, wx, wy) in enumerate(walls):
            c, s = math.cos(self.box_yaw), math.sin(self.box_yaw)
            self.publish_box(f"box_wall_{i}", [x+c*dx-s*dy, y+s*dx+c*dy, z],
                             [wx, wy, h], yaw=self.box_yaw)
        time.sleep(0.2)

    def attach_object(self, grasp_pose: Optional[Pose] = None) -> bool:
        aco = AttachedCollisionObject()
        aco.link_name = self.tcp_frame
        aco.object.id = "object"
        aco.object.header.frame_id = self.tcp_frame
        aco.object.header.stamp = self.get_clock().now().to_msg()
        aco.object.operation = CollisionObject.ADD
        # object pose relative to tool0 approx identity; use current world pose as attached
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [self.obj_sx, self.obj_sy, self.obj_sz]
        pose = Pose()
        pose.orientation.w = 1.0
        if self.use_tf_tcp:
            world = Pose()
            world.position.x, world.position.y, world.position.z = (
                self.scene_obj_x, self.scene_obj_y, self.scene_obj_z)
            yaw = quat_to_yaw(self.vision_pose.pose.orientation) if self.vision_pose is not None else 0.
            world.orientation = rpy_to_quat(0., 0., yaw)
            if self.plan_only:
                if grasp_pose is None:
                    return False
                # Pure planning has no actual grasp TF. Invert the planned pose.
                import numpy as np
                from scipy.spatial.transform import Rotation
                from geometry_msgs.msg import TransformStamped
                q = grasp_pose.orientation
                R = Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
                t = -R.T @ np.array(pose_to_xyz(grasp_pose))
                inv = Rotation.from_matrix(R.T).as_quat()
                transform = TransformStamped()
                transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float,t)
                (transform.transform.rotation.x, transform.transform.rotation.y,
                 transform.transform.rotation.z, transform.transform.rotation.w) = map(float,inv)
            else:
                try:
                    transform = self.tf_buffer.lookup_transform(self.tcp_frame, self.base_frame, Time(),
                                                                timeout=Duration(seconds=0.5))
                except Exception as exc:
                    self.log_stage("ATTACH_OBJECT", result="FAIL", reason=str(exc))
                    return False
            pose = do_transform_pose(world, transform)
        aco.object.primitives = [prim]
        aco.object.primitive_poses = [pose]
        self._held_object_offset = pose_to_xyz(pose)
        prefix = self.tcp_frame[:-len("tool0")] if self.tcp_frame.endswith("tool0") else ""
        aco.touch_links = [prefix + name for name in ("link6", "link7", "link8", "tool0")]
        # remove world object then attach
        self.publish_box(
            "object",
            [self.scene_obj_x, self.scene_obj_y, self.scene_obj_z],
            [self.obj_sx, self.obj_sy, self.obj_sz],
            operation=CollisionObject.REMOVE,
        )
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [aco]
        self.scene_pub.publish(scene)
        time.sleep(0.2)
        self.log_stage("ATTACH_OBJECT", result="PASS")
        return True

    def detach_object(self, place_xyz: Sequence[float]) -> None:
        # detach and place as world object
        aco = AttachedCollisionObject()
        aco.object.id = "object"
        aco.object.header.frame_id = self.tcp_frame
        aco.object.header.stamp = self.get_clock().now().to_msg()
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [aco]
        self.scene_pub.publish(scene)
        time.sleep(0.1)
        self.publish_box(
            "object",
            place_xyz,
            [self.obj_sx, self.obj_sy, self.obj_sz],
            operation=CollisionObject.ADD,
        )
        time.sleep(0.2)
        self.log_stage("DETACH_OBJECT", result="PASS", place=list(place_xyz))

    def move_joints(
        self,
        joints: Sequence[float],
        stage: str,
        velocity_scaling: float,
        acceleration_scaling: float,
    ) -> bool:
        assert self.q is not None
        g = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.group
        req.num_planning_attempts = self.max_attempts
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = velocity_scaling
        req.max_acceleration_scaling_factor = acceleration_scaling
        rs = RobotState()
        rs.joint_state.name = list(ARM_JOINTS)
        rs.joint_state.position = self.planning_start_joints()
        req.start_state = rs
        if stage == 'INITIAL_OBSERVATION':
            # Bound the home-to-observation plan near its joint endpoints.
            # OMPL still checks collisions; a required wide detour now fails
            # planning instead of executing an unexpectedly large sweep.
            corridor = Constraints()
            for name, start, target in zip(ARM_JOINTS, rs.joint_state.position, joints):
                constraint = JointConstraint()
                constraint.joint_name = name
                constraint.position = .5*(start+target)
                constraint.tolerance_above = constraint.tolerance_below = .5*abs(target-start)+.15
                constraint.weight = 1.0
                corridor.joint_constraints.append(constraint)
            req.path_constraints = corridor
        cons = Constraints()
        for n, v in zip(ARM_JOINTS, joints):
            jc = JointConstraint()
            jc.joint_name = n
            jc.position = float(v)
            jc.tolerance_above = self.joint_goal_tolerance
            jc.tolerance_below = self.joint_goal_tolerance
            jc.weight = 1.0
            cons.joint_constraints.append(jc)
        req.goal_constraints.append(cons)
        g.request = req
        g.planning_options = PlanningOptions()
        # Ask MoveGroup for a plan; execution is sent separately through
        # MoveIt's ExecuteTrajectory action so the controller manager remains
        # the single owner of hardware execution.
        g.planning_options.plan_only = True
        start_tcp = self.current_tcp()
        fut = self.move_ac.send_goal_async(g)
        if not self.wait_future(fut, 20.0):
            self.log_stage(stage, planning="FAIL", reason="goal timeout")
            return False
        gh = fut.result()
        if not gh or not gh.accepted:
            self.log_stage(stage, planning="FAIL", reason="goal not accepted")
            return False
        rf = gh.get_result_async()
        if not self.wait_future(rf, 120.0):
            self.log_stage(stage, planning="FAIL", reason="result timeout")
            return False
        res = rf.result().result
        planning_ok = res.error_code.val == 1
        ok = planning_ok
        if planning_ok:
            self._update_last_joints_from_result(res)
            if self.plan_only:
                self.last_joints = [float(v) for v in joints]
            else:
                ok = self.execute_moveit_trajectory(res.planned_trajectory, stage + "_EXEC")
        actual = self.current_tcp()
        self.log_stage(
            stage,
            planning="PASS" if planning_ok else f"FAIL({res.error_code.val})",
            execution="SKIP" if self.plan_only else ("PASS" if ok else "FAIL"),
            start_tcp=start_tcp,
            actual_tcp=actual,
            velocity_scaling=velocity_scaling,
            acceleration_scaling=acceleration_scaling,
        )
        return ok

    def move_pose(
        self,
        pose: Pose,
        stage: str,
        velocity_scaling: float,
        acceleration_scaling: float,
        try_pose_goal: bool = True,
    ) -> bool:
        assert self.q is not None
        # IK pre-solve for free-space pose goals, then joint-space plan (more reliable on this arm).
        joints = self.ik_joints_for_pose(pose)
        if joints is not None:
            if self.move_joints(
                joints, stage + "_IK", velocity_scaling, acceleration_scaling
            ):
                return True
        else:
            self.log_stage(stage + "_IK", planning="FAIL", reason="IK pre-solve failed")
        if not try_pose_goal:
            return False
        self.last_joints = list(self.q)
        self.get_logger().warn(
            f"{stage}: retry with Pose Goal from current joint state {self.last_joints}"
        )
        pose_stage = stage + "_POSE"
        g = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.group
        req.num_planning_attempts = self.max_attempts
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = velocity_scaling
        req.max_acceleration_scaling_factor = acceleration_scaling
        rs = RobotState()
        rs.joint_state.name = list(ARM_JOINTS)
        rs.joint_state.position = self.planning_start_joints()
        req.start_state = rs
        # position constraint around target with small box
        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.tcp_frame
        pc.weight = 1.0
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [0.01]
        pc.constraint_region.primitives = [prim]
        pc.constraint_region.primitive_poses = [pose]
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.tcp_frame
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.15
        oc.absolute_y_axis_tolerance = 0.15
        oc.absolute_z_axis_tolerance = 0.15
        oc.weight = 1.0
        cons = Constraints()
        cons.position_constraints.append(pc)
        cons.orientation_constraints.append(oc)
        req.goal_constraints.append(cons)
        g.request = req
        g.planning_options = PlanningOptions()
        g.planning_options.plan_only = True
        target = pose_to_xyz(pose)
        start_tcp = self.current_tcp()
        fut = self.move_ac.send_goal_async(g)
        if not self.wait_future(fut, 20.0):
            self.log_stage(pose_stage, planning="FAIL", reason="goal timeout", target_tcp=target)
            return False
        gh = fut.result()
        if not gh or not gh.accepted:
            self.log_stage(pose_stage, planning="FAIL", target_tcp=target, reason="not accepted")
            return False
        rf = gh.get_result_async()
        if not self.wait_future(rf, 120.0):
            self.log_stage(pose_stage, planning="FAIL", reason="result timeout", target_tcp=target)
            return False
        res = rf.result().result
        planning_ok = res.error_code.val == 1
        ok = planning_ok
        if planning_ok:
            self._update_last_joints_from_result(res)
            if not self.plan_only:
                ok = self.execute_moveit_trajectory(res.planned_trajectory, pose_stage + "_EXEC")
        actual = self.current_tcp()
        self.log_stage(
            pose_stage,
            planning="PASS" if planning_ok else f"FAIL({res.error_code.val})",
            execution="SKIP" if self.plan_only else ("PASS" if ok else "FAIL"),
            target_tcp=target,
            start_tcp=start_tcp,
            actual_tcp=actual,
            velocity_scaling=velocity_scaling,
            acceleration_scaling=acceleration_scaling,
        )
        return ok

    def cartesian_to(self, pose: Pose, stage: str, velocity_scaling=1.0,
                     acceleration_scaling=1.0) -> Tuple[bool, float]:
        assert self.q is not None
        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = self.group
        req.link_name = self.tcp_frame
        req.max_step = self.cart_step
        req.jump_threshold = self.jump_threshold
        req.avoid_collisions = True
        start_q = self.planning_start_joints()
        req.start_state.joint_state.name = list(ARM_JOINTS) + ['joint7', 'joint8']
        req.start_state.joint_state.position = start_q + [self.gripper_position]*2
        wp = Pose()
        wp.position = pose.position
        wp.orientation = pose.orientation
        req.waypoints = [wp]
        start_tcp = self.current_tcp()
        target = pose_to_xyz(pose)
        fut = self.cart_cli.call_async(req)
        if not self.wait_future(fut, 30.0):
            self.log_stage(stage, planning="FAIL", fraction=0.0, reason="service timeout")
            return False, 0.0
        res = fut.result()
        if res is None:
            self.log_stage(stage, planning="FAIL", fraction=0.0, reason="no response")
            return False, 0.0
        frac = float(res.fraction)
        if res.error_code.val != 1 or frac < self.min_frac:
            self.log_stage(
                stage,
                planning="FAIL",
                fraction=frac,
                target_tcp=target,
                start_tcp=start_tcp,
            )
            return False, frac
        try:
            res.solution = scale_cartesian_timing(res.solution, velocity_scaling, acceleration_scaling)
        except ValueError as exc:
            self.log_stage(stage, planning='FAIL', reason=str(exc))
            return False, frac
        self._update_last_joints_from_traj(res.solution)
        if self.plan_only:
            self.log_stage(
                stage,
                planning="PASS",
                execution="SKIP",
                fraction=frac,
                target_tcp=target,
                start_tcp=start_tcp,
            )
            return True, frac
        ok = self.execute_moveit_trajectory(res.solution, stage + "_EXEC")
        actual = self.current_tcp()
        self.log_stage(
            stage,
            planning="PASS",
            execution="PASS" if ok else "FAIL",
            fraction=frac,
            target_tcp=target,
            start_tcp=start_tcp,
            actual_tcp=actual,
        )
        return ok, frac


    def planning_start_joints(self) -> List[float]:
        if self.last_joints is not None:
            return list(self.last_joints)
        if self.q is not None:
            return list(self.q)
        raise RuntimeError("no joint state available")

    def _update_last_joints_from_result(self, res) -> None:
        try:
            traj = res.planned_trajectory.joint_trajectory
            if traj.points:
                names = list(traj.joint_names)
                pos = list(traj.points[-1].positions)
                self.last_joints = [float(pos[names.index(n)]) for n in ARM_JOINTS]
                return
        except Exception:
            pass
        if self.q is not None:
            self.last_joints = list(self.q)

    def _update_last_joints_from_traj(self, robot_traj: RobotTrajectory) -> None:
        try:
            traj = robot_traj.joint_trajectory
            if traj.points:
                names = list(traj.joint_names)
                pos = list(traj.points[-1].positions)
                self.last_joints = [float(pos[names.index(n)]) for n in ARM_JOINTS]
        except Exception:
            if self.q is not None:
                self.last_joints = list(self.q)


    def ik_seeds(self, x: Optional[float] = None, y: Optional[float] = None) -> List[List[float]]:
        seeds: List[List[float]] = []
        if self.last_joints is not None:
            seeds.append(list(self.last_joints))
        if x is not None and y is not None:
            yaw = math.atan2(y, x)
            seeds.append([yaw, 0.85, 0.95, -0.55, 0.0, 0.0])
        seeds.append(list(self.pre_grasp_ready_joints))
        unique: List[List[float]] = []
        for seed in seeds:
            if len(seed) < 6:
                continue
            if any(
                all(abs(a - b) < 1e-3 for a, b in zip(seed, existing))
                for existing in unique
            ):
                continue
            unique.append([float(v) for v in seed[:6]])
        return unique[:3]

    def ik_joints_for_pose(self, pose: Pose, collision_aware_only=False) -> Optional[List[float]]:
        if not self.ik_cli.wait_for_service(timeout_sec=2.0):
            return None
        seeds = self.ik_seeds(pose.position.x, pose.position.y)
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = pose
        req.ik_request.pose_stamped = ps
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = 200000000
        # Collision-aware first, then one collision-unaware pass on the last seed.
        attempts = [(seed, True) for seed in seeds]
        if not collision_aware_only:
            attempts.append((seeds[-1], False))
        for seed, avoid in attempts:
            req.ik_request.robot_state.joint_state.name = list(ARM_JOINTS) + ['joint7', 'joint8']
            req.ik_request.robot_state.joint_state.position = seed + [self.gripper_position]*2
            req.ik_request.avoid_collisions = avoid
            fut = self.ik_cli.call_async(req)
            if not self.wait_future(fut, 1.5):
                continue
            res = fut.result()
            if not res or res.error_code.val != 1:
                continue
            names = list(res.solution.joint_state.name)
            pos = list(res.solution.joint_state.position)
            try:
                return [float(pos[names.index(n)]) for n in ARM_JOINTS]
            except Exception:
                return None
        return None

    def vertical_move(
        self,
        pose: Pose,
        stage: str,
        fallback_velocity_scaling: float,
        fallback_acceleration_scaling: float,
    ) -> Tuple[bool, float]:
        """Prefer cartesian; fallback to IK+joint planning for vertical stages."""
        ok, frac = self.cartesian_to(pose, stage + "_CART",
                                    fallback_velocity_scaling, fallback_acceleration_scaling)
        if ok:
            return True, frac
        if self.require_cartesian:
            self.log_stage(stage, planning="FAIL", reason="complete cartesian path required; fallback disabled")
            return False, frac
        self.get_logger().warn(
            f"{stage}: cartesian fraction={frac:.3f}, fallback to MoveIt pose planning"
        )
        if self.move_pose(
            pose,
            stage + "_POSE",
            fallback_velocity_scaling,
            fallback_acceleration_scaling,
        ):
            return True, 1.0
        self.get_logger().warn(f"{stage}: pose planning failed, fallback to IK joint plan")
        joints = self.ik_joints_for_pose(pose)
        if joints is None:
            self.log_stage(stage, planning="FAIL", reason="IK failed", target_tcp=pose_to_xyz(pose))
            return False, frac
        ok2 = self.move_joints(
            joints,
            stage + "_IK",
            fallback_velocity_scaling,
            fallback_acceleration_scaling,
        )
        if ok2:
            return True, 1.0
        return False, frac

    def lift_with_reachable_height(self, start, orientation):
        """Retry shorter complete lifts only when planning failed before motion."""
        heights = [self.lift_h]
        height = self.lift_h-.02
        while height > self.minimum_lift_h+1e-6:
            heights.append(height)
            height -= .02
        if self.minimum_lift_h < self.lift_h-1e-6:
            heights.append(self.minimum_lift_h)
        fraction = 0.0
        for height in heights:
            if self._cancelled():
                return False, fraction
            lift = self.pose_rpy(start[0], start[1], start[2]+height, orientation)
            self.log_stage('LIFT_CANDIDATE', height_m=height)
            ok, fraction = self.vertical_move(lift, 'LIFT', self.lift_retreat_vel, self.lift_retreat_acc)
            # An execution failure reports a complete planned fraction: stop;
            # never retry after a trajectory may have partially executed.
            if ok or fraction >= self.min_frac:
                return ok, fraction
        return False, fraction

    def retreat_with_reachable_height(self, start, orientation):
        """After release, try shorter complete retreats before planned transit.

        No partial path is executed. Retry only a planning failure from the
        same start, and stop immediately after any possible execution failure.
        """
        heights = [self.retreat_h]
        height = self.retreat_h-.01
        while height >= .02-1e-6:
            heights.append(height)
            height -= .01
        fraction = 0.0
        for height in heights:
            if self._cancelled():
                return False, fraction
            retreat = self.pose_rpy(start[0], start[1], start[2]+height, orientation)
            self.log_stage('RETREAT_CANDIDATE', height_m=height)
            ok, fraction = self.vertical_move(retreat, 'RETREAT', self.lift_retreat_vel, self.lift_retreat_acc)
            if ok or fraction >= self.min_frac:
                return ok, fraction
        return False, fraction

    def make_pose(self, x: float, y: float, z: float) -> Pose:
        return pose_xyz_quat(x, y, z, self.grasp_quat)

    def make_transit_pose(self, x: float, y: float, z: float) -> Pose:
        return pose_xyz_quat(x, y, z, self.transit_quat)

    def pose_rpy(self, x: float, y: float, z: float, rpy: Sequence[float]) -> Pose:
        return pose_xyz_quat(x, y, z, rpy_to_quat(rpy[0], rpy[1], rpy[2]))

    def grasp_orientation_candidates(
        self,
        x: float,
        y: float,
        cube_yaw: Optional[float] = None,
    ) -> List[List[float]]:
        # Align finger pads with a pair of opposite cube faces. A 45-degree
        # offset closes on adjacent faces and the cube slips out.
        yaw0 = math.atan2(y, x)
        pitch = float(self.grasp_rpy[1]) if len(self.grasp_rpy) > 1 else 1.45
        cands: List[List[float]] = []
        if cube_yaw is not None:
            pitches = [pitch] + (self.grasp_pitch_fallbacks if self.approach_precheck else [])
            for candidate_pitch in pitches:
                for axis in (cube_yaw, cube_yaw + 0.5 * math.pi):
                    yaw = wrap_pi(axis)
                    cands.append([0.0, candidate_pitch, yaw])
                    cands.append([0.0, candidate_pitch, wrap_pi(yaw + math.pi)])
        else:
            cands.extend([[0.0, pitch, 0.0], [0.0, pitch, 0.5 * math.pi]])
        cands.extend([list(self.grasp_rpy), [0.0, pitch, 0.20], [0.0, pitch, -0.20]])
        if math.hypot(x, y) >= 0.42:
            cands.extend([[0.0, pitch, yaw0], [0.0, 1.30, yaw0]])
        return unique_rpy(cands, cap=20 if self.approach_precheck else 12)

    def set_preflight_target_present(self, present: bool) -> bool:
        """Synchronously change only the grasp target, keeping all obstacles."""
        if not self.apply_scene_cli.wait_for_service(timeout_sec=2.0):
            return False
        co = CollisionObject()
        co.id = 'object'
        co.header.frame_id = self.base_frame
        co.operation = CollisionObject.ADD if present else CollisionObject.REMOVE
        if present:
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [self.obj_sx, self.obj_sy, self.obj_sz]
            co.primitives = [primitive]
            co.primitive_poses = [self.pose_rpy(
                self.scene_obj_x, self.scene_obj_y, self.scene_obj_z, [0., 0., 0.])]
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.world.collision_objects = [co]
        future = self.apply_scene_cli.call_async(req)
        return (self.wait_future(future, 5.0) and future.result() is not None
                and bool(future.result().success))

    def precheck_approach_path(self, start_q, contact: Pose) -> bool:
        """Check the full descent from a virtual hover; never execute it here."""
        if self.servo_bounds_fault(start_q):
            return False
        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.group_name = self.group
        req.link_name = self.tcp_frame
        req.max_step = self.cart_step
        req.jump_threshold = self.jump_threshold
        req.avoid_collisions = True
        # Include the actual open fingers, which can hit a nearby box rim even
        # when the arm and a closed-gripper IK solution fit.
        req.start_state.joint_state.name = list(ARM_JOINTS) + ['joint7', 'joint8']
        req.start_state.joint_state.position = list(start_q) + [self.gripper_position]*2
        req.waypoints = [contact]
        future = self.cart_cli.call_async(req)
        if not self.wait_future(future, 10.0):
            return False
        res = future.result()
        if res is None:
            return False
        trajectory = res.solution.joint_trajectory
        complete = res.error_code.val == 1 and res.fraction >= max(self.min_frac, .9999)
        bounded = False
        if complete and trajectory.points and all(name in trajectory.joint_names for name in ARM_JOINTS):
            indices = [trajectory.joint_names.index(name) for name in ARM_JOINTS]
            bounded = all(not self.servo_bounds_fault([p.positions[i] for i in indices])
                          for p in trajectory.points)
        self.log_stage('APPROACH_PREFLIGHT', result='PASS' if complete and bounded else 'FAIL',
                       fraction=float(res.fraction), target_tcp=pose_to_xyz(contact))
        return bool(complete and bounded)

    def move_pregrasp_candidates(self, x, y, z, orientations, hover_height):
        """Reject unusable approaches before moving to an observation pose."""
        candidates = []
        restored = False
        try:
            if not self.set_preflight_target_present(False):
                return None
            for index, rpy in enumerate(orientations):
                if self._cancelled():
                    break
                backoff = self.reobserve_backoff if self.reobserve_enabled else 0.0
                hover = self.pose_rpy(x-backoff*math.cos(rpy[2]),
                                      y-backoff*math.sin(rpy[2]),
                                      z+hover_height, rpy)
                joints = self.ik_joints_for_pose(hover)
                if joints is None:
                    continue
                contact = self.pose_rpy(x, y, z, rpy)
                if self.precheck_approach_path(joints, contact):
                    candidates.append((index, list(rpy), joints))
                    if len(candidates) >= 3:
                        break
        finally:
            # Free-space transit must still avoid the target cube. If restoring
            # it fails, no hardware trajectory is permitted.
            restored = self.set_preflight_target_present(True)
        if not restored or self._cancelled():
            return None
        for index, rpy, joints in candidates:
            self.log_stage('PREGRASP_SELECTED', rpy=rpy, approach_prechecked=True)
            if self.move_joints(joints, f'MOVE_PRE_GRASP_{index+1}_IK',
                                self.transit_vel, self.transit_acc):
                return rpy
        return None

    def place_orientation_candidates(self, x: float, y: float) -> List[List[float]]:
        yaw0 = math.atan2(y, x)
        radius = math.hypot(x, y)
        # Far +Y box succeeded at pitch 1.00; try that before the vertical set.
        pitches = [1.00, 1.15, 1.30] if radius >= 0.42 else [1.15, 1.00, 1.30]
        cands = [[0.0, pitch, yaw0] for pitch in pitches]
        cands.append(list(self.transit_rpy))
        return unique_rpy(cands, cap=4)

    def move_pose_candidates(
        self,
        x: float,
        y: float,
        z: float,
        orientations: Sequence[Sequence[float]],
        stage: str,
        velocity_scaling: float,
        acceleration_scaling: float,
        xy_offsets: Optional[Sequence[Tuple[float, float]]] = None,
        observation_backoff_m: float = 0.0,
    ) -> Optional[List[float]]:
        offsets = list(xy_offsets) if xy_offsets is not None else [(0.0, 0.0)]
        for ox, oy in offsets:
            for index, rpy in enumerate(orientations):
                px = x + ox - observation_backoff_m*math.cos(rpy[2])
                py = y + oy - observation_backoff_m*math.sin(rpy[2])
                pose = self.pose_rpy(px, py, z, rpy)
                label = f"{stage}_{index + 1}"
                self.log_stage(
                    label + "_TRY",
                    xyz=(px, py, z),
                    rpy=[round(v, 3) for v in rpy],
                )
                last = (
                    ox == offsets[-1][0]
                    and oy == offsets[-1][1]
                    and index == len(orientations) - 1
                )
                if self.move_pose(
                    pose, label, velocity_scaling, acceleration_scaling,
                    try_pose_goal=last,
                ):
                    return list(rpy)
        return None

    def move_pre_grasp_ready(self, x: float, y: float) -> bool:
        joints = list(self.pre_grasp_ready_joints)
        if len(joints) < 6:
            joints = [0.72, 0.85, 0.95, -0.55, 0.0, 0.0]
        joints[0] = math.atan2(y, x)
        self.log_stage("MOVE_READY", joints=[round(v, 3) for v in joints])
        return self.move_joints(
            joints, "MOVE_READY", self.transit_vel, self.transit_acc
        )

    def move_initial_observation(self, observe_target: bool = True) -> bool:
        """Expose the workspace to the wrist camera before requesting detections."""
        if self._cancelled():
            return False
        if not self.plan_only and self.count_publishers('/arm_cmd') != 1:
            self.log_stage('INITIAL_OBSERVATION', result='FAIL',
                           reason='close manual arm rqt; /arm_cmd must have only the hardware adapter')
            return False
        # The table is known even when the home camera sees no objects. Keep
        # existing other obstacles; never use an unplanned home-to-ready jump.
        self.publish_box('table', [self.table_x, self.table_y, self.table_z],
                         [self.table_sx, self.table_sy, self.table_sz])
        time.sleep(.15)
        self.report_stage('MOVE_INITIAL_OBSERVATION')
        already_ready = (self.q is not None and len(self.q) == len(self.ready_joints) == 6
                         and all(abs(wrap_pi(a-b)) <= .01 for a,b in zip(self.q,self.ready_joints)))
        if not already_ready:
            if not self.move_joints(self.ready_joints, 'INITIAL_OBSERVATION',
                                    self.transit_vel, self.transit_acc):
                return False
        if not self.plan_only:
            self.last_joints = None
            # A recent frame from during the move must not establish the target.
            if observe_target and self.reobserve_enabled and self.wait_for_reobserved_cube() is None:
                return False
        return True

    def camera_facing_ready_joints(self) -> List[float]:
        joints = list(self.pre_grasp_ready_joints)
        if len(joints) < 6:
            joints = [0.72, 0.85, 0.95, -0.55, 0.0, 0.0]
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, Time(), timeout=Duration(seconds=0.3)
            )
            joints[0] = math.atan2(tf.transform.translation.y, tf.transform.translation.x)
        except Exception:
            pass
        return joints

    def log_calibration(self) -> None:
        try:
            from pathlib import Path

            import yaml
            from ament_index_python.packages import get_package_share_directory

            path = (Path(self.calibration_result_file) if self.calibration_result_file else
                    Path(get_package_share_directory("x5a_handeye")) / "config" / "handeye_result.yaml")
            data = yaml.safe_load(path.read_text())
            # yaml 里键存在但值为 null（如全样本训练时 holdout_error: null）
            hold = data.get("holdout_error") or {}
            samples = data.get("samples") or {}
            self.log_stage(
                "HANDEYE",
                file=str(path),
                source=samples.get("source"),
                solver=data.get("solver"),
                holdout_mean_mm=round(float(hold.get("translation_mean_m", 0.0)) * 1000.0, 2),
                holdout_max_mm=round(float(hold.get("translation_max_m", 0.0)) * 1000.0, 2),
            )
        except Exception as exc:
            self.get_logger().warn(f"HANDEYE yaml unread: {exc}")
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, Time(), timeout=Duration(seconds=0.5)
            )
            t = tf.transform.translation
            self.log_stage("TF_BASE_CAMERA_LINK", xyz=(t.x, t.y, t.z))
        except Exception as exc:
            self.get_logger().warn(f"TF base_link->camera_link unavailable: {exc}")

    def vision_snapshot(self):
        """Describe the same pose, freshness and stability gates used for grasping."""
        now = time.monotonic()

        def describe(pose, arrival, stable, max_age):
            xyz = pose_to_xyz(pose.pose) if pose is not None else None
            age = max(0.0, now - arrival) if arrival else None
            if xyz is None:
                status, label = 'missing', '未收到有效坐标'
            elif not all(math.isfinite(v) for v in xyz):
                status, label = 'invalid', '坐标无效'
            elif age is None or age > max_age:
                status, label = 'stale', '坐标已过期'
            elif not stable:
                status, label = 'unstable', '有坐标，但尚未稳定'
            else:
                status, label = 'stable', '定位稳定'
            return {'status': status, 'status_zh': label, 'ready': status == 'stable',
                    'age_s': age, 'xyz': xyz if xyz and all(math.isfinite(v) for v in xyz) else None}

        snapshot = {color: describe(self.cube_poses.get(color),
                                   self.cube_pose_arrivals.get(color, 0.0),
                                   self.cube_stable.get(color, False), self.vision_max_age) for color in ALLOWED_COLORS}
        snapshot['box'] = describe(self.box_pose, self.box_pose_arrival, self.box_stable, self.box_max_age)
        return snapshot

    def wait_for_frozen_vision(
        self,
        reuse_frozen_box: bool = False,
        destination_color: str = 'box',
        require_destination: bool = True,
    ) -> Optional[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ]:
        """Freeze the selected cube. Box is frozen once per sequence when requested."""
        self._vision_wait_failure = ''
        reuse_box = bool(reuse_frozen_box and self._session_box_xyz is not None)
        deadline = time.monotonic() + self.vision_wait_timeout
        self.log_stage(
            "WAIT_FOR_VISION",
            target_color=self.target_color,
            timeout=self.vision_wait_timeout,
            reuse_frozen_box=reuse_box,
        )
        if reuse_box:
            self.log_stage("REUSE_FROZEN_BOX", box_xyz=self._session_box_xyz)
        while time.monotonic() < deadline:
            if self._cancelled():
                self.log_stage(
                    "WAIT_FOR_VISION", result="CANCELLED",
                    target_color=self.target_color,
                )
                return None
            now = time.monotonic()
            pose = self.cube_poses.get(self.target_color)
            arrival = self.cube_pose_arrivals.get(self.target_color, 0.0)
            stable = self.cube_stable.get(self.target_color, False)
            cube_age = now - arrival
            cube_ok = (
                pose is not None
                and stable
                and cube_age <= self.vision_max_age
            )
            if not require_destination:
                box_ok, box_xyz, box_age = True, (0.0, 0.0, 0.0), 0.0
            elif destination_color != 'box':
                destination = self.cube_poses.get(destination_color)
                box_age = now - self.cube_pose_arrivals.get(destination_color, 0.0)
                box_ok = (destination is not None and self.cube_stable.get(destination_color, False)
                          and box_age <= self.vision_max_age)
                box_xyz = pose_to_xyz(destination.pose) if destination is not None else None
            elif reuse_box:
                box_ok = True
                box_xyz = self._session_box_xyz
                box_age = 0.0
            else:
                box_age = now - self.box_pose_arrival
                box_ok = (
                    self.box_pose is not None
                    and self.box_stable
                    and box_age <= self.box_max_age
                )
                box_xyz = pose_to_xyz(self.box_pose.pose) if self.box_pose is not None else None
            if cube_ok and box_ok and pose is not None and box_xyz is not None:
                cube_xyz = pose_to_xyz(pose.pose)
                if all(math.isfinite(v) for v in cube_xyz + box_xyz):
                    self.vision_pose = pose
                    self.vision_pose_arrival = arrival
                    self.vision_stable = True
                    if not require_destination:
                        pass
                    elif destination_color != 'box':
                        self._destination_yaw = quat_to_yaw(destination.pose.orientation)
                    elif not reuse_box:
                        self._session_box_xyz = box_xyz
                        if self.box_use_pose_yaw:
                            self.box_yaw = quat_to_yaw(self.box_pose.pose.orientation)
                    self.log_stage(
                        "STABLE_DETECTION",
                        target_color=self.target_color,
                        cube_age=cube_age,
                        box_age=box_age,
                        cube_xyz=cube_xyz,
                        box_xyz=box_xyz,
                        reused_box=reuse_box,
                    )
                    self.log_stage(
                        "FREEZE_POSES",
                        target_color=self.target_color,
                        cube_xyz=cube_xyz,
                        box_xyz=box_xyz,
                        reused_box=reuse_box,
                    )
                    return cube_xyz, box_xyz
            time.sleep(0.05)
        snapshot = self.vision_snapshot()
        label = {'red': '红块', 'white': '白块', 'orange': '橙块'}[self.target_color]
        box_status = '使用已冻结坐标' if reuse_box else snapshot[destination_color]['status_zh']
        destination_label = {'box': '盒子', 'red': '目标红块', 'white': '目标白块', 'orange': '目标橙块'}[destination_color]
        self._vision_wait_failure = (
            f"定位等待超时：{label}{snapshot[self.target_color]['status_zh']}；{destination_label}{box_status}。"
            '未开始抓取，请等定位恢复后直接重试抓取'
        )
        if not require_destination:
            self._vision_wait_failure = (f"定位等待超时：{label}{snapshot[self.target_color]['status_zh']}。"
                                         '未开始抓取，请等定位恢复后直接重试抓取')
        self.log_stage(
            "WAIT_FOR_VISION", result="FAIL", target_color=self.target_color,
            cube_stable=self.cube_stable.get(self.target_color, False),
            box_stable=self.box_stable if not reuse_box else True,
            cube_age=time.monotonic() - self.cube_pose_arrivals.get(self.target_color, 0.0),
            box_age=0.0 if reuse_box else time.monotonic() - self.box_pose_arrival,
            reuse_frozen_box=reuse_box,
        )
        return None

    def wait_for_reobserved_cube(self) -> Optional[Tuple[float, float, float]]:
        """Require fresh feedback and new camera frames after the arm settles.

        The box remains frozen in base coordinates; it need not still be visible
        in the close view. The selected cube's orientation remains unchanged.
        """
        deadline = time.monotonic() + self.reobserve_timeout
        anchor = None
        settled_since = time.monotonic()
        window = None
        while time.monotonic() < deadline:
            if self._cancelled():
                return None
            now = time.monotonic()
            q = None if self.q is None else list(self.q)
            if (q is None or now - self.joint_state_arrival > 0.5
                    or not all(math.isfinite(v) for v in q)):
                anchor, window = None, None
                settled_since = now
                time.sleep(0.02)
                continue
            if anchor is None or any(abs(wrap_pi(a - b)) > 0.003 for a, b in zip(q, anchor)):
                anchor, window = q, None
                settled_since = now
            if self.dq is not None and any(not math.isfinite(v) or abs(v) > 0.02 for v in self.dq):
                window = None
                settled_since = now
            if now - settled_since < self.reobserve_settle:
                time.sleep(0.02)
                continue
            if window is None:
                window = ReobserveWindow(self.get_clock().now().nanoseconds,
                                         max_std_m=self.reobserve_max_std,
                                         max_gap_s=self.reobserve_max_frame_gap)
            pose = self.cube_poses.get(self.target_color)
            arrival = self.cube_pose_arrivals.get(self.target_color, 0.0)
            if pose is not None and pose.header.frame_id == self.base_frame:
                stamp_ns = pose.header.stamp.sec * 1000000000 + pose.header.stamp.nanosec
                image_age = (self.get_clock().now().nanoseconds - stamp_ns) * 1e-9
                valid = (self.cube_stable.get(self.target_color, False)
                         and now - arrival <= self.vision_max_age
                         and -0.05 <= image_age <= self.vision_max_age)
                # Evaluate image age when a new frame arrives. A previously
                # accepted frame aging in the cache between camera callbacks
                # must not erase all the other fresh samples. A loss of stable
                # detection still clears the window immediately.
                xyz = None
                if stamp_ns > window.last_stamp or not self.cube_stable.get(self.target_color, False):
                    xyz = window.add(stamp_ns, pose_to_xyz(pose.pose), valid)
                if xyz is not None:
                    self.log_stage("REOBSERVE_CUBE", result="PASS", cube_xyz=xyz,
                                   image_age=image_age)
                    return xyz
            time.sleep(0.02)
        self.log_stage("REOBSERVE_CUBE", result="FAIL", reason="no fresh stationary detection")
        return None

    def correct_pregrasp(self, original_top, chosen_rpy, hover_height):
        self.report_stage("REOBSERVING")
        observed = self.wait_for_reobserved_cube()
        if observed is None:
            return None
        try:
            delta = bounded_correction(original_top, observed, self.reobserve_max_delta)
        except ValueError as exc:
            self.log_stage("REOBSERVE_CUBE", result="FAIL", reason=str(exc),
                           original_top=original_top, observed_top=observed)
            return None
        self.scene_obj_x, self.scene_obj_y = observed[:2]
        self.scene_obj_z = observed[2] - 0.5 * self.obj_sz
        self.publish_box("object", [self.scene_obj_x, self.scene_obj_y, self.scene_obj_z],
                         [self.obj_sx, self.obj_sy, self.obj_sz], frame=self.base_frame)
        # Replan from actual feedback, not the endpoint cached by the first plan.
        self.last_joints = None
        corrected = self.pose_rpy(observed[0] + self.gx-self.reobserve_backoff*math.cos(chosen_rpy[2]),
                                  observed[1] + self.gy-self.reobserve_backoff*math.sin(chosen_rpy[2]),
                                  observed[2] + self.gz + hover_height, chosen_rpy)
        self.report_stage("CORRECTING_PRE_GRASP")
        if not self.move_pose(corrected, "CORRECT_PRE_GRASP", self.transit_vel, self.transit_acc):
            return None
        self.report_stage("VERIFYING_CORRECTION")
        verified = self.wait_for_reobserved_cube()
        if verified is None:
            return None
        try:
            bounded_correction(observed, verified, self.reobserve_verify_tolerance)
        except ValueError:
            self.log_stage("VERIFY_CORRECTION", result="FAIL", reason="residual exceeds configured tolerance",
                           tolerance_m=self.reobserve_verify_tolerance)
            return None
        self.last_joints = None
        self.scene_obj_x, self.scene_obj_y = verified[:2]
        self.scene_obj_z = verified[2] - 0.5 * self.obj_sz
        self.log_stage("VERIFY_CORRECTION", result="PASS", correction=delta,
                       corrected_top=verified)
        return verified

    def _fail(self, message: str) -> Tuple[bool, str]:
        self.report_stage("FAILED", reason=message)
        return False, message

    def _cube_candidates(self, max_age: float = 2.0, require_stable: bool = False):
        now = time.monotonic()
        box = self.box_pose
        if box is None or (now - self.box_pose_arrival) > max_age:
            return [], "没有看到放置盒子"
        box_xy = (box.pose.position.x, box.pose.position.y)
        found = []
        for color in ALLOWED_COLORS:
            pose = self.cube_poses.get(color)
            arrival = self.cube_pose_arrivals.get(color, 0.0)
            if pose is None or (now - arrival) > max_age:
                continue
            if require_stable and not self.cube_stable.get(color, False):
                continue
            xyz = pose_to_xyz(pose.pose)
            if not all(math.isfinite(v) for v in xyz):
                continue
            dist = math.hypot(xyz[0] - box_xy[0], xyz[1] - box_xy[1])
            found.append((dist, color, xyz))
        if not found:
            return [], "没有看到可抓取的方块"
        found.sort(key=lambda item: item[0])
        return found, ""

    def select_color_by_box_distance(self, selector: str) -> Tuple[Optional[str], str]:
        """Pick the cube nearest to or farthest from the box center (XY)."""
        name = selector.strip().lower()
        if name not in ALLOWED_SELECTORS:
            return None, f"invalid selector: {selector!r}"
        deadline = time.monotonic() + max(3.0, min(self.vision_wait_timeout, 8.0))
        last_err = "没有看到可抓取的方块"
        while time.monotonic() < deadline:
            if self._cancelled():
                return None, "任务已取消"
            candidates, err = self._cube_candidates(require_stable=True)
            if not candidates:
                candidates, err = self._cube_candidates(require_stable=False)
            if candidates:
                chosen = candidates[0] if name == "nearest" else candidates[-1]
                detail = ", ".join(
                    f"{color}={dist:.3f}m" for dist, color, _xyz in candidates
                )
                self.report_stage(
                    "SELECT_BY_BOX",
                    selector=name,
                    chosen=chosen[1],
                    distance=chosen[0],
                    ranking=detail,
                )
                return chosen[1], ""
            last_err = err
            time.sleep(0.05)
        return None, last_err

    def on_object_picked(self, color):
        """Task-server hook for an actually attached object."""

    def on_object_placed(self, color, center, destination):
        """Task-server hook after successful release and detachment."""

    def execute_place_motion(self, place_xyz, place_oris, stack_center=None, skip_return_home=False):
        """Place the already attached cube; never repeat approach or grasp."""
        place_x, place_y, place_z = place_xyz
        stacking = stack_center is not None
        self.report_stage("MOVE_PRE_PLACE")
        chosen_place = self.move_pose_candidates(
            place_x, place_y, place_z + self.pre_place_h,
            place_oris, "MOVE_PRE_PLACE", self.transit_vel, self.transit_acc,
            xy_offsets=[(0.0, 0.0)] if stacking else [(0.0, 0.0), (0.01, 0.0), (0.0, -0.01)],
        )
        if chosen_place is None:
            return self._fail("pre-place planning failed")
        place = self.pose_rpy(place_x, place_y, place_z, chosen_place)
        self.report_stage("DESCENDING")
        ok, frac_desc = self.vertical_move(
            place, "DESCEND", self.precision_vel, self.precision_acc
        )
        if not ok:
            return self._fail("descend planning or execution failed")
        self.report_stage("RELEASING")
        released_color = getattr(self, '_held_color', None)
        if not self.call_gripper(True):
            return self._fail("release gripper failed")
        table_top = self.table_z + 0.5 * self.table_sz
        placed_center = stack_center if stacking else [place_x, place_y, table_top + 0.5 * self.obj_sz]
        self.detach_object(placed_center)
        callback = getattr(self, 'on_object_placed', None)
        if not self.plan_only and callback is not None:
            callback(released_color, placed_center, getattr(self, '_placement_destination', None))
        retreat_start = pose_to_xyz(place) if self.plan_only else self.current_tcp()
        if retreat_start is None:
            return self._fail("no TCP after place")
        retreat = self.pose_rpy(
            retreat_start[0], retreat_start[1], retreat_start[2] + self.retreat_h,
            chosen_place,
        )
        self.report_stage("RETREATING")
        if stacking:
            ok, frac_ret = self.retreat_with_reachable_height(retreat_start, chosen_place)
        else:
            ok, frac_ret = self.vertical_move(
                retreat, "RETREAT", self.lift_retreat_vel, self.lift_retreat_acc
            )
        if not ok:
            return self._fail("方块已放置，但撤离路径规划或执行失败")
        if skip_return_home:
            ready = self.camera_facing_ready_joints()
            self.report_stage("MOVE_READY", joints=[round(v, 3) for v in ready])
            if not self.move_joints(
                ready,
                "MOVE_READY",
                self.transit_vel,
                self.transit_acc,
            ):
                self.log_stage("MOVE_READY", result="WARN_CONTINUE")
        else:
            self.report_stage("RETURN_HOME")
            if not self.move_joints(
                self.ready_joints,
                "RETURN_HOME",
                self.transit_vel,
                self.transit_acc,
            ):
                self.log_stage("RETURN_HOME", result="WARN_CONTINUE")

        self._last_place_fractions = (frac_desc, frac_ret)
        return True, "任务完成"

    def execute_pick_place(
        self,
        target_color: str,
        skip_return_home: bool = False,
        reuse_frozen_box: bool = False,
        destination_color: str = 'box',
        pick_only: bool = False,
        hold_at_lift: bool = False,
    ) -> Tuple[bool, str]:
        """Run one verified pick-place cycle and stay alive for the next command."""
        color = str(target_color or "").strip().lower()
        if hold_at_lift and not pick_only:
            return self._fail('直接转移模式仅用于抓起后由同一任务继续放置')
        destination_color = str(destination_color or 'box').strip().lower()
        if pick_only:
            destination_color = 'box'
        stacking = destination_color != 'box'
        if destination_color not in ('box', *ALLOWED_COLORS) or (stacking and destination_color == color):
            return self._fail('放置目标必须是盒子或另一种颜色的方块')
        if stacking and (not self.vision_enabled or not self.use_tf_tcp or reuse_frozen_box):
            return self._fail('叠放需要实时视觉及实测夹持偏移')
        if not reuse_frozen_box:
            self._session_box_xyz = None
        if color in ALLOWED_SELECTORS:
            if not self.vision_enabled:
                return self._fail("nearest/farthest requires vision")
            self.report_stage("WAITING_VISION", selector=color)
            resolved, err = self.select_color_by_box_distance(color)
            if resolved is None:
                return self._fail(err)
            self.report_stage("TARGET_FOUND", selector=color, target_color=resolved)
            color = resolved
        if color not in ALLOWED_COLORS:
            return self._fail(f"invalid target_color: {target_color!r}")

        self.reset_task_state(color)
        self.clear_planning_scene_objects()

        if not self.wait_ready():
            self.get_logger().error("dependencies not ready")
            return self._fail("dependencies not ready")
        if self.require_servo_bounds and self.servo_readiness_fault():
            return self._fail(self.servo_readiness_fault())
        self.log_calibration()

        if self.vision_enabled:
            if not self.handeye_tf_ok():
                return self._fail("TF base_link->camera_link unavailable")
            if self.initial_observation_enabled and not self.move_initial_observation():
                return self._fail('initial observation move or fresh detection failed')
            self.report_stage("WAITING_VISION", target_color=color)
            kwargs = {'reuse_frozen_box': reuse_frozen_box}
            if pick_only:
                kwargs['require_destination'] = False
            if stacking:
                kwargs['destination_color'] = destination_color
            frozen = self.wait_for_frozen_vision(**kwargs)
            if frozen is None:
                if self._cancelled():
                    return self._fail("任务已取消")
                return self._fail(self._vision_wait_failure)
            self.report_stage("TARGET_FOUND", target_color=color)
            (top_x, top_y, top_z), (box_x, box_y, destination_top) = frozen
        else:
            top_x, top_y, top_z = self.obj_x, self.obj_y, self.obj_z + self.obj_sz * 0.5
            box_x, box_y = self.place_x, self.place_y

        # RGB-D reports the visible top surface. PlanningScene needs the cube center,
        # while the TCP grasp target uses the independently calibrated grasp offsets.
        self.scene_obj_x = top_x
        self.scene_obj_y = top_y
        self.scene_obj_z = top_z - self.obj_sz * 0.5
        grasp_x = top_x + self.gx
        grasp_y = top_y + self.gy
        grasp_z = top_z + self.gz
        place_x, place_y = box_x, box_y
        # Release height is configured separately from the visual box center.
        place_z = self.place_z
        stack_center = None
        if stacking:
            try:
                stack_center = stack_geometry((top_x, top_y, top_z), (box_x, box_y, destination_top),
                                              (self.obj_sx, self.obj_sy, self.obj_sz),
                                              self.table_z + .5*self.table_sz, self.stack_clearance)
            except ValueError as exc:
                return self._fail(str(exc))
        self.setup_scene()
        if not stacking and not pick_only:
            self.setup_box_walls(box_x, box_y)
        elif stacking and self.vision_snapshot()['box']['ready']:
            if self.box_use_pose_yaw:
                self.box_yaw = quat_to_yaw(self.box_pose.pose.orientation)
            self.setup_box_walls(*pose_to_xyz(self.box_pose.pose)[:2])
        if stacking:
            self.publish_box('cube_obstacle_' + destination_color,
                             [box_x, box_y, destination_top-self.obj_sz*.5],
                             [self.obj_sx, self.obj_sy, self.obj_sz], yaw=self._destination_yaw)

        cube_yaw = None
        if self.vision_enabled and self.vision_pose is not None:
            cube_yaw = quat_to_yaw(self.vision_pose.pose.orientation)
            self.log_stage("CUBE_YAW", yaw=cube_yaw)
        grasp_oris = self.grasp_orientation_candidates(grasp_x, grasp_y, cube_yaw)
        place_oris = self.place_orientation_candidates(place_x, place_y)
        self.log_stage(
            "POSES", visual_top=(top_x, top_y, top_z),
            collision_center=(self.scene_obj_x, self.scene_obj_y, self.scene_obj_z),
            grasp_xyz=(grasp_x, grasp_y, grasp_z),
            box_center_xy=(box_x, box_y),
            release_z=place_z,
            grasp_rpy_candidates=[[round(v, 3) for v in rpy] for rpy in grasp_oris],
            place_rpy_candidates=[[round(v, 3) for v in rpy] for rpy in place_oris],
        )

        if self._cancelled():
            return self._fail("任务已取消")
        if not self.call_gripper(True):
            return self._fail("open gripper failed")
        if self.pre_grasp_ready_enabled:
            self.report_stage("MOVE_READY")
            if not self.move_pre_grasp_ready(grasp_x, grasp_y):
                self.log_stage("MOVE_READY", result="WARN_CONTINUE")
        # Approach and contact share one orientation. Switching pitch/yaw on
        # the last 8 cm is what twisted the wrist and pinned joint4.
        if self._cancelled():
            return self._fail("任务已取消")
        self.report_stage("MOVE_PRE_GRASP")
        hover_height = max(self.pre_grasp_h, self.reobserve_height) if self.reobserve_enabled else self.pre_grasp_h
        if self.approach_precheck:
            chosen_grasp = self.move_pregrasp_candidates(
                grasp_x, grasp_y, grasp_z, grasp_oris, hover_height)
        else:
            chosen_grasp = self.move_pose_candidates(
                grasp_x, grasp_y, grasp_z + hover_height,
                grasp_oris, "MOVE_PRE_GRASP", self.transit_vel, self.transit_acc,
                observation_backoff_m=self.reobserve_backoff if self.reobserve_enabled else 0.,
            )
        if chosen_grasp is None:
            return self._fail("pre-grasp planning failed")
        if self.reobserve_enabled:
            if self.plan_only:
                self.log_stage("REOBSERVE_CUBE", result="NOT_EVALUATED",
                               reason="plan-only does not move the camera to the observation pose")
            else:
                corrected_top = self.correct_pregrasp((top_x, top_y, top_z), chosen_grasp, hover_height)
                if corrected_top is None:
                    return self._fail("pre-grasp reobservation or correction failed; descent blocked")
                top_x, top_y, top_z = corrected_top
                grasp_x, grasp_y, grasp_z = top_x + self.gx, top_y + self.gy, top_z + self.gz
        grasp = self.pose_rpy(grasp_x, grasp_y, grasp_z, chosen_grasp)
        # The visual object was used for scene construction. Remove it immediately
        # before contact so the gripper is allowed to enter the grasp volume.
        self.publish_box(
            "object", [self.scene_obj_x, self.scene_obj_y, self.scene_obj_z],
            [self.obj_sx, self.obj_sy, self.obj_sz], operation=CollisionObject.REMOVE,
        )
        time.sleep(0.15)
        if self._cancelled():
            return self._fail("任务已取消")
        self.report_stage("APPROACH")
        ok, frac_app = self.vertical_move(
            grasp, "APPROACH", self.precision_vel, self.precision_acc
        )
        if not ok:
            return self._fail("approach planning or execution failed")
        self.report_stage("GRASPING")
        if not self.call_gripper(False):
            return self._fail("close gripper failed")
        if not self.attach_object(grasp):
            return self._fail("could not attach held object at its measured TCP offset")
        self._held_color = color
        self._held_grasp_rpy = list(chosen_grasp)
        self._held_cube_yaw = cube_yaw if cube_yaw is not None else chosen_grasp[2]
        self._held_source_top = (top_x, top_y, top_z)
        callback = getattr(self, 'on_object_picked', None)
        if not self.plan_only and callback is not None:
            callback(color)

        # Construct lift from the actual TCP after grasp. In plan-only mode the
        # planned grasp pose is the current logical TCP.
        lift_start = pose_to_xyz(grasp) if self.plan_only else self.current_tcp()
        if lift_start is None:
            return self._fail("no TCP after grasp")
        self.report_stage("LIFTING")
        ok, frac_lift = self.lift_with_reachable_height(lift_start, chosen_grasp)
        if not ok:
            return self._fail("lift planning or execution failed")
        if self._cancelled():
            return self._fail("任务已取消")
        if pick_only:
            if hold_at_lift:
                self.report_stage('HOLDING_ABOVE_SOURCE')
                return True, '上层方块已抓起并抬升，继续本次暂放任务'
            self.report_stage('RETURN_HOME')
            if not self.move_joints(self.ready_joints, 'RETURN_WITH_HELD_OBJECT', self.transit_vel, self.transit_acc):
                return self._fail('已抓起方块，但返回观察位失败；夹持状态保留')
            self.report_stage('WAITING_PLACE')
            name = {'red': '红块', 'white': '白块', 'orange': '橙块'}[color]
            return True, f'已抓起{name}并返回观察位，等待放置指令'
        if stacking:
            place_oris = [aligned_stack_rpy(chosen_grasp, cube_yaw, self._destination_yaw)]
            place_x, place_y, place_z = tcp_for_object_center(
                stack_center, self._held_object_offset, place_oris[0])
            self.log_stage('STACK_TARGET', destination_color=destination_color,
                           object_center=tuple(stack_center), tcp=(place_x, place_y, place_z),
                           held_offset=self._held_object_offset)
        previous_destination = getattr(self, '_placement_destination', None)
        self._placement_destination = destination_color if stacking else None
        try:
            ok, message = self.execute_place_motion(
                (place_x, place_y, place_z), place_oris, stack_center, skip_return_home)
        finally:
            self._placement_destination = previous_destination
        if not ok:
            return False, message
        frac_desc, frac_ret = self._last_place_fractions

        self.log_stage(
            "SUMMARY",
            approach_fraction=frac_app,
            lift_fraction=frac_lift,
            descend_fraction=frac_desc,
            retreat_fraction=frac_ret,
            plan_only=self.plan_only,
            target_color=self.target_color if self.vision_enabled else "fixed",
            frozen_box_xy=(box_x, box_y),
            skip_return_home=skip_return_home,
            reuse_frozen_box=reuse_frozen_box,
        )
        if self.plan_only and self.vision_enabled:
            self.get_logger().info("VISION PICK AND PLACE PLAN: PASS")
        elif self.plan_only:
            self.get_logger().info("FIXED-POSE PICK AND PLACE PLAN: PASS")
        elif self.vision_enabled:
            self.get_logger().info("VISION PICK AND PLACE: PASS")
        else:
            self.get_logger().info("FIXED-POSE PICK AND PLACE: PASS")
        self.report_stage("SUCCESS")
        return True, "任务完成"

    def run(self) -> int:
        ok, message = self.execute_pick_place(self.target_color)
        if ok:
            return 0
        if message == "dependencies not ready":
            return 2
        return 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickPlaceNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    def spin():
        executor.spin()

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    code = 1
    try:
        code = node.run()
    except Exception as exc:
        node.get_logger().error(f"pick_place exception: {exc}")
        code = 1
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass
    sys.exit(code)


if __name__ == "__main__":
    main()
