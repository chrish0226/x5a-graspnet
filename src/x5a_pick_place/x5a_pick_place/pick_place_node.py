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
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from x5a_handeye.x5a_fk import fk_base_tool0


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
    def __init__(self, node_name: str = "x5a_pick_place") -> None:
        super().__init__(node_name)
        self.cb = ReentrantCallbackGroup()
        self._declare_params()
        self._load_params()

        self.q: Optional[List[float]] = None
        self.dq: Optional[List[float]] = None
        self.gripper_position = 0.0
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
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

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
        self.declare_parameter("execution.action_name", "/execute_trajectory")
        self.declare_parameter("execution.timeout", 120.0)
        self.declare_parameter("state.joint_state_topic", "/joint_states")
        self.declare_parameter("vision.enabled", False)
        self.declare_parameter("target_color", "red")
        self.declare_parameter("vision.max_pose_age", 0.5)
        self.declare_parameter("vision.wait_timeout", 10.0)
        self.declare_parameter("box.pose_topic", "/x5a_vision/box_pose")
        self.declare_parameter("box.stable_topic", "/x5a_vision/box_stable")
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
        self.max_attempts = int(g("planning.max_attempts").value)
        self.execute_action = str(g("execution.action_name").value)
        self.execution_timeout = float(g("execution.timeout").value)
        self.joint_state_topic = str(g("state.joint_state_topic").value)
        self.vision_enabled = bool(g("vision.enabled").value)
        self.target_color = str(g("target_color").value).strip().lower()
        if self.target_color not in ALLOWED_COLORS:
            raise ValueError(
                f"target_color must be red, white, or orange; got {self.target_color!r}"
            )
        self.vision_pose_topic = f"/x5a_vision/{self.target_color}_cube_pose"
        self.vision_stable_topic = f"/x5a_vision/{self.target_color}_cube_stable"
        self.vision_max_age = float(g("vision.max_pose_age").value)
        self.vision_wait_timeout = float(g("vision.wait_timeout").value)
        self.box_pose_topic = str(g("box.pose_topic").value)
        self.box_stable_topic = str(g("box.stable_topic").value)
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
        self.table_x = float(g("table.x").value)
        self.table_y = float(g("table.y").value)
        self.table_z = float(g("table.z").value)
        self.table_sx = float(g("table.size_x").value)
        self.table_sy = float(g("table.size_y").value)
        self.table_sz = float(g("table.size_z").value)
        self.pre_grasp_h = float(g("motion.pre_grasp_height").value)
        self.lift_h = float(g("motion.lift_height").value)
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

    def reset_task_state(self, target_color: str) -> None:
        """Drop frozen poses from the previous task before a new color is selected."""
        color = target_color.strip().lower()
        self.target_color = color
        self.vision_pose_topic = f"/x5a_vision/{color}_cube_pose"
        self.vision_stable_topic = f"/x5a_vision/{color}_cube_stable"
        for name in self.allowed_colors:
            self.cube_poses[name] = None
            self.cube_pose_arrivals[name] = 0.0
            self.cube_stable[name] = False
        self.vision_pose = None
        self.vision_pose_arrival = 0.0
        self.vision_stable = False
        self.box_pose = None
        self.box_pose_arrival = 0.0
        self.box_stable = False
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
        time.sleep(0.1)

    def handeye_tf_ok(self) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                "base_link", "camera_link", Time(), timeout=Duration(seconds=0.5)
            )
            return True
        except Exception as exc:
            self.get_logger().error(f"TF base_link->camera_link unavailable: {exc}")
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
        result = result_future.result().result
        ok = bool(result.reached_goal)
        self.log_stage(
            stage,
            result="PASS" if ok else "FAIL",
            target_m=target,
            feedback_m=float(result.position),
            interface="control_msgs/GripperCommand",
        )
        return ok

    def publish_box(
        self,
        name: str,
        xyz: Sequence[float],
        size: Sequence[float],
        frame: str = "base_link",
        operation=CollisionObject.ADD,
    ) -> None:
        co = CollisionObject()
        co.id = name
        co.header.frame_id = frame
        co.header.stamp = self.get_clock().now().to_msg()
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(size[0]), float(size[1]), float(size[2])]
        pose = Pose()
        pose.orientation.w = 1.0
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
        time.sleep(0.3)
        self.log_stage("SCENE", table="PASS", object="PASS")

    def attach_object(self) -> None:
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
        aco.object.primitives = [prim]
        aco.object.primitive_poses = [pose]
        aco.touch_links = ["link6", "link7", "link8", "tool0"]
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
        cons = Constraints()
        for n, v in zip(ARM_JOINTS, joints):
            jc = JointConstraint()
            jc.joint_name = n
            jc.position = float(v)
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
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

    def cartesian_to(self, pose: Pose, stage: str) -> Tuple[bool, float]:
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
        req.start_state.joint_state.name = list(ARM_JOINTS)
        req.start_state.joint_state.position = start_q
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
        if frac < self.min_frac:
            self.log_stage(
                stage,
                planning="FAIL",
                fraction=frac,
                target_tcp=target,
                start_tcp=start_tcp,
            )
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

    def ik_joints_for_pose(self, pose: Pose) -> Optional[List[float]]:
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
        attempts.append((seeds[-1], False))
        for seed, avoid in attempts:
            req.ik_request.robot_state.joint_state.name = list(ARM_JOINTS)
            req.ik_request.robot_state.joint_state.position = seed
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
        ok, frac = self.cartesian_to(pose, stage + "_CART")
        if ok:
            return True, frac
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
            for axis in (cube_yaw, cube_yaw + 0.5 * math.pi):
                yaw = wrap_pi(axis)
                cands.append([0.0, pitch, yaw])
                cands.append([0.0, pitch, wrap_pi(yaw + math.pi)])
        else:
            cands.extend([[0.0, pitch, 0.0], [0.0, pitch, 0.5 * math.pi]])
        cands.extend([list(self.grasp_rpy), [0.0, pitch, 0.20], [0.0, pitch, -0.20]])
        if math.hypot(x, y) >= 0.42:
            cands.extend([[0.0, pitch, yaw0], [0.0, 1.30, yaw0]])
        return unique_rpy(cands)

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
    ) -> Optional[List[float]]:
        offsets = list(xy_offsets) if xy_offsets is not None else [(0.0, 0.0)]
        for ox, oy in offsets:
            px, py = x + ox, y + oy
            for index, rpy in enumerate(orientations):
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

    def camera_facing_ready_joints(self) -> List[float]:
        joints = list(self.pre_grasp_ready_joints)
        if len(joints) < 6:
            joints = [0.72, 0.85, 0.95, -0.55, 0.0, 0.0]
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", "camera_link", Time(), timeout=Duration(seconds=0.3)
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

            path = Path(get_package_share_directory("x5a_handeye")) / "config" / "handeye_result.yaml"
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
                "base_link", "camera_link", Time(), timeout=Duration(seconds=0.5)
            )
            t = tf.transform.translation
            self.log_stage("TF_BASE_CAMERA_LINK", xyz=(t.x, t.y, t.z))
        except Exception as exc:
            self.get_logger().warn(f"TF base_link->camera_link unavailable: {exc}")

    def wait_for_frozen_vision(
        self,
        reuse_frozen_box: bool = False,
    ) -> Optional[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ]:
        """Freeze the selected cube. Box is frozen once per sequence when requested."""
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
            if reuse_box:
                box_ok = True
                box_xyz = self._session_box_xyz
                box_age = 0.0
            else:
                box_age = now - self.box_pose_arrival
                box_ok = (
                    self.box_pose is not None
                    and self.box_stable
                    and box_age <= self.vision_max_age
                )
                box_xyz = pose_to_xyz(self.box_pose.pose) if self.box_pose is not None else None
            if cube_ok and box_ok and pose is not None and box_xyz is not None:
                cube_xyz = pose_to_xyz(pose.pose)
                if all(math.isfinite(v) for v in cube_xyz + box_xyz):
                    self.vision_pose = pose
                    self.vision_pose_arrival = arrival
                    self.vision_stable = True
                    if not reuse_box:
                        self._session_box_xyz = box_xyz
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
        self.log_stage(
            "WAIT_FOR_VISION", result="FAIL", target_color=self.target_color,
            cube_stable=self.cube_stable.get(self.target_color, False),
            box_stable=self.box_stable if not reuse_box else True,
            cube_age=time.monotonic() - self.cube_pose_arrivals.get(self.target_color, 0.0),
            box_age=0.0 if reuse_box else time.monotonic() - self.box_pose_arrival,
            reuse_frozen_box=reuse_box,
        )
        return None

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

    def execute_pick_place(
        self,
        target_color: str,
        skip_return_home: bool = False,
        reuse_frozen_box: bool = False,
    ) -> Tuple[bool, str]:
        """Run one verified pick-place cycle and stay alive for the next command."""
        color = str(target_color or "").strip().lower()
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
        self.log_calibration()

        if self.vision_enabled:
            if not self.handeye_tf_ok():
                return self._fail("TF base_link->camera_link unavailable")
            self.report_stage("WAITING_VISION", target_color=color)
            frozen = self.wait_for_frozen_vision(reuse_frozen_box=reuse_frozen_box)
            if frozen is None:
                if self._cancelled():
                    return self._fail("任务已取消")
                return self._fail(f"vision not stable for {color}")
            self.report_stage("TARGET_FOUND", target_color=color)
            (top_x, top_y, top_z), (box_x, box_y, _) = frozen
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
        # Movable box contributes center XY only. Keep the previously verified
        # fixed release height so the gripper remains above the box interior.
        place_z = self.place_z
        self.setup_scene()

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
        self.report_stage("MOVE_READY")
        if not self.move_pre_grasp_ready(grasp_x, grasp_y):
            self.log_stage("MOVE_READY", result="WARN_CONTINUE")
        # Approach and contact share one orientation. Switching pitch/yaw on
        # the last 8 cm is what twisted the wrist and pinned joint4.
        if self._cancelled():
            return self._fail("任务已取消")
        self.report_stage("MOVE_PRE_GRASP")
        chosen_grasp = self.move_pose_candidates(
            grasp_x, grasp_y, grasp_z + self.pre_grasp_h,
            grasp_oris, "MOVE_PRE_GRASP", self.transit_vel, self.transit_acc,
        )
        if chosen_grasp is None:
            return self._fail("pre-grasp planning failed")
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
        self.attach_object()

        # Construct lift from the actual TCP after grasp. In plan-only mode the
        # planned grasp pose is the current logical TCP.
        lift_start = pose_to_xyz(grasp) if self.plan_only else self.current_tcp()
        if lift_start is None:
            return self._fail("no TCP after grasp")
        lift = self.pose_rpy(
            lift_start[0], lift_start[1], lift_start[2] + self.lift_h, chosen_grasp
        )
        self.report_stage("LIFTING")
        ok, frac_lift = self.vertical_move(
            lift, "LIFT", self.lift_retreat_vel, self.lift_retreat_acc
        )
        if not ok:
            return self._fail("lift planning or execution failed")
        if self._cancelled():
            return self._fail("任务已取消")
        self.report_stage("MOVE_PRE_PLACE")
        chosen_place = self.move_pose_candidates(
            place_x, place_y, place_z + self.pre_place_h,
            place_oris, "MOVE_PRE_PLACE", self.transit_vel, self.transit_acc,
            xy_offsets=[(0.0, 0.0), (0.01, 0.0), (0.0, -0.01)],
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
        if not self.call_gripper(True):
            return self._fail("release gripper failed")
        table_top = self.table_z + 0.5 * self.table_sz
        self.detach_object([place_x, place_y, table_top + 0.5 * self.obj_sz])
        retreat_start = pose_to_xyz(place) if self.plan_only else self.current_tcp()
        if retreat_start is None:
            return self._fail("no TCP after place")
        retreat = self.pose_rpy(
            retreat_start[0], retreat_start[1], retreat_start[2] + self.retreat_h,
            chosen_place,
        )
        self.report_stage("RETREATING")
        ok, frac_ret = self.vertical_move(
            retreat, "RETREAT", self.lift_retreat_vel, self.lift_retreat_acc
        )
        if not ok:
            return self._fail("retreat planning or execution failed")
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
