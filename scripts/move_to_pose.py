#!/usr/bin/env python3
"""Move the X5A arm to a recorded end-effector pose via MoveIt 2.

Default target is the snapshot recorded in arm/logs/end_pose_20260828_173732.yaml
(base_link -> tool0, computed by x5a_handeye.x5a_fk from joint feedback).  A
different snapshot or explicit xyz/rpy can be given on the command line.

Planning idiom copied from x5a_pick_place.pick_place_node:
  IK pre-solve -> joint-space MoveGroup goal (plan_only) -> ExecuteTrajectory,
with a Pose-goal fallback when IK fails.  MoveIt execution goes through the
standard ExecuteTrajectory action; the official trajectory adapter turns the
FollowJointTrajectory call into vendor /arm_cmd commands.

Requires: move_group running (x5a_real_moveit.launch.py) and /joint_states
published by the official adapter.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    RobotState,
    RobotTrajectory,
)
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

try:
    from x5a_handeye.x5a_fk import fk_base_tool0
except ImportError:  # run from a bare terminal without sourcing x5a_handeye
    sys.path.insert(0, "/home/quella/arx/arm/src/x5a_handeye")
    from x5a_handeye.x5a_fk import fk_base_tool0

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GROUP = "arm"
BASE_FRAME = "base_link"
TCP_FRAME = "tool0"

# Snapshot of 2026-08-28 17:37 (arm/logs/end_pose_20260828_173732.yaml),
# base_link -> tool0 from joint feedback via fk_base_tool0.
DEFAULT_POSE_FILE = "/home/quella/arx/arm/logs/end_pose_20260828_173732.yaml"
FALLBACK_POSITION = (0.184622, 0.171395, 0.169435)
FALLBACK_RPY = (0.325476, 1.439038, 0.988323)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


def load_pose_file(path: str):
    """Return (position_m, rpy_rad, joints_rad) from a saved pose snapshot.

    joints_rad is the first 6 arm joints recorded in the file (None if
    missing) — an exact IK seed for moving back to that pose.
    """
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    pose = data["base_link_to_tool0"]
    joints = data.get("joint_pos_rad")
    if joints is not None:
        joints = [float(v) for v in joints[:6]]
    return list(pose["position_m"]), list(pose["rpy_rad"]), joints


class MoveToPoseNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("move_to_pose")
        self.cb = ReentrantCallbackGroup()
        self.plan_only = bool(args.plan_only)
        self.velocity_scaling = float(args.vx)
        self.acceleration_scaling = float(args.ax)
        self.planning_time = float(args.time)
        self.max_attempts = int(args.attempts)
        self.execution_timeout = float(args.execution_timeout)

        position, rpy, seed_joints = FALLBACK_POSITION, FALLBACK_RPY, None
        if args.pose_file:
            position, rpy, seed_joints = load_pose_file(args.pose_file)
        if any(v is not None for v in (args.x, args.y, args.z)):
            position = [
                float(args.x if args.x is not None else position[0]),
                float(args.y if args.y is not None else position[1]),
                float(args.z if args.z is not None else position[2]),
            ]
        if args.rpy:
            rpy = [float(v) for v in args.rpy]
        qx, qy, qz, qw = rpy_to_quat(*rpy)
        self.target_position = [float(v) for v in position]
        self.target_quat = (qx, qy, qz, qw)
        self.target_rpy = [float(v) for v in rpy]
        # exact joint solution for the loaded pose — best IK seed available
        self.extra_seed: Optional[List[float]] = seed_joints
        # logical current joints: end of the last planned/executed move, so a
        # plan-only rehearsal chains segments as if execution had happened
        self.last_joints: Optional[List[float]] = None

        self.q: Optional[List[float]] = None
        # Monotonic receive time of the latest complete arm JointState.  A
        # cached position is not proof that the driver is still online.
        self.q_stamp = 0.0
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10,
            callback_group=self.cb,
        )

        self.move_ac = ActionClient(self, MoveGroup, "move_action", callback_group=self.cb)
        self.execute_ac = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory", callback_group=self.cb
        )
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik", callback_group=self.cb)

        self.get_logger().info(
            f"move_to_pose ready plan_only={self.plan_only} "
            f"target_xyz={self.target_position} target_rpy={self.target_rpy}"
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        index = {name: i for i, name in enumerate(msg.name)}
        if any(name not in index for name in ARM_JOINTS):
            return
        self.q = [float(msg.position[index[name]]) for name in ARM_JOINTS]
        self.q_stamp = time.monotonic()

    def joint_state_age(self) -> float:
        if self.q is None or self.q_stamp <= 0.0:
            return float("inf")
        return time.monotonic() - self.q_stamp

    def wait_for_fresh_joint_state(
        self, timeout: float = 5.0, max_age: float = 0.5
    ) -> bool:
        """Wait until feedback is live, not merely present in the cache."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.q is not None and self.joint_state_age() <= max_age:
                return True
            time.sleep(0.05)
        self.get_logger().error(
            f"joint feedback unavailable/stale (age={self.joint_state_age():.2f}s)"
        )
        return False

    def log(self, stage: str, **kwargs) -> None:
        parts = [f"[{stage}]"]
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        self.get_logger().info(" ".join(parts))

    @staticmethod
    def wait_future(future, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.done()

    def current_tcp(self) -> Optional[Tuple[float, float, float]]:
        if self.q is None:
            return None
        t = fk_base_tool0(self.q)[:3, 3]
        return (float(t[0]), float(t[1]), float(t[2]))

    def wait_ready(self, timeout: float = 30.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if (
                self.q is not None
                and self.joint_state_age() <= 0.5
                and self.move_ac.wait_for_server(timeout_sec=0.1)
                and self.ik_cli.wait_for_service(timeout_sec=0.1)
                and (self.plan_only or self.execute_ac.wait_for_server(timeout_sec=0.1))
            ):
                return True
            time.sleep(0.05)
        missing = []
        if self.q is None or self.joint_state_age() > 0.5:
            missing.append("/joint_states (start the official adapter)")
        if not self.move_ac.wait_for_server(timeout_sec=0.1):
            missing.append("move_action (start move_group)")
        if not self.ik_cli.wait_for_service(timeout_sec=0.1):
            missing.append("/compute_ik")
        if not self.plan_only and not self.execute_ac.wait_for_server(timeout_sec=0.1):
            missing.append("/execute_trajectory")
        self.get_logger().error(f"dependencies not ready: {', '.join(missing)}")
        return False

    def target_pose(self) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = self.target_position
        qx, qy, qz, qw = self.target_quat
        pose.orientation.x, pose.orientation.y = qx, qy
        pose.orientation.z, pose.orientation.w = qz, qw
        return pose

    def ik_joints_for_pose(self, pose: Pose) -> Optional[List[float]]:
        if not self.ik_cli.wait_for_service(timeout_sec=2.0):
            return None
        seeds: List[List[float]] = []
        # The live state is normally the closest IK branch.  The recorded
        # pose remains a useful fallback, especially on the first move.
        if self.q is not None:
            seeds.append(list(self.q))
        if self.extra_seed is not None:
            seeds.append(list(self.extra_seed))
        yaw = math.atan2(pose.position.y, pose.position.x)
        seeds.append([yaw, 0.85, 0.95, -0.55, 0.0, 0.0])
        # Collision-aware first, then one collision-unaware pass on the last seed.
        attempts = [(seed, True) for seed in seeds]
        attempts.append((seeds[-1], False))
        for seed, avoid in attempts:
            req = GetPositionIK.Request()
            req.ik_request.group_name = GROUP
            ps = PoseStamped()
            ps.header.frame_id = BASE_FRAME
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose = pose
            req.ik_request.pose_stamped = ps
            req.ik_request.timeout.sec = 0
            req.ik_request.timeout.nanosec = 200000000
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

    def plan_and_execute(self, req: MotionPlanRequest, stage: str) -> bool:
        g = MoveGroup.Goal()
        g.request = req
        g.planning_options = PlanningOptions()
        g.planning_options.plan_only = True
        fut = self.move_ac.send_goal_async(g)
        if not self.wait_future(fut, 20.0):
            self.log(stage, planning="FAIL", reason="goal timeout")
            return False
        gh = fut.result()
        if not gh or not gh.accepted:
            self.log(stage, planning="FAIL", reason="goal not accepted")
            return False
        rf = gh.get_result_async()
        if not self.wait_future(rf, 120.0):
            cancel_future = gh.cancel_goal_async()
            self.wait_future(cancel_future, 2.0)
            self.log(stage, planning="FAIL", reason="result timeout")
            return False
        res = rf.result().result
        planning_ok = res.error_code.val == 1
        if not planning_ok:
            self.log(stage, planning=f"FAIL({res.error_code.val})")
            return False
        end = self._traj_end_joints(res.planned_trajectory)
        if self.plan_only:
            if end is not None:
                self.last_joints = list(end)
            self.log(stage, planning="PASS", execution="SKIP",
                     plan_end_joints=self._traj_end_joints(res.planned_trajectory, 3))
            return True
        ok = self.execute_trajectory(res.planned_trajectory, stage + "_EXEC")
        if ok and end is not None:
            self.last_joints = list(end)
        elif not ok and self.q is not None:
            # execution failed: re-sync the logical state to the real joints,
            # otherwise the next plan starts from a phantom pose
            self.last_joints = list(self.q)
        return ok

    @staticmethod
    def _traj_end_joints(traj: RobotTrajectory, digits: Optional[int] = None) -> Optional[List[float]]:
        jt = traj.joint_trajectory
        if not jt.points:
            return None
        names = list(jt.joint_names)
        pos = [float(jt.points[-1].positions[names.index(n)]) for n in ARM_JOINTS]
        if digits is not None:
            pos = [round(v, digits) for v in pos]
        return pos

    def planning_start_joints(self) -> List[float]:
        # A real execution must always be planned from fresh hardware
        # feedback.  last_joints is only a virtual state for --plan-only.
        if self.plan_only and self.last_joints is not None:
            return list(self.last_joints)
        if self.q is not None and self.joint_state_age() <= 0.5:
            return list(self.q)
        raise RuntimeError(
            f"no fresh joint state available (age={self.joint_state_age():.2f}s)"
        )

    def execute_trajectory(self, trajectory: RobotTrajectory, stage: str) -> bool:
        if not trajectory.joint_trajectory.points:
            self.log(stage, execution="FAIL", reason="empty trajectory")
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        future = self.execute_ac.send_goal_async(goal)
        if not self.wait_future(future, 10.0):
            self.log(stage, execution="FAIL", reason="goal timeout")
            return False
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.log(stage, execution="FAIL", reason="goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        if not self.wait_future(result_future, self.execution_timeout):
            # Do not leave an unknown active goal behind.  Without cancelling,
            # a recovery attempt is rejected as "another goal is active".
            cancel_future = goal_handle.cancel_goal_async()
            self.wait_future(cancel_future, 2.0)
            self.log(stage, execution="FAIL", reason="result timeout")
            return False
        result = result_future.result().result
        ok = result.error_code.val == 1
        self.log(
            stage,
            execution="PASS" if ok else f"FAIL({result.error_code.val})",
            actual_tcp=self.current_tcp(),
        )
        return ok

    def move_joints(self, joints: Sequence[float], stage: str) -> bool:
        if not self.plan_only and not self.wait_for_fresh_joint_state():
            self.log(stage, planning="FAIL", reason="joint feedback offline")
            return False
        req = MotionPlanRequest()
        req.group_name = GROUP
        req.num_planning_attempts = self.max_attempts
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.acceleration_scaling
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
        return self.plan_and_execute(req, stage)

    def move_pose_goal(self, pose: Pose, stage: str) -> bool:
        if not self.plan_only and not self.wait_for_fresh_joint_state():
            self.log(stage, planning="FAIL", reason="joint feedback offline")
            return False
        req = MotionPlanRequest()
        req.group_name = GROUP
        req.num_planning_attempts = self.max_attempts
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.acceleration_scaling
        rs = RobotState()
        rs.joint_state.name = list(ARM_JOINTS)
        rs.joint_state.position = self.planning_start_joints()
        req.start_state = rs
        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = TCP_FRAME
        pc.weight = 1.0
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [0.02]
        pc.constraint_region.primitives = [prim]
        pc.constraint_region.primitive_poses = [pose]
        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = TCP_FRAME
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.25
        oc.absolute_y_axis_tolerance = 0.25
        oc.absolute_z_axis_tolerance = 0.25
        oc.weight = 1.0
        cons = Constraints()
        cons.position_constraints.append(pc)
        cons.orientation_constraints.append(oc)
        req.goal_constraints.append(cons)
        return self.plan_and_execute(req, stage)

    def goto_pose(self, pose: Pose, stage: str = "MOVE") -> bool:
        """IK pre-solve -> joint-space plan -> execute, with Pose-goal fallback."""
        joints = self.ik_joints_for_pose(pose)
        if joints is not None:
            self.log(stage + "_IK", joints=[round(v, 3) for v in joints])
            if self.move_joints(joints, stage + "_IK"):
                return True
            self.get_logger().warn(f"{stage}: IK joint plan failed, retry with Pose goal")
        else:
            self.get_logger().warn(f"{stage}: IK pre-solve failed, retry with Pose goal")
        return self.move_pose_goal(pose, stage + "_POSE")

    def run(self) -> int:
        if not self.wait_ready():
            return 2
        pose = self.target_pose()
        self.log(
            "START",
            current_tcp=self.current_tcp(),
            target_xyz=self.target_position,
            target_rpy=[round(v, 4) for v in self.target_rpy],
            current_joints=[round(v, 3) for v in self.q] if self.q else None,
        )
        if self.goto_pose(pose, "MOVE"):
            self.log("DONE", result="PASS")
            return 0
        self.log("DONE", result="FAIL")
        return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose-file", default=DEFAULT_POSE_FILE,
                   help="saved end_pose snapshot yaml (default: recorded snapshot)")
    p.add_argument("--x", type=float, help="target x in base_link (m), overrides pose file")
    p.add_argument("--y", type=float, help="target y in base_link (m)")
    p.add_argument("--z", type=float, help="target z in base_link (m)")
    p.add_argument("--rpy", type=float, nargs=3, metavar=("R", "P", "Y"),
                   help="target rpy in rad, overrides pose file")
    p.add_argument("--plan-only", action="store_true",
                   help="plan and print the result without moving the arm")
    p.add_argument("--vx", type=float, default=0.35, help="velocity scaling (default 0.35)")
    p.add_argument("--ax", type=float, default=0.15, help="acceleration scaling (default 0.15)")
    p.add_argument("--time", type=float, default=5.0, help="allowed planning time (s)")
    p.add_argument("--attempts", type=int, default=8, help="planning attempts")
    p.add_argument("--execution-timeout", type=float, default=120.0)
    return p.parse_args()


def main(args=None) -> None:
    ns = parse_args()
    rclpy.init(args=args)
    node = MoveToPoseNode(ns)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        code = node.run()
    except Exception as exc:
        node.get_logger().error(f"move_to_pose exception: {exc}")
        code = 1
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
    sys.exit(code)


if __name__ == "__main__":
    main()
