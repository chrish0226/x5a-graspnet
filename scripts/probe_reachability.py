#!/usr/bin/env python3
"""Sweep /compute_ik over the workspace and record what the arm can actually reach.

Read-only: the probe never commands motion. It applies the same table/camera
collision boxes the plan node uses (so measurements match its planning scene),
then asks MoveIt for an IK solution at each (x, y, z, pitch, yaw) grid point with
the same seed set the node uses. Results go to a JSON table that the node reads
back to order its pitch ladder and to explain unreachable targets.

    ./scripts/probe_reachability.sh                    # 全量（默认 3cm 步长）
    python3 scripts/probe_reachability.py --step 0.05 --max-minutes 3   # 小网格验证
Ctrl+C 或超时会保存已测部分，重跑自动续测（--fresh 从头扫）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

DEFAULT_CONFIG = Path("/home/quella/arx/arm/src/x5a_remote_grasp/config/remote_grasp.yaml")
DEFAULT_OUT = Path("/home/quella/.local/share/x5a_remote_grasp/reachability.json")

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
MOVEIT_SUCCESS = 1


def load_params(path: Path) -> dict:
    """Flatten the ROS 2 parameter file: /**/ros__parameters holds the keys."""
    with Path(path).open() as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    wildcard = payload.get("/**")
    if isinstance(wildcard, dict):
        block = wildcard.get("ros__parameters")
        if isinstance(block, dict):
            return block
        return wildcard
    block = payload.get("ros__parameters")
    return block if isinstance(block, dict) else payload


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS fixed-axis RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quaternion_from_matrix(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    """Largest-component (Shepperd) conversion: stable for any rotation."""
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return x / norm, y / norm, z / norm, w / norm


def grid_values(low: float, high: float, step: float) -> List[float]:
    count = int(math.floor((high - low) / step + 1e-9))
    return [round(low + index * step, 4) for index in range(count + 1)]


def cell_key(x: float, y: float, z: float) -> str:
    return f"{x:.3f},{y:.3f},{z:.3f}"


class ReachabilityProbe:
    """Thin ROS wrapper: IK + state validity, matching the plan node's checks."""

    def __init__(self, params: dict, args: argparse.Namespace) -> None:
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from geometry_msgs.msg import Pose, PoseStamped
        from moveit_msgs.msg import CollisionObject, PlanningScene, RobotState
        from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK, GetStateValidity
        from shape_msgs.msg import SolidPrimitive
        from tf2_ros import Buffer, TransformListener

        self._Pose, self._PoseStamped = Pose, PoseStamped
        self._RobotState = RobotState
        self._CollisionObject, self._PlanningScene, self._SolidPrimitive = (
            CollisionObject,
            PlanningScene,
            SolidPrimitive,
        )
        self._GetPositionIK = GetPositionIK
        self._GetStateValidity = GetStateValidity
        self._ApplyPlanningScene = ApplyPlanningScene

        rclpy.init(args=None)
        self.node = Node("reachability_probe")
        self.group = ReentrantCallbackGroup()
        planning = params.get("planning", {})
        self.group_name = str(planning.get("group", "arm"))
        self.tcp_frame = str(planning.get("tcp_frame", "tool0"))
        self.ik_timeout = float(planning.get("ik_timeout_s", 0.5))
        self.base_frame = str(params.get("frames", {}).get("base", "base_link"))
        self.ik_client = self.node.create_client(
            GetPositionIK,
            str(planning.get("compute_ik_service", "/compute_ik")),
            callback_group=self.group,
        )
        self.validity_client = self.node.create_client(
            GetStateValidity,
            str(planning.get("state_validity_service", "/check_state_validity")),
            callback_group=self.group,
        )
        self.scene_client = self.node.create_client(
            ApplyPlanningScene,
            str(planning.get("apply_scene_service", "/apply_planning_scene")),
            callback_group=self.group,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.params = params
        self.args = args
        self.calls = 0
        self.started = time.monotonic()
        self.ready_joints = [
            float(v)
            for v in params.get("poses", {}).get(
                "ready_joints", [0.72, 0.85, 0.95, -0.55, 0.0, 0.0]
            )
        ]

    # ---------- ROS plumbing ----------
    def _spin(self, timeout: float) -> None:
        self.executor.spin_once(timeout_sec=timeout)

    def wait_for(self, client, name: str, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if client.wait_for_service(timeout_sec=0.5):
                return
        raise RuntimeError(f"{name} 不可用：MoveIt 和适配器是否在运行？")

    def call(self, client, request, timeout: float):
        self._rate_limit()
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._spin(0.005)
            if future.done():
                return future.result()
        return None

    def _rate_limit(self) -> None:
        if self.args.rate <= 0:
            return
        self.calls += 1
        target = self.calls / self.args.rate
        elapsed = time.monotonic() - self.started
        if target > elapsed:
            time.sleep(min(target - elapsed, 1.0))

    # ---------- scene ----------
    def apply_scene(self) -> bool:
        objects = []
        scene_cfg = self.params.get("scene", {})
        if self.args.scene:
            table_center, table_size = scene_cfg.get("table_center"), scene_cfg.get("table_size")
            if table_center and table_size:
                objects.append(self._box("remote_grasp_table", table_center, table_size))
            camera_size = scene_cfg.get("camera_size")
            if camera_size:
                center = self._camera_center(scene_cfg)
                if center is not None:
                    objects.append(self._box("remote_grasp_camera", center, camera_size))
        scene = self._PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objects
        response = self.call(
            self.scene_client, self._ApplyPlanningScene.Request(scene=scene), 5.0
        )
        applied = bool(response is not None and response.success)
        print(f"[probe] planning scene {'已应用' if applied else '应用失败'}（{len(objects)} 个盒子）")
        return applied

    def _camera_center(self, scene_cfg: dict) -> Optional[np.ndarray]:
        frame = str(scene_cfg.get("camera_frame", "camera_link"))
        offset = np.asarray(
            [float(v) for v in scene_cfg.get("camera_center_in_camera_link", [0.0, 0.0, 0.0])]
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self._spin(0.05)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, frame, self.node.get_clock().now().to_msg()
                )
            except Exception:
                continue
            rotation, translation = transform.transform.rotation, transform.transform.translation
            matrix = self._quaternion_matrix((rotation.x, rotation.y, rotation.z, rotation.w))
            return matrix @ offset + np.array([translation.x, translation.y, translation.z])
        print(f"[probe] 警告：TF 取不到 {self.base_frame}->{frame}，相机盒子未加入")
        return None

    def _box(self, name: str, center: Sequence[float], size: Sequence[float]):
        collision = self._CollisionObject()
        collision.header.frame_id = self.base_frame
        collision.id = name
        primitive = self._SolidPrimitive()
        primitive.type = self._SolidPrimitive.BOX
        primitive.dimensions = [float(item) for item in size]
        pose = self._Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(item) for item in center)
        pose.orientation.w = 1.0
        collision.primitives = [primitive]
        collision.primitive_poses = [pose]
        collision.operation = self._CollisionObject.ADD
        return collision

    @staticmethod
    def _quaternion_matrix(quaternion: Sequence[float]) -> np.ndarray:
        x, y, z, w = quaternion
        norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    # ---------- IK ----------
    def _pose_stamped(self, position: Sequence[float], rotation: np.ndarray):
        pose = self._PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        (qx, qy, qz, qw) = quaternion_from_matrix(rotation)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def _state_valid(self, state) -> bool:
        request = self._GetStateValidity.Request()
        request.robot_state = state
        response = self.call(self.validity_client, request, self.ik_timeout + 1.0)
        return bool(response is not None and response.valid)

    def solve(self, position: Sequence[float], pitch: float, yaw: float) -> Optional[List[float]]:
        """Joints for the first seed that yields a collision-free IK solution."""
        pose = self._pose_stamped(position, rpy_matrix(0.0, pitch, yaw))
        seeds = [[float(yaw), 0.85, 0.95, -0.55, 0.0, 0.0], list(self.ready_joints)]
        if self.args.seeds >= 3:
            seeds.append([0.0, 0.85, 0.95, -0.55, 0.0, 0.0])
        for joints in seeds[: max(1, self.args.seeds)]:
            request = self._GetPositionIK.Request()
            request.ik_request.group_name = self.group_name
            request.ik_request.ik_link_name = self.tcp_frame
            request.ik_request.pose_stamped = pose
            request.ik_request.avoid_collisions = True
            seconds = max(0.01, self.ik_timeout)
            request.ik_request.timeout.sec = int(seconds)
            request.ik_request.timeout.nanosec = int((seconds - int(seconds)) * 1.0e9)
            state = self._RobotState()
            state.joint_state.name = list(ARM_JOINTS)
            state.joint_state.position = [float(v) for v in joints]
            state.is_diff = False
            request.ik_request.robot_state = state
            response = self.call(self.ik_client, request, self.ik_timeout + 1.5)
            if response is None or response.error_code.val != MOVEIT_SUCCESS:
                continue
            solution = response.solution
            names = list(solution.joint_state.name)
            if not all(joint in names for joint in ARM_JOINTS):
                continue
            if not self._state_valid(solution):
                continue
            return [float(solution.joint_state.position[names.index(j)]) for j in ARM_JOINTS]
        return None

    def over_budget(self) -> bool:
        if self.args.max_minutes <= 0:
            return False
        return (time.monotonic() - self.started) > self.args.max_minutes * 60.0


def parse_out(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save(path: Path, results: dict, meta: dict, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {**meta, "summary": summary, "saved_at": datetime.now().isoformat(timespec="seconds")}, "summary": summary, "results": results}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="X5A IK 可达性扫描（只读，不动机械臂）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--step", type=float, default=0.03, help="x/y 网格步长（米）")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max", type=float, default=0.29)
    parser.add_argument("--z-step", type=float, default=0.03)
    parser.add_argument("--pitches", type=float, nargs="+", default=[1.45, 1.15, 0.90, 0.75])
    parser.add_argument(
        "--yaw-rels", type=int, nargs="+", default=[0, 45, 180],
        help="相对「朝向机械臂」的 yaw 偏角（度）",
    )
    parser.add_argument("--seeds", type=int, default=2, help="每个姿态尝试的 IK 种子数（1-3）")
    parser.add_argument("--rate", type=float, default=40.0, help="每秒最多 IK 调用数")
    parser.add_argument("--max-minutes", type=float, default=0.0, help="0 = 不限时")
    parser.add_argument("--scene", dest="scene", action="store_true", default=True)
    parser.add_argument("--no-scene", dest="scene", action="store_false")
    parser.add_argument("--fresh", action="store_true", help="忽略已有结果，从头扫")
    parser.add_argument("--save-every-s", type=float, default=30.0)
    args = parser.parse_args(argv)

    # 从包里拿汇总函数，保证表和节点读到的是同一套格式。
    try:
        from x5a_remote_grasp.reachability import build_summary
    except ImportError:
        build_root = Path(__file__).resolve().parent.parent / "build"
        sys.path.insert(0, str(build_root))
        try:
            from x5a_remote_grasp.reachability import build_summary
        except ImportError as exc:  # pragma: no cover
            print(f"无法导入 x5a_remote_grasp.reachability（{exc}）：先 colcon build 并 source stack_env.sh")
            return 2

    params = load_params(args.config)
    workspace = params.get("workspace", {})
    x_min = args.x_min if args.x_min is not None else float(workspace.get("x_min", 0.08))
    x_max = args.x_max if args.x_max is not None else float(workspace.get("x_max", 0.41))
    y_min = args.y_min if args.y_min is not None else float(workspace.get("y_min", 0.08))
    y_max = args.y_max if args.y_max is not None else float(workspace.get("y_max", 0.48))
    r_max = float(workspace.get("r_max", 0.54))

    probe = ReachabilityProbe(params, args)
    probe.wait_for(probe.ik_client, "compute_ik")
    probe.wait_for(probe.validity_client, "check_state_validity")
    if args.scene:
        probe.apply_scene()

    existing = {} if args.fresh else parse_out(args.out)
    results: Dict[str, dict] = existing.get("results", {}) if isinstance(existing.get("results"), dict) else {}

    xs = grid_values(x_min, x_max, args.step)
    ys = grid_values(y_min, y_max, args.step)
    zs = grid_values(args.z_min, args.z_max, args.z_step)
    total = len(xs) * len(ys) * len(zs)
    print(
        f"[probe] 网格 {len(xs)}x{len(ys)}x{len(zs)} = {total} 点，"
        f"{len(args.pitches)} 个 pitch x {len(args.yaw_rels)} 个 yaw，"
        f"每点最多 {len(args.pitches) * len(args.yaw_rels)} 次 IK；半径上限 {r_max} m"
    )

    meta = {
        "version": 1,
        "group_name": probe.group_name,
        "tcp_frame": probe.tcp_frame,
        "frame": probe.base_frame,
        "ik_timeout_s": probe.ik_timeout,
        "seeds": args.seeds,
        "pitches": [float(p) for p in args.pitches],
        "yaw_rels_deg": [int(v) for v in args.yaw_rels],
        "grid": {
            "x": [xs[0], xs[-1], args.step],
            "y": [ys[0], ys[-1], args.step],
            "z": [zs[0], zs[-1], args.z_step],
        },
        "r_max": r_max,
        "scene_applied": bool(args.scene),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    done = skipped = 0
    last_save = time.monotonic()
    interrupted = False
    try:
        for x in xs:
            for y in ys:
                if math.hypot(x, y) > r_max:
                    continue
                for z in zs:
                    key = cell_key(x, y, z)
                    cell = results.get(key)
                    if (
                        isinstance(cell, dict)
                        and all(f"{p:.2f}" in cell for p in args.pitches)
                    ):
                        skipped += 1
                        continue
                    if probe.over_budget():
                        print(f"[probe] 达到 --max-minutes {args.max_minutes}，停止并保存")
                        interrupted = True
                        break
                    cell = cell if isinstance(cell, dict) else {}
                    yaw_face = math.atan2(y, x)
                    for pitch in args.pitches:
                        row = cell.get(f"{pitch:.2f}")
                        row = row if isinstance(row, dict) else {}
                        for yaw_rel in args.yaw_rels:
                            entry = row.get(str(int(yaw_rel)))
                            if isinstance(entry, dict):
                                continue
                            yaw = yaw_face + math.radians(float(yaw_rel))
                            joints = probe.solve((x, y, z), float(pitch), yaw)
                            row[str(int(yaw_rel))] = (
                                {"ok": True, "joints": [round(v, 5) for v in joints]}
                                if joints is not None
                                else {"ok": False}
                            )
                        cell[f"{pitch:.2f}"] = row
                    results[key] = cell
                    done += 1
                    if time.monotonic() - last_save >= args.save_every_s:
                        save(args.out, results, meta, build_summary(results))
                        last_save = time.monotonic()
                        elapsed = time.monotonic() - probe.started
                        rate = done / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[probe] 已测 {done} 点（跳过 {skipped} 点）"
                            f" {rate:.1f} 点/秒  最近 {key}"
                        )
                if interrupted:
                    break
            if interrupted:
                break
    except KeyboardInterrupt:
        print("[probe] 收到 Ctrl+C，保存已测部分")
    finally:
        summary = build_summary(results)
        save(args.out, results, meta, summary)
        probe.executor.shutdown()
        probe.node.destroy_node()
        import rclpy

        rclpy.shutdown()

    print(f"[probe] 完成：{done} 个新网格点，{skipped} 个已有；结果写入 {args.out}")
    caps = ", ".join(f"pitch {pitch}: z<={height:.2f}" for pitch, height in summary["pitch_max_z"].items())
    if caps:
        print(f"[probe] 实测可达高度上限  {caps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
