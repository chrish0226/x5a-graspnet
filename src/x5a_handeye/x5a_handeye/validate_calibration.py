#!/usr/bin/env python3
"""Hold-out validation of eye-to-hand calibration with new robot poses."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration as MsgDuration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from x5a_handeye.transforms import inv_T, pose_to_T, se3_errors

ARM = [f"joint{i}" for i in range(1, 7)]
QMIN = np.array([-3.14, -0.05, -0.1, -1.6, -1.57, -2.0])
QMAX = np.array([2.618, 3.50, 3.20, 1.55, 1.57, 2.0])


def clamp_q(q):
    return [float(min(QMAX[i] - 0.05, max(QMIN[i] + 0.05, q[i]))) for i in range(6)]


class Validator(Node):
    def __init__(self, result_yaml: str):
        super().__init__("x5a_handeye_validate")
        self.result = yaml.safe_load(Path(result_yaml).read_text())
        if self.result.get('calibration_type') == 'eye_in_hand':
            raise ValueError('this automatic validator is eye-to-hand only; use the read-only sampler and solver --holdout for eye-in-hand')
        self.T_base_cam = np.asarray(self.result["T_base_camera"], float)
        self.T_tool_board = np.asarray(self.result["T_tool_board"], float)
        self.q = None
        self.board = None
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.create_subscription(PoseStamped, "/calibration_board/pose", self.on_board, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.fjt = ActionClient(self, FollowJointTrajectory, "/x5a_arm_controller/follow_joint_trajectory")

    def on_js(self, msg):
        try:
            self.q = [float(msg.position[msg.name.index(n)]) for n in ARM]
        except Exception:
            pass

    def on_board(self, msg):
        self.board = msg

    def move(self, target, duration=3.5):
        traj = JointTrajectory()
        traj.joint_names = ARM
        p0 = JointTrajectoryPoint()
        p0.positions = list(self.q)
        p0.time_from_start = MsgDuration(sec=0)
        p1 = JointTrajectoryPoint()
        p1.positions = clamp_q(target)
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        p1.time_from_start = MsgDuration(sec=sec, nanosec=nsec)
        traj.points = [p0, p1]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        fut = self.fjt.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10)
        gh = fut.result()
        if not gh or not gh.accepted:
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=60)
        return rf.result().result.error_code == 0

    def lookup_base_tool(self):
        tf = self.tf_buffer.lookup_transform("base_link", "tool0", Time(), timeout=Duration(seconds=1.0))
        t = tf.transform.translation
        r = tf.transform.rotation
        return pose_to_T([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])

    def stable_board(self, n=10):
        poses = []
        t0 = time.time()
        while len(poses) < n and time.time() - t0 < 3.0:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.board is None:
                continue
            p = self.board.pose
            poses.append(
                (
                    [p.position.x, p.position.y, p.position.z],
                    [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
                )
            )
            time.sleep(0.03)
        if len(poses) < 5:
            return None
        xyz = np.mean([p[0] for p in poses], axis=0)
        quat = np.array([p[1] for p in poses], float)
        for i in range(1, len(quat)):
            if np.dot(quat[0], quat[i]) < 0:
                quat[i] *= -1
        q = np.mean(quat, axis=0)
        q /= np.linalg.norm(q)
        if q[3] < 0:
            q = -q
        return pose_to_T(xyz.tolist(), q.tolist())

    def run(self):
        t0 = time.time()
        while self.q is None and time.time() - t0 < 15:
            rclpy.spin_once(self, timeout_sec=0.05)
        assert self.q is not None
        assert self.fjt.wait_for_server(10)
        q0 = list(self.q)
        # hold-out poses different from typical sampling deltas
        deltas = [
            [0.22, 0.28, 0.18, -0.22, 0.28, -0.18],
            [-0.22, 0.32, 0.22, -0.28, -0.28, 0.22],
            [0.08, 0.42, 0.28, -0.18, 0.18, 0.28],
        ]
        rows = []
        for i, d in enumerate(deltas):
            tgt = clamp_q([q0[j] + d[j] for j in range(6)])
            print(f"holdout move {i+1}")
            if not self.move(tgt):
                print(f"Pose {chr(65+i)} motion FAIL")
                continue
            time.sleep(0.9)
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.05)
            T_bt = self.lookup_base_tool()
            T_cb = self.stable_board()
            if T_bt is None or T_cb is None:
                print(f"Pose {chr(65+i)} detection FAIL")
                continue
            T_base_board_vis = self.T_base_cam @ T_cb
            T_base_board_rob = T_bt @ self.T_tool_board
            te, re = se3_errors(T_base_board_vis, T_base_board_rob)
            rows.append((te, re))
            print(f"Pose {chr(65+i)} t_err={te*1000:.2f} mm r_err={re:.2f} deg")
        self.move(q0, duration=4.0)
        if not rows:
            print("NO_HOLDOUT")
            return 1
        tmean = float(np.mean([r[0] for r in rows]))
        tmax = float(np.max([r[0] for r in rows]))
        print("mean_mm", tmean * 1000, "max_mm", tmax * 1000)
        out = {
            "holdout": [
                {"pose": chr(65 + i), "translation_m": float(r[0]), "rotation_deg": float(r[1])}
                for i, r in enumerate(rows)
            ],
            "mean_m": tmean,
            "max_m": tmax,
        }
        output_path = Path.home() / ".ros" / "x5a_handeye" / "holdout.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2))
        print("holdout_output", output_path)
        return 0 if tmean < 0.02 else 1


def main(args=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result",
        default=str(Path.home() / "arx/x5a_ws/src/x5a_handeye/config/handeye_result.yaml"),
    )
    ns, _ = ap.parse_known_args()
    rclpy.init(args=args)
    node = Validator(ns.result)
    code = 1
    try:
        code = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
