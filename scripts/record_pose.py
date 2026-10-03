#!/usr/bin/env python3
"""Snapshot the arm's current pose (from /joint_states via FK) to a yaml file.

Used to teach the fixed place points A/B: move the arm to the desired pose by
any means (gravity teach / vendor GUI with MoveIt stopped), then run:

    /usr/bin/python3 ~/arx/arm/scripts/record_pose.py --name place_a

The output format matches the end_pose snapshots, so ring_pick_place.py and
move_to_pose.py can load it directly (--place-a/--pose-file).
"""
from __future__ import annotations

import argparse
import datetime
import math
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

try:
    from x5a_handeye.x5a_fk import fk_base_tool0
except ImportError:
    sys.path.insert(0, "/home/quella/arx/arm/src/x5a_handeye")
    from x5a_handeye.x5a_fk import fk_base_tool0

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_DIR = "/home/quella/arx/arm/logs/poses"


def mat_to_rpy(R) -> tuple:
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def mat_to_quat(R) -> tuple:
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = (R[2, 1] - R[1, 2]) / (4.0 * w)
    y = (R[0, 2] - R[2, 0]) / (4.0 * w)
    z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    return x, y, z, w


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="pose name, e.g. place_a")
    p.add_argument("--out-dir", default=DEFAULT_DIR)
    args = p.parse_args()

    rclpy.init()
    node = Node("record_pose")
    latest: dict = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: latest.update(msg=m), 10)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        msg = latest.get("msg")
        if msg is None:
            continue
        index = {n: i for i, n in enumerate(msg.name)}
        if all(n in index for n in ARM_JOINTS):
            break
    else:
        print("ERROR: no /joint_states with joint1..6 (start the MoveIt adapter first)",
              file=sys.stderr)
        return 2

    index = {n: i for i, n in enumerate(latest["msg"].name)}
    joints = [float(latest["msg"].position[index[n]]) for n in ARM_JOINTS]
    gripper = 0.0
    if "joint7" in index:
        gripper = float(latest["msg"].position[index["joint7"]])
    node.destroy_node()
    rclpy.shutdown()

    T = fk_base_tool0(joints)
    roll, pitch, yaw = mat_to_rpy(T[:3, :3])
    qx, qy, qz, qw = mat_to_quat(T[:3, :3])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name}.yaml"
    with open(path, "w") as f:
        f.write(f"# pose snapshot '{args.name}' saved_at_local: "
                f"{datetime.datetime.now().isoformat()}\n")
        f.write("joint_pos_rad:  # 6 arm joints from /joint_states feedback\n")
        for v in joints:
            f.write(f"  - {v}\n")
        f.write(f"  # gripper joint7 = {gripper}\n")
        f.write("base_link_to_tool0:  # via x5a_handeye.x5a_fk.fk_base_tool0\n")
        f.write(f"  position_m: [{T[0, 3]:.6f}, {T[1, 3]:.6f}, {T[2, 3]:.6f}]\n")
        f.write(f"  rpy_rad: [{roll:.6f}, {pitch:.6f}, {yaw:.6f}]\n")
        f.write(f"  quaternion_xyzw: [{qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}]\n")

    print(f"saved: {path}")
    print(f"base_link -> tool0 position (m): [{T[0, 3]:.4f}, {T[1, 3]:.4f}, {T[2, 3]:.4f}]")
    print(f"base_link -> tool0 rpy (rad):    [{roll:.4f}, {pitch:.4f}, {yaw:.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
