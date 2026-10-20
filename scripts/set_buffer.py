#!/usr/bin/env python3
"""Read-only ROS observation; optionally save a verified fixed table buffer.

Does not publish commands, call robot services, or move the arm.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener, TransformException
import yaml

from x5a_vision.clearance import DEFAULT_BUFFER_FILE, base_cloud, buffer_surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--near', choices=['red', 'white', 'orange'], required=True)
    parser.add_argument('--gap', type=float, default=.05, help='cube edge gap, metres')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--file', default=DEFAULT_BUFFER_FILE)
    args = parser.parse_args()
    if not math.isfinite(args.gap) or not .04 <= args.gap <= .10:
        parser.error('gap must be 0.04–0.10 m')
    root = Path(__file__).resolve().parents[1]
    vision = yaml.safe_load((root/'config/vision.yaml').read_text())['/**']['ros__parameters']
    task = yaml.safe_load((root/'config/task_execute.yaml').read_text())['/**']['ros__parameters']
    size, table_top = vision['object']['size_x'], vision['table']['z']
    rclpy.init()
    node = rclpy.create_node('wrist_buffer_readonly_configuration')
    tf_buffer = Buffer()
    listener = TransformListener(tf_buffer, node)
    state = {'side': 'current', 'stable': False}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    def remember(key):
        def cb(msg):
            state[key] = msg
            state[key+'_arrival'] = time.monotonic()
        return cb
    subscriptions = [
        node.create_subscription(Image, '/wrist_camera/depth/image_raw', remember('depth'), qos),
        node.create_subscription(CameraInfo, '/wrist_camera/color/camera_info', remember('info'), qos),
        node.create_subscription(PoseStamped, '/wrist_preview/'+args.near+'_cube_pose', remember('pose'), 10),
        node.create_subscription(Bool, '/wrist_preview/'+args.near+'_cube_stable',
                                 lambda m: state.update(stable=m.data), 10),
        node.create_subscription(String, '/wrist_pick/observation_side',
            lambda m: state.update(side=m.data), QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
    ]
    bridge = CvBridge()
    deadline, result, error, confirmations, last_stamp = time.monotonic()+20, None, '', [], None
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            if not all(k in state for k in ('depth', 'info', 'pose')) or not state['stable']:
                continue
            if any(time.monotonic()-state[k+'_arrival'] > 1.0 for k in ('depth', 'pose')):
                continue
            depth_msg, pose = state['depth'], state['pose']
            stamp = (depth_msg.header.stamp.sec, depth_msg.header.stamp.nanosec)
            if stamp == last_stamp:
                continue
            frame = 'wrist_preview/base_link'
            if pose.header.frame_id != frame:
                continue
            try:
                transform = tf_buffer.lookup_transform(frame, state['info'].header.frame_id,
                            Time.from_msg(depth_msg.header.stamp), timeout=Duration(seconds=0.0)).transform
            except TransformException:
                continue
            last_stamp = stamp
            q = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
            q /= np.linalg.norm(q)
            x, y, z, w = q
            matrix = np.eye(4)
            matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
            matrix[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
            depth = bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(float)
            if depth_msg.encoding == '16UC1':
                depth *= vision['depth']['scale_16uc1']
            elif depth_msg.encoding != '32FC1':
                raise ValueError('unsupported depth encoding')
            cloud, _, _ = base_cloud(depth, state['info'].k, matrix,
                                    vision['depth']['min_m'], vision['depth']['max_m'])
            source = np.array([pose.pose.position.x, pose.pose.position.y])
            table = task['table']
            table_y = table['y'] if state['side'] == 'current' else -table['y']
            choices = []
            for offset in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                xy = source+np.array(offset)*(size+args.gap)
                if (abs(xy[0]-table['x'])+.04 >= table['size_x']*.5
                        or abs(xy[1]-table_y)+.04 >= table['size_y']*.5):
                    continue
                if buffer_surface(cloud, xy, table_top)['empty']:
                    choices.append(xy)
            if not choices:
                confirmations.clear()
                error = '距方块边缘约 5 cm 的四个候选位置均未完整可见、被占用或超出桌面'
                continue
            chosen = choices[0]
            if confirmations and np.linalg.norm(chosen-confirmations[-1]) > .003:
                confirmations.clear()
            confirmations.append(chosen)
            if len(confirmations) >= 4:
                result = {'enabled': True, 'base_frame': frame, 'observation_side': state['side'],
                          'xy_m': np.median(confirmations[-4:], axis=0).tolist(),
                          'source_color': args.near, 'cube_edge_gap_m': args.gap}
                break
        if result is None:
            raise RuntimeError(error or '未获得新鲜、稳定的方块定位和深度图，暂放位未写入')
        if args.save:
            target = Path(args.file)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as output:
                yaml.safe_dump(result, output, allow_unicode=True, sort_keys=False)
                filename = output.name
            os.replace(filename, target)
        print(json.dumps({'saved': args.save, **result}, ensure_ascii=False))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
