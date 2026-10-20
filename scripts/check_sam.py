#!/usr/bin/env python3
"""Read Gemini RGB-D/TF and optionally call SAM/GraspNet; never plan or move."""
import argparse
import json
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import TransformException

from x5a_remote_grasp.geometry import parse_api_grasps
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', help='Also submit this object description to the original inference service')
    parser.add_argument('--timeout', type=float, default=20.)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rclpy.init(args=['--ros-args', '--params-file', str(root/'config/sam_grasp.yaml'),
        '-p', 'planning.enabled:=false', '-p', 'action_name:=/wrist_sam/check/plan',
        '-p', 'health_service:=/wrist_sam/check/health',
        '-p', 'validation_service:=/wrist_sam/check/validate',
        '-p', 'save_snapshot_service:=/wrist_sam/check/snapshot',
        '-r', '/joint_states:=/wrist_preview/joint_states'])
    node = RemoteGraspNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    worker = threading.Thread(target=executor.spin, daemon=True)
    worker.start()
    result = {'ok': False, 'robot_motion_sent': False, 'planning_requested': False}
    directory = root/'reports/sam-integration'
    directory.mkdir(parents=True, exist_ok=True)
    try:
        deadline = time.monotonic()+args.timeout
        while True:
            try:
                encoded = node._encode_latest_frame()
                transform = node._lookup_matrix(node.base_frame, encoded.source_frame, encoded.stamp)
                break
            except (RuntimeError, TransformException):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.1)
        capture = node._save_capture(encoded)
        result.update(rgbd_ok=True, tf_ok=True, source_frame=encoded.source_frame,
            base_frame=node.base_frame, resolution=[encoded.width, encoded.height],
            stamp_delta_s=encoded.stamp_delta_s, depth_scale_m_per_unit=encoded.depth_scale_m_per_unit,
            transform_base_camera=transform.tolist(), capture=str(capture))
        if args.prompt:
            payload = node._infer_target_grasps(encoded, {
                'prompt': args.prompt, 'fx': encoded.fx, 'fy': encoded.fy,
                'cx': encoded.cx, 'cy': encoded.cy, 'depth_scale': encoded.depth_scale_divisor,
                'min_depth': node.min_depth_default, 'max_depth': node.max_depth_default,
                'threshold': node.threshold, 'mask_dilation_px': node.mask_dilation_px,
                'num_point': node.num_point, 'top_k': node.top_k_default,
                'collision_thresh': node.collision_thresh, 'voxel_size': node.voxel_size,
                'target_distance_thresh': node.target_distance_thresh,
                'max_gripper_width': node.max_gripper_width}, transform, capture)
            grasps = parse_api_grasps(payload)
            target, scene, _, _ = node._quality_clouds(encoded, payload, transform)
            (capture/'grasp_response.json').write_text(json.dumps(payload, indent=2)+'\n')
            result.update(inference_ok=True, candidate_count=len(grasps),
                          target_points=len(target), scene_points=len(scene))
        result['ok'] = True
    except Exception as exc:
        result['error'] = str(exc)
    finally:
        executor.shutdown(timeout_sec=3.)
        worker.join(timeout=3.)
        node.destroy_node()
        rclpy.try_shutdown()
    (directory/'latest_check.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
