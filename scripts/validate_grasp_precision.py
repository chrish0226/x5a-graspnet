#!/usr/bin/env python3
"""Replay archived RGB-D + exact SAM masks; optionally plan, never execute.

Run with scripts/stack_env.sh sourced. The inference endpoint must support
quality_schema=2 and an explicit mask upload. Results are review artifacts.
"""
import argparse
import base64
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import requests
import rclpy
from rclpy.executors import MultiThreadedExecutor
from x5a_remote_grasp.geometry import parse_api_grasps
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:18303')
    parser.add_argument('--plan', action='store_true', help='use MoveIt plan-only services')
    parser.add_argument('--place-xyz', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='also test a synthetic placement point, in base metres')
    args = parser.parse_args()
    if args.place_xyz and not args.plan:
        parser.error('--place-xyz requires --plan')
    root = Path(__file__).resolve().parents[1]
    review = root/'docs/reviews/graspnet_20261003'
    records = json.loads((review/'sam_replay.json').read_text())
    rclpy.init(args=['--ros-args', '--params-file', str(root/'src/x5a_remote_grasp/config/remote_grasp.yaml'),
        '-r', '__node:=x5a_precision_review',
        '-p', 'action_name:=/x5a_precision_review/plan',
        '-p', 'health_service:=/x5a_precision_review/health',
        '-p', 'save_snapshot_service:=/x5a_precision_review/save_snapshot',
        '-p', 'validation_service:=/x5a_precision_review/validate_latest',
        '-p', 'topics.display_path:=/x5a_precision_review/display_path',
        '-p', 'markers.topic:=/x5a_precision_review/candidates'])
    node = RemoteGraspNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    output = []
    try:
        for record in records:
            capture = Path.home()/'.local/share/x5a_remote_grasp/captures'/record['capture']
            metadata = json.loads((capture/'camera_info.json').read_text())
            validation = json.loads((capture/'tf_validation.json').read_text())
            fields = {key: metadata[key] for key in ('fx', 'fy', 'cx', 'cy')}
            fields.update(prompt='archived target', depth_scale=metadata['depth_scale_divisor'],
                min_depth=0.10, max_depth=1.50, mask_dilation_px=2,
                num_point=20000, top_k=20, collision_thresh=0.01,
                voxel_size=0.005, target_distance_thresh=0.006, max_gripper_width=0.088)
            mask = base64.b64decode(max(record['objects'], key=lambda item: item['score'])['mask_png_b64'])
            files = {'rgb': ('color.png', (capture/'color.png').read_bytes(), 'image/png'),
                'depth': ('depth.png', (capture/'depth.png').read_bytes(), 'image/png'),
                'mask': ('mask.png', mask, 'image/png')}
            response = requests.post(args.endpoint+'/v1/grasp', data=fields, files=files, timeout=120)
            response.raise_for_status()
            payload = response.json()
            assert payload['quality_schema'] == 2
            assert payload['collision_checked'] and payload['target_distance_thresh_applied']
            repeated = requests.post(args.endpoint+'/v1/grasp', data=fields, files=files, timeout=120)
            repeated.raise_for_status()
            assert repeated.json()['grasps'] == payload['grasps'], 'identical frame produced different grasps'
            (review/(record['capture']+'_precision_response.json')).write_text(json.dumps(payload, indent=2)+'\n')
            encoded = SimpleNamespace(depth_png=(capture/'depth.png').read_bytes(),
                **{key: metadata[key] for key in ('fx', 'fy', 'cx', 'cy', 'depth_scale_m_per_unit')})
            transform = np.asarray(validation['transform_base_camera'])
            target, scene, center, band = node._quality_clouds(encoded, payload, transform)
            node.target_cloud, node.scene_cloud = target, scene
            grasps = parse_api_grasps(payload)
            base_grasps = [transform@g.transform_camera_grasp for g in grasps]
            goal = SimpleNamespace(is_cancel_requested=False, publish_feedback=lambda _: None)
            report = []
            suffix = '_precision_pick_place' if args.place_xyz else ('_precision_plan' if args.plan else '_precision_geometry')
            directory = review/(record['capture']+suffix)
            directory.mkdir(exist_ok=True)
            start = time.monotonic()
            selection = node._select_candidate(goal, grasps, base_grasps, node.get_clock().now().to_msg(),
                args.plan, capture_directory=directory, report=report,
                centroid_locate=lambda: center.copy(), center_band=band,
                place_target=None if args.place_xyz is None else (np.array(args.place_xyz), 'on'))
            result = {'capture': record['capture'], 'original_target_points': len(target),
                'context_inference_points': payload['num_points'], 'native_candidates': len(grasps),
                'deterministic_replay': True, 'height_base_m': band,
                'selected': selection is not None, 'elapsed_s': round(time.monotonic()-start, 3),
                'planning_requested': args.plan, 'failures': report}
            if selection is not None:
                selected, waypoints, plan = selection
                result.update(kind=node.selection_kind, index=selected.index,
                    required_opening_m=selected.width_m, tool0=waypoints['grasp'].tolist(),
                    stages=[] if plan is None else plan.stages)
            output.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('failures', 'tool0')}, ensure_ascii=False), flush=True)
        filename = 'precision_pick_place_replay.json' if args.place_xyz else ('precision_plan_replay.json' if args.plan else 'precision_geometry_replay.json')
        (review/filename).write_text(
            json.dumps(output, indent=2, ensure_ascii=False)+'\n')
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2)


if __name__ == '__main__':
    main()
