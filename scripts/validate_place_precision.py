#!/usr/bin/env python3
"""Replay the missed red-cube placement through perception and MoveIt plan-only.

Uses archived RGB-D, an isolated planner namespace/topic, and no execution
service or controller action. All reported object bottoms are estimates from
the observed cloud, not physical placement verification.
"""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
from x5a_remote_grasp.place_geometry import object_in_tool
from x5a_remote_grasp.geometry import parse_api_grasps


def main():
    root = Path(__file__).resolve().parents[1]
    captures = Path.home()/'.local/share/x5a_remote_grasp/captures'
    grasp_dir = captures/'20261003_154105_469549'
    place_dir = captures/'20261003_154136_413750'
    output = root/'docs/reviews/place_precision_20261003'
    output.mkdir(exist_ok=True)
    rclpy.init(args=['--ros-args', '--params-file', str(root/'src/x5a_remote_grasp/config/remote_grasp.yaml'),
        '-r', '__node:=x5a_place_review',
        '-p', 'action_name:=/x5a_place_review/plan',
        '-p', 'health_service:=/x5a_place_review/health',
        '-p', 'save_snapshot_service:=/x5a_place_review/save_snapshot',
        '-p', 'validation_service:=/x5a_place_review/validate_latest',
        '-p', 'topics.display_path:=/x5a_place_review/display_path',
        '-p', 'markers.topic:=/x5a_place_review/candidates'])
    node = RemoteGraspNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        metadata = json.loads((place_dir/'camera_info.json').read_text())
        transform = np.asarray(json.loads((grasp_dir/'tf_validation.json').read_text())['transform_base_camera'])
        tool = np.asarray(json.loads((grasp_dir/'selected_grasp.json').read_text())['tool0'])
        cloud = np.load(grasp_dir/'target_cloud_base.npy')
        node.place_object_cloud_tool = object_in_tool(cloud, tool)
        node.place_grasp_rotation = tool[:3, :3]
        stamp = node.get_clock().now().to_msg()
        encoded = NS(rgb_png=(place_dir/'color.png').read_bytes(),
            depth_png=(place_dir/'depth.png').read_bytes(),
            **{k:metadata[k] for k in ('fx', 'fy', 'cx', 'cy', 'depth_scale_divisor', 'depth_scale_m_per_unit')})
        goal = NS(is_cancel_requested=False, publish_feedback=lambda _: None)
        target = node._locate_place_target(goal, encoded, transform, 'red cube', 'on', .1, 1.5, output)
        if target is None:
            raise RuntimeError('archived support localization failed')
        ok, reason = node.planner.ready()
        if not ok:
            raise RuntimeError(reason)
        tf_deadline = time.monotonic()+5.0
        while not node.tf_buffer.can_transform(node.base_frame, node.camera_frame, rclpy.time.Time()):
            if time.monotonic() > tf_deadline:
                raise RuntimeError('camera collision TF unavailable for plan-only replay')
            time.sleep(0.1)
        stamp = node.get_clock().now().to_msg()
        ok, reason = node.planner.apply_scene(node._safety_scene(stamp))
        if not ok:
            raise RuntimeError(reason)
        failures = []
        start = time.monotonic()
        selection = node._plan_place_only(goal, target, stamp, start+node.plan_budget_s, failures, output)
        result = {'plan_only': True, 'controller_actions_sent': 0,
            'selected': selection is not None, 'elapsed_s': round(time.monotonic()-start, 3),
            'support_surface_base': target[0].tolist(),
            'allowed_object_center_points': [p.tolist() for p in node._place_point_candidates(target)],
            'failures': failures}
        if selection is not None:
            waypoints, plan = selection
            place = waypoints['place']
            world = node.place_object_cloud_tool @ place[:3, :3].T+place[:3, 3]
            lo, hi = np.percentile(world, [2, 98], axis=0)
            result.update(stages=plan.stages, place_tool0=place.tolist(),
                estimated_object_center_xy=((lo[:2]+hi[:2])/2).tolist(),
                estimated_bottom_base=float(lo[2]), bottom_clearance_m=float(lo[2]-target[0][2]))
        old = json.loads((place_dir/'place_locate.json').read_text())
        old_point = np.asarray(old['place_point_base'])
        old_R = node._rpy_matrix(0, 1.45, np.arctan2(old_point[1], old_point[0]))
        old_world = node.place_object_cloud_tool @ old_R.T+old_point
        result['old_estimated_bottom_base_p5'] = float(np.percentile(old_world[:, 2], 5))
        result['old_support_top_base'] = old['z_top_base']
        # Also exercise the combined grasp/place path with the same archive.
        grasp_metadata = json.loads((grasp_dir/'camera_info.json').read_text())
        grasp_encoded = NS(depth_png=(grasp_dir/'depth.png').read_bytes(),
            **{k:grasp_metadata[k] for k in ('fx', 'fy', 'cx', 'cy', 'depth_scale_m_per_unit')})
        payload = json.loads((grasp_dir/'grasp_response.json').read_text())
        node.target_cloud, node.scene_cloud, center, band = node._quality_clouds(grasp_encoded, payload, transform)
        grasps = parse_api_grasps(payload)
        reports = []
        picked = node._select_candidate(goal, grasps, [transform@g.transform_camera_grasp for g in grasps],
            stamp, True, capture_directory=output, place_target=target,
            report=reports, centroid_locate=lambda: center.copy(), center_band=band)
        result['combined_plan_selected'] = picked is not None
        result['combined_plan_stages'] = [] if picked is None else picked[2].stages
        result['combined_failures'] = reports
        (output/'replay.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2)


if __name__ == '__main__':
    main()
