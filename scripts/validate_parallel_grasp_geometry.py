#!/usr/bin/env python3
"""Compare complete candidate preparation using archived frames; no MoveIt calls.

The planner is stubbed to reject every candidate, exercising both native and
geometric phases. No controller goal, planning-scene update or plan publication
is sent. Hashes compare every candidate's requested stages and tool poses.
"""
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import rclpy
import x5a_remote_grasp.remote_grasp_node as module
from x5a_remote_grasp.geometry import parse_api_grasps


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'docs/reviews/planning_speed_20261003'
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=['--ros-args', '--params-file', str(root/'src/x5a_remote_grasp/config/remote_grasp.yaml'),
        '-r', '__node:=x5a_parallel_geometry_review', '-p', 'action_name:=/x5a_parallel_geometry_review/plan',
        '-p', 'health_service:=/x5a_parallel_geometry_review/health',
        '-p', 'save_snapshot_service:=/x5a_parallel_geometry_review/save_snapshot',
        '-p', 'validation_service:=/x5a_parallel_geometry_review/validate_latest',
        '-p', 'topics.display_path:=/x5a_parallel_geometry_review/display_path',
        '-p', 'markers.topic:=/x5a_parallel_geometry_review/candidates'])
    node = module.RemoteGraspNode()
    real_planner = node.planner
    prepared_class = module.PreparedGraspScene
    # Scene construction/TF are irrelevant to this pure candidate comparison;
    # the archived camera transform below supplies the exact perception frame.
    node._safety_scene = lambda stamp: None
    results = []
    try:
        for name in ('20261003_165902_803341', '20261003_170234_987941', '20261003_170536_818755'):
            capture = Path.home()/'.local/share/x5a_remote_grasp/captures'/name
            metadata = json.loads((capture/'camera_info.json').read_text())
            encoded = NS(depth_png=(capture/'depth.png').read_bytes(),
                **{k:metadata[k] for k in ('fx', 'fy', 'cx', 'cy', 'depth_scale_m_per_unit')})
            transform = np.asarray(json.loads((capture/'tf_validation.json').read_text())['transform_base_camera'])
            payload = json.loads((capture/'grasp_response.json').read_text())
            node.target_cloud, node.scene_cloud, center, band = node._quality_clouds(encoded, payload, transform)
            grasps = parse_api_grasps(payload)
            reference = None
            for mode, workers in (('original', 1), ('prepared_serial', 1), ('prepared_parallel', 2)):
                module.PreparedGraspScene = (lambda scene: scene) if mode == 'original' else prepared_class
                node.geometry_workers = workers
                times = []
                for _ in range(3):
                    trace = []; reports = []
                    def reject(poses):
                        trace.append({'stages': [stage for stage, _ in poses],
                            'poses': [[p.pose.position.x, p.pose.position.y, p.pose.position.z,
                                p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z,
                                p.pose.orientation.w] for _, p in poses],
                            'expected_finger_position': node.planner.expected_grasp_finger_position})
                        return None, 'offline all-candidate audit'
                    node.planner = NS(ready=lambda: (True, ''), apply_scene=lambda _: (True, ''),
                        plan_sequence=reject, last_solution_seed=None)
                    start = time.perf_counter()
                    selected = node._select_candidate(NS(is_cancel_requested=False, publish_feedback=lambda _: None),
                        grasps, [transform@g.transform_camera_grasp for g in grasps], node.get_clock().now().to_msg(),
                        True, report=reports, centroid_locate=lambda: center.copy(), center_band=band)
                    times.append(time.perf_counter()-start)
                    assert selected is None
                    if reference is None:
                        reference = trace
                    assert trace == reference, 'Candidate order, tool poses or finger opening changed'
                digest = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
                result = {'capture': name, 'mode': mode, 'workers': workers,
                    'median_s': round(statistics.median(times), 5), 'candidate_plan_requests': len(trace),
                    'candidate_trace_sha256': digest, 'identical_to_original': True}
                results.append(result)
                print(json.dumps(result), flush=True)
        (output/'integrated_geometry_replay.json').write_text(json.dumps({
            'controller_actions_sent': 0, 'scene_changes_sent': 0,
            'moveit_requests_sent': 0, 'results': results}, indent=2)+'\n')
    finally:
        module.PreparedGraspScene = prepared_class
        node.planner = real_planner
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
