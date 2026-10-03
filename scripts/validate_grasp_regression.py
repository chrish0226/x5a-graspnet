#!/usr/bin/env python3
"""Replay an archived grasp from simulated READY. Never executes motion."""
import argparse
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from moveit_msgs.msg import RobotState
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
from x5a_remote_grasp.geometry import parse_api_grasps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', default='20261003_163107_150215')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    capture = Path.home()/'.local/share/x5a_remote_grasp/captures'/args.capture
    output = root/'docs/reviews/grasp_reference_regression_20261003'/args.capture
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=['--ros-args', '--params-file', str(root/'src/x5a_remote_grasp/config/remote_grasp.yaml'),
        '-r', '__node:=x5a_grasp_regression_review', '-p', 'action_name:=/x5a_grasp_regression_review/plan',
        '-p', 'health_service:=/x5a_grasp_regression_review/health',
        '-p', 'save_snapshot_service:=/x5a_grasp_regression_review/save_snapshot',
        '-p', 'validation_service:=/x5a_grasp_regression_review/validate_latest',
        '-p', 'topics.display_path:=/x5a_grasp_regression_review/display_path',
        '-p', 'markers.topic:=/x5a_grasp_regression_review/candidates'])
    node = RemoteGraspNode();executor = MultiThreadedExecutor(num_threads=4);executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True);thread.start()
    try:
        deadline = time.monotonic()+5
        while not node.tf_buffer.can_transform(node.base_frame, node.camera_frame, Time()):
            if time.monotonic()>deadline:raise RuntimeError('camera TF unavailable')
            time.sleep(.1)
        metadata = json.loads((capture/'camera_info.json').read_text())
        encoded = NS(depth_png=(capture/'depth.png').read_bytes(),
            **{k:metadata[k] for k in ('fx', 'fy', 'cx', 'cy', 'depth_scale_m_per_unit')})
        transform = np.asarray(json.loads((capture/'tf_validation.json').read_text())['transform_base_camera'])
        payload = json.loads((capture/'grasp_response.json').read_text())
        node.target_cloud, node.scene_cloud, center, band = node._quality_clouds(encoded, payload, transform)
        ready = RobotState();ready.joint_state.name = ['joint'+str(i) for i in range(1, 9)]
        ready.joint_state.position = list(node.ready_joints)+[.044, .044]
        original = node.planner._plan_stage
        node.planner._plan_stage = lambda stage, pose, start, v, a: original(stage, pose, ready if start is None else start, v, a)
        grasps = parse_api_grasps(payload);report = [];start = time.monotonic()
        selected = node._select_candidate(NS(is_cancel_requested=False, publish_feedback=lambda _: None),
            grasps, [transform@g.transform_camera_grasp for g in grasps], node.get_clock().now().to_msg(),
            True, capture_directory=output, report=report, centroid_locate=lambda: center.copy(), center_band=band)
        result = {'capture': args.capture, 'plan_only': True, 'controller_actions_sent': 0,
            'reference_offset_m': node.gripper_geometry.tip_offset, 'selected': selected is not None,
            'elapsed_s': round(time.monotonic()-start, 3), 'failures': report}
        if selected is not None:
            grasp, waypoints, plan = selected
            failed_tool = np.asarray(json.loads((capture/'selected_grasp.json').read_text())['tool0'])
            result.update(stages=plan.stages, selected_tool0=waypoints['grasp'].tolist(),
                previous_tool_height_m=float(failed_tool[2, 3]), corrected_tool_height_m=float(waypoints['grasp'][2, 3]),
                required_opening_m=grasp.width_m)
        (output/'replay.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='failures'}, ensure_ascii=False), flush=True)
    finally:
        executor.shutdown();node.destroy_node();rclpy.shutdown();thread.join(timeout=2)


if __name__ == '__main__':
    main()
