#!/usr/bin/env python3
"""Replay the white-cube stop without sending any execution/controller goal."""
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
    root = Path(__file__).resolve().parents[1]
    output = root/'docs/reviews/place_contact_20261003'
    captures = Path.home()/'.local/share/x5a_remote_grasp/captures'
    grab, place = captures/'20261003_160859_754402', captures/'20261003_160928_915199'
    rclpy.init(args=['--ros-args', '--params-file', str(root/'src/x5a_remote_grasp/config/remote_grasp.yaml'),
        '-r', '__node:=x5a_contact_review', '-p', 'action_name:=/x5a_contact_review/plan',
        '-p', 'health_service:=/x5a_contact_review/health',
        '-p', 'save_snapshot_service:=/x5a_contact_review/save_snapshot',
        '-p', 'validation_service:=/x5a_contact_review/validate_latest',
        '-p', 'topics.display_path:=/x5a_contact_review/display_path',
        '-p', 'markers.topic:=/x5a_contact_review/candidates'])
    node = RemoteGraspNode()
    executor = MultiThreadedExecutor(num_threads=4); executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True); thread.start()
    try:
        transform = np.asarray(json.loads((grab/'tf_validation.json').read_text())['transform_base_camera'])
        def encoded(directory):
            metadata = json.loads((directory/'camera_info.json').read_text())
            return NS(rgb_png=(directory/'color.png').read_bytes(), depth_png=(directory/'depth.png').read_bytes(),
                **{k:metadata[k] for k in ('fx', 'fy', 'cx', 'cy', 'depth_scale_divisor', 'depth_scale_m_per_unit')})
        goal = NS(is_cancel_requested=False, publish_feedback=lambda _: None)
        target = node._locate_place_target(goal, encoded(place), transform, 'white cube', 'on', .1, 1.5, output)
        tf_deadline = time.monotonic()+5
        while not node.tf_buffer.can_transform(node.base_frame, node.camera_frame, Time()):
            if time.monotonic()>tf_deadline:raise RuntimeError('camera TF unavailable')
            time.sleep(.1)
        stamp = node.get_clock().now().to_msg()
        ok, reason = node.planner.ready()
        if not ok:raise RuntimeError(reason)
        ok, reason = node.planner.apply_scene(node._safety_scene(stamp))
        if not ok:
            time.sleep(0.5)
            ok, reason = node.planner.apply_scene(node._safety_scene(node.get_clock().now().to_msg()))
        if not ok:raise RuntimeError(reason)
        held = json.loads((output/'executor_state.json').read_text())['held_geometry']
        state = RobotState()
        state.joint_state.name = ['joint'+str(i) for i in range(1, 9)]
        state.joint_state.position = [.9314, 1.9085, 1.4067, -.9434, -.0091, -.0648, .0137133, .0137133]
        old_valid, old_reason = node.planner.state_valid(state)
        # Simulate a fresh grasp from READY, without changing the real arm.
        ready = RobotState()
        ready.joint_state.name = list(state.joint_state.name)
        ready.joint_state.position = list(node.ready_joints)+[.044, .044]
        original = node.planner._plan_stage
        node.planner._plan_stage = lambda stage, pose, start, v, a: original(stage, pose, ready if start is None else start, v, a)
        payload = json.loads((grab/'grasp_response.json').read_text())
        node.target_cloud, node.scene_cloud, center, band = node._quality_clouds(encoded(grab), payload, transform)
        grasps = parse_api_grasps(payload)
        report = []; start = time.monotonic()
        selection = node._select_candidate(goal, grasps, [transform@g.transform_camera_grasp for g in grasps],
            stamp, True, capture_directory=output, place_target=target, report=report,
            centroid_locate=lambda: center.copy(), center_band=band)
        result = {'plan_only': True, 'controller_actions_sent': 0,
            'old_release_state_valid_with_support': old_valid, 'old_release_collision': old_reason,
            'tip_offset_m': node.gripper_geometry.tip_offset,
            'fresh_combined_plan_selected': selection is not None,
            'elapsed_s': round(time.monotonic()-start, 3), 'failures': report}
        if selection is not None:
            grasp, waypoints, plan = selection
            result.update(stages=plan.stages, grasp_tool0=waypoints['grasp'].tolist(),
                release_tool0=waypoints['place'].tolist(), expected_total_gap_m=grasp.width_m)
        (output/'contact_replay.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='failures'}, ensure_ascii=False), flush=True)
    finally:
        executor.shutdown();node.destroy_node();rclpy.shutdown();thread.join(timeout=2)


if __name__ == '__main__':
    main()
