#!/usr/bin/env python3
"""Replay the last two failed placements with recorded held geometry, plan-only."""
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
from sensor_msgs.msg import JointState
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
from x5a_remote_grasp.place_geometry import support_surface


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'docs/reviews/place_solvability_20261003'
    captures = Path.home()/'.local/share/x5a_remote_grasp/captures'
    held = json.loads((output/'held_state.json').read_text())['held_geometry']
    rclpy.init(args=['--ros-args', '--params-file', str(root/'src/x5a_remote_grasp/config/remote_grasp.yaml'),
        '-r', '__node:=x5a_place_solvability_review', '-p', 'action_name:=/x5a_place_solvability_review/plan',
        '-p', 'health_service:=/x5a_place_solvability_review/health',
        '-p', 'save_snapshot_service:=/x5a_place_solvability_review/save_snapshot',
        '-p', 'validation_service:=/x5a_place_solvability_review/validate_latest',
        '-p', 'topics.display_path:=/x5a_place_solvability_review/display_path',
        '-p', 'markers.topic:=/x5a_place_solvability_review/candidates'])
    node = RemoteGraspNode();executor = MultiThreadedExecutor(num_threads=4);executor.add_node(node)
    live = {};node.create_subscription(JointState, '/joint_states', lambda msg: live.update(joints=msg), 10)
    thread = threading.Thread(target=executor.spin, daemon=True);thread.start()
    try:
        deadline = time.monotonic()+6
        while 'joints' not in live or not node.tf_buffer.can_transform(node.base_frame, node.camera_frame, Time()):
            if time.monotonic()>deadline:raise RuntimeError('fresh TF/joints unavailable')
            time.sleep(.1)
        actual = dict(zip(live['joints'].name, live['joints'].position))
        ready = RobotState();ready.joint_state.name = ['joint'+str(i) for i in range(1, 9)]
        ready.joint_state.position = list(node.ready_joints)+[actual['joint7'], actual['joint8']]
        original = node.planner._plan_stage
        node.planner._plan_stage = lambda stage, pose, start, v, a: original(stage, pose, ready if start is None else start, v, a)
        node.place_object_cloud_tool = np.asarray(held['cloud_tool'])
        node.place_grasp_rotation = np.asarray(held['grasp_rotation'])
        node.place_relation = 'on'
        results = []
        for name in ('20261003_164100_000967', '20261003_164147_521557'):
            capture = captures/name; support = np.load(capture/'place_support_cloud_base.npy')
            surface, node.place_support_hull = support_surface(support)
            node.place_support_zrange = tuple(np.percentile(support[:, 2], [5, 95]))
            stamp = node.get_clock().now().to_msg()
            ok, reason = node.planner.ready()
            if not ok:raise RuntimeError(reason)
            ok, reason = node.planner.apply_scene(node._safety_scene(stamp))
            if not ok:raise RuntimeError(reason)
            reports = []; start = time.monotonic()
            selected = node._plan_place_only(NS(is_cancel_requested=False, publish_feedback=lambda _: None),
                (surface, 'on'), stamp, start+node.plan_budget_s, reports)
            result = {'capture': name, 'plan_only': True, 'controller_actions_sent': 0,
                'selected': selected is not None, 'elapsed_s': round(time.monotonic()-start, 3),
                'surface': surface.tolist(), 'failures': reports}
            if selected is not None:
                waypoints, plan = selected
                result.update(stages=plan.stages, release_tool0=waypoints['place'].tolist(),
                    observed_bottom_clearance_m=node._place_clearance(waypoints['place'], surface, node.place_object_cloud_tool))
            results.append(result);node.planner.deadline = None
            print(json.dumps(result, ensure_ascii=False), flush=True)
        (output/'replay.json').write_text(json.dumps(results, indent=2, ensure_ascii=False)+'\n')
    finally:
        executor.shutdown();node.destroy_node();rclpy.shutdown();thread.join(timeout=2)


if __name__ == '__main__':
    main()
