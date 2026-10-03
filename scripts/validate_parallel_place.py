#!/usr/bin/env python3
"""Recorded object-attachment placement replay; plans only, never publishes.

Uses simulated READY and 15 mm finger joint positions. Temporarily imports each
recorded support into MoveIt and restores the original scene in finally.
"""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import rclpy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
from x5a_remote_grasp.place_geometry import support_surface
from validate_parallel_planning import idle


def main():
    if not idle():raise RuntimeError('Task active; placement replay deferred')
    root=Path(__file__).resolve().parents[1]
    held=json.loads((root/'docs/reviews/place_solvability_20261003/held_state.json').read_text())['held_geometry']
    rclpy.init(args=['--ros-args','--params-file',str(root/'src/x5a_remote_grasp/config/remote_grasp.yaml'),
        '-r','__node:=x5a_parallel_place_review','-p','action_name:=/x5a_parallel_place_review/plan',
        '-p','health_service:=/x5a_parallel_place_review/health',
        '-p','save_snapshot_service:=/x5a_parallel_place_review/snapshot',
        '-p','validation_service:=/x5a_parallel_place_review/validate',
        '-p','topics.display_path:=/x5a_parallel_place_review/display_path',
        '-p','markers.topic:=/x5a_parallel_place_review/candidates'])
    node=RemoteGraspNode();executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    original=None;results=[];backend=node.planner.parallel_ik
    def call(client,request):
        if not client.wait_for_service(timeout_sec=4):raise RuntimeError('Scene service unavailable')
        future=client.call_async(request)
        if not node.planner.wait_future(future,5) or future.result() is None:raise RuntimeError('Scene service timeout')
        return future.result()
    try:
        request=GetPlanningScene.Request();request.components.components=1023
        original=call(node.planner.scene_query_client,request).scene
        end=time.monotonic()+5
        while not node.tf_buffer.can_transform(node.base_frame,node.camera_frame,Time()):
            if time.monotonic()>end:raise RuntimeError('Camera TF unavailable')
            time.sleep(.05)
        if not backend.client.wait_for_service(timeout_sec=4):raise RuntimeError('Independent IK service unavailable')
        ready=RobotState();ready.joint_state.name=['joint'+str(i) for i in range(1,9)]
        ready.joint_state.position=list(node.ready_joints)+[.015,.015]
        stage=node.planner._plan_stage
        node.planner._plan_stage=lambda name,pose,start,v,a:stage(name,pose,ready if start is None else start,v,a)
        node.place_object_cloud_tool=np.asarray(held['cloud_tool'])
        node.place_grasp_rotation=np.asarray(held['grasp_rotation'])
        node.place_relation='on'
        for name in ('20261003_164100_000967','20261003_164147_521557'):
            support=np.load(Path.home()/'.local/share/x5a_remote_grasp/captures'/name/'place_support_cloud_base.npy')
            surface,node.place_support_hull=support_surface(support)
            node.place_support_zrange=tuple(np.percentile(support[:,2],[5,95]))
            for mode in ('serial','parallel'):
                if not idle():raise RuntimeError('User task started; aborting replay')
                node.planner.parallel_ik=None if mode=='serial' else backend
                stamp=node.get_clock().now().to_msg();node.planner.deadline=None
                ok,reason=node.planner.apply_scene(node._safety_scene(stamp))
                if not ok:raise RuntimeError(reason)
                start=time.monotonic()
                selected=node._plan_place_only(NS(is_cancel_requested=False,publish_feedback=lambda _:None),
                    (surface,'on'),stamp,start+node.plan_budget_s,[])
                result={'capture':name,'mode':mode,'elapsed_s':round(time.monotonic()-start,3),
                    'selected':selected is not None,'simulated_finger_position_m':.015}
                if selected is not None:
                    waypoints,plan=selected
                    result.update(tool0=waypoints['place'].tolist(),stages=plan.stages,
                        bottom_clearance_m=node._place_clearance(waypoints['place'],surface,node.place_object_cloud_tool))
                results.append(result);print(json.dumps(result),flush=True)
                assert selected is not None,'Placement lost solvability: '+mode
    finally:
        node.planner.deadline=None
        if original is not None:
            request=GetPlanningScene.Request();request.components.components=1023
            original.robot_state=call(node.planner.scene_query_client,request).scene.robot_state
            restore=ApplyPlanningScene.Request();restore.scene=original
            if not call(node.planner.scene_client,restore).success:raise RuntimeError('Scene restore failed')
        output=root/'docs/reviews/parallel_ik_20261003';output.mkdir(parents=True,exist_ok=True)
        (output/'placement_replay.json').write_text(json.dumps({'scene_restored':original is not None,
            'controller_actions_sent':0,'results':results},indent=2)+'\n')
        executor.shutdown();node.destroy_node();rclpy.shutdown();thread.join(timeout=2)


if __name__=='__main__':main()
