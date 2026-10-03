#!/usr/bin/env python3
"""Archived grasp planning comparison. Never publishes or executes a trajectory.

Temporarily applies the archived safety world to MoveIt and restores the complete
original scene in finally. Run only with the user task pipeline idle.
"""
import argparse
import copy
import json
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
from x5a_remote_grasp.geometry import parse_api_grasps


def idle():
    state=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/state',timeout=3))
    return not state.get('busy') and state.get('robot_state')=='IDLE'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',nargs='+',default=['20261003_170234_987941','20261003_170536_818755'])
    args=parser.parse_args()
    if not idle():raise RuntimeError('Task pipeline active; read-only trajectory replay deferred')
    root=Path(__file__).resolve().parents[1];output=root/'docs/reviews/parallel_ik_20261003'
    output.mkdir(parents=True,exist_ok=True)
    rclpy.init(args=['--ros-args','--params-file',str(root/'src/x5a_remote_grasp/config/remote_grasp.yaml'),
        '-r','__node:=x5a_parallel_plan_review','-p','action_name:=/x5a_parallel_plan_review/plan',
        '-p','health_service:=/x5a_parallel_plan_review/health',
        '-p','save_snapshot_service:=/x5a_parallel_plan_review/snapshot',
        '-p','validation_service:=/x5a_parallel_plan_review/validate',
        '-p','topics.display_path:=/x5a_parallel_plan_review/display_path',
        '-p','markers.topic:=/x5a_parallel_plan_review/candidates'])
    node=RemoteGraspNode();executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    original=None;backend=node.planner.parallel_ik;results=[]
    def future_result(client,request):
        if not client.wait_for_service(timeout_sec=4):raise RuntimeError('Scene service unavailable')
        future=client.call_async(request)
        if not node.planner.wait_future(future,5) or future.result() is None:raise RuntimeError('Scene service timeout')
        return future.result()
    try:
        request=GetPlanningScene.Request();request.components.components=1023
        original=copy.deepcopy(future_result(node.planner.scene_query_client,request).scene)
        end=time.monotonic()+5
        while node.latest_joints is None or not node.tf_buffer.can_transform(node.base_frame,node.camera_frame,Time()):
            if time.monotonic()>end:raise RuntimeError('Fresh arm state or camera TF unavailable')
            time.sleep(.05)
        if not backend.client.wait_for_service(timeout_sec=4):raise RuntimeError('Independent IK service unavailable')
        for name in args.captures:
            capture=Path.home()/'.local/share/x5a_remote_grasp/captures'/name
            metadata=json.loads((capture/'camera_info.json').read_text())
            encoded=NS(depth_png=(capture/'depth.png').read_bytes(),
                **{k:metadata[k] for k in ('fx','fy','cx','cy','depth_scale_m_per_unit')})
            payload=json.loads((capture/'grasp_response.json').read_text())
            transform=np.asarray(json.loads((capture/'tf_validation.json').read_text())['transform_base_camera'])
            node.target_cloud,node.scene_cloud,center,band=node._quality_clouds(encoded,payload,transform)
            grasps=parse_api_grasps(payload)
            for mode in ('serial','parallel'):
                if not idle():raise RuntimeError('User task started; aborting replay')
                node.planner.parallel_ik=None if mode=='serial' else backend
                node.last_successful_seed=None
                report=[];start=time.monotonic()
                selected=node._select_candidate(NS(is_cancel_requested=False,publish_feedback=lambda _:None),
                    grasps,[transform@g.transform_camera_grasp for g in grasps],node.get_clock().now().to_msg(),
                    True,report=report,centroid_locate=lambda:center.copy(),center_band=band)
                result={'capture':name,'mode':mode,'elapsed_s':round(time.monotonic()-start,3),
                    'selected':selected is not None,'controller_actions_sent':0,
                    'planning_rejections':len(report)}
                if selected is not None:
                    grasp,waypoints,plan=selected
                    result.update(source=node.selection_kind,tool0=waypoints['grasp'].tolist(),
                        required_opening_m=grasp.width_m,stages=plan.stages,
                        segment_durations_s=[t.joint_trajectory.points[-1].time_from_start.sec+
                            t.joint_trajectory.points[-1].time_from_start.nanosec*1e-9 for t in plan.trajectories])
                results.append(result);print(json.dumps(result),flush=True)
                assert selected is not None,'Recorded grasp lost solvability: '+mode
    finally:
        node.planner.deadline=None
        if original is not None:
            # Restore the world/ACM/padding but keep the fresh live arm state.
            request=GetPlanningScene.Request();request.components.components=1023
            current=future_result(node.planner.scene_query_client,request).scene
            original.robot_state=current.robot_state
            restore=ApplyPlanningScene.Request();restore.scene=original
            if not future_result(node.planner.scene_client,restore).success:
                raise RuntimeError('Failed to restore authoritative scene')
        (output/'planning_replay.json').write_text(json.dumps({'scene_restored':original is not None,
            'controller_actions_sent':0,'results':results},indent=2)+'\n')
        executor.shutdown();node.destroy_node();rclpy.shutdown();thread.join(timeout=2)


if __name__=='__main__':main()
