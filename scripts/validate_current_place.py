#!/usr/bin/env python3
"""Plan a recorded support placement with the currently held object, without execution.

Uses the existing MoveIt world; does not apply a scene or publish a trajectory.
The provided support capture must match the current scene. Isolated ROS action
and visualization names prevent the executor from consuming the result.
"""
import argparse
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
from x5a_remote_grasp.place_geometry import support_surface
from validate_parallel_planning import idle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not idle():raise RuntimeError('Task active; replay deferred')
    root=Path(__file__).resolve().parents[1]
    rclpy.init(args=['--ros-args','--params-file',str(root/'src/x5a_remote_grasp/config/remote_grasp.yaml'),
        '-r','__node:=x5a_current_place_review','-p','action_name:=/x5a_current_place_review/plan',
        '-p','health_service:=/x5a_current_place_review/health',
        '-p','save_snapshot_service:=/x5a_current_place_review/snapshot',
        '-p','validation_service:=/x5a_current_place_review/validate',
        '-p','topics.display_path:=/x5a_current_place_review/display_path',
        '-p','markers.topic:=/x5a_current_place_review/candidates'])
    node=RemoteGraspNode();executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    def state():
        client=node.executor_state_client
        if not client.wait_for_service(timeout_sec=4):raise RuntimeError('Executor unavailable')
        future=client.call_async(Trigger.Request())
        if not node.planner.wait_future(future,4):raise RuntimeError('Executor state timeout')
        response=future.result()
        if response is None or not response.success:raise RuntimeError('Executor state unavailable')
        return json.loads(response.message)
    results={}
    try:
        before=state()
        held=before.get('held_geometry')
        if before.get('running') or before.get('holding_state')!='held' or not held:
            raise RuntimeError('Requires idle executor with confirmed held geometry')
        if not node.planner.parallel_ik.client.wait_for_service(timeout_sec=4):
            raise RuntimeError('Independent IK service unavailable')
        node.place_object_cloud_tool=np.asarray(held['cloud_tool'])
        node.place_grasp_rotation=np.asarray(held['grasp_rotation'])
        support=np.load(args.capture/'place_support_cloud_base.npy')
        surface,node.place_support_hull=support_surface(support)
        node.place_support_zrange=tuple(np.percentile(support[:,2],[5,95]))
        node.place_relation='on'
        if not idle():raise RuntimeError('User task started; replay deferred')
        report=[];start=time.monotonic();stamp=node.get_clock().now().to_msg()
        selected=node._plan_place_only(NS(is_cancel_requested=False,publish_feedback=lambda _:None),
            (surface,'on'),stamp,start+node.plan_budget_s,report)
        results={'capture':str(args.capture),'elapsed_s':round(time.monotonic()-start,3),
                 'selected':selected is not None,'attempts':len(report),
                 'release_clearances_m':node.place_release_clearances,
                 'scene_changes_sent':0,'controller_actions_sent':0}
        if selected is not None:
            waypoints,plan=selected
            world=node.place_object_cloud_tool@waypoints['place'][:3,:3].T+waypoints['place'][:3,3]
            lo,hi=np.percentile(world,[2,98],axis=0)
            results.update(tool0=waypoints['place'].tolist(),stages=plan.stages,
                bottom_clearance_m=node._place_clearance(waypoints['place'],surface,node.place_object_cloud_tool),
                object_xy_error_m=float(np.linalg.norm((lo[:2]+hi[:2])/2-surface[:2])))
        else:results['failures']=report
        after=state()
        results['held_geometry_preserved']=before['held_geometry']==after.get('held_geometry')
        results['still_holding']=after.get('holding')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results),flush=True)
        assert selected is not None,'No complete collision-free placement found'
        assert .009999 <= results['bottom_clearance_m'] <= .020001
        assert results['object_xy_error_m']<1e-6
        assert results['held_geometry_preserved'] and results['still_holding']
    finally:
        node.planner.deadline=None
        executor.shutdown();node.destroy_node();rclpy.shutdown();thread.join(timeout=2)


if __name__=='__main__':main()
