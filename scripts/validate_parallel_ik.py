#!/usr/bin/env python3
"""Exercise isolated IK snapshots and compare 1/4 workers; never moves the arm."""
import copy
import hashlib
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit_msgs.msg import RobotState, PositionIKRequest, CollisionObject
from moveit_msgs.srv import GetPlanningScene, GetPositionFK, GetPositionIK, GetStateValidity
from rcl_interfaces.srv import GetParameters
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from x5a_task_interfaces.srv import ParallelIK


def main():
    rclpy.init();node=Node('x5a_parallel_ik_review');cb=ReentrantCallbackGroup()
    executor=MultiThreadedExecutor(num_threads=2);executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    clients={}
    def call(kind, service, request):
        key=(kind,service)
        if key not in clients:clients[key]=node.create_client(kind,service,callback_group=cb)
        client=clients[key]
        if not client.wait_for_service(timeout_sec=4):raise RuntimeError('Service unavailable: '+service)
        future=client.call_async(request);deadline=time.monotonic()+8
        while not future.done():
            if time.monotonic()>deadline:raise RuntimeError('Service timed out: '+service)
            time.sleep(.002)
        return future.result()
    try:
        request=GetPlanningScene.Request();request.components.components=1023
        scene=call(GetPlanningScene,'/get_planning_scene',request).scene
        request=GetParameters.Request();request.names=['robot_description','robot_description_semantic']
        values=call(GetParameters,'/move_group/get_parameters',request).values
        digest=hashlib.sha256((values[0].string_value+'\0'+values[1].string_value).encode()).hexdigest()
        ready=RobotState();ready.joint_state.name=['joint'+str(i) for i in range(1,9)]
        ready.joint_state.position=[.72,.85,.95,-.55,0.,0.,.044,.044]
        fk=GetPositionFK.Request();fk.header.frame_id='base_link';fk.fk_link_names=['tool0'];fk.robot_state=ready
        target=call(GetPositionFK,'/compute_fk',fk).pose_stamped[0]
        seeds=[]
        for yaw in (.72,.74,.70,.78,.68,.80,.65,.82):
            seed=copy.deepcopy(ready);seed.joint_state.position[0]=yaw;seeds.append(seed)
        def batch(pose, workers, local_scene=scene):
            request=ParallelIK.Request();request.scene=local_scene;request.model_digest=digest
            request.workers=workers;request.total_budget_s=3.
            for seed in seeds:
                item=PositionIKRequest();item.group_name='arm';item.ik_link_name='tool0'
                item.pose_stamped=pose;item.robot_state=copy.deepcopy(seed);item.avoid_collisions=True
                item.timeout.nanosec=150000000;request.requests.append(item)
            return request
        results=[]
        unreachable=copy.deepcopy(target);unreachable.pose.position.x=2.;unreachable.pose.position.z=1.
        for case,pose in [('reachable',target),('unreachable',unreachable)]:
            for workers in (1,4):
                elapsed=[];server=[]
                for _ in range(3):
                    request=batch(pose,workers);start=time.monotonic()
                    result=call(ParallelIK,'/x5a_parallel_ik/solve',request)
                    elapsed.append(time.monotonic()-start);server.append(result.elapsed_s)
                    assert result.success, result.message
                    codes=[code.val for code in result.error_codes]
                    if case=='unreachable':assert all(code==-31 for code in codes),codes
                    else:
                        assert codes[0]==1, codes
                        for code,solution in zip(codes,result.solutions):
                            if code!=1:continue
                            validity=GetStateValidity.Request();validity.group_name='arm';validity.robot_state=solution
                            assert call(GetStateValidity,'/check_state_validity',validity).valid
                            fk.robot_state=solution;actual=call(GetPositionFK,'/compute_fk',fk).pose_stamped[0].pose
                            delta=np.linalg.norm([actual.position.x-pose.pose.position.x,
                                actual.position.y-pose.pose.position.y,actual.position.z-pose.pose.position.z])
                            assert delta<.003
                            a=actual.orientation;b=pose.pose.orientation
                            angle=2*np.arccos(np.clip(abs(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w),0,1))
                            assert angle<.05
                            names=list(solution.joint_state.name)
                            assert abs(solution.joint_state.position[names.index('joint7')]-.044)<1e-9
                            assert abs(solution.joint_state.position[names.index('joint8')]-.044)<1e-9
                results.append({'case':case,'workers':workers,'median_wall_s':round(statistics.median(elapsed),4),
                    'median_server_s':round(statistics.median(server),4),'seed_count':8,'last_error_codes':codes})
                print(json.dumps(results[-1]),flush=True)
        # Obstacle lives ONLY in the request snapshot, never in real MoveIt.
        blocked=copy.deepcopy(scene);obstacle=CollisionObject();obstacle.id='review_tcp_blocker'
        obstacle.header.frame_id='base_link';obstacle.operation=CollisionObject.ADD
        obstacle.primitives=[SolidPrimitive(type=SolidPrimitive.BOX,dimensions=[.2,.2,.2])]
        placement=Pose();placement.position=copy.deepcopy(target.pose.position);placement.orientation.w=1.
        obstacle.primitive_poses=[placement];blocked.world.collision_objects.append(obstacle)
        result=call(ParallelIK,'/x5a_parallel_ik/solve',batch(target,4,blocked))
        assert result.success and all(code.val==-31 for code in result.error_codes),result.message
        contracts={'snapshot_obstacle_rejected':True}
        result=call(ParallelIK,'/x5a_parallel_ik/solve',batch(target,4))
        assert result.success and result.error_codes[0].val==1,result.message
        contracts['previous_snapshot_obstacles_cleared']=True
        for name, mutate in [('wrong_model',lambda req:setattr(req,'model_digest','wrong')),
                             ('diff_scene',lambda req:setattr(req.scene,'is_diff',True)),
                             ('missing_fingers',lambda req:setattr(req.scene.robot_state.joint_state,'position',[]))]:
            request=batch(target,4,copy.deepcopy(scene));mutate(request)
            result=call(ParallelIK,'/x5a_parallel_ik/solve',request)
            assert not result.success,name;contracts[name+'_rejected']=True
        # Arm-only seeds must preserve the snapshot's CURRENT finger positions.
        request=batch(target,4,copy.deepcopy(scene))
        for item in request.requests:
            item.robot_state.joint_state.name=item.robot_state.joint_state.name[:6]
            item.robot_state.joint_state.position=item.robot_state.joint_state.position[:6]
            item.robot_state.is_diff=True
        result=call(ParallelIK,'/x5a_parallel_ik/solve',request)
        assert result.success and result.error_codes[0].val==1,result.message
        original=dict(zip(scene.robot_state.joint_state.name,scene.robot_state.joint_state.position))
        actual=dict(zip(result.solutions[0].joint_state.name,result.solutions[0].joint_state.position))
        assert all(abs(actual[j]-original[j])<1e-9 for j in ('joint7','joint8'))
        contracts['observed_fingers_preserved']=True
        root=Path(__file__).resolve().parents[1];output=root/'docs/reviews/parallel_ik_20261003'
        output.mkdir(parents=True,exist_ok=True)
        (output/'service_validation.json').write_text(json.dumps({'controller_actions_sent':0,
            'scene_changes_sent':0,'model_digest':digest,'results':results,'contracts':contracts},indent=2)+'\n')
        print(json.dumps(contracts),flush=True)
    finally:
        executor.shutdown();node.destroy_node();rclpy.shutdown();thread.join(timeout=2)


if __name__=='__main__':main()
