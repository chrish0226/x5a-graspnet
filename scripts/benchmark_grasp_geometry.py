#!/usr/bin/env python3
"""Archived-frame CPU benchmark; sends no ROS or controller requests."""
import argparse
import json, base64, time, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from x5a_remote_grasp.grasp_quality import PreparedGraspScene, GripperGeometry, assess_tool_grasp, recenter_tool, depth_cloud, height_statistics, voxel_cloud, graspnet_tool_transform
from x5a_remote_grasp.geometry import parse_api_grasps, rpy_rotation, parallel_jaw_rotations
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture',default='20261003_170536_818755')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
capture=Path.home()/'.local/share/x5a_remote_grasp/captures'/args.capture
meta=json.loads((capture/'camera_info.json').read_text());payload=json.loads((capture/'grasp_response.json').read_text())
transform=np.asarray(json.loads((capture/'tf_validation.json').read_text())['transform_base_camera'])
depth=cv2.imread(str(capture/'depth.png'),cv2.IMREAD_UNCHANGED).astype(float)*meta['depth_scale_m_per_unit']
mask=cv2.imdecode(np.frombuffer(base64.b64decode(payload['target_mask_png_b64']),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
cloud=depth_cloud(depth,meta['fx'],meta['fy'],meta['cx'],meta['cy']);valid=np.isfinite(depth)&(depth>=.1)&(depth<=1.5)
target,center,low,high=height_statistics(cloud[(mask>127)&valid],transform)
full=cloud[valid]@transform[:3,:3].T+transform[:3,3]
lo,hi=np.min(target,axis=0)-.22,np.max(target,axis=0)+.22
scene=voxel_cloud(full[np.all((full>=lo)&(full<=hi),axis=1)])
geometry=GripperGeometry();mapping=np.eye(4);mapping[0,3]=-.015
tools=[]
for g in parse_api_grasps(payload):
    base=transform@g.transform_camera_grasp
    for rotation in parallel_jaw_rotations(base[:3,:3]):
        native=base.copy();native[:3,:3]=rotation
        tools.append((graspnet_tool_transform(native,g.depth_m,mapping),.008))
for fraction in (.35,.55,.75):
    point=center.copy();point[2]=max(-.006+.006,low+fraction*(high-low))
    for pitch in (1.45,1.15,.90,.75):
        for degrees in (0,45,-45,180,15,-15):
            rotation=rpy_rotation(0,pitch,np.arctan2(point[1],point[0])+np.deg2rad(degrees))
            tool=np.eye(4);tool[:3,:3]=rotation;tool[:3,3]=point-rotation[:,0]*.015
            tools.append((tool,.02))
prepared=PreparedGraspScene(scene)
def work(item, source):
    tool=recenter_tool(target,item[0],geometry,max_shift=item[1])
    return {'tool':tool.tolist(), 'measure':assess_tool_grasp(target,source,tool,geometry,.05)}
reference=[work(item,scene) for item in tools]
results=[]
for cached in (False,True):
    source=prepared if cached else scene
    for workers in (1,2,4,8):
        times=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(4):
                start=time.perf_counter()
                output=list(pool.map(lambda item:work(item,source),tools)) if workers>1 else [work(item,source) for item in tools]
                times.append(time.perf_counter()-start)
                if output!=reference:raise AssertionError('different geometry or collision result')
        results.append({'prepared_scene':cached,'workers':workers,'median_s':round(statistics.median(times),5),'runs_s':[round(t,5) for t in times],'identical_to_original':True})
output={'capture':capture.name,'proposals':len(tools),'target_points':len(target),'scene_voxels':len(scene),'controller_actions_sent':0,'results':results}
output_dir=root/'docs/reviews/planning_speed_20261003'
output_dir.mkdir(parents=True,exist_ok=True)
(output_dir/'geometry_benchmark.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
