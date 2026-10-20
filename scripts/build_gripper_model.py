#!/usr/bin/env python3
"""Derive a shared stock-jaw contact/collision model from the accepted URDF.

No camera, ROS, network, planning or motion. Physical installation confirmation
is recorded separately from dimensional metrology: CAD is not a caliper reading.
"""
import argparse,hashlib,json,struct,xml.etree.ElementTree as ET
import yaml
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from x5a_remote_grasp.gripper_model import model_digest


def origin(element):
    value=np.eye(4)
    if element is not None:
        value[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        value[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return value


def build(root):
    measurement_path=root/'config/gripper_measurements.yaml';measurement=yaml.safe_load(measurement_path.read_text()) if measurement_path.exists() else {}
    correction=0.
    if measurement.get('body_front_reference_confirmed'):
        if not measurement.get('flat_contact_length_m'):raise ValueError('measured contact face length required')
        correction=float(measurement['reported_body_front_to_tip_m'])-.0735700034570694
        if not 0<=correction<=.015:raise ValueError('measurement does not match stock jaw family')
    urdf=root/'config/calibration/robot_model_candidate.urdf';robot=ET.parse(urdf).getroot()
    joint=next(j for j in robot.findall('joint') if j.find('child').get('link')=='tool0')
    tool=origin(joint.find('origin'));sources={str(urdf.relative_to(root)):hashlib.sha256(urdf.read_bytes()).hexdigest()};bodies=[];contacts=[]
    for name in ['link6','link7','link8']:
        link=robot.find(f"link[@name='{name}']");collision=link.find('collision');mesh=collision.find('geometry/mesh')
        if mesh is None or len(link.findall('collision'))!=1:raise ValueError('expected one stock CAD mesh per body')
        path=root/'src'/mesh.get('filename').replace('package://','');sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        raw=path.read_bytes();count=int.from_bytes(raw[80:84],'little')
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attrs','<u2')])
        if len(raw)!=84+50*count:raise ValueError('expected binary STL')
        triangles=np.frombuffer(raw[84:],dtype=dtype)['vertices'].astype(float)
        link_pose=np.eye(4);axis=np.zeros(3)
        if name!='link6':
            j=next(j for j in robot.findall('joint') if j.find('child').get('link')==name)
            if j.get('type')!='prismatic' or j.find('parent').get('link')!='link6':raise ValueError('unexpected jaw kinematics')
            link_pose=origin(j.find('origin'));axis=(np.linalg.inv(tool)@link_pose)[:3,:3]@np.fromstring(j.find('axis').get('xyz'),sep=' ')
        transform=np.linalg.inv(tool)@link_pose@origin(collision.find('origin'));scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        triangles=(triangles*scale)@transform[:3,:3].T+transform[:3,3]
        points=np.unique(triangles.reshape(-1,3),axis=0)
        if name!='link6' and correction:
            # Cover original CAD AND the measured forward extent. This never
            # shrinks collision geometry or assumes how the extra length mounts.
            points=np.vstack((points,points+[correction,0,0]))
        hull=ConvexHull(points);vertices=points[hull.vertices]
        mesh_output=root/f'models/x5a_stock_collision_{name}.stl';mesh_output.parent.mkdir(exist_ok=True)
        local=(points[hull.simplices]-transform[:3,3])@transform[:3,:3]
        contents=bytearray(80)+bytearray(struct.pack('<I',len(local)))
        for triangle_index,triangle in enumerate(local):
            cross=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal=cross/np.linalg.norm(cross)
            if np.dot(normal,hull.equations[triangle_index,:3]@transform[:3,:3])<0:
                triangle=triangle[[0,2,1]];normal=-normal
            contents.extend(struct.pack('<12fH',*(normal.tolist()+triangle.ravel().tolist()),0))
        mesh_output.write_bytes(contents)
        bodies.append(dict(name=name,jaw_axis=axis.tolist(),vertices_tool0_m=vertices.tolist(),planes_tool0=np.unique(np.round(hull.equations,12),axis=0).tolist(),collision_mesh=str(mesh_output.relative_to(root)),collision_mesh_sha256=hashlib.sha256(contents).hexdigest()))
        if name=='link6':continue
        cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);area=np.linalg.norm(cross,axis=1);normals=cross/np.maximum(area[:,None],1e-20)
        # Original tapered jaws have a flat inner front face on Y=0 at q=0.
        inner=(normals[:,1]*np.sign(axis[1])<-.999)&(area>2e-6)&(triangles.mean(axis=1)[:,0]>0)&(np.max(abs(triangles[:,:,1]),axis=1)<.0001)
        face=triangles[inner]
        if len(face)<2:raise ValueError('no flat stock-jaw contact face; manual profile required')
        if correction:face=face+[correction,0,0]
        front=float(face[:,:,0].max());back=float(face[:,:,0].min());end=face.reshape(-1,3)[np.isclose(face.reshape(-1,3)[:,0],front,atol=1e-6)]
        height=2*min(float(end[:,2].max()),float(-end[:,2].min()))
        contacts.append(dict(name=name,tip_x_m=front,start_x_m=back,height_m=height,triangles_tool0_m=face.tolist()))
    tip=min(x['tip_x_m'] for x in contacts);start=max(x['start_x_m'] for x in contacts)
    if correction:start=tip-float(measurement['flat_contact_length_m'])
    height=min(x['height_m'] for x in contacts)
    data=dict(native_grasp_reference_offset_m=.015,schema=1,name='x5a_stock_tapered_cad_v1',tcp_frame='wrist_preview/tool0',max_width_m=.088,contact=dict(tip_x_m=tip,length_m=tip-start,height_m=height,faces=contacts),finger_length_m=tip-min(np.min(x['vertices_tool0_m'],axis=0)[0] for x in bodies if x['name']!='link6'),finger_thickness_m=.01,bodies=bodies,source_hashes=sources,evidence=dict(installation='user_confirmed_original_unmodified_2026-10-16',dimensions='derived_from_vendor_CAD_not_physical_caliper_measurement',tcp_preserved=True,collision='convex CAD envelopes; never smaller than source mesh'))
    data['contact_validation']=dict(frame='gravity_aligned',scope='thin_elongated_tools',tip_inset_m=.005,rear_inset_m=.002,height_inset_m=.001,pair_distance_m=.0025,min_pair_voxels=3,thin_tool_min_closing_angle_deg=70.,thin_tool_max_vertical_closing_deg=20.)
    data['physical_measurement']=measurement
    data['forward_collision_extension_m']=correction
    if measurement_path.exists():data['source_hashes'][str(measurement_path.relative_to(root))]=hashlib.sha256(measurement_path.read_bytes()).hexdigest()
    data['model_id']=model_digest(data);return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output');args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    output=Path(args.output) if args.output else root/'models/x5a_stock_gripper.json';data=build(root);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(path=str(output),model_id=data['model_id'],tip_mm=data['contact']['tip_x_m']*1000,pad_length_mm=data['contact']['length_m']*1000,pad_height_mm=data['contact']['height_m']*1000,hull_planes=[len(b['planes_tool0']) for b in data['bodies']]),ensure_ascii=False))
