"""Contact geometry and whole-CAD regression tests, never send robot goals."""
import copy,json
from pathlib import Path

import numpy as np
import pytest
from x5a_remote_grasp.gripper_model import GripperModel,model_digest
from x5a_remote_grasp.grasp_quality import assess_tool_grasp,graspnet_tool_transform
from x5a_remote_grasp.geometry import rpy_rotation

ROOT=Path(__file__).resolve().parents[3]

@pytest.fixture
def model():
    return GripperModel.load(ROOT/'models/x5a_stock_gripper.json')


def contact():
    x,y,z=np.meshgrid(np.linspace(.038,.049,10),np.linspace(-.01,.01,17),np.linspace(-.003,.003,4))
    return np.column_stack([x.ravel(),y.ravel(),z.ravel()])


def test_model_matches_unchanged_urdf_and_stock_flat_face(model):
    model.validate_sources(ROOT)
    assert model.tip_offset==pytest.approx(.05123,abs=1e-7)
    assert model.tip_offset-(-.028770001132488246)==pytest.approx(.080,abs=1e-7)
    assert model.pad_length==pytest.approx(.015,abs=1e-7)
    assert model.pad_height==pytest.approx(.010,abs=1e-7)
    assert model.native_reference==.015  # successful insertion reference is separate
    for body in model.bodies:
        assert np.max(body.vertices@body.planes[:,:3].T+body.planes[:,3])<1e-7
        with pytest.raises(ValueError):body.vertices[0,0]=1.


def test_tampered_geometry_cannot_keep_old_model_id(model):
    data=copy.deepcopy(model.data);data['contact']['tip_x_m']=.015
    with pytest.raises(ValueError,match='hash'):GripperModel(data)


def test_source_mesh_drift_is_rejected_before_planning(model,tmp_path):
    with pytest.raises(FileNotFoundError):model.validate_sources(tmp_path)


def test_flat_contact_passes_but_sloped_backing_is_not_a_contact_pad(model):
    points=contact();g=model.geometry()
    assert assess_tool_grasp(points,points,np.eye(4),g)['valid']
    shifted=points-[.015,0,0]
    result=assess_tool_grasp(shifted,shifted,np.eye(4),g)
    assert result['reason']=='insufficient_inset_contact'


def test_obstacle_at_real_forward_finger_is_rejected(model):
    target=contact();x,y,z=np.meshgrid(np.linspace(.034,.039,5),np.linspace(.045,.047,5),np.linspace(-.003,.003,4))
    obstacle=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    checked=assess_tool_grasp(target,np.vstack([target,obstacle]),np.eye(4),model.geometry())
    assert not checked['valid'] and checked['reason']=='scene_collision'


def test_neighbour_closing_contact_cannot_be_excused_as_target(model):
    target=contact();x,y,z=np.meshgrid(np.linspace(.034,.040,5),np.linspace(.022,.026,5),np.linspace(-.003,.003,4))
    neighbour=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    checked=assess_tool_grasp(target,np.vstack([target,neighbour]),np.eye(4),model.geometry())
    assert not checked['valid'] and checked['reason']=='closing_collision'


@pytest.mark.parametrize('depth',[.01,.02,.03,.04])
def test_native_mapping_preserves_empirical_insertion_without_lying_about_tip(model,depth):
    grasp=np.eye(4);grasp[:3,:3]=rpy_rotation(.3,1.1,.7);grasp[:3,3]=[.2,.3,.1]
    mapping=np.eye(4);mapping[0,3]=-model.native_reference
    tool=graspnet_tool_transform(grasp,depth,mapping)
    physical_tip=tool[:3,3]+model.tip_offset*tool[:3,0]
    np.testing.assert_allclose(physical_tip,grasp[:3,3]+(depth+model.tip_offset-model.native_reference)*grasp[:3,0])


def test_cartesian_sweep_contains_translated_entire_mesh(model):
    for body in model.bodies:
        planes=body.swept_planes(.05)
        points=np.vstack([body.vertices,body.vertices-[.05,0,0]])
        assert np.max(points@planes[:,:3].T+planes[:,3])<1e-7


def test_moveit_uses_same_envelopes_without_changing_calibrated_joints(model):
    from x5a_handeye.robot_model import model_signature
    import xml.etree.ElementTree as ET
    source=(ROOT/'config/calibration/robot_model_candidate.urdf').read_text()
    updated=model.apply_to_urdf(source,ROOT)
    assert model_signature(updated)==model_signature(source)
    robot=ET.fromstring(updated)
    for body in model.data['bodies']:
        uri=robot.find("link[@name='%s']/collision/geometry/mesh"%body['name']).get('filename')
        assert uri==(ROOT/body['collision_mesh']).resolve().as_uri()


def test_original_finger_vertices_remain_inside_measured_extended_hulls(model):
    import struct,xml.etree.ElementTree as ET
    robot=ET.parse(ROOT/'config/calibration/robot_model_candidate.urdf').getroot()
    for body in model.bodies:
        if body.name=='link6':continue
        raw=(ROOT/f'src/X5A/meshes/{body.name}.STL').read_bytes()
        a=np.frombuffer(raw[84:],dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]))['v'].reshape(-1,3)
        j=next(j for j in robot.findall('joint') if j.find('child').get('link')==body.name)
        original=a+np.fromstring(j.find('origin').get('xyz'),sep=' ')-[.11277,0,0]
        assert np.max(original@body.planes[:,:3].T+body.planes[:,3])<1e-7


def thin_shaft(model):
    tool=np.eye(4);tool[:3,:3]=np.array([[0,1,0],[0,0,-1],[-1,0,0]])
    tool[:3,3]=[.25,.25,.06]
    x,y,z=np.meshgrid(np.linspace(model.tip_offset-.011,model.tip_offset-.006,8),np.linspace(-.006,.006,17),np.linspace(-.06,.06,81))
    local=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    return local@tool[:3,:3].T+tool[:3,3],tool


def test_thin_shaft_requires_crosswise_closing_and_inset_support(model):
    target,tool=thin_shaft(model)
    good=model.contact_quality(target,tool)
    assert good['valid'] and good['closing_to_tool_axis_deg']>=70
    assert good['contact_insertion_p10_m']>=.005
    bad=tool.copy();bad[:3,:3]=tool[:3,:3]@rpy_rotation(np.pi/2,0,0)
    assert model.contact_quality(target,bad)['reason']=='closing_along_tool'
    high=tool.copy();high[:3,3]-=.010*tool[:3,0]
    assert model.contact_quality(target,high)['reason']=='insufficient_inset_contact'


def test_opposite_sides_must_overlap_in_approach_and_pad_height(model):
    target,tool=thin_shaft(model);local=(target-tool[:3,3])@tool[:3,:3]
    # Two disjoint end patches cannot be called opposing pad contact.
    keep=((local[:,1]<-.002)&(local[:,0]<model.tip_offset-.009))|((local[:,1]>.002)&(local[:,0]>model.tip_offset-.007))
    checked=model.contact_quality(target[keep],tool)
    assert not checked['valid'] and checked['reason']=='nonopposing_contact'


def test_saved_real_empty_grasp_is_rejected(model):
    sample=np.load(Path(__file__).parent/'data/tool_empty_20261016.npz')
    assert not model.contact_quality(sample['target'],sample['tool'])['valid']


def test_actual_tcp_contact_check_rejects_shallow_arrival_before_closing(model):
    from types import SimpleNamespace as NS
    from x5a_remote_grasp.remote_grasp_executor import RemoteGraspExecutor
    target,tool=thin_shaft(model);checked=dict(geometry_model_id=model.model_id,object_cloud_base=target.tolist())
    fake=NS(gripper_model=model)
    RemoteGraspExecutor._verify_contact_geometry(fake,checked,tool)
    shifted=tool.copy();shifted[:3,3]-=.010*tool[:3,0]
    with pytest.raises(RuntimeError,match='未合爪'):
        RemoteGraspExecutor._verify_contact_geometry(fake,checked,shifted)


def test_tool_contact_is_consistent_between_base_and_camera_frames(model):
    from x5a_remote_grasp.gripper_model import gravity_aligned_rotation
    target,tool=thin_shaft(model);rotation=rpy_rotation(.7,1.2,-.5)
    camera_target=target@rotation;camera_tool=tool.copy();camera_tool[:3,:3]=rotation.T@tool[:3,:3];camera_tool[:3,3]=rotation.T@tool[:3,3]
    aligned=gravity_aligned_rotation(rotation.T@np.array([0.,0.,1.]))
    check=camera_tool.copy();check[:3,:3]=aligned@camera_tool[:3,:3];check[:3,3]=aligned@camera_tool[:3,3]
    assert model.contact_quality(camera_target@aligned.T,check)['valid']
    assert np.linalg.det(aligned)==pytest.approx(1.)
    with pytest.raises(ValueError):gravity_aligned_rotation([0,0,0])
