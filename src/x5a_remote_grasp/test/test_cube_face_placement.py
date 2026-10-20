"""Cube face equivalence, rather than keeping a particular top face upward."""
import numpy as np
from x5a_remote_grasp.geometry import rpy_rotation
from x5a_remote_grasp.place_geometry import cube_face_placement_rotations,object_in_tool,placement_tool


def cube(size=(.03,.03,.03)):
    x,y,z=np.meshgrid(*(np.linspace(-s/2,s/2,15) for s in size))
    return np.c_[x.ravel(),y.ravel(),z.ravel()] @ rpy_rotation(0,0,.31).T


def test_cube_has_24_distinct_flat_face_placements_and_keeps_destination_edge_axes():
    points=cube();grasp=np.eye(4);grasp[:3,:3]=rpy_rotation(-1.2,.6,.5)
    grasp[:3,3]=[.025,.01,-.005]
    local=object_in_tool(points,grasp)
    rotations=cube_face_placement_rotations(local,grasp[:3,:3],dict(yaw_rad=.2))
    assert len(rotations)==24
    assert len({tuple(np.round(r.ravel(),6)) for r in rotations})==24
    source=rpy_rotation(0,0,.31);target=rpy_rotation(0,0,.2)
    upward=[]
    for rotation in rotations:
        symmetry=target.T@rotation@grasp[:3,:3].T@source
        np.testing.assert_allclose(symmetry,np.round(symmetry),atol=1e-5)
        assert np.isclose(np.linalg.det(rotation),1.)
        upward.append(symmetry[2,2]>.99)
        tool=placement_tool(local,rotation,[.2,.3,.05],.01)
        placed=local@rotation.T+tool[:3,3]
        low,high=np.percentile(placed,[2,98],axis=0)
        assert np.isclose(low[2],.06)
        np.testing.assert_allclose((low[:2]+high[:2])/2,[.2,.3],atol=1e-9)
    assert sum(upward)==4  # the other 20 valid orientations were previously excluded
    assert abs(rotations[0][2,1]) < abs(grasp[2,1])


def test_flat_square_and_non_cube_shapes_cannot_use_face_equivalence():
    grasp=rpy_rotation(-1.2,.6,.5)
    for size in [(.03,.03,.004),(.03,.03,.05),(.03,.045,.03),(.08,.08,.08)]:
        world=cube(size)
        assert not cube_face_placement_rotations(world@grasp,grasp)
