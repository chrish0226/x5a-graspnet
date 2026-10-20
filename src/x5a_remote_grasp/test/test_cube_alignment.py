import numpy as np
import pytest
from x5a_remote_grasp.geometry import rpy_rotation
from x5a_remote_grasp.place_geometry import (
    cube_face_axes, cube_alignment_rotations, cube_alignment_report,
    object_in_tool, placement_tool, preserves_upright,
)


def face(angle, noise=0.):
    x, y = np.meshgrid(np.linspace(-.015, .015, 35), np.linspace(-.015, .015, 35))
    points = np.c_[x.ravel(), y.ravel(), np.full(x.size, .03)] @ rpy_rotation(0, 0, angle).T
    return points + np.random.default_rng(6).normal(0., noise, points.shape)


@pytest.mark.parametrize('angle', [.05, .4, .78, 1.15])
def test_square_edges_are_recovered_even_though_pca_is_ambiguous(angle):
    axes = cube_face_axes(face(angle, .0002))
    assert axes is not None
    error = (axes['yaw_rad']-angle+np.pi/4) % (np.pi/2)-np.pi/4
    assert abs(error) < np.radians(2.)


def test_place_rotates_cube_about_world_vertical_and_preserves_center_and_height():
    source = face(.55)
    support = cube_face_axes(face(-.2))
    grasp = np.eye(4)
    grasp[:3, :3] = rpy_rotation(0, .5, 1.1)
    grasp[:3, 3] = [.02, -.01, .03]
    local = object_in_tool(source, grasp)
    rotations = cube_alignment_rotations(local, grasp[:3, :3], support)
    assert len(rotations) == 4
    for rotation in rotations:
        assert preserves_upright(grasp[:3, :3], rotation)
        tool = placement_tool(local, rotation, [.2, .3, .06], .01)
        report = cube_alignment_report(local, tool, support)
        assert report['error_deg'] < .1
        placed = local @ rotation.T+tool[:3, 3]
        lo, hi = np.percentile(placed, [2, 98], axis=0)
        np.testing.assert_allclose((lo[:2]+hi[:2])/2, [.2, .3], atol=1e-10)
        assert lo[2] == pytest.approx(.07)


def test_round_and_narrow_shapes_do_not_get_an_invented_cube_orientation():
    angle = np.linspace(0, 2*np.pi, 180, endpoint=False)
    circle = np.c_[.015*np.cos(angle), .015*np.sin(angle), np.full(len(angle), .03)]
    assert cube_face_axes(circle) is None
    narrow = face(0.)
    narrow[:, 1] *= .2
    assert cube_face_axes(narrow) is None
