import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from x5a_pick_place.stacking import stack_geometry, tcp_for_object_center, aligned_stack_rpy


def test_stack_height_is_destination_top_plus_held_half_height_and_small_gap():
    center = stack_geometry((.18,.33,.024), (.30,.30,.024), (.03,.03,.03), -.006)
    np.testing.assert_allclose(center, [.30,.30,.041])
    assert center[2]-.015 == pytest.approx(.024+.002)


@pytest.mark.parametrize('rpy', [[0.,1.1,0.], [.1,.9,.8], [0.,1.1,2.]])
def test_release_compensates_measured_held_offset_after_orientation_changes(rpy):
    offset = np.array([.01,-.004,-.028])
    center = np.array([.30,.30,.041])
    tcp = tcp_for_object_center(center, offset, rpy)
    np.testing.assert_allclose(tcp+Rotation.from_euler('xyz',rpy).apply(offset),center)


def test_stack_orientation_preserves_pitch_and_uses_square_symmetry():
    grasp = [0.,1.1,.2]
    aligned = aligned_stack_rpy(grasp, .02, math.pi/2-.02)
    assert aligned[:2] == grasp[:2]
    assert aligned[2] == pytest.approx(.16)


@pytest.mark.parametrize('source,destination', [
    ((.18,.33,.084),(.30,.30,.024)),
    ((.18,.33,.024),(.30,.30,.054)),
    ((.18,.33,.024),(.19,.33,.024)),
    ((.18,.33,float('nan')),(.30,.30,.024)),
])
def test_towers_overlapping_cubes_and_bad_depth_are_rejected(source,destination):
    with pytest.raises(ValueError):
        stack_geometry(source,destination,(.03,.03,.03),-.006)


def test_exposed_upper_cube_can_be_reused_on_a_standalone_cube():
    center = stack_geometry((.18,.33,.054), (.30,.30,.024), (.03,.03,.03), -.006)
    np.testing.assert_allclose(center, [.30,.30,.041])
