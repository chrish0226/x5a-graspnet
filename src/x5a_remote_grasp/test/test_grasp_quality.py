"""Regressions for observed missed-grasp geometry, without ROS/hardware."""
import numpy as np
import pytest
from x5a_remote_grasp.geometry import rpy_rotation
from x5a_remote_grasp.grasp_quality import (
    GripperGeometry, assess_tool_grasp, graspnet_tool_transform,
    height_statistics, held_from_gap, nms_indices, opening_measurement,
    recenter_tool,
)


def contact_cloud(x_span=(-0.01, 0.01), y_span=(-0.02, 0.02)):
    x, y, z = np.meshgrid(np.linspace(*x_span, 8),
                          np.linspace(*y_span, 15), np.linspace(-0.004, 0.004, 3))
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def test_nms_preserves_different_closing_axis_at_same_position():
    rows = np.zeros((3, 17))
    rows[:, 0] = [0.9, 0.8, 0.7]
    rows[0, 4:13] = np.eye(3).ravel()
    rows[1, 4:13] = rpy_rotation(np.pi/2, 0, 0).ravel()
    rows[2, 4:13] = np.diag([1, -1, -1]).ravel()
    assert nms_indices(rows, 10) == [0, 1]


@pytest.mark.parametrize('depth', [0.01, 0.02, 0.03, 0.04])
def test_tool_tip_matches_graspnet_insertion_depth(depth):
    grasp = np.eye(4)
    grasp[:3, :3] = rpy_rotation(0.3, 1.1, 0.7)
    grasp[:3, 3] = [0.2, 0.3, 0.1]
    mapping = np.eye(4)
    mapping[0, 3] = -0.015
    tool = graspnet_tool_transform(grasp, depth, mapping)
    physical_tip = tool[:3, 3]+0.015*tool[:3, 0]
    np.testing.assert_allclose(physical_tip, grasp[:3, 3]+depth*grasp[:3, 0])


def test_narrow_but_offcenter_target_is_rejected():
    target = contact_cloud(y_span=(0.05, 0.07))
    measured = opening_measurement(target, np.eye(3), np.zeros(3), 0.015, 0.03, 0.02)
    assert measured['span_m'] < 0.03
    assert measured['required_opening_m'] > 0.13
    assert assess_tool_grasp(target, target, np.eye(4), GripperGeometry())['reason'] == 'opening_or_offcenter'


def test_recenter_is_limited_and_rechecked():
    target = contact_cloud(y_span=(-0.01, 0.03))
    centered = recenter_tool(target, np.eye(4), GripperGeometry(), max_shift=0.015)
    assert centered[1, 3] == pytest.approx(0.01)
    assert assess_tool_grasp(target, target, centered, GripperGeometry())['valid']
    np.testing.assert_array_equal(recenter_tool(target, np.eye(4), GripperGeometry(), 0.005), np.eye(4))


def test_collision_outside_target_in_approach_sweep_is_rejected():
    target = contact_cloud()
    obstacles = contact_cloud(x_span=(-0.125, -0.115), y_span=(-0.025, 0.025))
    assert assess_tool_grasp(target, target, np.eye(4), GripperGeometry())['valid']
    checked = assess_tool_grasp(target, np.vstack([target, obstacles]), np.eye(4), GripperGeometry())
    assert checked['reason'] == 'scene_collision'
    assert checked['collision_voxels'] >= 3


def test_rotating_pose_requires_new_aperture_check():
    target = contact_cloud(y_span=(-0.02, 0.02))
    target[:, 2] *= 15  # long dimension along old Z, narrow along old Y
    original = assess_tool_grasp(target, target, np.eye(4), GripperGeometry())
    rotated = np.eye(4)
    rotated[:3, :3] = rpy_rotation(np.pi/2, 0, 0)
    assert original['valid']
    assert not assess_tool_grasp(target, target, rotated, GripperGeometry())['valid']


def test_neighbour_between_open_jaws_blocks_closing():
    target = contact_cloud()
    neighbour = contact_cloud(y_span=(0.03, 0.038))
    checked = assess_tool_grasp(target, np.vstack([target, neighbour]), np.eye(4), GripperGeometry())
    assert checked['collision_voxels'] == 0
    assert not checked['valid'] and checked['reason'] == 'closing_collision'


def test_invalid_noise_margin_cannot_silently_disable_collisions():
    with pytest.raises(ValueError):
        GripperGeometry(noise_margin=0.01)


def test_opening_clears_target_along_full_finger_body():
    target = np.vstack([contact_cloud(y_span=(-0.015, 0.015)),
                       contact_cloud(x_span=(-0.07, -0.05), y_span=(-0.03, 0.03))])
    checked = assess_tool_grasp(target, target, np.eye(4), GripperGeometry())
    assert checked['valid']
    assert checked['required_opening_m'] > 0.05


def test_full_cloud_height_preserves_plane_under_tilted_camera():
    x, y = np.meshgrid(np.linspace(0.1, 0.3, 40), np.linspace(0.2, 0.4, 40))
    base = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.035)])
    transform = np.eye(4)
    transform[:3, :3] = rpy_rotation(0.3, 1.2, 0.2)
    transform[:3, 3] = [0.4, 0.2, 0.5]
    camera = (base-transform[:3, 3])@transform[:3, :3]
    _, _, lo, hi = height_statistics(camera, transform)
    assert hi-lo == pytest.approx(0, abs=1e-12)


def test_empty_close_and_stale_feedback_do_not_claim_holding():
    assert held_from_gap(0.0, True) == 'empty'
    assert held_from_gap(0.025, True) == 'held'
    assert held_from_gap(0.025, False) == 'unknown'
    assert held_from_gap(None, True) == 'unknown'
