import numpy as np
import pytest

from x5a_remote_grasp.geometry import (
    approach_from_below,
    capped_pregrasp_distance,
    candidate_waypoints,
    is_steep_topdown,
    make_grasp_tool_transform,
    matrix_to_quaternion,
    parallel_jaw_rotations,
    rpy_rotation,
    parse_api_grasps,
    validate_rotation_matrix,
)


def response(rotation=None, translation=None):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    translation = np.array([0.1, -0.2, 0.5]) if translation is None else np.asarray(translation)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return {
        "frame": "camera_optical_frame",
        "convention": {"x": "right", "y": "down", "z": "forward", "units": "metres"},
        "grasps": [{
            "score": 0.9, "width_m": 0.06, "height_m": 0.02, "depth_m": 0.03,
            "translation_m": translation.tolist(),
            "rotation_matrix": rotation.tolist(),
            "transform_camera_grasp": transform.tolist(),
        }],
    }


def test_parse_and_quaternion_are_valid():
    grasp = parse_api_grasps(response())[0]
    np.testing.assert_allclose(grasp.transform_camera_grasp[:3, 3], [0.1, -0.2, 0.5])
    quaternion = matrix_to_quaternion(grasp.transform_camera_grasp[:3, :3])
    np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0])
    assert abs(np.linalg.norm(quaternion) - 1.0) < 1.0e-12


def test_reflection_rotation_is_rejected():
    with pytest.raises(ValueError, match="invalid rotation matrix"):
        validate_rotation_matrix(np.diag([1.0, 1.0, -1.0]))


def test_transform_fields_must_agree():
    payload = response()
    payload["grasps"][0]["translation_m"][0] += 0.01
    with pytest.raises(ValueError, match="translation disagrees"):
        parse_api_grasps(payload)


def test_pregrasp_moves_against_positive_x_approach():
    transform = np.eye(4)
    transform[:3, 3] = [0.25, 0.30, 0.10]
    waypoints = candidate_waypoints(transform, np.eye(4), 0.10, 0.08, 0.08)
    np.testing.assert_allclose(waypoints["grasp"][:3, 3], [0.25, 0.30, 0.10])
    np.testing.assert_allclose(waypoints["pregrasp"][:3, 3], [0.15, 0.30, 0.10])
    np.testing.assert_allclose(waypoints["lift"][:3, 3], [0.25, 0.30, 0.18])
    np.testing.assert_allclose(waypoints["retreat"][:3, 3], [0.17, 0.30, 0.18])


def test_side_grasp_keeps_orientation_and_lifts_world_z():
    transform = np.eye(4)
    # Approach along +Y (side grasp on a bottle).
    transform[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
    )
    transform[:3, 3] = [0.25, 0.30, 0.12]
    waypoints = candidate_waypoints(transform, np.eye(4), 0.10, 0.08, 0.08)
    np.testing.assert_allclose(waypoints["grasp"][:3, 3], [0.25, 0.30, 0.12])
    np.testing.assert_allclose(waypoints["pregrasp"][:3, 3], [0.25, 0.20, 0.12])
    np.testing.assert_allclose(waypoints["lift"][:3, 3], [0.25, 0.30, 0.20])
    np.testing.assert_allclose(waypoints["lift"][:3, :3], transform[:3, :3])
    np.testing.assert_allclose(waypoints["retreat"][:3, 3], [0.25, 0.22, 0.20])


def test_approach_offset_pulls_tool0_toward_palm():
    transform = np.eye(4)
    transform[:3, 3] = [0.25, 0.30, 0.10]
    tool = make_grasp_tool_transform(0.02)
    waypoints = candidate_waypoints(transform, tool, 0.10, 0.08, 0.08)
    np.testing.assert_allclose(waypoints["grasp"][:3, 3], [0.23, 0.30, 0.10])
    np.testing.assert_allclose(waypoints["pregrasp"][:3, 3], [0.13, 0.30, 0.10])


def test_parallel_jaw_flip_preserves_approach():
    rotation = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
    )
    original, flipped = parallel_jaw_rotations(rotation)
    np.testing.assert_allclose(original[:, 0], flipped[:, 0])
    np.testing.assert_allclose(flipped[:, 1], -original[:, 1])
    np.testing.assert_allclose(flipped[:, 2], -original[:, 2])


def test_approach_from_below_detects_upward_x():
    top_down = np.eye(4)
    # +X approach = world -Z (gripper above the object).
    top_down[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float
    )
    assert not approach_from_below(top_down)
    upward = np.eye(4)
    # +X approach = world +Z (gripper under the object).
    upward[:3, :3] = np.array(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=float
    )
    assert approach_from_below(upward)


def test_capped_pregrasp_keeps_steep_tool_under_ik_ceiling():
    # Tape-roll top-down: grasp z=0.062, approach_z=-0.83, requested 0.10
    # would put pregrasp at ~0.145. Cap at 0.11.
    distance = capped_pregrasp_distance(0.062, -0.83, 0.10, max_pregrasp_z=0.11)
    assert 0.03 <= distance < 0.10
    assert 0.062 + 0.83 * distance <= 0.11 + 1.0e-9


def test_steep_lift_uses_shallower_transit_orientation():
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float
    )
    transform[:3, 3] = [0.20, 0.30, 0.06]
    assert is_steep_topdown(transform)
    transit = rpy_rotation(0.0, 1.15, 0.0)
    waypoints = candidate_waypoints(
        transform, np.eye(4), 0.10, 0.08, 0.08, transit_rotation=transit, max_pregrasp_z=0.11
    )
    assert waypoints["pregrasp"][2, 3] <= 0.11 + 1.0e-9
    np.testing.assert_allclose(waypoints["grasp"][:3, :3], transform[:3, :3])
    np.testing.assert_allclose(waypoints["pregrasp"][:3, :3], transform[:3, :3])
    np.testing.assert_allclose(waypoints["lift"][:3, :3], transit)
    np.testing.assert_allclose(waypoints["lift"][:3, 3], [0.20, 0.30, 0.14])
    np.testing.assert_allclose(waypoints['retreat'][:3, 3],
                               waypoints['lift'][:3, 3]-transit[:, 0]*0.08)
