"""Geometry validation and frame conversions independent of ROS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np


ROTATION_TOLERANCE = 2.0e-3


@dataclass(frozen=True)
class RemoteGrasp:
    index: int
    score: float
    width_m: float
    height_m: float
    depth_m: float
    transform_camera_grasp: np.ndarray
    # Centered opening measured from the original target cloud. Robot-side
    # geometry is checked again after depth mapping and any pose modification.
    required_opening_m: Optional[float] = None


def validate_rotation_matrix(rotation: np.ndarray, tolerance: float = ROTATION_TOLERANCE) -> None:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix must be finite 3x3")
    error = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(matrix))
    if error > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError(
            f"invalid rotation matrix (orthogonality_error={error:.3g}, det={determinant:.6g})"
        )


def validate_homogeneous(transform: np.ndarray) -> None:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("grasp transform must be finite 4x4")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6):
        raise ValueError("grasp transform has invalid homogeneous bottom row")
    validate_rotation_matrix(matrix[:3, :3])


def parse_api_grasps(payload: Mapping[str, object]) -> List[RemoteGrasp]:
    convention = payload.get("convention")
    if not isinstance(convention, dict):
        raise ValueError("response convention is missing")
    expected = {"x": "right", "y": "down", "z": "forward", "units": "metres"}
    for key, value in expected.items():
        if str(convention.get(key, "")).lower() != value:
            raise ValueError(f"unsupported response convention {key}={convention.get(key)!r}")
    raw_grasps = payload.get("grasps")
    if not isinstance(raw_grasps, list):
        raise ValueError("response grasps must be a list")
    output: List[RemoteGrasp] = []
    for index, raw in enumerate(raw_grasps):
        if not isinstance(raw, dict):
            raise ValueError(f"grasp {index} is not an object")
        transform = np.asarray(raw.get("transform_camera_grasp"), dtype=float)
        validate_homogeneous(transform)
        translation = np.asarray(raw.get("translation_m"), dtype=float)
        rotation = np.asarray(raw.get("rotation_matrix"), dtype=float)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError(f"grasp {index} translation_m must be finite length 3")
        validate_rotation_matrix(rotation)
        if not np.allclose(transform[:3, 3], translation, atol=1.0e-6):
            raise ValueError(f"grasp {index} translation disagrees with transform")
        if not np.allclose(transform[:3, :3], rotation, atol=1.0e-6):
            raise ValueError(f"grasp {index} rotation disagrees with transform")
        score = _finite_float(raw, "score", index)
        width = _finite_float(raw, "width_m", index)
        height = _finite_float(raw, "height_m", index)
        depth = _finite_float(raw, "depth_m", index)
        if width < 0.0 or height < 0.0 or depth < 0.0:
            raise ValueError(f"grasp {index} has a negative dimension")
        required_raw = raw.get("required_opening_m")
        try:
            required = float(required_raw) if required_raw is not None else None
        except (TypeError, ValueError):
            required = None
        if required is not None and (not np.isfinite(required) or required < 0.0):
            required = None
        output.append(
            RemoteGrasp(
                index=index,
                score=score,
                width_m=width,
                height_m=height,
                depth_m=depth,
                transform_camera_grasp=transform,
                required_opening_m=required,
            )
        )
    return output


def _finite_float(raw: Mapping[str, object], key: str, index: int) -> float:
    try:
        value = float(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"grasp {index} {key} is invalid") from exc
    if not np.isfinite(value):
        raise ValueError(f"grasp {index} {key} is not finite")
    return value


def transform_from_xyz_quat(xyz, quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("TF quaternion has zero or invalid norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    validate_homogeneous(transform)
    return transform


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    validate_rotation_matrix(matrix)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quat = np.asarray([x, y, z, w], dtype=float)
    quat /= np.linalg.norm(quat)
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def make_grasp_tool_transform(approach_offset_m: float = 0.0) -> np.ndarray:
    """tool0 in the GraspNet grasp frame.

    X5A URDF: tool0 is identity-rotated vs link6, +X toward the fingers,
    +Y along the jaw opening. That matches GraspNet (+X approach, +Y
    between jaws). A positive offset pulls tool0 back along -X (toward
    the palm) so the fingers wrap deeper than the predicted center.
    """
    matrix = np.eye(4)
    matrix[0, 3] = -float(approach_offset_m)
    validate_homogeneous(matrix)
    return matrix


def parallel_jaw_rotations(rotation: np.ndarray) -> List[np.ndarray]:
    """Equivalent parallel-jaw orientations: original and 180 deg about +X.

    Column 0 (approach) is unchanged; the two fingers swap sides.
    """
    matrix = np.asarray(rotation, dtype=float)
    validate_rotation_matrix(matrix)
    flipped = matrix @ np.diag([1.0, -1.0, -1.0])
    validate_rotation_matrix(flipped)
    return [matrix, flipped]


def approach_from_below(
    transform_base_grasp: np.ndarray, max_approach_z: float = 0.35
) -> bool:
    """True when GraspNet +X points mostly upward, i.e. the gripper is under the object."""
    validate_homogeneous(transform_base_grasp)
    return float(transform_base_grasp[2, 0]) > float(max_approach_z)


def is_steep_topdown(
    transform_base_grasp: np.ndarray, min_approach_z: float = -0.65
) -> bool:
    """True when GraspNet +X points mostly downward (top-down grasp)."""
    validate_homogeneous(transform_base_grasp)
    return float(transform_base_grasp[2, 0]) < float(min_approach_z)


def rpy_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS fixed-axis RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = float(np.cos(roll)), float(np.sin(roll))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    matrix = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )
    validate_rotation_matrix(matrix)
    return matrix


def approach_yaw(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation, dtype=float)
    validate_rotation_matrix(matrix)
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def capped_pregrasp_distance(
    tool_z: float,
    approach_z: float,
    requested: float,
    max_pregrasp_z: float = 0.11,
    min_distance: float = 0.03,
) -> float:
    """Keep a top-down pregrasp below X5A's steep-tool IK ceiling (~0.13 m)."""
    requested = max(float(min_distance), float(requested))
    if float(approach_z) >= -1.0e-3:
        return requested
    room = float(max_pregrasp_z) - float(tool_z)
    if room <= 0.0:
        return float(min_distance)
    return min(requested, max(float(min_distance), room / (-float(approach_z))))


def candidate_waypoints(
    transform_base_grasp: np.ndarray,
    transform_grasp_tool: np.ndarray,
    pregrasp_distance: float,
    lift_distance: float,
    retreat_distance: float,
    transit_rotation: np.ndarray | None = None,
    max_pregrasp_z: float = 0.11,
    insertion_depth: float = 0.0,
) -> Dict[str, np.ndarray]:
    """GraspNet waypoints: approach along +X, lift along world +Z, retreat along -X.

    GRASP keeps the predicted orientation. PREGRASP stays on the approach axis
    but is shortened so a steep tool does not rise above ``max_pregrasp_z``.
    LIFT/RETREAT may use ``transit_rotation`` (shallower pitch) because this
    arm cannot hold a near-vertical tool above ~0.13 m.
    """
    validate_homogeneous(transform_base_grasp)
    validate_homogeneous(transform_grasp_tool)
    mapping = np.array(transform_grasp_tool, copy=True)
    mapping[0, 3] += float(insertion_depth)
    tool = transform_base_grasp @ mapping
    validate_homogeneous(tool)
    approach = transform_base_grasp[:3, 0]
    distance = capped_pregrasp_distance(
        float(tool[2, 3]), float(approach[2]), pregrasp_distance, max_pregrasp_z
    )
    pregrasp = tool.copy()
    pregrasp[:3, 3] -= approach * distance
    if transit_rotation is None:
        lift = tool.copy()
        lift[2, 3] += float(lift_distance)
    else:
        rotation = np.asarray(transit_rotation, dtype=float)
        validate_rotation_matrix(rotation)
        # Lift and carry may use the shallower tool.
        # Keep the final approach on the validated grasp axis. Rotating while
        # entering the object invalidates the checked finger swept volume.
        lift = np.eye(4)
        lift[:3, :3] = rotation
        lift[:3, 3] = tool[:3, 3] + np.array([0.0, 0.0, float(lift_distance)])
        validate_homogeneous(lift)
        validate_homogeneous(pregrasp)
    retreat = lift.copy()
    retreat[:3, 3] -= lift[:3, 0] * float(retreat_distance)
    return {"pregrasp": pregrasp, "grasp": tool, "lift": lift, "retreat": retreat}
