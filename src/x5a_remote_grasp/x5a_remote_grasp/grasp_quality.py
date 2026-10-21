"""Point-cloud grasp checks shared by the cloud service and robot planner.

Coordinates follow GraspNet: X approach, Y closing, Z finger height.
No ROS, GPU, or network dependencies. Distances are metres.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GripperGeometry:
    tip_offset: float = 0.015  # tool0 -> physical fingertip along +X
    finger_length: float = 0.10
    finger_thickness: float = 0.01
    finger_height: float = 0.02
    pad_length: float = 0.03
    max_width: float = 0.088
    opening_margin: float = 0.006
    noise_margin: float = 0.002
    min_contact_points: int = 12
    min_collision_voxels: int = 3
    cad_model: object = None

    def __post_init__(self):
        values = [self.tip_offset, self.finger_length, self.finger_thickness,
                  self.finger_height, self.pad_length, self.max_width,
                  self.opening_margin, self.noise_margin]
        if not np.all(np.isfinite(values)) or min(values[1:]) <= 0:
            raise ValueError('invalid gripper geometry')
        if self.pad_length > self.finger_length or self.opening_margin >= self.max_width:
            raise ValueError('pad/opening geometry is inconsistent')
        if (2*self.noise_margin >= min(self.finger_thickness, self.finger_height)
                or min(self.min_contact_points, self.min_collision_voxels) < 1):
            raise ValueError('noise/count limits would disable collision checks')


def finite_cloud(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError('point cloud must be finite Nx3')
    return points


def voxel_cloud(points, size=0.003):
    points = finite_cloud(points)
    if not len(points):
        return points
    cells = np.floor(points / size).astype(np.int64)
    low, high = cells.min(axis=0), cells.max(axis=0)
    shape = [int(b)-int(a)+1 for a, b in zip(low, high)]
    if math.prod(shape) <= np.iinfo(np.int64).max:
        relative = cells-low
        keys = (relative[:, 0]*shape[1]+relative[:, 1])*shape[2]+relative[:, 2]
        _, indices = np.unique(keys, return_index=True)
    else:
        _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


@dataclass(frozen=True)
class PreparedGraspScene:
    """Immutable, once-voxelized scene shared by checks in ONE captured frame.

    Constructing a fresh instance performs the same 3 mm reduction used by
    assess_tool_grasp. It changes neither obstacle margins nor point counts.
    """
    points: np.ndarray

    def __post_init__(self):
        points = np.array(voxel_cloud(self.points, 0.003), copy=True)
        points.setflags(write=False)
        object.__setattr__(self, 'points', points)


def depth_cloud(depth_m, fx, fy, cx, cy):
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or not np.all(np.isfinite([fx, fy, cx, cy])) or min(fx, fy) <= 0:
        raise ValueError('invalid RGB-D intrinsics/depth')
    y, x = np.indices(depth.shape)
    return np.stack([(x-cx)*depth/fx, (y-cy)*depth/fy, depth], axis=-1)


def depth_points(depth_m, mask, fx, fy, cx, cy):
    """Project only selected pixels, preserving row-major order and raw depth."""
    depth, mask = np.asarray(depth_m, float), np.asarray(mask, bool)
    if (depth.ndim != 2 or mask.shape != depth.shape
            or not np.all(np.isfinite([fx, fy, cx, cy])) or min(fx, fy) <= 0):
        raise ValueError('invalid RGB-D intrinsics/depth/mask')
    y, x = np.nonzero(mask)
    z = depth[mask]
    return np.column_stack(((x-cx)*z/fx, (y-cy)*z/fy, z))


def height_statistics(points, transform):
    points = finite_cloud(points)
    transform = np.asarray(transform, dtype=float)
    if len(points) < 30 or transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError('insufficient target geometry or invalid transform')
    base = points @ transform[:3, :3].T + transform[:3, 3]
    low, high = np.percentile(base, [5, 95], axis=0)
    center = (low + high) / 2
    return base, center, float(low[2]), float(high[2])


def rotation_distance(a, b):
    """SO(3) distance with the parallel-jaw 180-degree X symmetry."""
    a, b = np.asarray(a), np.asarray(b)
    traces = [np.trace(a.T @ b), np.trace(a.T @ b @ np.diag([1, -1, -1]))]
    return float(np.arccos(np.clip((max(traces)-1)/2, -1, 1)))


def nms_indices(rows, top_k, translation_threshold=0.008, angle_threshold=np.pi/6):
    rows = np.asarray(rows)
    keep = []
    for index in np.argsort(-rows[:, 0], kind='stable'):
        r = rows[index, 4:13].reshape(3, 3)
        t = rows[index, 13:16]
        if any(np.linalg.norm(t-rows[j, 13:16]) < translation_threshold
               and rotation_distance(r, rows[j, 4:13].reshape(3, 3)) < angle_threshold
               for j in keep):
            continue
        keep.append(int(index))
        if len(keep) >= top_k:
            break
    return keep


def opening_measurement(points, rotation, translation, tip_x, length, height, min_points=12):
    local = (finite_cloud(points)-translation) @ rotation
    contact = local[(local[:, 0] >= tip_x-length) & (local[:, 0] <= tip_x+0.002)
                    & (np.abs(local[:, 2]) <= height/2+0.002)]
    if len(contact) < min_points:
        return None
    lo, hi = np.percentile(contact[:, 1], [2, 98])
    return {'required_opening_m': float(2*max(abs(lo), abs(hi))),
            'span_m': float(hi-lo), 'center_offset_m': float((lo+hi)/2),
            'contact_points': int(len(contact))}


def assess_tool_grasp(target, scene, tool, geometry, approach_distance=0.04):
    """Check target engagement and actual OPEN gripper/approach swept volume.

    Two independent clouds: target measures engagement, scene checks obstacles.
    This validates a geometric grasp, not friction/force closure or physical
    grasp success. Unobserved surfaces remain a sensor limitation.
    """
    tool = np.asarray(tool, dtype=float)
    if tool.shape != (4, 4) or not np.all(np.isfinite(tool)):
        raise ValueError('invalid tool transform')
    if geometry.cad_model is not None:
        return geometry.cad_model.assess(target, scene, tool, geometry, approach_distance)
    r, t = tool[:3, :3], tool[:3, 3]
    measure = opening_measurement(target, r, t, geometry.tip_offset,
                                  geometry.pad_length, geometry.finger_height,
                                  geometry.min_contact_points)
    if measure is None:
        return {'valid': False, 'reason': 'empty_or_sparse_contact'}
    if measure['span_m'] < 0.003:
        return dict(measure, valid=False, reason='insufficient_closing_span')
    body = opening_measurement(target, r, t, geometry.tip_offset,
                               geometry.finger_length, geometry.finger_height,
                               geometry.min_contact_points)
    if body is not None:
        measure['required_opening_m'] = max(measure['required_opening_m'], body['required_opening_m'])
    if measure['required_opening_m'] + geometry.opening_margin > geometry.max_width:
        return dict(measure, valid=False, reason='opening_or_offcenter')
    # Quantize before counting so multiple depth pixels do not magnify noise.
    scene_points = scene.points if isinstance(scene, PreparedGraspScene) else voxel_cloud(scene, 0.003)
    local = (scene_points-t) @ r
    n = geometry.noise_margin
    half = geometry.max_width/2
    tip = geometry.tip_offset
    back = tip-geometry.finger_length
    height = np.abs(local[:, 2]) < geometry.finger_height/2-n
    fingers = ((local[:, 0] > back-approach_distance+n)
               & (local[:, 0] < tip-n)
               & (np.abs(local[:, 1]) > half+n)
               & (np.abs(local[:, 1]) < half+geometry.finger_thickness-n))
    palm = ((local[:, 0] > back-geometry.finger_thickness-approach_distance+n)
            & (local[:, 0] < back-n)
            & (np.abs(local[:, 1]) < half+geometry.finger_thickness-n))
    collisions = int(np.count_nonzero(height & (fingers | palm)))
    # At the final pose, the fingers close from their full opening toward the
    # measured target width. A neighbour inside the open jaws can block closing
    # without touching the OPEN fingers, and must not masquerade as a grasp.
    closing = ((local[:, 0] > back+n) & (local[:, 0] < tip-n)
               & (np.abs(local[:, 1]) > measure['required_opening_m']/2+n)
               & (np.abs(local[:, 1]) < half+geometry.finger_thickness-n))
    closing_collisions = int(np.count_nonzero(height & closing))
    reason = ('scene_collision' if collisions >= geometry.min_collision_voxels else
              'closing_collision' if closing_collisions >= geometry.min_collision_voxels else 'ok')
    return dict(measure, valid=reason == 'ok', reason=reason,
                collision_voxels=collisions,
                closing_collision_voxels=closing_collisions,
                open_width_m=min(geometry.max_width, measure['required_opening_m']+geometry.opening_margin))


def recenter_tool(target, tool, geometry, max_shift=0.015):
    tool = np.array(tool, copy=True)
    measure = (geometry.cad_model.contact_measurement(target, tool, geometry.min_contact_points)
               if geometry.cad_model is not None else
               opening_measurement(target, tool[:3, :3], tool[:3, 3],
                   geometry.tip_offset, geometry.pad_length,
                   geometry.finger_height, geometry.min_contact_points))
    if measure is not None:
        shift = float(measure['center_offset_m'])
        if abs(shift) <= max_shift:
            tool[:3, 3] += tool[:3, 1]*shift
    return tool


def graspnet_tool_transform(transform, depth, mapping):
    """Mapping translation is tip calibration, separate from network insertion.

    GraspNet fingertip = t + depth*R[:,0]. For the current X5A mapping,
    mapping[0,3] is the negative insertion reference. Stock wrist CAD
    keeps its measured physical tip separate from the successful native insertion
    reference; contact and collision always use the actual CAD model.
    """
    if not np.isfinite(depth) or not 0 < depth <= 0.10:
        raise ValueError('invalid GraspNet insertion depth')
    offset = np.array(mapping, copy=True)
    offset[0, 3] += depth
    return np.asarray(transform) @ offset


def held_from_gap(gap, fresh, min_gap=0.004):
    if not fresh or gap is None or not np.isfinite(gap):
        return 'unknown'
    return 'empty' if gap < min_gap else 'held'


def trajectory_fingerprint(trajectories):
    import hashlib
    import json
    values = []
    for trajectory in trajectories:
        joint = trajectory.joint_trajectory
        values.append([list(joint.joint_names), [[list(p.positions), p.time_from_start.sec,
                       p.time_from_start.nanosec] for p in joint.points]])
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
