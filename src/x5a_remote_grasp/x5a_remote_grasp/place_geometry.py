"""Placement geometry in metres; no ROS or controller dependencies.

The observed object cloud is attached to tool0 at the measured grasp pose.
It is transformed again for every release orientation, including its bottom.
"""
import itertools
import numpy as np
from .grasp_quality import finite_cloud


def object_in_tool(points_base, base_tool):
    points = finite_cloud(points_base)
    tool = np.asarray(base_tool, dtype=float)
    if len(points) < 30 or tool.shape != (4, 4) or not np.all(np.isfinite(tool)):
        raise ValueError('缺少已夹持物体的点云或实际抓取位姿，请重新抓取')
    return (points-tool[:3, 3]) @ tool[:3, :3]


def placement_tool(points_tool, rotation, surface_point, clearance=0.002):
    points = finite_cloud(points_tool)
    rotation = np.asarray(rotation, dtype=float)
    surface = np.asarray(surface_point, dtype=float)
    if len(points) < 30 or rotation.shape != (3, 3) or surface.shape != (3,):
        raise ValueError('invalid placement geometry')
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(surface)) or clearance < 0:
        raise ValueError('invalid placement pose/clearance')
    low, high = np.percentile(points @ rotation.T, [2, 98], axis=0)
    tool = np.eye(4)
    tool[:3, :3] = rotation
    tool[:2, 3] = surface[:2]-(low[:2]+high[:2])/2
    tool[2, 3] = surface[2]+clearance-low[2]
    return tool


def _hull(points):
    points = sorted(set(map(tuple, points)))
    def half(seq):
        result = []
        for p in seq:
            while len(result) >= 2:
                a, b = np.asarray(result[-2]), np.asarray(result[-1])
                v, w = b-a, np.asarray(p)-b
                if v[0]*w[1]-v[1]*w[0] > 0:
                    break
                result.pop()
            result.append(p)
        return result
    return np.asarray(half(points)[:-1]+half(points[::-1])[:-1])


def support_surface(points_base):
    points = finite_cloud(points_base)
    if len(points) < 30:
        raise ValueError('放置支撑面点云不足')
    top = np.percentile(points[:, 2], 95)
    face = points[points[:, 2] >= top-0.004]
    lo, hi = np.percentile(face[:, :2], [1, 99], axis=0)
    face = face[np.all((face[:, :2] >= lo) & (face[:, :2] <= hi), axis=1)]
    hull = _hull(face[:, :2])
    if len(hull) < 3:
        raise ValueError('无法确定放置支撑面的有效边界')
    # Convex polygon centroid, rather than a side-face-biased cloud median.
    nxt = np.roll(hull, -1, axis=0)
    cross = hull[:, 0]*nxt[:, 1]-nxt[:, 0]*hull[:, 1]
    if cross.sum() <= 1e-8:
        raise ValueError('放置支撑面过小')
    center = ((hull+nxt)*cross[:, None]).sum(axis=0)/(3*cross.sum())
    return np.r_[center, np.median(face[:, 2])], hull


def cube_face_axes(points_base):
    """Estimate square top-face edges modulo 90 degrees. PCA is unstable for
    squares, so fit a minimum-area rectangle to the observed top-face hull.
    Return None when the face is too small, elongated, or poorly observed.
    """
    points = finite_cloud(points_base)
    if len(points) < 40:
        return None
    face = points[points[:, 2] >= np.percentile(points[:, 2], 95)-.004]
    if len(face) < 30:
        return None
    lo, hi = np.percentile(face[:, :2], [1, 99], axis=0)
    # Do not clip genuine rotated corners to axis-aligned percentile bounds:
    # that invents horizontal edges when a cube is almost axis-aligned.
    xy = face[np.all((face[:, :2] >= lo-.001) & (face[:, :2] <= hi+.001), axis=1), :2]
    hull = _hull(xy)
    if len(hull) < 4:
        return None
    edge = np.roll(hull, -1, axis=0)-hull
    angles = np.unique(np.round(np.arctan2(edge[:, 1], edge[:, 0]) % (np.pi/2), 6))
    best = None
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        local = hull @ np.array([[c, -s], [s, c]])
        size = np.ptp(local, axis=0)
        area = float(np.prod(size))
        if best is None or area < best[0]:
            best = (area, angle, size, (local.min(axis=0)+local.max(axis=0))/2)
    area, angle, size, local_center = best
    hull_area = abs(float(np.sum(hull[:, 0]*np.roll(hull[:, 1], -1)
                                 - hull[:, 1]*np.roll(hull[:, 0], -1))))/2
    coverage = hull_area/max(area, 1e-12)
    if np.min(size) < .012 or np.max(size) > .12 or np.max(size)/np.min(size) > 1.35 or coverage < .86:
        return None
    c, s = np.cos(angle), np.sin(angle)
    center = local_center @ np.array([[c, s], [-s, c]])
    return dict(yaw_rad=float(angle), size_xy_m=size.tolist(), center_xy_m=center.tolist(),
                coverage=float(coverage), points=len(xy))


def cube_placement_cloud(points_base, table_top=-.006):
    """Complete a recognized cube's hidden bottom from its observed square face.

    Depth often sees only its top. Treating that top as the object's bottom
    lowers the release by a full cube height. This prior is placement-only.
    """
    points = finite_cloud(points_base)
    axes = cube_face_axes(points)
    if axes is None:
        return points
    size = np.asarray(axes['size_xy_m'])
    # Restrict the cube prior to the project's small blocks, not general boxes.
    if np.min(size) < .020 or np.max(size) > .055:
        return points
    height = float(np.mean(size))
    top = float(np.percentile(points[:, 2], 95))
    bottom = max(float(table_top), top-height)
    if top-bottom < height*.75:
        return points  # geometry inconsistent with a cube on this table
    a = np.linspace(-.5, .5, 21)
    x, y = np.meshgrid(a*size[0], a*size[1])
    c, s = np.cos(axes['yaw_rad']), np.sin(axes['yaw_rad'])
    xy = np.c_[x.ravel(), y.ravel()] @ np.array([[c, s], [-s, c]])+axes['center_xy_m']
    # Dense bottom samples survive robust 2nd-percentile placement bounds.
    hidden_bottom = np.c_[xy, np.full(len(xy), bottom)]
    return np.vstack([points, hidden_bottom])


def cube_alignment_rotations(points_tool, grasp_rotation, support_axes):
    source = cube_face_axes(np.asarray(points_tool) @ np.asarray(grasp_rotation).T)
    if source is None or support_axes is None:
        return []
    quarter = np.pi/2
    delta = (support_axes['yaw_rad']-source['yaw_rad']+quarter/2) % quarter-quarter/2
    rotations = []
    for angle in (delta, delta+quarter, delta-quarter, delta+np.pi):
        c, s = np.cos(angle), np.sin(angle)
        rotations.append(np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ grasp_rotation)
    return rotations


def cube_face_placement_rotations(points_tool, grasp_rotation, support_axes=None):
    """All 24 face-equivalent placements of a small, fully modeled cube.

    A cube may change which face points upward without losing a flat support
    face or aligned edges. General objects keep the original upright policy.
    """
    grasp = np.asarray(grasp_rotation, dtype=float)
    world = finite_cloud(points_tool) @ grasp.T
    axes = cube_face_axes(world)
    if axes is None:
        return []
    size = np.asarray(axes['size_xy_m'])
    height = float(np.ptp(np.percentile(world[:, 2], [2, 98])))
    if (np.min(size) < .020 or np.max(size) > .055
            or np.max(size)/np.min(size) > 1.15
            or not .80 <= height/np.mean(size) <= 1.20):
        return []
    def rz(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    source = rz(axes['yaw_rad'])
    target = rz(axes['yaw_rad'] if support_axes is None else support_axes['yaw_rad'])
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            symmetry = np.eye(3)[:, permutation]*np.asarray(signs)
            if np.linalg.det(symmetry) < .5:
                continue
            rotation = target @ symmetry @ source.T @ grasp
            rotations.append(rotation)
    # Prefer a horizontal jaw closing axis: one downward-pointing finger was
    # the limiting collision in the observed failures. Full paths still decide.
    rotations.sort(key=lambda rotation: (
        abs(float(rotation[2, 1])),
        abs(float(np.arctan2(-rotation[2, 0], np.linalg.norm(rotation[:2, 0])))-.6),
        float(np.arccos(np.clip((np.trace(grasp.T@rotation)-1)/2, -1., 1.)))))
    return rotations


def cube_alignment_report(points_tool, place_tool, support_axes):
    if support_axes is None:
        return None
    placed = cube_face_axes(np.asarray(points_tool) @ place_tool[:3, :3].T)
    if placed is None:
        return None
    quarter = np.pi/2
    error = (placed['yaw_rad']-support_axes['yaw_rad']+quarter/2) % quarter-quarter/2
    return dict(error_deg=float(abs(np.degrees(error))), support=support_axes, placed=placed)


def supported_center(point_xy, hull, margin=0.004):
    hull = np.asarray(hull)
    edges = np.roll(hull, -1, axis=0)-hull
    offset = np.asarray(point_xy)-hull
    lengths = np.linalg.norm(edges, axis=1)
    signed = (edges[:, 0]*offset[:, 1]-edges[:, 1]*offset[:, 0])/lengths
    return bool(np.all(signed >= margin))


def preserves_upright(grasp_rotation, place_rotation, tolerance=0.08):
    relative = np.asarray(place_rotation) @ np.asarray(grasp_rotation).T
    return bool(relative[2, 2] >= np.cos(tolerance))


def support_prism(hull, z_low, z_high, margin=0.002):
    """Closed, outward-oriented mesh of the support, inflated laterally."""
    hull = np.asarray(hull, dtype=float)
    if hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 3 or z_high <= z_low:
        raise ValueError('invalid support collision prism')
    center = hull.mean(axis=0)
    edges = np.roll(hull, -1, axis=0)-hull
    inward = np.c_[-edges[:, 1], edges[:, 0]]/np.linalg.norm(edges, axis=1)[:, None]
    radius = np.min(np.sum((center-hull)*inward, axis=1))
    if radius <= 0:
        raise ValueError('support hull must be convex and counterclockwise')
    hull = center+(hull-center)*(1+margin/radius)
    count = len(hull)
    vertices = np.vstack([np.c_[hull, np.full(count, z_low)],
                          np.c_[hull, np.full(count, z_high)]])
    triangles = []
    for i in range(1, count-1):
        triangles.extend([(0, i+1, i), (count, count+i, count+i+1)])
    for i in range(count):
        j = (i+1) % count
        triangles.extend([(i, j, count+j), (i, count+j, count+i)])
    return vertices, triangles
