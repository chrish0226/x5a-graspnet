"""Placement geometry in metres; no ROS or controller dependencies.

The observed object cloud is attached to tool0 at the measured grasp pose.
It is transformed again for every release orientation, including its bottom.
"""
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
