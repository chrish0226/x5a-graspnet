"""Extract a cube's horizontal top from an oblique RGB-D view."""
import cv2
import numpy as np


def rgbd_dimensions_match(color_shape, depth_shape, info_width, info_height):
    return (len(color_shape) == 3 and len(depth_shape) == 2
            and tuple(color_shape[:2]) == tuple(depth_shape)
            and tuple(depth_shape) == (info_height, info_width))


def deproject(xs, ys, z, K):
    K = np.asarray(K, float).reshape(3, 3)
    return np.column_stack(((xs-K[0, 2])*z/K[0, 0], (ys-K[1, 2])*z/K[1, 1], z))


def height_mask(depth, K, T, minimum_z, maximum_z, depth_min, depth_max):
    K, T = np.asarray(K).reshape(3, 3), np.asarray(T)
    # Broadcast one row and column instead of allocating two HD pixel grids.
    xs = np.arange(depth.shape[1])[None, :]
    ys = np.arange(depth.shape[0])[:, None]
    base_z = depth*((xs-K[0, 2])/K[0, 0]*T[2, 0]
                    + (ys-K[1, 2])/K[1, 1]*T[2, 1]+T[2, 2])+T[2, 3]
    return (np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
            & (base_z >= minimum_z) & (base_z <= maximum_z))


def fit_horizontal_top(points, tolerance=.002, max_tilt_deg=25., minimum_support=20):
    """RANSAC rejects vertical sides; return actual inlier points in base frame.

    Raw 3D points retain their own depths; combining a mean pixel with an
    unrelated median depth creates a point that may lie on neither face.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('need Nx3 base-frame points')
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < minimum_support:
        return None
    if len(points) > 3000:
        points = points[np.linspace(0, len(points)-1, 3000, dtype=int)]
    # Start at the high part of the object in base Z. A long visible side can
    # otherwise dominate a tilted slice that contains only the top edge.
    points = points[points[:, 2] >= np.percentile(points[:, 2], 75.)-tolerance]
    if len(points) < minimum_support:
        return None
    minimum_normal_z = np.cos(np.deg2rad(max_tilt_deg))
    rng = np.random.default_rng(0)
    # Preserve all 128 seeded trials and first-wins ties, but evaluate them
    # together. The support, tilt and final fit guards remain unchanged.
    samples = points[np.array([rng.choice(len(points), 3, replace=False) for _ in range(128)])]
    anchors = samples[:, 0]
    normals = np.cross(samples[:, 1]-anchors, samples[:, 2]-anchors)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths >= 1e-10
    normals[valid] /= lengths[valid, None]
    valid &= np.abs(normals[:, 2]) >= minimum_normal_z
    if not np.any(valid):
        return None
    normals, anchors = normals[valid], anchors[valid]
    inliers = np.abs(points @ normals.T - np.sum(anchors*normals, axis=1)) <= tolerance
    counts = np.count_nonzero(inliers, axis=0)
    winner = int(np.argmax(counts))
    best, best_count = inliers[:, winner], counts[winner]
    if best_count < minimum_support:
        return None
    selected = points[best]
    center = selected.mean(axis=0)
    _, _, vt = np.linalg.svd(selected-center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    if normal[2] < minimum_normal_z:
        return None
    selected = points[np.abs((points-center) @ normal) <= tolerance]
    if len(selected) < minimum_support:
        return None
    # Exclude narrow stripe fits and large background planes.
    _, singular, _ = np.linalg.svd(selected-selected.mean(axis=0), full_matrices=False)
    if singular[1]/np.sqrt(len(selected)) < .002 or np.max(np.ptp(selected[:, :2], axis=0)) > .065:
        return None
    return selected


def plane_rectangle_center(points):
    """A face center should not move with unequal pixel density on its edges."""
    points = np.asarray(points, float)
    low, high = np.percentile(points[:, :2], [1., 99.], axis=0)
    inner = points[np.all((points[:, :2] >= low) & (points[:, :2] <= high), axis=1)]
    rect = cv2.minAreaRect(inner[:, :2].astype(np.float32))
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points-mean, full_matrices=False)
    normal = vt[-1]
    center = np.array([rect[0][0], rect[0][1], mean[2]])
    center[2] -= np.dot(normal[:2], center[:2]-mean[:2])/normal[2]
    return center, rect
