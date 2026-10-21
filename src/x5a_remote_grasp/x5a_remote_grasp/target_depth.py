"""Select an observed target surface without altering the obstacle cloud."""
import numpy as np
from scipy import ndimage

from x5a_remote_grasp.grasp_quality import depth_points


def target_surface(depth_m, mask, intrinsics, transform, *, table_top,
                   max_height=0.5, max_radius=0.54):
    """Return a subset of the SAM mask containing coherent foreground depth.

    The nearest substantial 3D component wins, rather than a background plane
    with more depth pixels. No holes are filled and no depth is synthesized.
    The caller must keep the original depth for obstacle/collision checks.
    """
    depth = np.asarray(depth_m, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    transform = np.asarray(transform, dtype=float)
    if (mask.shape != depth.shape or transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or not np.isfinite([table_top, max_height, max_radius]).all()
            or max_height <= table_top or not 0 < max_radius <= 2):
        raise ValueError('invalid target depth geometry')
    observed = mask & np.isfinite(depth) & (depth > 0)
    camera = depth_points(depth, observed, *intrinsics)
    base = camera @ transform[:3, :3].T + transform[:3, 3]
    plausible = ((base[:, 2] >= table_top-0.003)
                 & (base[:, 2] <= max_height)
                 & (np.linalg.norm(base[:, :2], axis=-1) <= max_radius))
    points = base[plausible]
    if len(points) < 50:
        raise ValueError('目标有效深度不足：桌面/工作范围检查后少于 50 个点')
    # A bounded voxel grid avoids quadratic neighbour comparisons on HD masks.
    voxels, inverse = np.unique(np.floor(points/0.006).astype(np.int64),
                                axis=0, return_inverse=True)
    cells = voxels-voxels.min(axis=0)
    shape = cells.max(axis=0)+1
    if np.prod(shape, dtype=np.int64) > 10_000_000:
        raise ValueError('目标深度分布过于分散，请重新观察')
    grid = np.zeros(tuple(shape), dtype=bool)
    grid[tuple(cells.T)] = True
    # Group neighbouring observed fragments across small missing-depth gaps.
    # Dilation is ONLY used for component labels: returned points are still
    # an exact subset of the original observation, with no synthesized depth.
    structure = np.ones((3, 3, 3))
    linked = ndimage.binary_dilation(grid, structure=structure)
    labels, _ = ndimage.label(linked, structure=structure)
    voxel_labels = labels[tuple(cells.T)]
    point_labels = voxel_labels[inverse]
    counts = np.bincount(point_labels)
    voxel_counts = np.bincount(voxel_labels, minlength=len(counts))
    minimum = max(50, int(np.ceil(counts.max()*0.2)))
    eligible = np.flatnonzero((counts >= minimum) & (voxel_counts >= 12))
    eligible = eligible[eligible != 0]
    if not len(eligible):
        raise ValueError('目标深度缺少连续表面，请调整观察位置')
    camera_depth = camera[plausible, 2]
    chosen = min(eligible, key=lambda label: (
        float(np.median(camera_depth[point_labels == label])), -counts[label]))
    keep = point_labels == chosen
    # Off-workspace/background returns have already been rejected. Counting
    # them again in the denominator falsely rejects a well-observed foreground
    # surface whenever a reflective bottle produces many distant returns.
    if keep.sum() < max(50, int(np.ceil(plausible.sum()*0.1))):
        raise ValueError('目标深度异常点过多，无法可靠定位，请重新观察')
    clean = np.zeros(mask.shape, dtype=bool)
    clean.flat[np.flatnonzero(observed)[plausible][keep]] = True
    selected = points[keep]
    low, high = np.percentile(selected, [5, 95], axis=0)
    report = dict(raw_points=int(observed.sum()), plausible_points=int(plausible.sum()),
                  selected_points=int(keep.sum()), components=int(len(counts)-1),
                  rejected_points=int(observed.sum()-keep.sum()),
                  plausible_coverage=float(keep.sum()/plausible.sum()),
                  center_base=((low+high)/2).tolist(),
                  height_band_base=[float(low[2]), float(high[2])])
    return clean, report


def target_motion_landmarks(mask, surface_depths, intrinsics, transform):
    """Motion references only; these points must never become grasp poses.

    Track the SAM silhouette box with the observed foreground's near-surface
    depth (10th percentile). Unlike a 3D extent midpoint, this does not move
    downward when additional, deeper neck/body pixels become visible. Corners
    also expose changes in silhouette position/scale hidden by its center.
    """
    mask = np.asarray(mask, bool)
    depths = np.asarray(surface_depths, float)
    transform = np.asarray(transform, float)
    fx, fy, cx, cy = intrinsics
    if (mask.ndim != 2 or not mask.any() or depths.ndim != 1 or len(depths) < 50
            or not np.isfinite(depths).all() or np.min(depths) <= 0
            or not np.isfinite(intrinsics).all() or min(fx, fy) <= 0
            or transform.shape != (4, 4) or not np.isfinite(transform).all()):
        raise ValueError('目标运动复核缺少有效轮廓或前表面深度')
    y, x = np.nonzero(mask)
    left, right, top, bottom = x.min(), x.max(), y.min(), y.max()
    uv = np.array([[(left+right)/2, (top+bottom)/2],
                   [left, top], [right, top], [left, bottom], [right, bottom]])
    depth = float(np.percentile(depths, 10))
    camera = np.column_stack(((uv[:, 0]-cx)*depth/fx,
                              (uv[:, 1]-cy)*depth/fy, np.full(5, depth)))
    return camera @ transform[:3, :3].T + transform[:3, 3]


def stable_target_motion(reference, centers, max_motion=0.01, max_spread=0.003):
    """Require two fresh consistent observations, retaining the motion limit."""
    reference, centers = np.asarray(reference, float), np.asarray(centers, float)
    if (reference.ndim not in (1, 2) or reference.shape[-1] != 3
            or (reference.ndim == 2 and len(reference) < 1)
            or centers.shape != (2,)+reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(centers).all()
            or not np.isfinite([max_motion, max_spread]).all()
            or min(max_motion, max_spread) <= 0):
        raise ValueError('目标复核缺少两次有效定位')
    spread = float(np.max(np.linalg.norm(centers[1]-centers[0], axis=-1)))
    if spread > max_spread:
        raise ValueError(f'目标定位不稳定：两次复核相差 {spread*1000:.1f}mm，请重新观察')
    shift = float(np.max(np.linalg.norm(centers-reference, axis=-1)))
    if shift > max_motion:
        raise ValueError(f'目标定位变化 {shift*1000:.1f}mm，超过 {max_motion*1000:.1f}mm，请重新规划')
    return shift, spread
