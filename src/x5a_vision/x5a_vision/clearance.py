"""Conservative two-layer support and fixed-buffer checks on live RGB-D points."""
from pathlib import Path
import math

import numpy as np
import yaml


DEFAULT_BUFFER_FILE = str(Path.home() / 'arx/wrist_ws/config/buffer.yaml')


def buffer_policy(filename):
    data = yaml.safe_load(Path(filename).read_text()) or {}
    if data.get('enabled') is not True:
        raise ValueError('自动移开功能未启用')
    mode = data.get('mode', 'fixed')
    gap = data.get('cube_edge_gap_m', .05)
    if (mode not in ('auto', 'fixed') or isinstance(gap, bool)
            or not isinstance(gap, (int, float)) or not math.isfinite(gap) or not .04 <= gap <= .10):
        raise ValueError('暂放模式或方块边缘间距配置无效')
    return mode, float(gap)


class BufferSurfaceIndex:
    """Conservative integral-grid queries over observed depth, built once/frame.

    Clutter queries include boundary bins; coverage queries use only complete
    bins inside each cell. Rounding can reject a marginal patch, but cannot
    turn an unseen or occupied patch into an accepted placement surface.
    """
    def __init__(self, points, table_top, bounds, resolution=.001):
        self.origin = np.array([bounds['x_min'], bounds['y_min']], float)
        self.resolution = resolution
        shape = np.ceil((np.array([bounds['x_max'], bounds['y_max']])-self.origin)/resolution).astype(int)
        points = np.asarray(points, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        ij = np.floor((points[:, :2]-self.origin)/resolution).astype(int)
        keep = np.all((ij >= 0) & (ij < shape), axis=1)
        ij, z = ij[keep], points[keep, 2]
        self.integrals = []
        for valid in (np.abs(z-table_top) <= .008, z > table_top+.010):
            counts = np.bincount(ij[valid, 0]*shape[1]+ij[valid, 1], minlength=int(np.prod(shape))).reshape(shape)
            self.integrals.append(np.pad(counts.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0))))
        self.shape = shape

    def _count(self, integral, low, high, outer=False):
        a, b = (np.asarray(low)-self.origin)/self.resolution, (np.asarray(high)-self.origin)/self.resolution
        start = (np.floor(a) if outer else np.ceil(a)).astype(int)
        stop = (np.ceil(b) if outer else np.floor(b)).astype(int)
        # Coverage beyond the indexed workspace cannot be certified.
        if np.any(start < 0) or np.any(stop > self.shape) or np.any(stop <= start):
            return None
        x0, y0 = start
        x1, y1 = stop
        return integral[x1, y1]-integral[x0, y1]-integral[x1, y0]+integral[x0, y0]

    def check(self, xy, half_width=.04, cells=6):
        xy = np.asarray(xy, float)
        raised = self._count(self.integrals[1], xy-half_width, xy+half_width, outer=True)
        if raised is not None and raised >= 3:
            return {'empty': False, 'reason': '暂放位或夹爪周围有物体占用'}
        if raised is None:
            return {'empty': False, 'reason': '暂放位未完整可见或深度不足'}
        edges = xy[:, None] + np.linspace(-half_width, half_width, cells+1)
        start = np.ceil((edges[:, :-1]-self.origin[:, None])/self.resolution).astype(int)
        stop = np.floor((edges[:, 1:]-self.origin[:, None])/self.resolution).astype(int)
        if (np.any(start < 0) or np.any(stop > self.shape[:, None]) or np.any(stop <= start)):
            return {'empty': False, 'reason': '暂放位未完整可见或深度不足'}
        x0, y0 = start[0, :, None], start[1, None, :]
        x1, y1 = stop[0, :, None], stop[1, None, :]
        integral = self.integrals[0]
        counts = integral[x1, y1]-integral[x0, y1]-integral[x1, y0]+integral[x0, y0]
        if np.any(counts < 3):
            return {'empty': False, 'reason': '暂放位未完整可见或深度不足'}
        return {'empty': True, 'reason': '暂放位桌面可见且无占用'}


def buffer_candidates(points, source_xy, size, table_top, bounds, gap=.05, surface_index=None, diagnostics=None):
    """Search sixteen directions and five rings; reject unobserved space.

    Radii include the square footprint projected onto the search direction,
    so diagonal candidates retain at least the requested edge separation.
    """
    source = np.asarray(source_xy, float)
    candidates = []
    stats = {'tested': 0, 'outside_workspace': 0, 'occupied': 0, 'insufficient_depth': 0, 'available': 0}
    surface_index = surface_index or BufferSurfaceIndex(points, table_top, bounds)
    directions = [(-1,0),(0,1),(1,0),(0,-1),(-1,1),(1,1),(1,-1),(-1,-1)]
    directions += [(math.cos(angle), math.sin(angle)) for angle in np.arange(8)*math.pi/4+math.pi/8]
    for extra in (0., .02, .04, .07, .11):
        for dx, dy in directions:
            stats['tested'] += 1
            direction = np.array([dx,dy], float)/math.hypot(dx,dy)
            # The lower cube yaw is hidden. Reserve its maximum projected half
            # footprint as well as that of the held upper cube.
            radius = gap+extra+size*math.sqrt(2)
            xy = source+direction*radius
            if (xy[0]-.04 < bounds['x_min'] or xy[0]+.04 > bounds['x_max']
                    or xy[1]-.04 < bounds['y_min'] or xy[1]+.04 > bounds['y_max']
                    or math.hypot(*xy)+.04 > bounds['r_max']):
                stats['outside_workspace'] += 1
                continue
            surface = surface_index.check(xy)
            if surface['empty']:
                candidates.append({'xy_m': xy.tolist(), 'edge_gap_m': gap+extra})
            else:
                stats['occupied' if '占用' in surface['reason'] else 'insufficient_depth'] += 1
    stats['available'] = len(candidates)
    if diagnostics is not None:
        diagnostics.update(stats)
    return candidates


def read_buffer(filename, frame, side):
    data = yaml.safe_load(Path(filename).read_text()) or {}
    if data.get('enabled') is not True:
        raise ValueError('暂放位尚未设置，请先用标记方块记录位置')
    if data.get('base_frame') != frame or data.get('observation_side') != side:
        raise ValueError('暂放位与当前坐标系或观察侧不一致，请重新设置')
    xy = data.get('xy_m')
    if (not isinstance(xy, list) or len(xy) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in xy)):
        raise ValueError('暂放位需要两个有限的基座 X、Y 坐标（米）')
    return tuple(float(x) for x in xy)


def base_cloud(depth, intrinsics, transform, depth_min, depth_max, stride=3):
    """Return sampled points and their original image pixels, without depth filling."""
    ys, xs = np.indices(depth[::stride, ::stride].shape)
    xs, ys = xs.ravel()*stride, ys.ravel()*stride
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z >= depth_min) & (z <= depth_max)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    k, t = np.asarray(intrinsics).reshape(3, 3), np.asarray(transform)
    camera = np.column_stack(((xs-k[0, 2])*z/k[0, 0], (ys-k[1, 2])*z/k[1, 1], z))
    return camera @ t[:3, :3].T + t[:3, 3], xs, ys


def blockers_from_sides(points, pixel_colors, visible_tops, unavailable, table_top, size=.03):
    """Require colored side depth under one stable second-layer top.

    Old hidden target poses are never used. Ambiguous towers produce no relation.
    """
    result = {}
    for lower in unavailable:
        side = points[pixel_colors == lower]
        candidates = []
        for upper, top in visible_tops.items():
            if upper == lower or abs(top[2]-(table_top+2*size)) > .012:
                continue
            near = side[(np.max(np.abs(side[:, :2]-top[:2]), axis=1) <= size*.5+.006)
                        & (side[:, 2] >= table_top+.004) & (side[:, 2] <= table_top+size-.003)]
            if len(near) >= 20 and np.ptp(near[:, 2]) >= .006:
                candidates.append(upper)
        if len(candidates) == 1:
            result[lower] = candidates[0]
    return result


def buffer_surface(points, xy, table_top, half_width=.04, cells=6):
    """Every grid cell must have observed table depth; clutter or gaps reject."""
    points = np.asarray(points, float).reshape(-1, 3)
    selected = points[np.all(np.abs(points[:, :2]-xy) < half_width, axis=1)]
    if np.count_nonzero(selected[:, 2] > table_top+.010) >= 3:
        return {'empty': False, 'reason': '暂放位或夹爪周围有物体占用'}
    flat = selected[np.abs(selected[:, 2]-table_top) <= .008]
    grid = ((flat[:, :2]-xy+half_width) / (2*half_width)*cells).astype(int)
    grid = np.clip(grid, 0, cells-1)
    counts = np.zeros((cells, cells), dtype=int)
    if len(grid):
        np.add.at(counts, (grid[:, 0], grid[:, 1]), 1)
    if np.any(counts < 3):
        return {'empty': False, 'reason': '暂放位未完整可见或深度不足'}
    return {'empty': True, 'reason': '暂放位桌面可见且无占用'}
