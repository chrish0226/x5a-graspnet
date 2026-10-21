"""Geometry for placing one measured cube on another, with the held TCP offset."""
import math
import numpy as np

MAX_STACK_LAYERS = 3


def cube_top_layer(z, height, table_top, max_layers=MAX_STACK_LAYERS):
    """Accept measured tops near discrete layers, including small release gaps."""
    for layer in range(1, max_layers+1):
        if abs(z-(table_top+layer*height)) <= .012:
            return layer
    raise ValueError(f'方块顶面高度不在 1～{max_layers} 层范围内')


def stack_geometry(source_top, destination_top, size, table_top, clearance=.002):
    source, destination = np.asarray(source_top, float), np.asarray(destination_top, float)
    size = np.asarray(size, float)
    if (not np.isfinite(np.r_[source, destination, size, table_top, clearance]).all()
            or source.shape != (3,) or destination.shape != (3,) or size.shape != (3,)
            or np.any(size <= 0) or not 0 <= clearance <= .005):
        raise ValueError('叠放坐标或尺寸无效')
    cube_top_layer(source[2], size[2], table_top)
    cube_top_layer(destination[2], size[2], table_top, MAX_STACK_LAYERS-1)
    if np.linalg.norm(source[:2] - destination[:2]) < max(size[:2]) + .01:
        raise ValueError('两个方块距离过近或重叠，请先分开摆放')
    center = destination.copy()
    center[2] += size[2] / 2 + clearance
    return center


def tcp_for_object_center(center, held_offset, rpy):
    center, offset = np.asarray(center, float), np.asarray(held_offset, float)
    if not np.isfinite(np.r_[center, offset, rpy]).all():
        raise ValueError('夹持偏移或姿态无效')
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]])
    return center - rotation @ offset


def aligned_stack_rpy(grasp_rpy, source_yaw, destination_yaw):
    # Square faces repeat every 90 degrees; use the smallest yaw change.
    delta = (destination_yaw - source_yaw + math.pi/4) % (math.pi/2) - math.pi/4
    return [grasp_rpy[0], grasp_rpy[1], grasp_rpy[2] + delta]


def aligned_stack_candidates(grasp_rpy, source_yaw, destination_yaw):
    """All four upright square alignments, ordered by tool yaw travel."""
    nearest = aligned_stack_rpy(grasp_rpy, source_yaw, destination_yaw)
    wrap = lambda angle: (angle + math.pi) % (2*math.pi) - math.pi
    candidates = [[nearest[0], nearest[1], wrap(nearest[2] + quarter*math.pi/2)]
                  for quarter in range(4)]
    return sorted(candidates, key=lambda rpy: abs(wrap(rpy[2]-grasp_rpy[2])))


def held_stack_center(destination_top, size, table_top, clearance=.002):
    """An already lifted cube only needs the destination's current geometry."""
    destination, size = np.asarray(destination_top, float), np.asarray(size, float)
    if (destination.shape != (3,) or size.shape != (3,)
            or not np.isfinite(np.r_[destination, size, table_top, clearance]).all()
            or np.any(size <= 0) or not 0 <= clearance <= .005):
        raise ValueError('叠放坐标或尺寸无效')
    cube_top_layer(destination[2], size[2], table_top, MAX_STACK_LAYERS-1)
    center = destination.copy()
    center[2] += size[2] / 2 + clearance
    return center
