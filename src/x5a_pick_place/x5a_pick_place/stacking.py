"""Geometry for placing one measured cube on another, with the held TCP offset."""
import math
import numpy as np


def stack_geometry(source_top, destination_top, size, table_top, clearance=.002):
    source, destination = np.asarray(source_top, float), np.asarray(destination_top, float)
    size = np.asarray(size, float)
    if (not np.isfinite(np.r_[source, destination, size, table_top, clearance]).all()
            or np.any(size <= 0) or not 0 <= clearance <= .005):
        raise ValueError('叠放坐标或尺寸无效')
    # An exposed upper cube can be reused after a previous stack. The new
    # destination remains a standalone table cube, so no taller tower is built.
    if min(abs(source[2] - (table_top + layer*size[2])) for layer in (1, 2)) > .012:
        raise ValueError('源方块必须在桌面或两层叠放的上层，并露出完整顶面')
    if abs(destination[2] - (table_top + size[2])) > .012:
        raise ValueError('目标方块需要单独放在桌面上')
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


def held_stack_center(destination_top, size, table_top, clearance=.002):
    """An already lifted cube only needs the destination's current geometry."""
    destination, size = np.asarray(destination_top, float), np.asarray(size, float)
    if (destination.shape != (3,) or size.shape != (3,)
            or not np.isfinite(np.r_[destination, size, table_top, clearance]).all()
            or np.any(size <= 0) or not 0 <= clearance <= .005):
        raise ValueError('叠放坐标或尺寸无效')
    if abs(destination[2] - (table_top + size[2])) > .012:
        raise ValueError('目标方块需要单独放在桌面上')
    center = destination.copy()
    center[2] += size[2] / 2 + clearance
    return center
