"""Bounded finger/table contact without disabling the table collision model."""
import copy

import numpy as np
from moveit_msgs.msg import AllowedCollisionEntry

TABLE_ID = 'remote_grasp_table'
TABLE_LIMIT_ID = 'remote_grasp_table_contact_limit'


def contact_limit_box(center, size, tolerance):
    """Keep the table bottom fixed; lower only the fingers' hard upper limit."""
    center = np.asarray(center, dtype=float).copy()
    size = np.asarray(size, dtype=float).copy()
    if (center.shape != (3,) or size.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(size).all()
            or np.any(size <= 0) or not np.isfinite(tolerance)
            or not 0 <= tolerance <= .005 or tolerance >= size[2]):
        raise ValueError('table contact tolerance must be 0..5 mm and smaller than table thickness')
    center[2] -= tolerance / 2
    size[2] -= tolerance
    return center, size


def merge_collision_pairs(matrix, pairs):
    """Change explicit pairs, preserving SRDF entries and scene defaults.

    A nonempty ACM in a PlanningScene diff replaces the entire ACM in MoveIt.
    Sending just the table/finger rows would therefore erase self-collision
    exclusions. Expand a copy of the queried matrix instead.
    """
    result = copy.deepcopy(matrix)
    names = list(result.entry_names)
    if len(result.entry_values) != len(names) or any(
            len(row.enabled) != len(names) for row in result.entry_values):
        raise ValueError('malformed current allowed collision matrix')
    defaults = dict(zip(result.default_entry_names, result.default_entry_values))

    def default_pair(a, b):
        values = [defaults[name] for name in (a, b) if name in defaults]
        return bool(values) and all(values)

    for pair in pairs:
        for name in pair:
            if name in names:
                continue
            for other, row in zip(names, result.entry_values):
                row.enabled.append(default_pair(other, name))
            result.entry_values.append(AllowedCollisionEntry(
                enabled=[default_pair(name, other) for other in names] + [default_pair(name, name)]))
            names.append(name)
    result.entry_names = names
    for (a, b), enabled in pairs.items():
        i, j = names.index(a), names.index(b)
        result.entry_values[i].enabled[j] = bool(enabled)
        result.entry_values[j].enabled[i] = bool(enabled)
    return result
