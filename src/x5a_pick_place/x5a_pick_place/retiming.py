"""Apply stage speed/acceleration scaling to Humble Cartesian trajectories."""
from copy import deepcopy
import math


def scale_cartesian_timing(trajectory, velocity_scale, acceleration_scale):
    if any(not math.isfinite(v) or not 0 < v <= 1
           for v in (velocity_scale, acceleration_scale)):
        raise ValueError('trajectory scaling must be within (0, 1]')
    # Uniform time scaling preserves the collision-checked geometric path.
    # v scales with s, a with s²; choose s to respect both requested limits.
    speed = min(velocity_scale, math.sqrt(acceleration_scale))
    out = deepcopy(trajectory)
    if not out.joint_trajectory.points:
        raise ValueError('empty Cartesian trajectory')
    previous_time, previous_position = -1, None
    for point in out.joint_trajectory.points:
        ns = point.time_from_start.sec*10**9 + point.time_from_start.nanosec
        if ns < 0 or ns < previous_time or not all(math.isfinite(v) for v in
                [*point.positions, *point.velocities, *point.accelerations]):
            raise ValueError('malformed Cartesian timing')
        if ns == previous_time and previous_position != list(point.positions):
            raise ValueError('nonzero motion at zero duration')
        previous_time, previous_position = ns, list(point.positions)
        scaled = round(ns/speed)
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(scaled,10**9)
        point.velocities = [v*speed for v in point.velocities]
        point.accelerations = [a*speed*speed for a in point.accelerations]
    return out
