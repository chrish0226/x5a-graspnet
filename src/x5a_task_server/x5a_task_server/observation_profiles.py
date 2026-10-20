"""Home-plane reflection helpers for the wrist-only two-sided workspace."""
import math


def mirrored_seed(joints, home):
    # Z yaw (1,5) and X roll (6) reverse under reflection in the home XZ plane.
    signs = (-1, 1, 1, 1, -1, -1)
    return [h+s*(q-h) for q, h, s in zip(joints, home, signs)]


def crosses_home_plane(current, target, home):
    if current is None or len(current) != 6:
        return False
    a = math.atan2(math.sin(current[0]-home[0]), math.cos(current[0]-home[0]))
    b = math.atan2(math.sin(target[0]-home[0]), math.cos(target[0]-home[0]))
    return abs(a) > .2 and abs(b) > .2 and a*b < 0
