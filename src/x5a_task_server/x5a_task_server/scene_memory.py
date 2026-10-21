"""Session memory of robot-executed stacks of up to three cubes, verified by the visible top."""
import math
import threading
import time


COLORS = ('red', 'white', 'orange')


class SceneMemory:
    def __init__(self, ttl_s=3600.0):
        self.ttl_s = ttl_s
        self._stacks = {}
        self._invalidations = {}
        self._lock = threading.RLock()

    def _expire(self):
        now = time.monotonic()
        for lower, entry in list(self._stacks.items()):
            if now-entry['recorded_at'] > self.ttl_s:
                del self._stacks[lower]
                self._invalidations.pop(lower, None)

    def moved(self, color):
        with self._lock:
            for lower, entry in list(self._stacks.items()):
                if color in (lower, entry['upper']):
                    del self._stacks[lower]
                    self._invalidations.pop(lower, None)

    def record_stack(self, upper, lower, center, size, gap, frame, side, stamp_ns):
        if (upper not in COLORS or lower not in COLORS or upper == lower
                or len(center) != 3 or not all(math.isfinite(v) for v in center)
                or not math.isfinite(size) or size <= 0 or not math.isfinite(gap) or not 0 <= gap <= .005):
            raise ValueError('invalid stack memory geometry')
        upper_top = [center[0], center[1], center[2]+size*.5]
        lower_top = [center[0], center[1], center[2]-size*.5-gap]
        with self._lock:
            self._expire()
            if lower in self._stacks and self._stacks[lower]['upper'] != upper:
                raise ValueError('destination already has a remembered upper cube')
            self.moved(upper)
            # A new top must retain the support chain below its destination.
            # Discard a stale relation only when measured placement contradicts it.
            for entry in list(self._stacks.values()):
                if entry['upper'] == lower and (entry['base_frame'] != frame
                        or entry['observation_side'] != side
                        or math.dist(entry['upper_top'], lower_top) > .015):
                    self.moved(lower)
                    break
            self._stacks[lower] = {'upper':upper, 'lower':lower, 'upper_top':upper_top,
                'lower_top':lower_top, 'base_frame':frame, 'observation_side':side,
                'stamp_ns':stamp_ns, 'recorded_at':time.monotonic()}

    def observe(self, color, top, frame, side, stamp_ns, upper_observation=None, scene=None):
        """Require three distinct stable exposures to invalidate a stack.

        A visible lower fragment does not prove uncovering when the expected
        upper is still stably present. The caller excludes moving camera views.
        """
        with self._lock:
            self._expire()
            for lower, entry in list(self._stacks.items()):
                if frame != entry['base_frame'] or side != entry['observation_side'] or stamp_ns <= entry['stamp_ns']:
                    continue
                if color == lower:
                    if scene is not None and self.candidate(lower, scene, frame, side) is not None:
                        self._invalidations.pop(lower, None)
                        continue
                    upper = upper_observation or {}
                    xyz = upper.get('xyz')
                    if (upper.get('ready') and xyz is not None
                            and math.dist(xyz[:2], entry['upper_top'][:2]) <= .015
                            and abs(xyz[2]-entry['upper_top'][2]) <= .015):
                        self._invalidations.pop(lower, None)
                        continue
                    reason = 'lower_exposed'
                elif color == entry['upper'] and (math.dist(top[:2],entry['upper_top'][:2]) > .015
                                                 or abs(top[2]-entry['upper_top'][2]) > .015):
                    reason = 'upper_moved'
                elif color == entry['upper']:
                    self._invalidations.pop(lower, None)
                    continue
                else:
                    continue
                evidence = self._invalidations.get(lower, {})
                if stamp_ns <= evidence.get('stamp_ns', -1):
                    continue
                count = evidence.get('count', 0)+1 if evidence.get('reason') == reason else 1
                self._invalidations[lower] = {'reason':reason, 'count':count, 'stamp_ns':stamp_ns}
                if count >= 3:
                    del self._stacks[lower]
                    self._invalidations.pop(lower, None)

    def has_lower(self, color):
        with self._lock:
            self._expire()
            return color in self._stacks

    def candidate(self, lower, snapshot, frame, side):
        """The lower may be completely hidden; the upper must be freshly measured."""
        with self._lock:
            self._expire()
            entry = self._stacks.get(lower)
            visited = {lower}
            previous_top = None
            for _ in range(len(COLORS)-1):
                if not entry or frame != entry['base_frame'] or side != entry['observation_side']:
                    return None
                if previous_top is not None and math.dist(previous_top, entry['lower_top']) > .015:
                    return None
                upper = entry['upper']
                if upper in visited:
                    return None
                visited.add(upper)
                next_entry = self._stacks.get(upper)
                if next_entry is None:
                    item = snapshot.get(upper, {})
                    xyz = item.get('xyz')
                    if (not item.get('ready') or xyz is None or len(xyz) != 3
                            or not all(math.isfinite(v) for v in xyz)):
                        return None
                    if (math.dist(xyz[:2],entry['upper_top'][:2]) > .015
                            or abs(xyz[2]-entry['upper_top'][2]) > .015):
                        return None
                    return upper
                previous_top, entry = entry['upper_top'], next_entry
            return None

    def snapshot(self):
        with self._lock:
            self._expire()
            return [{k:v[:] if isinstance(v,list) else v for k,v in entry.items() if k != 'recorded_at'}
                    for entry in self._stacks.values()]
