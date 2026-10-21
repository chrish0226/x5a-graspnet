"""Fresh, stationary RGB-D observations for a stop-and-look grasp correction."""
from collections import deque
import math
from statistics import median, pstdev


class ReobserveWindow:
    def __init__(self, after_ns, samples=5, span_s=0.25, max_std_m=0.0025, max_gap_s=0.2):
        self.after_ns = after_ns
        self.samples = deque(maxlen=samples)
        self.count = samples
        self.span_ns = int(span_s * 1e9)
        self.max_std = max_std_m
        self.last_stamp = after_ns
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('max_gap_s must be finite and positive')
        self.max_gap_ns = int(max_gap_s*1e9)

    def add(self, stamp_ns, xyz, stable):
        if not stable or len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            self.samples.clear()
            self.last_stamp = max(self.last_stamp, stamp_ns)
            return None
        # Repeated reads of a cached pose and frames captured before stopping
        # cannot count toward a new observation.
        if stamp_ns <= self.last_stamp:
            return None
        if self.samples and stamp_ns - self.last_stamp > self.max_gap_ns:
            self.samples.clear()
        self.last_stamp = stamp_ns
        self.samples.append((stamp_ns, tuple(xyz)))
        if len(self.samples) < self.count:
            return None
        if self.samples[-1][0] - self.samples[0][0] < self.span_ns:
            return None
        axes = list(zip(*(p for _, p in self.samples)))
        if any(pstdev(axis) > self.max_std for axis in axes):
            return None
        return tuple(median(axis) for axis in axes)


def bounded_correction(original, observed, max_delta_m):
    if len(original) != 3 or len(observed) != 3 or not math.isfinite(max_delta_m) or max_delta_m <= 0:
        raise ValueError('invalid correction geometry or limit')
    if not all(math.isfinite(v) for v in (*original, *observed)):
        raise ValueError('non-finite target')
    delta = tuple(b - a for a, b in zip(original, observed))
    if math.sqrt(sum(v * v for v in delta)) > max_delta_m:
        raise ValueError('reobserved target exceeds correction limit')
    return delta
