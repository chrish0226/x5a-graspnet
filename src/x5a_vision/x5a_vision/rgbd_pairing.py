"""Match exposure timestamps independently of transport arrival order."""
from collections import deque


def stamp_ns(msg):
    return msg.header.stamp.sec*1_000_000_000 + msg.header.stamp.nanosec


class RgbdPairBuffer:
    """Bounded raw-frame history; caller serializes access with its intake lock."""
    def __init__(self, max_gap_s=.04, max_age_s=1.5, history=20):
        self.colors = deque(maxlen=history)
        self.depths = deque(maxlen=history)
        self.max_gap_ns = round(max_gap_s*1e9)
        self.max_age_ns = round(max_age_s*1e9)
        self.last_processed_ns = -1

    def take_latest(self, now_ns):
        # A newly arriving RGB can correspond to a depth received several
        # callbacks ago. Pair against retained depths, never just the latest.
        best = None
        for depth in self.depths:
            stamp = stamp_ns(depth)
            if stamp <= self.last_processed_ns or not -50_000_000 <= now_ns-stamp <= self.max_age_ns:
                continue
            if not self.colors:
                break
            color = min(self.colors, key=lambda c: abs(stamp_ns(c)-stamp))
            if abs(stamp_ns(color)-stamp) > self.max_gap_ns:
                continue
            if best is None or stamp > stamp_ns(best[1]):
                best = color, depth
        if best is not None:
            self.last_processed_ns = stamp_ns(best[1])
        return best
