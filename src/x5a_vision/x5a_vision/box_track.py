"""Short-lived box tracking through missing depth, without refreshing old poses."""
from collections import deque
import numpy as np


class BoxTrack:
    def __init__(self, window, max_std, reset_distance, grace):
        self.history = deque(maxlen=window)
        self.window, self.max_std = window, max_std
        self.reset_distance, self.grace = reset_distance, grace
        self.last_seen = None
        self.stable = False

    def clear(self):
        self.history.clear()
        self.stable = False
        self.last_seen = None

    def update(self, point, now):
        # HD processing can skip one camera update. A newly visible, consistent
        # measurement may bridge that interval; repeated missing frames still
        # expire through missing(), and a long camera outage rebuilds the window.
        if self.last_seen is not None and now-self.last_seen > max(3.0, self.grace*2):
            self.clear()
        if self.history and np.linalg.norm(point-np.median(self.history,axis=0)) > self.reset_distance:
            self.clear()
        self.history.append(np.asarray(point,float))
        self.last_seen = now
        self.stable = len(self.history) >= self.window and bool(np.all(np.std(self.history,axis=0) <= self.max_std))
        return np.median(self.history,axis=0), self.stable

    def missing(self, now):
        if self.last_seen is None or now-self.last_seen > self.grace:
            self.clear()
        return self.stable
