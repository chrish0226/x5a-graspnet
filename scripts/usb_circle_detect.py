#!/usr/bin/env python3
"""Detect circles in a USB camera feed (HDMI capture card -> /dev/video0).

Pipeline: capture (V4L2/MJPG) -> grayscale -> blur -> adaptive HoughCircles
(param2 ladder, strictest first) -> concentric dedupe -> annotated overlay
+ pixel coordinates.  No scene means no detection: nothing is reported when
the strict ladder finds nothing.

Modes:
  --save DIR   single shot: write raw.jpg / annotated.jpg (+ gray.jpg with --debug)
  --show       live preview window with overlay (run on the desktop)
  default      single shot, print result only

Exit codes: 0 circle found, 1 no circle, 2 camera/capture error.
"""
from __future__ import annotations

import argparse
import datetime
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # let auto exposure settle before the first real frame
    for _ in range(15):
        cap.read()
    return cap


def capture_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ok = False
    frame = None
    for _ in range(5):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.1)
    if not ok or frame is None:
        raise RuntimeError("failed to read frame from camera")
    return frame


def dedupe_concentric(circles: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Collapse concentric detections (e.g. outer ring + inner bore) to the
    largest radius of each group, so one physical ring always yields one
    stable circle instead of flipping between inner and outer."""
    if circles is None or len(circles) == 0:
        return None
    kept: List[List[float]] = []
    for x, y, r in circles[0]:
        for k in kept:
            if math.dist((float(x), float(y)), (k[0], k[1])) < 0.5 * max(float(r), k[2]):
                if float(r) > k[2]:
                    k[0], k[1], k[2] = float(x), float(y), float(r)
                break
        else:
            kept.append([float(x), float(y), float(r)])
    return np.array([kept], dtype=np.float32)


class CircleTracker:
    """Lock onto one circle and keep reporting the same one frame to frame.

    Concentric detections (outer ring vs inner bore) share a centre and
    Hough sometimes only finds one of them per frame, which makes the raw
    output flip between rings.  Once locked, the tracker keeps the locked
    circle identity (fixed radius, centre follows small drift) and matches
    later frames by centre distance, so the output stays stable.
    """

    def __init__(self, center_tol_px: float = 60.0, coast_sec: float = 0.3) -> None:
        self.center_tol = float(center_tol_px)
        self.coast_sec = float(coast_sec)
        self.locked: Optional[Tuple[float, float, float]] = None  # (x, y, r)
        self.last_seen = 0.0

    def reset(self) -> None:
        self.locked = None
        self.last_seen = 0.0

    def update(self, circles) -> Optional[Tuple[float, float, float]]:
        cands = [(float(x), float(y), float(r)) for x, y, r in circles] \
            if circles is not None else []
        if not cands:
            # brief misses (motion blur, exposure flicker) coast on the last
            # result instead of dropping the circle for one frame
            if self.locked is not None and time.monotonic() - self.last_seen <= self.coast_sec:
                return self.locked
            self.locked = None  # circle genuinely gone: allow re-acquiring
            return None
        if self.locked is None:
            best = max(cands, key=lambda c: c[2])
            self.locked = best
            self.last_seen = time.monotonic()
            return self.locked
        near = [c for c in cands
                if math.dist((c[0], c[1]), (self.locked[0], self.locked[1]))
                <= self.center_tol]
        if not near:
            if time.monotonic() - self.last_seen <= self.coast_sec:
                return self.locked
            self.locked = None  # locked circle gone: re-acquire on next frame
            return None
        best = max(near, key=lambda c: c[2])
        # keep the locked radius (ring identity); follow small centre drift
        self.locked = (best[0], best[1], self.locked[2])
        self.last_seen = time.monotonic()
        return self.locked


def detect_circles(
    frame: np.ndarray,
    min_radius: int,
    max_radius: int,
    min_dist_frac: float = 0.08,
    expected_radius: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Return (circles on original frame, gray used, debug notes).

    expected_radius: when a circle is already being tracked, one extra
    radius-constrained lenient pass runs if the strict ladder fails — this
    recovers blurred/odd-angle frames without reopening global false
    positives.
    """
    notes: List[str] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best: Optional[np.ndarray] = None
    # param2 ladder: high -> strict (few, good circles); low -> lenient
    for param2 in (60, 45, 35, 25, 18):
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(20, int(min(frame.shape[:2]) * min_dist_frac)),
            param1=120,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None and len(circles) > 0:
            best = circles
            notes.append(f"param2={param2} hits={len(circles)}")
            break
        notes.append(f"param2={param2} hits=0")
    if best is None and expected_radius:
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        best = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1,
            minDist=max(20, int(min(frame.shape[:2]) * min_dist_frac)),
            param1=120, param2=22,
            minRadius=int(expected_radius * 0.6),
            maxRadius=int(expected_radius * 1.5),
        )
        notes.append(f"guided_retry r~{int(expected_radius)} "
                     f"hits={0 if best is None else len(best)}")
    return dedupe_concentric(best), gray, notes


def annotate(frame: np.ndarray, circles: np.ndarray) -> np.ndarray:
    out = frame.copy()
    h, w = frame.shape[:2]
    for (x, y, r) in np.round(circles[0]).astype(int):
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(out, f"({x},{y}) r{r}", (max(0, x - r), max(12, y - r - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"{len(circles[0])} circle(s)  {w}x{h}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def report(circles: np.ndarray, frame_shape: Tuple[int, int]) -> None:
    h, w = frame_shape
    print(f"detected {len(circles[0])} circle(s), frame {w}x{h}:")
    for (x, y, r) in np.round(circles[0]).astype(int):
        print(f"  center_px=({x}, {y}) radius_px={r} "
              f"center_norm=({x / w:.3f}, {y / h:.3f})")


def run_once(args: argparse.Namespace) -> int:
    cap = open_camera(args.device, args.width, args.height)
    try:
        frame = capture_frame(cap)
    finally:
        cap.release()
    circles, gray, notes = detect_circles(frame, args.min_radius, args.max_radius)
    for n in notes:
        print(n)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.save) if args.save else Path("/tmp/circle_detect")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"raw_{stamp}.jpg"), frame)
    if args.debug:
        cv2.imwrite(str(out_dir / f"gray_{stamp}.jpg"), gray)
    if circles is not None:
        report(circles, frame.shape[:2])
        cv2.imwrite(str(out_dir / f"annotated_{stamp}.jpg"), annotate(frame, circles))
        print(f"saved: {out_dir}/raw_{stamp}.jpg, annotated_{stamp}.jpg")
        return 0
    print("NO CIRCLE DETECTED")
    print(f"saved: {out_dir}/raw_{stamp}.jpg")
    return 1


def run_live(args: argparse.Namespace) -> int:
    cap = open_camera(args.device, args.width, args.height)
    tracker = CircleTracker()
    found_last = False
    try:
        while True:
            frame = capture_frame(cap)
            circles, _, _ = detect_circles(frame, args.min_radius, args.max_radius)
            tracked = tracker.update(circles[0] if circles is not None else None)
            shown = frame
            if tracked is not None:
                found_last = True
                shown = annotate(frame, np.array([[tracked]], dtype=np.float32))
            else:
                cv2.putText(shown, "NO CIRCLE (r: re-lock)", (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("circle detect (q quit, s save, r re-lock)", shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.reset()
            if key == ord("s"):
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out = Path(args.save or "/tmp/circle_detect")
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"live_{stamp}.jpg"), shown)
                print(f"saved {out}/live_{stamp}.jpg")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0 if found_last else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0, help="/dev/videoN index (default 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--min-radius", type=int, default=15, help="min circle radius px")
    p.add_argument("--max-radius", type=int, default=0, help="0 = half frame")
    p.add_argument("--show", action="store_true", help="live preview window")
    p.add_argument("--save", default="", help="output dir for saved frames")
    p.add_argument("--debug", action="store_true", help="also save grayscale frame")
    args = p.parse_args()
    if args.max_radius <= 0:
        args.max_radius = args.height // 2
    try:
        return run_live(args) if args.show else run_once(args)
    except RuntimeError as exc:
        print(f"CAMERA ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
