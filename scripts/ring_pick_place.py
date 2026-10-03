#!/usr/bin/env python3
"""Fixed-position ring pick and place (conveyor + good/bad sorting).

Flow per workpiece:
  1. A ring passes the camera's detection zone (frame centre +-zone-frac).
     When the circle holds position within --move-tol-px for --stable-sec
     (default 0.65 s), the countdown starts.
  2. The conveyor carries the ring to the grasp point during --countdown-sec
     (default 6 s).  The ring leaving the zone during the countdown is the
     expected transport, NOT an abort condition; the countdown always runs
     to completion.
  3. After the countdown the arm picks at the fixed pre-grasp pose (directly
     above the conveyor grasp point): open -> cartesian descend
     --grasp-depth -> close -> lift.
  4. Placement is quality sorting: workpiece occurrence #1 is a good part ->
     --place-good point, #2 is a bad part -> --place-bad point, then
     alternate (odd occurrences good, even occurrences bad).  A real visual
     quality classifier can replace the parity rule in classify_workpiece().
  5. Return to pre-grasp and wait for the next ring.

Place points are taught with record_pose.py and passed explicitly; the task
refuses to move the arm without them.  Use --plan-only to rehearse the whole
loop without executing, or --monitor-only to test just the camera FSM.

Requires: move_group + official adapter running, /arm driver up, rqt and any
other /arm_cmd publisher closed (adapter rejects execution otherwise).
"""
from __future__ import annotations

import argparse
import datetime
import math
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import rclpy
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath

from usb_circle_detect import (
    CircleTracker,
    capture_frame,
    detect_circles,
    open_camera,
)
from move_to_pose import ARM_JOINTS, MoveToPoseNode, load_pose_file, rpy_to_quat

DEFAULT_PRE_GRASP = "/home/quella/arx/arm/logs/end_pose_20260828_173732.yaml"


class RingMonitor(threading.Thread):
    """Watch the camera, detect the ring and run the stable/countdown FSM."""

    def __init__(
        self,
        log,
        device: int,
        width: int,
        height: int,
        zone_frac: float,
        move_tol_px: float,
        stable_sec: float,
        countdown_sec: float,
        save_dir: Path,
        camera_retries: int,
        camera_retry_sec: float,
    ) -> None:
        super().__init__(daemon=True)
        self.log = log
        self.device = device
        self.width = width
        self.height = height
        self.zone_frac = zone_frac
        self.move_tol = move_tol_px
        self.stable_sec = stable_sec
        self.countdown_sec = countdown_sec
        self.save_dir = save_dir
        self.camera_retries = max(0, int(camera_retries))
        self.camera_retry_sec = max(0.1, float(camera_retry_sec))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.state = "paused"  # paused|watch|countdown|trigger|error
        self.stable_t = 0.0
        self.countdown_remaining = 0.0
        self.last_center: Optional[Tuple[int, int]] = None
        self.frozen_center: Optional[Tuple[int, int]] = None
        self.countdown_start = 0.0
        self.left_zone_logged = False
        self.tracker = CircleTracker()
        # armed = the zone has been seen clear, so the next circle is a NEW
        # arriving workpiece (not the ring just placed or a leftover)
        self.armed = False
        self.empty_t = 0.0
        self.wait_t = 0.0
        # Non-zero only while VideoCapture.read() is in progress.  Circle
        # detection can legitimately take seconds and must not be mistaken
        # for a blocked USB camera read.
        self.capture_started_time = 0.0
        self._last_watch_save = 0.0
        self.trigger_evt = threading.Event()
        self._stop_flag = False

    # ---- control (called from the task thread) ----
    def pause(self) -> None:
        with self.lock:
            self.state = "paused"
            self.trigger_evt.clear()
            self.stable_t = 0.0
            self.last_center = None
            self.frozen_center = None

    def resume(self) -> None:
        with self.lock:
            if self.state != "error":
                self.state = "watch"
                self.stable_t = 0.0
                self.countdown_remaining = 0.0
                self.last_center = None
                self.frozen_center = None
                self.left_zone_logged = False
                self.tracker.reset()
                self.armed = False
                self.empty_t = 0.0
                self.wait_t = 0.0
                self.trigger_evt.clear()

    def stop(self) -> None:
        self._stop_flag = True

    def snapshot_state(self) -> Tuple[str, float, float]:
        with self.lock:
            return self.state, self.stable_t, self.countdown_remaining

    # ---- internals ----
    def _save(self, frame, tag: str) -> None:
        stamp = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
        cv2.imwrite(str(self.save_dir / f"{tag}_{stamp}.jpg"), frame)

    def _in_zone(self, x: int, y: int) -> bool:
        w, h = self.width, self.height
        return (0.5 - self.zone_frac) * w <= x <= (0.5 + self.zone_frac) * w and \
               (0.5 - self.zone_frac) * h <= y <= (0.5 + self.zone_frac) * h

    def run(self) -> None:
        def _open_camera_with_retry():
            last_error = "unknown camera error"
            for attempt in range(self.camera_retries + 1):
                try:
                    return open_camera(self.device, self.width, self.height)
                except RuntimeError as exc:
                    last_error = str(exc)
                    if attempt < self.camera_retries:
                        self.log(
                            f"[RING] camera open failed ({last_error}); "
                            f"retry {attempt + 1}/{self.camera_retries}"
                        )
                        time.sleep(self.camera_retry_sec)
            raise RuntimeError(last_error)

        try:
            cap = _open_camera_with_retry()
        except RuntimeError as exc:
            with self.lock:
                self.state = "error"
            self.log(f"monitor camera error: {exc}")
            return

        def _watchdog() -> None:
            # A V4L2 read can block forever (uninterruptible ioctl) when the
            # USB capture stalls; that used to hang the whole task silently
            # after cycle 1.  Detect the stall and abort the wait visibly.
            while not self._stop_flag:
                time.sleep(0.5)
                with self.lock:
                    st = self.state
                capture_started = self.capture_started_time
                if st in ("watch", "countdown") and capture_started and \
                        time.monotonic() - capture_started > 2.0:
                    stalled = time.monotonic() - capture_started
                    with self.lock:
                        if self.state in ("watch", "countdown"):
                            self.state = "error"
                    self.log(f"[RING] camera stalled: no frame for {stalled:.1f}s, "
                             f"aborting wait (replug the capture device if this repeats)")
                    return

        threading.Thread(target=_watchdog, daemon=True).start()

        prev = time.monotonic()
        while not self._stop_flag:
            time.sleep(0.1)
            now = time.monotonic()
            dt = now - prev
            prev = now
            with self.lock:
                state = self.state
            if state in ("paused", "error"):
                continue
            self.capture_started_time = time.monotonic()
            try:
                frame = capture_frame(cap)
            except RuntimeError as exc:
                self.capture_started_time = 0.0
                self.log(f"[RING] camera read failed: {exc}; reopening device")
                cap.release()
                try:
                    cap = _open_camera_with_retry()
                except RuntimeError as reopen_exc:
                    with self.lock:
                        self.state = "error"
                    self.log(f"monitor capture error after retries: {reopen_exc}")
                    break
                self.tracker.reset()
                self.log("[RING] camera reconnected")
                continue
            finally:
                self.capture_started_time = 0.0
            locked_r = self.tracker.locked[2] if self.tracker.locked else None
            circles, _, _ = detect_circles(
                frame, 15, self.height // 2, expected_radius=locked_r
            )
            in_zone = []
            if circles is not None:
                in_zone = [
                    c for c in circles[0]
                    if self._in_zone(int(round(c[0])), int(round(c[1])))
                ]
            # tracker keeps reporting the same ring instead of flipping
            # between concentric inner/outer detections
            best = self.tracker.update(in_zone)
            center = (int(round(best[0])), int(round(best[1]))) if best is not None else None

            with self.lock:
                if state == "watch":
                    if center is not None and not self.armed:
                        # an object already sits in the zone (e.g. the first
                        # ring at task start): arm after a bounded grace so
                        # the task still triggers on it
                        self.wait_t += dt
                        if self.wait_t >= 3.0:
                            self.armed = True
                            self.log("[RING] zone not clear for 3.0s, "
                                     "arming on the sitting object")
                    elif center is None:
                        self.wait_t += dt
                        self.stable_t = 0.0
                        self.last_center = None
                        self.empty_t += dt
                        if not self.armed and self.empty_t >= 0.3:
                            self.armed = True
                            self.log("[RING] zone clear, armed for next arrival")
                    elif self.armed:
                        if self.last_center is not None and math.dist(center, self.last_center) <= self.move_tol:
                            self.stable_t += dt
                        else:
                            self.stable_t = 0.0
                        self.last_center = center
                        if self.stable_t >= self.stable_sec:
                            self.frozen_center = center
                            self.state = "countdown"
                            self.countdown_start = now
                            self.countdown_remaining = self.countdown_sec
                            self.left_zone_logged = False
                            self.log(f"[RING] stable {self.stable_sec:.1f}s at {center}, "
                                     f"countdown {self.countdown_sec:.0f}s started")
                            self._save(frame, "stable")
                elif state == "countdown":
                    # The ring is being carried to the grasp point by the
                    # conveyor: leaving the zone is expected, never an abort.
                    self.countdown_remaining = self.countdown_sec - (now - self.countdown_start)
                    if center is None and not self.left_zone_logged:
                        self.left_zone_logged = True
                        self.log(f"[RING] ring left the detection zone "
                                 f"({self.countdown_remaining:.1f}s left) - in transport")
                    if self.countdown_remaining <= 0:
                        self.state = "trigger"
                        self.trigger_evt.set()
                        self.log(f"[RING] countdown finished, TRIGGER grasp "
                                 f"(zone circle last at {self.frozen_center})")
                        self._save(frame, "trigger")
            if state in ("watch", "countdown") and now - self._last_watch_save > 2.0:
                self._last_watch_save = now
                cv2.imwrite(str(self.save_dir / "watch_latest.jpg"), frame)
        cap.release()


class RingTaskNode(MoveToPoseNode):
    def __init__(self, args: argparse.Namespace) -> None:
        # map ring-task args onto the MoveToPoseNode constructor interface
        parent_args = argparse.Namespace(
            pose_file=args.pre_grasp, x=None, y=None, z=None, rpy=None,
            plan_only=args.plan_only, vx=args.vx, ax=args.ax,
            time=5.0, attempts=8, execution_timeout=120.0,
        )
        super().__init__(parent_args)  # pre-grasp pose loaded as the node target
        self.grasp_depth = float(args.grasp_depth)
        self.place_height = float(args.place_height)
        self.min_cart_frac = float(args.min_cartesian_fraction)
        self.cartesian_step = float(args.cartesian_step)
        self.motion_retries = max(1, int(args.motion_retries))
        self.recovery_wait = max(0.1, float(args.recovery_wait))
        self.home_joints = [float(v) for v in args.home_joints]
        self.gripper_open_pos = float(args.gripper_open)
        self.gripper_close_pos = float(args.gripper_close)
        self.place_good = load_pose_file(args.place_good)
        self.place_bad = load_pose_file(args.place_bad)
        self.cart_cli = self.create_client(
            GetCartesianPath, "/compute_cartesian_path", callback_group=self.cb
        )
        self.gripper_ac = ActionClient(
            self, GripperCommand, args.gripper_action, callback_group=self.cb
        )
        self.monitor = RingMonitor(
            self.get_logger().info,
            device=args.device,
            width=args.width,
            height=args.height,
            zone_frac=args.zone_frac,
            move_tol_px=args.move_tol_px,
            stable_sec=args.stable_sec,
            countdown_sec=args.countdown_sec,
            save_dir=Path(args.save_dir),
            camera_retries=args.camera_retries,
            camera_retry_sec=args.camera_retry_sec,
        )
        self.get_logger().info(
            f"ring task ready place_good_xyz={self.place_good[0]} "
            f"place_bad_xyz={self.place_bad[0]} "
            f"grasp_depth={self.grasp_depth} pre_grasp={args.pre_grasp}"
        )

    @staticmethod
    def classify_workpiece(occurrence: int) -> str:
        """Quality by order of appearance for now: odd = good, even = bad.

        Swap this for a real visual classifier when available.
        """
        return "good" if occurrence % 2 == 1 else "bad"

    # ---- hardware primitives ----
    def _call_gripper_once(self, open_gripper: bool, stage: str) -> bool:
        if self.plan_only:
            self.log(stage, result="SKIP_PLAN_ONLY")
            return True
        if not self.gripper_ac.wait_for_server(timeout_sec=5.0):
            self.log(stage, result="FAIL", reason="GripperCommand unavailable")
            return False
        target = self.gripper_open_pos if open_gripper else self.gripper_close_pos
        goal = GripperCommand.Goal()
        goal.command.position = float(target)
        goal.command.max_effort = 0.0
        future = self.gripper_ac.send_goal_async(goal)
        if not self.wait_future(future, 5.0):
            self.log(stage, result="FAIL", reason="goal timeout")
            return False
        gh = future.result()
        if not gh or not gh.accepted:
            self.log(stage, result="FAIL", reason="goal rejected")
            return False
        rf = gh.get_result_async()
        if not self.wait_future(rf, 10.0):
            cancel_future = gh.cancel_goal_async()
            self.wait_future(cancel_future, 2.0)
            self.log(stage, result="FAIL", reason="result timeout")
            return False
        wrapped = rf.result()
        if wrapped is None:
            self.log(stage, result="FAIL", reason="empty action result")
            return False
        result = wrapped.result
        ok = bool(result.reached_goal)
        self.log(stage, result="PASS" if ok else "FAIL",
                 target_m=target, feedback_m=float(result.position))
        return ok

    def call_gripper(self, open_gripper: bool, stage: str) -> bool:
        """Run an idempotent gripper command with bounded reconnect retries."""
        for attempt in range(1, self.motion_retries + 1):
            if self._call_gripper_once(open_gripper, stage):
                return True
            if attempt >= self.motion_retries:
                break
            self.get_logger().warn(
                f"{stage} attempt {attempt}/{self.motion_retries} failed; "
                f"waiting {self.recovery_wait:.1f}s for feedback/action recovery"
            )
            time.sleep(self.recovery_wait)
            if not self.wait_ready(timeout=max(5.0, self.recovery_wait + 2.0)):
                continue
            if not self.plan_only:
                self.last_joints = None
        return False

    def retry_motion(self, operation, stage: str) -> bool:
        """Retry a motion only after re-synchronising with the real arm.

        Replanning makes retries safe after a partial execution: the new
        trajectory starts at fresh joint feedback and targets the same pose.
        """
        for attempt in range(1, self.motion_retries + 1):
            if operation():
                return True
            if attempt >= self.motion_retries:
                break
            self.get_logger().warn(
                f"{stage} attempt {attempt}/{self.motion_retries} failed; "
                f"waiting {self.recovery_wait:.1f}s, then replanning from hardware"
            )
            time.sleep(self.recovery_wait)
            if not self.wait_ready(timeout=max(5.0, self.recovery_wait + 2.0)):
                continue
            if not self.plan_only:
                self.last_joints = None
        self.get_logger().error(
            f"{stage} failed after {self.motion_retries} attempt(s)"
        )
        return False

    def cartesian_move(self, pose, stage: str) -> bool:
        if not self.plan_only and not self.wait_for_fresh_joint_state():
            self.log(stage, planning="FAIL", reason="joint feedback offline")
            return False
        if not self.cart_cli.wait_for_service(timeout_sec=5.0):
            self.log(stage, planning="FAIL", reason="cartesian service unavailable")
            return False
        req = GetCartesianPath.Request()
        req.header.frame_id = "base_link"
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = "arm"
        req.link_name = "tool0"
        req.max_step = self.cartesian_step
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state.joint_state.name = list(ARM_JOINTS)
        req.start_state.joint_state.position = self.planning_start_joints()
        req.waypoints = [pose]
        fut = self.cart_cli.call_async(req)
        if not self.wait_future(fut, 30.0):
            self.log(stage, planning="FAIL", reason="service timeout")
            return False
        res = fut.result()
        if res is None:
            self.log(stage, planning="FAIL", reason="no response")
            return False
        frac = float(res.fraction)
        if frac < self.min_cart_frac:
            self.log(stage, planning="FAIL", fraction=frac,
                     target=(pose.position.x, pose.position.y, pose.position.z))
            return False
        end = self._traj_end_joints(res.solution)
        if self.plan_only:
            if end is not None:
                self.last_joints = list(end)
            self.log(stage, planning="PASS", execution="SKIP", fraction=frac)
            return True
        ok = self.execute_trajectory(res.solution, stage + "_EXEC")
        if ok and end is not None:
            self.last_joints = list(end)
        elif not ok and self.q is not None:
            self.last_joints = list(self.q)  # re-sync to reality after failure
        self.log(stage, planning="PASS", execution="PASS" if ok else "FAIL",
                 fraction=frac, actual_tcp=self.current_tcp())
        return ok

    def pose_at(self, xyz: Sequence[float], rpy: Sequence[float]) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in xyz)
        qx, qy, qz, qw = rpy_to_quat(*rpy)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
        return pose

    # ---- task ----
    def _wait_trigger(self) -> bool:
        while not self.monitor.trigger_evt.wait(0.2):
            state, _, _ = self.monitor.snapshot_state()
            if state == "error":
                return False
        return True

    def run_task(self, cycles: int) -> int:
        if not self.wait_ready():
            return 2
        if not self.plan_only and not self.gripper_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("gripper action unavailable")
            return 2
        pre_grasp = self.target_pose()
        pre_rpy = list(self.target_rpy)
        pre_xyz = list(self.target_position)

        self.log("GOTO_PRE_GRASP", target=pre_xyz)
        if not self.retry_motion(
            lambda: self.goto_pose(pre_grasp, "GOTO_PRE_GRASP"),
            "GOTO_PRE_GRASP",
        ):
            return 1
        self.monitor.start()
        cycle = 0
        try:
            while cycles <= 0 or cycle < cycles:
                occurrence = cycle + 1
                quality = self.classify_workpiece(occurrence)
                self.log("WAIT_RING", occurrence=occurrence, quality=quality)
                self.monitor.resume()
                if not self._wait_trigger():
                    self.get_logger().error("monitor failed, aborting")
                    return 1
                self.monitor.pause()

                if not self.call_gripper(True, "OPEN_GRIPPER"):
                    return 1
                pre_approach_joints = self.planning_start_joints()
                grasp_xyz = [pre_xyz[0], pre_xyz[1], pre_xyz[2] - self.grasp_depth]
                if not self.retry_motion(
                    lambda: self.cartesian_move(
                        self.pose_at(grasp_xyz, pre_rpy), "APPROACH"
                    ),
                    "APPROACH",
                ):
                    return 1
                if not self.call_gripper(False, "CLOSE_GRIPPER"):
                    return 1
                # lift retraces the descend in joint space: the vertical
                # cartesian line stalls near joint limits in reverse
                if not self.retry_motion(
                    lambda: self.move_joints(pre_approach_joints, "LIFT"), "LIFT"
                ):
                    return 1

                place_xyz, place_rpy, place_joints = (
                    self.place_good if quality == "good" else self.place_bad
                )
                pre_place_xyz = [place_xyz[0], place_xyz[1], place_xyz[2] + self.place_height]
                self.log("MOVE_PRE_PLACE", quality=quality, xyz=pre_place_xyz)
                # The taught place joint snapshot is a much better IK branch
                # seed than the generic pre-grasp snapshot.
                previous_seed = self.extra_seed
                if place_joints is not None:
                    self.extra_seed = list(place_joints)
                try:
                    if not self.retry_motion(
                        lambda: self.goto_pose(
                            self.pose_at(pre_place_xyz, place_rpy), "MOVE_PRE_PLACE"
                        ),
                        "MOVE_PRE_PLACE",
                    ):
                        return 1
                finally:
                    self.extra_seed = previous_seed
                pre_descend_joints = self.planning_start_joints()
                if not self.retry_motion(
                    lambda: self.cartesian_move(
                        self.pose_at(place_xyz, place_rpy), "DESCEND"
                    ),
                    "DESCEND",
                ):
                    return 1
                if not self.call_gripper(True, "RELEASE"):
                    return 1
                if not self.retry_motion(
                    lambda: self.move_joints(pre_descend_joints, "RETREAT"),
                    "RETREAT",
                ):
                    return 1
                if not self.retry_motion(
                    lambda: self.goto_pose(pre_grasp, "RETURN_PRE_GRASP"),
                    "RETURN_PRE_GRASP",
                ):
                    return 1
                cycle += 1
                self.log("CYCLE_DONE", completed=cycle, quality=quality,
                         tcp=self.current_tcp())
        except KeyboardInterrupt:
            self.get_logger().info("interrupted by user")
        finally:
            self.monitor.stop()
        if cycles > 0 and cycle >= cycles:
            # finite run finished: return to home and close the gripper
            self.log("TASK_FINISH", completed=cycle)
            if not self.retry_motion(
                lambda: self.move_joints(self.home_joints, "RETURN_HOME"),
                "RETURN_HOME",
            ):
                return 1
            if not self.call_gripper(False, "FINAL_CLOSE_GRIPPER"):
                return 1
            self.log("HOME_DONE", joints=self.home_joints)
        self.log("TASK_DONE", completed_cycles=cycle)
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pre-grasp", default=DEFAULT_PRE_GRASP,
                   help="pre-grasp pose yaml above the ring (default: recorded snapshot)")
    p.add_argument("--place-good", default="",
                   help="place point for GOOD workpieces yaml (record_pose.py)")
    p.add_argument("--place-bad", default="",
                   help="place point for BAD workpieces yaml (record_pose.py)")
    p.add_argument("--grasp-depth", type=float, default=0.04,
                   help="descend depth below pre-grasp z for grasping (m)")
    p.add_argument("--place-height", type=float, default=0.08,
                   help="approach height above place point (m)")
    p.add_argument("--min-cartesian-fraction", type=float, default=0.95)
    p.add_argument("--cartesian-step", type=float, default=0.005,
                   help="cartesian interpolation step in metres (smaller is more robust)")
    p.add_argument("--motion-retries", type=int, default=3,
                   help="bounded attempts for planning/execution and gripper recovery")
    p.add_argument("--recovery-wait", type=float, default=2.5,
                   help="seconds to wait for adapter/feedback recovery before retry")
    p.add_argument("--gripper-action", default="/x5a_gripper_controller/gripper_cmd")
    p.add_argument("--gripper-open", type=float, default=0.044)
    p.add_argument("--gripper-close", type=float, default=0.0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--camera-retries", type=int, default=5,
                   help="attempts to reopen the USB camera after a read/open failure")
    p.add_argument("--camera-retry-sec", type=float, default=1.0,
                   help="seconds between USB camera reopen attempts")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--zone-frac", type=float, default=0.25,
                   help="detection zone = frame centre +-this fraction")
    p.add_argument("--move-tol-px", type=float, default=12.0,
                   help="circle may drift up to this many px and still count as stopped")
    p.add_argument("--stable-sec", type=float, default=0.65)
    p.add_argument("--countdown-sec", type=float, default=6.0)
    p.add_argument("--save-dir", default="/home/quella/arx/arm/logs/ring_task")
    p.add_argument("--cycles", type=int, default=0, help="0 = run until Ctrl-C")
    p.add_argument("--plan-only", action="store_true",
                   help="full loop with detection and planning, no execution")
    p.add_argument("--monitor-only", action="store_true",
                   help="only run the camera FSM (no ROS, no arm); exits after first trigger")
    p.add_argument("--vx", type=float, default=0.70,
                   help="velocity scaling for joint-space moves (grasp "
                        "descent is cartesian and unaffected)")
    p.add_argument("--ax", type=float, default=0.25)
    p.add_argument("--home-joints", type=float, nargs=6,
                   default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   help="home joint angles used after the last cycle")
    return p.parse_args()


def run_monitor_only(args: argparse.Namespace) -> int:
    mon = RingMonitor(
        print,
        device=args.device,
        width=args.width,
        height=args.height,
        zone_frac=args.zone_frac,
        move_tol_px=args.move_tol_px,
        stable_sec=args.stable_sec,
        countdown_sec=args.countdown_sec,
        save_dir=Path(args.save_dir),
        camera_retries=args.camera_retries,
        camera_retry_sec=args.camera_retry_sec,
    )
    mon.start()
    mon.resume()
    print(f"monitor-only: watching detection zone "
          f"(centre +-{args.zone_frac:.0%}), Ctrl-C to stop", flush=True)
    try:
        while not mon.trigger_evt.wait(0.2):
            st, _, _ = mon.snapshot_state()
            if st == "error":
                print("monitor error (camera unavailable?) - exiting")
                return 2
        print("TRIGGER fired - FSM verified (stable -> countdown -> grasp trigger)")
        return 0
    except KeyboardInterrupt:
        print("stopped by user")
        return 0
    finally:
        mon.stop()
        mon.join(timeout=2.0)


def main() -> int:
    args = parse_args()
    if args.monitor_only:
        return run_monitor_only(args)
    if not args.place_good or not args.place_bad:
        print("ERROR: --place-good and --place-bad are required "
              "(teach them with record_pose.py first)", file=sys.stderr)
        return 2
    rclpy.init()
    node = RingTaskNode(args)
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        code = node.run_task(args.cycles)
    except Exception as exc:
        node.get_logger().error(f"ring task exception: {exc}")
        code = 1
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
    sys.exit(code)


if __name__ == "__main__":
    main()
