#!/usr/bin/env python3
"""Long-running Action server that reuses the verified pick-place cycle."""
from __future__ import annotations

import sys
import threading

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, String
from x5a_pick_place.pick_place_node import ALLOWED_TARGETS, PickPlaceNode
from x5a_task_interfaces.action import X5aPickPlace
from x5a_task_server.mqtt_link import publish_task_finished

TERMINAL_STAGES = frozenset({"SUCCESS", "FAILED"})


class X5aTaskServer(PickPlaceNode):
    def __init__(self) -> None:
        super().__init__(node_name="x5a_task_server")
        self._busy = False
        self._busy_lock = threading.Lock()
        self._mqtt_lock = threading.Lock()
        self._mqtt_completion_pending = False
        self._mqtt_defer_success = False
        self._last_stage = "IDLE"
        self._action_cb = ReentrantCallbackGroup()
        self.set_stage_callback(self._on_stage)
        self.stage_pub = self.create_publisher(String, "/x5a/task_stage", 10)
        self.busy_pub = self.create_publisher(Bool, "/x5a/task_busy", 10)
        self.create_timer(0.5, self._republish_status)
        self._action_server = ActionServer(
            self,
            X5aPickPlace,
            "/x5a/pick_place",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._action_cb,
        )
        self._set_idle("IDLE")
        self.get_logger().info(
            "X5A TASK SERVER: READY action=/x5a/pick_place "
            f"plan_only={self.plan_only} vision={self.vision_enabled}"
        )

    def _set_idle(self, stage: str) -> None:
        with self._busy_lock:
            self._busy = False
        self._publish_stage(stage)
        self._publish_busy(False)
        if stage in TERMINAL_STAGES:
            with self._mqtt_lock:
                pending = self._mqtt_completion_pending
                defer_success = self._mqtt_defer_success
                self._mqtt_completion_pending = False
                self._mqtt_defer_success = False
            if pending and not (stage == "SUCCESS" and defer_success):
                publish_task_finished(self.get_logger())
        self.get_logger().info(f"X5A TASK SERVER: IDLE ({stage})")

    def goal_callback(self, goal_request: X5aPickPlace.Goal) -> GoalResponse:
        color = str(goal_request.target_color or "").strip().lower()
        if color not in ALLOWED_TARGETS:
            self.get_logger().warn(f"rejecting unsupported target_color={color!r}")
            return GoalResponse.REJECT
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn(
                    f"rejecting {color}: previous task still running"
                )
                return GoalResponse.REJECT
            self._busy = True
        with self._mqtt_lock:
            self._mqtt_completion_pending = True
            self._mqtt_defer_success = bool(goal_request.skip_return_home)
        self._publish_busy(True)
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle) -> CancelResponse:
        self.get_logger().warn("cancel requested; will stop after the current motion")
        self.request_cancel()
        return CancelResponse.ACCEPT

    def _publish_stage(self, stage: str) -> None:
        self._last_stage = stage
        msg = String()
        msg.data = stage
        self.stage_pub.publish(msg)

    def _publish_busy(self, busy: bool) -> None:
        msg = Bool()
        msg.data = bool(busy)
        self.busy_pub.publish(msg)

    def _republish_status(self) -> None:
        self._publish_stage(self._last_stage)
        with self._busy_lock:
            busy = self._busy
        if self._last_stage in TERMINAL_STAGES:
            busy = False
        self._publish_busy(busy)

    def _on_stage(self, stage: str) -> None:
        # Clear the lock as soon as pick-place reports a terminal stage.
        # Do not wait for the Action execute_callback to return.
        if stage in TERMINAL_STAGES:
            self._set_idle(stage)
            return
        self._publish_stage(stage)
        self._publish_busy(True)

    def execute_callback(self, goal_handle):
        """Synchronous execute. Do not use asyncio — rclpy has no asyncio loop."""
        result = X5aPickPlace.Result()
        req = goal_handle.request
        color = str(req.target_color or "").strip().lower()
        try:
            ok, message = self.execute_pick_place(
                color,
                skip_return_home=bool(req.skip_return_home),
                reuse_frozen_box=bool(req.reuse_frozen_box),
            )
            result.success = bool(ok)
            result.message = message
            if ok:
                self._set_idle("SUCCESS")
                goal_handle.succeed()
            elif goal_handle.is_cancel_requested or message == "任务已取消":
                self._set_idle("FAILED")
                goal_handle.canceled()
            else:
                self._set_idle("FAILED")
                goal_handle.abort()
            return result
        except Exception as exc:
            self.get_logger().error(f"task server exception: {exc}")
            result.success = False
            result.message = f"task exception: {exc}"
            self._set_idle("FAILED")
            try:
                goal_handle.abort()
            except Exception:
                pass
            return result
        finally:
            with self._busy_lock:
                self._busy = False
            self._publish_busy(False)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = X5aTaskServer()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)

    def spin():
        executor.spin()

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    if not node.wait_ready():
        node.get_logger().error("dependencies not ready")
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(2)
    node.log_calibration()
    node.get_logger().info("X5A TASK SERVER: WAITING FOR GOALS")
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
