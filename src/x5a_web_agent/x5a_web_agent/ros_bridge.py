"""ROS 2 Action client for structured pick-place tasks.

This process never publishes /arm_cmd, joint commands, or trajectories.
It only talks to /x5a/pick_place and reads /x5a/task_stage.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from x5a_task_interfaces.action import PlanRemoteGrasp, X5aPickPlace


FeedbackCallback = Callable[[str], None]
BUSY_MESSAGE = "机器人正在执行上一任务，请稍后。"
ALLOWED_COLORS = ("red", "white", "orange", "nearest", "farthest")
TERMINAL_STAGES = frozenset({"SUCCESS", "FAILED", "IDLE", "DRY_RUN"})


class RosBridge:
    def __init__(
        self,
        dry_run: bool = False,
        action_name: str = "/x5a/pick_place",
    ) -> None:
        self.dry_run = bool(dry_run)
        self.action_name = action_name
        self._lock = threading.Lock()
        self._busy = False
        self._node: Optional[Node] = None
        self._executor: Optional[MultiThreadedExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[ActionClient] = None
        self._goal_handle = None
        self._rclpy_owned = False
        self._stage = "IDLE"
        self._server_busy = False
        self._stage_cb: Optional[FeedbackCallback] = None

    def set_stage_callback(self, callback: Optional[FeedbackCallback]) -> None:
        self._stage_cb = callback

    @property
    def current_stage(self) -> str:
        return self._stage

    def start(self) -> None:
        if self._node is not None:
            return
        if not rclpy.ok():
            rclpy.init(args=None)
            self._rclpy_owned = True
        self._node = Node("x5a_web_agent")
        self._client = ActionClient(self._node, X5aPickPlace, self.action_name)
        self._remote_client = ActionClient(
            self._node, PlanRemoteGrasp, "/x5a_remote_grasp/plan"
        )
        self._execute_client = self._node.create_client(
            Trigger, "/x5a_remote_grasp/execute"
        )
        self._go_home_client = self._node.create_client(
            Trigger, "/x5a_remote_grasp/go_home"
        )
        self._go_ready_client = self._node.create_client(
            Trigger, "/x5a_remote_grasp/go_ready"
        )
        self._release_client = self._node.create_client(
            Trigger, "/x5a_remote_grasp/release"
        )
        self._state_client = self._node.create_client(
            Trigger, "/x5a_remote_grasp/executor_state"
        )
        self._node.create_subscription(
            String, "/x5a/task_stage", self._on_stage_topic, 10
        )
        self._node.create_subscription(
            Bool, "/x5a/task_busy", self._on_busy_topic, 10
        )
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        self._node.get_logger().info(
            f"web agent ROS bridge ready action={self.action_name} dry_run={self.dry_run}"
        )

    def _on_stage_topic(self, msg: String) -> None:
        stage = str(msg.data or "").strip() or "IDLE"
        self._stage = stage
        callback = self._stage_cb
        if callback is not None:
            try:
                callback(stage)
            except Exception:
                pass

    def _on_busy_topic(self, msg: Bool) -> None:
        self._server_busy = bool(msg.data)
        if not msg.data and self._stage in TERMINAL_STAGES:
            callback = self._stage_cb
            if callback is not None:
                try:
                    callback(self._stage or "IDLE")
                except Exception:
                    pass

    def shutdown(self) -> None:
        try:
            if self._executor is not None:
                self._executor.shutdown()
        except Exception:
            pass
        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        self._node = None
        self._client = None
        if self._rclpy_owned and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

    @property
    def busy(self) -> bool:
        with self._lock:
            local_busy = self._busy
        if self._stage in TERMINAL_STAGES and not self._server_busy:
            return False
        return local_busy

    def server_ready(self, timeout: float = 0.5) -> bool:
        if self.dry_run:
            return True
        if self._client is None:
            return False
        return bool(self._client.wait_for_server(timeout_sec=timeout))

    def cancel(self) -> bool:
        handle = self._goal_handle
        if handle is None:
            return False
        try:
            handle.cancel_goal_async()
            return True
        except Exception:
            return False

    def send_pick_place(
        self,
        target_color: str,
        on_feedback: Optional[FeedbackCallback] = None,
        timeout: float = 180.0,
        skip_return_home: bool = False,
        reuse_frozen_box: bool = False,
    ) -> Dict[str, Any]:
        color = str(target_color or "").strip().lower()
        if color not in ALLOWED_COLORS:
            return {
                "success": False,
                "message": f"invalid target_color: {target_color!r}",
                "stage": "FAILED",
            }
        if self.dry_run:
            if on_feedback is not None:
                on_feedback("DRY_RUN")
            return {
                "success": True,
                "message": "DRY_RUN: 已解析任务，未发送 ROS Action",
                "stage": "DRY_RUN",
            }

        with self._lock:
            if self._busy:
                return {
                    "success": False,
                    "message": BUSY_MESSAGE,
                    "stage": "BUSY",
                }
            self._busy = True

        try:
            return self._send_locked(
                color,
                on_feedback,
                timeout,
                skip_return_home=skip_return_home,
                reuse_frozen_box=reuse_frozen_box,
            )
        finally:
            with self._lock:
                self._busy = False
                self._goal_handle = None

    def _send_locked(
        self,
        color: str,
        on_feedback: Optional[FeedbackCallback],
        timeout: float,
        skip_return_home: bool = False,
        reuse_frozen_box: bool = False,
    ) -> Dict[str, Any]:
        if self._client is None or self._node is None:
            return {
                "success": False,
                "message": "ROS bridge 未启动",
                "stage": "FAILED",
            }
        if not self._client.wait_for_server(timeout_sec=2.0):
            return {
                "success": False,
                "message": "Task Server 未就绪（/x5a/pick_place）",
                "stage": "FAILED",
            }

        last_stage = "ACCEPTED"

        def _note(stage: str) -> None:
            nonlocal last_stage
            last_stage = stage
            if on_feedback is not None:
                on_feedback(stage)

        prev_cb = self._stage_cb

        def _topic_and_user(stage: str) -> None:
            _note(stage)
            if prev_cb is not None and prev_cb is not on_feedback:
                prev_cb(stage)

        self._stage_cb = _topic_and_user

        try:
            goal = X5aPickPlace.Goal()
            goal.target_color = color
            goal.skip_return_home = bool(skip_return_home)
            goal.reuse_frozen_box = bool(reuse_frozen_box)
            send_future = self._client.send_goal_async(goal)
            if not self._wait(send_future, 5.0):
                return {
                    "success": False,
                    "message": "发送 Action Goal 超时",
                    "stage": "FAILED",
                }
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                return {
                    "success": False,
                    "message": BUSY_MESSAGE,
                    "stage": "BUSY",
                }
            self._goal_handle = goal_handle
            result_future = goal_handle.get_result_async()
            if not self._wait(
                result_future,
                timeout,
                extra_done=lambda: last_stage in TERMINAL_STAGES
                or self._stage in TERMINAL_STAGES,
            ):
                stage = last_stage if last_stage in TERMINAL_STAGES else self._stage
                if stage in TERMINAL_STAGES:
                    return {
                        "success": stage == "SUCCESS",
                        "message": "任务完成" if stage == "SUCCESS" else "任务失败",
                        "stage": stage,
                    }
                return {
                    "success": False,
                    "message": "等待 Task Server 结果超时",
                    "stage": stage or "FAILED",
                }
            wrapped = result_future.result()
            result = wrapped.result
            return {
                "success": bool(result.success),
                "message": result.message or ("任务完成" if result.success else "任务失败"),
                "stage": "SUCCESS" if result.success else (last_stage or "FAILED"),
            }
        finally:
            self._stage_cb = prev_cb

    @staticmethod
    def _parse_trigger_message(text: str) -> Dict[str, Any]:
        """执行器把真实执行结果放在 Trigger 的 message 里（JSON 字符串）。

        老版本返回纯文本，解析失败时返回空字典，调用方回退到 success 字段。
        """
        payload_text = str(text or "").strip()
        if not payload_text.startswith("{"):
            return {}
        try:
            payload = json.loads(payload_text)
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _call_trigger(self, client, name: str, timeout: float = 60.0) -> Dict[str, Any]:
        if self.dry_run:
            return {"success": True, "message": f"DRY_RUN: 已跳过 {name}", "stage": "DRY_RUN"}
        if not client.wait_for_service(timeout_sec=2.0):
            return {"success": False, "message": f"{name} 服务未就绪", "stage": "FAILED"}
        done = threading.Event()
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            return {"success": False, "message": f"{name} 响应超时", "stage": "FAILED"}
        response = future.result()
        payload = self._parse_trigger_message(str(response.message or ""))
        return {
            "success": bool(response.success),
            "message": str(payload.get("message") or response.message or ""),
            "stage": str(payload.get("stage") or ("SUCCESS" if response.success else "FAILED")),
            "holding": payload.get("holding"),
            "raw": str(response.message or ""),
        }

    def execute_latest(self, timeout: float = 120.0) -> Dict[str, Any]:
        """Trigger the remote grasp executor (real motion) and WAIT for the real
        outcome: the executor blocks until the sequence ends and reports the
        stage plus whether the gripper is now holding an object."""
        result = self._call_trigger(self._execute_client, "抓取执行", timeout=timeout)
        if "执行器未就绪" in result.get("message", ""):
            result["message"] = "执行器未就绪（/x5a_remote_grasp/execute，终端 7 是否以 execution.enabled:=true 启动？）"
        return result

    def send_remote_place(
        self,
        target: str,
        relation: str = "on",
        on_feedback: Optional[FeedbackCallback] = None,
        timeout: float = 120.0,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Plan a place-only sequence for the object already in the gripper.

        Uses the same PlanRemoteGrasp action with a place-only prompt, which the
        node recognises (no grasp target → skip /v1/grasp, plan TRANSPORT/PLACE/
        RETREAT from the current pose)."""
        prompt = str(target or "").strip()
        if not prompt:
            return {"success": False, "message": "放置指令缺少目标物名称", "stage": "FAILED"}
        place_prompt = f"place: {prompt}; prel: {relation or 'on'}"
        return self.send_remote_grasp(
            place_prompt,
            on_feedback=on_feedback,
            timeout=timeout,
            top_k=top_k,
            plan_motion=True,
        )

    def release(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Open the gripper in place (drop the held object)."""
        return self._call_trigger(self._release_client, "放下(release)", timeout=timeout)

    def executor_state(self, timeout: float = 10.0) -> Dict[str, Any]:
        """Read the executor's own state: {enabled, running, holding, ...}.

        Used to re-sync the agent session after a web_agent restart."""
        result = self._call_trigger(self._state_client, "执行器状态", timeout=timeout)
        payload = self._parse_trigger_message(result.get("raw", ""))
        result.update(payload)
        return result

    def pose_home(self) -> Dict[str, Any]:
        return self._call_trigger(self._go_home_client, "归位(go_home)")

    def pose_ready(self) -> Dict[str, Any]:
        return self._call_trigger(self._go_ready_client, "预抓取就绪(go_ready)")

    @staticmethod
    def _wait(future, timeout: float, extra_done=None) -> bool:
        deadline = time.monotonic() + timeout
        seen_terminal = None
        while not future.done() and time.monotonic() < deadline:
            if extra_done is not None and extra_done():
                if seen_terminal is None:
                    seen_terminal = time.monotonic()
                elif time.monotonic() - seen_terminal >= 0.3:
                    return future.done()
            time.sleep(0.02)
        return future.done()

    def send_remote_grasp(
        self,
        prompt: str,
        on_feedback: Optional[FeedbackCallback] = None,
        timeout: float = 120.0,
        top_k: int = 20,
        plan_motion: bool = False,
    ) -> Dict[str, Any]:
        """Plan-only remote SAM + GraspNet grasp. Never moves the arm: the
        /x5a_remote_grasp node is plan-only by design (safety.plan_only=true)."""
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"success": False, "message": "远程抓取缺少目标物名称", "stage": "FAILED"}
        if self.dry_run:
            if on_feedback is not None:
                on_feedback("DRY_RUN")
            return {
                "success": True,
                "message": f"DRY_RUN: 远程抓取规划已解析（prompt={prompt}），未发送 ROS Action",
                "stage": "DRY_RUN",
            }
        with self._lock:
            if self._busy:
                return {"success": False, "message": BUSY_MESSAGE, "stage": "BUSY"}
            self._busy = True
        try:
            return self._send_remote_locked(
                prompt, on_feedback, timeout, top_k, plan_motion
            )
        finally:
            with self._lock:
                self._busy = False
                self._goal_handle = None

    def _send_remote_locked(
        self,
        prompt: str,
        on_feedback: Optional[FeedbackCallback],
        timeout: float,
        top_k: int,
        plan_motion: bool,
    ) -> Dict[str, Any]:
        if self._node is None or self._remote_client is None:
            return {"success": False, "message": "ROS bridge 未启动", "stage": "FAILED"}
        if not self._remote_client.wait_for_server(timeout_sec=5.0):
            # 5 秒而不是 2 秒：节点刚重启时 action 发现可能还没完成，误报
            # 「未就绪」会让用户以为服务挂了
            return {
                "success": False,
                "message": "远程抓取节点未就绪（/x5a_remote_grasp/plan）",
                "stage": "FAILED",
            }

        def _feedback_cb(wrapper) -> None:
            feedback = getattr(wrapper, "feedback", None)
            if feedback is None or on_feedback is None:
                return
            stage = str(getattr(feedback, "stage", "") or "")
            detail = str(getattr(feedback, "detail", "") or "")
            try:
                on_feedback(f"{stage}:{detail}" if detail else stage)
            except Exception:
                pass

        goal = PlanRemoteGrasp.Goal()
        goal.prompt = prompt
        goal.top_k = int(top_k)
        goal.plan_motion = bool(plan_motion)
        send_future = self._remote_client.send_goal_async(
            goal, feedback_callback=_feedback_cb
        )
        if not self._wait(send_future, 5.0):
            return {"success": False, "message": "发送远程抓取 Goal 超时", "stage": "FAILED"}
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {"success": False, "message": BUSY_MESSAGE, "stage": "BUSY"}
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        if not self._wait(result_future, timeout):
            return {"success": False, "message": "等待远程抓取规划结果超时", "stage": "FAILED"}
        wrapped = result_future.result()
        result = wrapped.result
        error_message = str(getattr(result, "error_message", "") or "")
        candidates = int(getattr(result, "candidate_count", 0) or 0)
        if not bool(result.success):
            # 安全门内的「成功」：候选和 TF 已发布，只是 tcp_mapping_confirmed=false
            # 停在 IK 之前。这不算失败，向用户如实说明状态。
            if candidates > 0 and "unconfirmed" in error_message:
                return {
                    "success": True,
                    "message": (
                        f"云端分割与抓取候选已发布到 RViz（{candidates} 个候选）。"
                        "TCP 映射未确认，IK 与执行未运行，机械臂不会动。"
                    ),
                    "stage": "SUCCESS",
                }
            return {
                "success": False,
                "message": error_message or "远程抓取规划失败",
                "stage": "FAILED",
            }
        candidates = int(result.candidate_count)
        score = float(result.score)
        plan_note = "，MoveIt 规划已生成" if result.plan_valid else "（仅候选可视化，未做 IK/规划）"
        stages = ",".join(str(item) for item in getattr(result, "planned_stages", []) or [])
        if candidates <= 0 and "PLACE" in stages:
            # 纯放置计划：本来就没有抓取候选，别报成「没有可用候选」
            return {
                "success": True,
                "message": (
                    f"放置规划完成：{stages}"
                    + ("，MoveIt 规划已生成" if result.plan_valid else "（未做 IK/规划）")
                ),
                "stage": "SUCCESS",
            }
        return {
            "success": True,
            "message": (
                f"远程抓取规划完成：{candidates} 个候选，"
                f"选中 #{result.selected_index} score={score:.3f}{plan_note}"
                if candidates
                else "远程抓取完成，但没有可用候选"
            ),
            "stage": "SUCCESS",
        }
