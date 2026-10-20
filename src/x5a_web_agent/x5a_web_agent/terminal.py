"""终端文字输入层：用键盘代替语音转文字，接入手机语音用的那条链路。

手机链路是 Web Speech API / faster-whisper -> handle_text_command -> 云端解析
-> ROS 2 Action。这一层只替换最前面的「语音转文字」，后面的解析、白名单检查、
数字人握手和执行完全复用，既不新增 /arm_cmd 出口，也不自己规划轨迹。

两种用法：

- ``web_agent --terminal``：终端层跑在 agent 进程里，直接调
  ``handle_text_command``，和手机说话进的是同一个函数。
- ``ros2 run x5a_web_agent text_cli``：独立进程，通过 ``/api/command`` 接入
  已经在跑的 agent，测试时不用重启服务。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

from x5a_web_agent.command_parser import ALLOWED_TARGETS, parse_command
from x5a_web_agent.task_sequence import describe_task

# 任务真的结束了才值得单独打一行结果，ACCEPTED / 各运动阶段只打阶段行。
RESULT_STAGES = frozenset({"SUCCESS", "FAILED", "CANCELED", "DRY_RUN"})

# 结果行前面标出这条结果属于哪个动作：真机链路是异步的，结果可能在你敲下
# 下一条指令之后才回来，不标一下就像新指令「不动却完成了」。
ACTION_ZH = {
    "remote_grasp": "抓取",
    "remote_place": "放置",
    "release": "放下",
    "pick_place": "抓放",
    "prepare": "准备",
    "go_home": "归位",
    "go_ready": "就位",
    "sequence": "序列",
    "cancel": "取消",
}

HELP_TEXT = """SAM 腕部抓放：
  准备                    移动到观察位，不抓取
  回原点 / 归原位 / 返回原位 返回原始收起姿态
  抓红色方块              抓住后保持夹持，等待放置指令
  放到白色方块上          放置手里的物体
  把白色方块放到橙色方块上，然后把红色方块放到白色方块上
                          按顺序执行，每步重新观察和规划
  松开夹爪                原地松开，之后重新准备
  取消                    当前动作结束后停止任务
  /state /help /cancel /task red /quit
"""


class TerminalBackend:
    """终端层后端：把一行文字交给语音链路用的那个入口。

    返回 dict 的话由终端层打印（例如被 BUSY 拒绝时的错误）；返回 None 表示
    结果会通过状态广播/polling 异步到达，不用在这里等。
    """

    name = "?"

    def submit_text(self, text: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def submit_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def cancel(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def state(self) -> Dict[str, Any]:
        raise NotImplementedError


class TerminalConsole:
    """stdin 一行 -> 语音链路；同时把广播回来的阶段/结果打到终端。"""

    def __init__(
        self,
        backend: TerminalBackend,
        *,
        prompt: str = "x5a> ",
        show_help: bool = True,
    ) -> None:
        self.backend = backend
        self.prompt = prompt
        self.show_help = show_help
        self._lock = threading.Lock()
        self._last_stage = "IDLE"
        self._last_message = ""
        self._context = ''
        self.verbose = os.environ.get('X5A_VERBOSE_TASK_LOGS', '0') == '1'

    # ------------------------------------------------------------------ 输出
    def _line(self, tag: str, text: str) -> None:
        message = str(text or "").strip()
        if not message:
            return
        with self._lock:
            print(f"{time.strftime('%H:%M:%S')} {self._context}[{tag}] {message}", flush=True)

    def note(self, text: str) -> None:
        self._line("提示", text)

    def error(self, text: str) -> None:
        self._line("错误", text)

    def _message_once(self, tag: str, text: str, label: str = "") -> None:
        """同一句话只打一次：state 里的 last_result 和随后的 result 广播内容相同。"""
        message = str(text or "").strip()
        key = (self._context, message)
        if not message or key == self._last_message:
            return
        self._last_message = key
        if label:
            message = f"{label}: {message}"
        self._line(tag, message)

    @staticmethod
    def _payload_label(payload: Dict[str, Any]) -> str:
        """从 parsed 里取一个短标签，例如「抓取『black pen』」「归位」。"""
        parsed = payload.get("parsed")
        if not isinstance(parsed, dict):
            return ""
        action = str(parsed.get("action") or "").strip()
        if not action:
            return ""
        return describe_task(parsed)

    def _set_context(self, payload):
        number = payload.get('command_id')
        if number is None:
            return
        context = f'[任务 {number} · {payload.get("step_index", 1)}/{payload.get("step_total", 1)}] '
        if context != self._context:
            self._context = context
            self._last_stage = ''
            self._last_message = ''

    def on_payload(self, payload: Any) -> None:
        """同时是 hub 广播的观察者，和 HTTP 响应的打印入口。"""
        if not isinstance(payload, dict):
            return
        self._set_context(payload)
        kind = str(payload.get("type") or "").strip()
        if kind == "transcript":
            return  # 就是刚刚敲进去的那句话，不用回显
        if kind == "parsed":
            self._context = ''
            self._print_parsed(payload.get("result"))
            return
        if kind == 'step':
            self._line('步骤', str(payload.get('message') or ''))
            return
        label = self._payload_label(payload)
        if payload.get("message"):
            if kind == "error" or payload.get("ok") is False:
                tag = "错误"
            elif kind == "result":
                tag = "结果"
            else:
                tag = "提示"
            message = str(payload.get('message'))
            if kind == 'result':
                duration = payload.get('duration_s')
                planning = payload.get('planning_s')
                if duration is not None:
                    message += (f'（规划 {planning:.1f}s / 本步 {duration:.1f}s）'
                                if planning is not None else f'（总耗时 {duration:.1f}s）')
                if payload.get('holding') is True:
                    message += '；仍在持物'
                elif payload.get('holding') is False:
                    message += '；夹爪已释放'
            self._message_once(tag, message, label if kind != 'result' and not self._context else '')
        stage = str(payload.get("stage") or "").strip()
        if stage:
            self._print_stage(stage, payload)

    def on_state(self, state: Any) -> None:
        """轮询 /api/state 拿到的快照（没有 type 字段）。"""
        if not isinstance(state, dict):
            return
        self._set_context(state)
        if state.get("message") or state.get("type"):
            self.on_payload(state)
            return
        stage = str(state.get("stage") or "").strip()
        if stage:
            self._print_stage(stage, state)
        elif state:
            self.print_state(state)

    def _print_stage(self, stage: str, payload: Dict[str, Any]) -> None:
        if stage == 'CANDIDATE_REJECT' and not self.verbose:
            return  # per-candidate diagnostics are retained in backend logs
        if stage == self._last_stage:
            return
        self._last_stage = stage
        if stage not in RESULT_STAGES:
            elapsed = payload.get('elapsed_s')
            suffix = f'（+{elapsed:.1f}s）' if elapsed is not None else ''
            self._line("阶段", str(payload.get("stage_zh") or stage)+suffix)
        if stage not in RESULT_STAGES:
            return
        result = payload.get("last_result")
        if isinstance(result, dict) and not payload.get('busy'):
            self._message_once(
                "结果", str(result.get("message") or ""), self._payload_label(payload)
            )

    def _print_parsed(self, parsed: Any) -> None:
        if not isinstance(parsed, dict):
            return
        if parsed.get('action') == 'sequence':
            tasks = parsed.get('tasks', [])
            self._line('解析', f'连续任务，共 {len(tasks)} 步')
            for index, task in enumerate(tasks, 1):
                self._line('计划', f'{index}. {describe_task(task)}')
        else:
            self._line('解析', describe_task(parsed) if parsed.get('action') != 'unknown'
                       else str(parsed.get('message') or '无法理解指令'))
        if self.verbose:
            self._line('解析详情', json.dumps(parsed, ensure_ascii=False, sort_keys=True))

    def print_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or not state:
            self.error("拿不到 agent 状态")
            return
        held = state.get("held_object") if state.get("holding") else None
        if isinstance(held, dict):
            held_text = str(held.get("label") or held.get("prompt") or "物体")
        else:
            held_text = "空"
        self._line(
            "状态",
            "阶段={stage} 忙={busy} 手上={held} 规划/执行服务={server}".format(
                stage=state.get("stage_zh") or state.get("stage") or "?",
                busy="是" if state.get("busy") else "否",
                held=held_text,
                server="就绪" if state.get("server_ready") else "未就绪",
                backend=self.backend.name,
            ),
        )

    # ------------------------------------------------------------------ 输入
    def start(self) -> threading.Thread:
        """后台线程模式（agent 进程内用，不抢主线程）。"""
        thread = threading.Thread(target=self.run, name="x5a-terminal", daemon=True)
        thread.start()
        return thread

    def run(self) -> None:
        if self.show_help:
            print(HELP_TEXT, flush=True)
        print(f"[提示] 终端输入层已就绪（后端 {self.backend.name}）", flush=True)
        try:
            state = self.backend.state()
            # 先把当前阶段记下来，免得随后第一条广播/polling 快照把同一阶段又打一遍
            self._last_stage = str(state.get("stage") or self._last_stage)
            self.print_state(state)
        except Exception as exc:
            self.error(f"读取状态失败: {exc}")
        while True:
            with self._lock:
                sys.stdout.write(self.prompt)
                sys.stdout.flush()
            try:
                line = sys.stdin.readline()
            except (KeyboardInterrupt, ValueError):
                print("", flush=True)
                self.note("终端输入层退出，agent 进程继续运行")
                return
            if line == "":  # EOF（Ctrl-D 或管道读完）
                print("", flush=True)
                self.note("输入结束（EOF），终端输入层退出，agent 进程继续运行")
                return
            text = line.strip()
            if not text:
                continue
            if text.startswith("/") or text in {"?", "？"}:
                if self._handle_slash(text):
                    return
                continue
            try:
                payload = self.backend.submit_text(text)
            except Exception as exc:
                self.error(f"发送失败: {exc}")
                continue
            if payload:
                self.on_payload(payload)

    def _handle_slash(self, text: str) -> bool:
        """True 表示退出输入层。"""
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""
        if command in {"/quit", "/exit", "/q"}:
            self.note("终端输入层退出，agent 进程继续运行")
            return True
        if command in {"/help", "/h", "/?"}:
            print(HELP_TEXT, flush=True)
            return False
        if command == "/state":
            try:
                self.print_state(self.backend.state())
            except Exception as exc:
                self.error(f"读取状态失败: {exc}")
            return False
        if command == "/cancel":
            try:
                payload = self.backend.cancel()
            except Exception as exc:
                self.error(f"取消失败: {exc}")
                return False
            self.on_payload(payload or {"type": "info", "message": "已请求停止当前任务"})
            return False
        if command == "/task":
            target = argument.lower()
            if target not in ALLOWED_TARGETS:
                self.error(
                    "用法: /task " + " | ".join(sorted(ALLOWED_TARGETS))
                    + "（跳过云端解析，直接发结构化任务）"
                )
                return False
            try:
                payload = self.backend.submit_task(
                    {"action": "pick_place", "target_color": target}
                )
            except Exception as exc:
                self.error(f"发送失败: {exc}")
                return False
            if payload:
                self.on_payload(payload)
            return False
        if command == "/parse":
            if not argument:
                self.error("用法: /parse <文字>（只跑本地规则解析器，不发任务）")
                return False
            self._print_parsed(parse_command(argument))
            return False
        self.error(f"未知命令: {command}（/help 看用法）")
        return False
