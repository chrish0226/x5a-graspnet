#!/usr/bin/env python3
"""独立进程的终端文字输入层：接入已经在跑的 x5a_web_agent。

``web_agent --terminal`` 是把终端层塞进 agent 进程里，要重启 agent 才生效。
这个 CLI 走 agent 的 HTTP 接口（/api/command、/api/task、/api/cancel、
/api/state），所以联调中途也能直接开一个终端打字测试，不用打断正在跑的
机械臂任务。

它和手机页一样只是前端：解析、白名单、数字人握手、ROS 2 Action 全部由
agent 决定，这里不发任何 /arm_cmd。
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from x5a_web_agent.terminal import TerminalBackend, TerminalConsole

_URL_DEFAULT = "http://127.0.0.1:8001"
POLL_INTERVAL = 1.0


class HttpBackend(TerminalBackend):
    """通过 web_agent 的 HTTP 接口接入，等价于手机页的一个客户端。"""

    name = "http"

    def __init__(self, url: str, timeout: float = 15.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.name = f"http {self.url}"

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(
            self.url + path, data=data, headers=headers, method="POST" if data else "GET"
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            # BUSY / 参数错误也带 JSON 正文，照样解析出来打印。
            body = exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"连不上 web_agent（{self.url}）: {exc}") from exc
        try:
            result = json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            raise RuntimeError(f"web_agent 返回了非 JSON 内容: {body[:200]!r}")
        return result if isinstance(result, dict) else {"message": str(result)}

    def submit_text(self, text: str) -> Optional[Dict[str, Any]]:
        return self._request("/api/command", {"text": text})

    def submit_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._request(
            "/api/task",
            {
                "action": str(task.get("action") or "pick_place"),
                "target_color": str(task.get("target_color") or ""),
            },
        )

    def cancel(self) -> Optional[Dict[str, Any]]:
        return self._request("/api/cancel", {})

    def state(self) -> Dict[str, Any]:
        return self._request("/api/state")

    def health(self) -> Dict[str, Any]:
        return self._request("/api/health")


def start_state_poller(
    backend: HttpBackend, console: TerminalConsole, interval: float = POLL_INTERVAL
) -> threading.Thread:
    """轮询 /api/state，把阶段变化打到终端（等价于 agent 内的广播观察者）。"""

    def _loop() -> None:
        offline = False
        while True:
            time.sleep(interval)
            try:
                console.on_state(backend.state())
                offline = False
            except Exception as exc:
                if not offline:
                    console.error(str(exc))
                    offline = True

    thread = threading.Thread(target=_loop, name="x5a-text-cli-poll", daemon=True)
    thread.start()
    return thread


def _health(backend: HttpBackend, console: TerminalConsole) -> None:
    try:
        health = backend.health()
    except Exception as exc:
        console.error(str(exc))
        return
    console.note(
        "agent: dry_run={dry} task_server={server} asr={asr}".format(
            dry=bool(health.get("dry_run")),
            server="就绪" if health.get("task_server") else "未就绪",
            asr=health.get("asr") or "?",
        )
    )
    if not health.get("task_server"):
        console.error("task_server 未就绪：先启动 x5a_task_server，否则任务会被拒绝")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="X5A 终端文字输入层（接入已在运行的 web_agent）"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("X5A_TEXT_CLI_URL", _URL_DEFAULT),
        help=f"web_agent 地址，默认 {_URL_DEFAULT}",
    )
    parser.add_argument("--prompt", default="x5a> ", help="输入提示符")
    parser.add_argument(
        "--no-poll",
        action="store_true",
        help="不轮询 /api/state（只看每次提交的返回，不打阶段变化）",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL,
        help=f"轮询 /api/state 的间隔秒数，默认 {POLL_INTERVAL}（每次轮询会在 agent 的访问日志里留一行）",
    )
    parsed = parser.parse_args(argv)

    backend = HttpBackend(parsed.url)
    console = TerminalConsole(backend, prompt=parsed.prompt)
    _health(backend, console)
    if not parsed.no_poll:
        start_state_poller(backend, console, max(0.2, parsed.poll_interval))
    try:
        console.run()
    except KeyboardInterrupt:
        print("", flush=True)


if __name__ == "__main__":
    main()
