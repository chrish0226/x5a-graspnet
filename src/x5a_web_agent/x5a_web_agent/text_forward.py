"""Forward the final speech transcript to the digital-human HTTP API."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

_HTTP_URL_DEFAULT = "http://118.253.179.80:8210/api/v1/interaction/text"


def forward_url() -> str:
    return str(
        os.environ.get("X5A_TEXT_FORWARD_URL", _HTTP_URL_DEFAULT) or ""
    ).strip()


def enabled() -> bool:
    # 默认关闭：不再等待数字人服务器返回才允许机械臂动作。
    # 需要联动时显式设置 X5A_TEXT_FORWARD=1。
    if os.environ.get("X5A_TEXT_FORWARD", "0") in {"0", "false", "FALSE", "no"}:
        return False
    return bool(forward_url())


@dataclass(frozen=True)
class ForwardResult:
    ok: bool
    status: int = 0
    error: str = ""


def forward_text(text: str) -> ForwardResult:
    """POST the final Chinese transcript. Blocks until the digital human answers.

    200: accepted, will play “正在执行中” then MQTT true.
    400: bad parameters.
    409: a linked task is already running.
    """
    payload_text = (text or "").strip()
    if not payload_text:
        return ForwardResult(ok=False, error="empty transcript")
    if not enabled():
        return ForwardResult(ok=True, status=0)
    try:
        status = _post_http(forward_url(), payload_text)
    except Exception as exc:
        print(f"[x5a_web_agent] transcript HTTP forward failed: {exc}")
        return ForwardResult(ok=False, error=str(exc))
    if status == 200:
        return ForwardResult(ok=True, status=200)
    if status == 409:
        return ForwardResult(ok=False, status=409, error="数字人已有联动任务")
    if status == 400:
        return ForwardResult(ok=False, status=400, error="数字人拒绝该文本（参数错误）")
    return ForwardResult(ok=False, status=status, error=f"数字人 HTTP {status}")


def _post_http(url: str, text: str) -> int:
    import urllib.error
    import urllib.request

    body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=5.0) as resp:
            resp.read()
            status = int(getattr(resp, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        response = exc.read().decode("utf-8", errors="replace").strip()
        detail = f": {response}" if response else ""
        print(f"[x5a_web_agent] transcript HTTP forward HTTP {exc.code}{detail}")
        return int(exc.code)
    print(f"[x5a_web_agent] forwarded transcript to {url}: {text}")
    return status
