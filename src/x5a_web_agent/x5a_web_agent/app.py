#!/usr/bin/env python3
"""FastAPI + WebSocket front door for phone voice control.

Never publishes /arm_cmd. The only robot interface is ros_bridge -> Action.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from x5a_web_agent.asr import WhisperSpeechToText, create_asr
from x5a_web_agent.cloud_parse import parse_text as cloud_parse_text
from x5a_web_agent.cloud_parse import enabled as cloud_parse_enabled
from x5a_web_agent.command_parser import (
    ALLOWED_TARGETS,
    RuleBasedCommandParser,
    follow_up_intent,
    parse_command,
    sanitize_task,
)
from x5a_web_agent.qq_bot import extract_group_command, fetch_record_wav, send_group_msg
from x5a_web_agent.mqtt_start import (
    cancel_wait as cancel_mqtt_start_wait,
    host_port_topic as mqtt_host_port_topic,
    prepare_for_start,
    start_listener as start_mqtt_listener,
    start_timeout as mqtt_start_timeout,
    stop_listener as stop_mqtt_listener,
    wait_for_true as wait_mqtt_true,
    wait_start_enabled,
    wait_was_cancelled,
)
from x5a_web_agent.ros_bridge import BUSY_MESSAGE, RosBridge
from x5a_web_agent.session import Session
from x5a_web_agent.terminal import TerminalBackend, TerminalConsole
from x5a_web_agent.text_forward import enabled as text_forward_enabled
from x5a_web_agent.text_forward import forward_text, forward_url

COLOR_ZH = {
    "red": "红色",
    "white": "白色",
    "orange": "橙色",
    "nearest": "离盒子最近的",
    "farthest": "离盒子最远的",
}


def auto_execute_enabled() -> bool:
    """规划成功后自动触发执行器（真机动作）。X5A_AUTO_EXECUTE=0 关闭，
    关闭后仍为「规划 + RViz 可视化」两步式流程。"""
    return os.environ.get("X5A_AUTO_EXECUTE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def terminal_enabled() -> bool:
    """终端文字输入层（键盘代替语音转文字）。web_agent --terminal 打开，
    或 X5A_WEB_TERMINAL=1。关掉它就是原来的纯语音/手机页行为。"""
    return os.environ.get("X5A_WEB_TERMINAL", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


STAGE_ZH = {
    "WAITING_VISION": "正在寻找目标",
    "TARGET_FOUND": "已定位目标",
    "MOVE_READY": "正在移动到准备姿态",
    "MOVE_PRE_GRASP": "正在移动到抓取位置",
    "APPROACH": "正在接近目标",
    "GRASPING": "正在抓取",
    "LIFTING": "已抓取，正在抬起",
    "MOVE_PRE_PLACE": "正在前往放置区域",
    "DESCENDING": "正在下降到放置位置",
    "RELEASING": "正在放置",
    "RETREATING": "正在撤离",
    "RETURN_HOME": "正在返回 Home",
    "SUCCESS": "任务完成",
    "FAILED": "任务失败",
    "BUSY": "机器人忙碌",
    "IDLE": "空闲，等待指令",
    "DRY_RUN": "演练模式：未发送运动指令",
    "ACCEPTED": "任务已接受",
    "WAITING_START": "等待数字人播报完成",
    "SELECT_BY_BOX": "正在比较与盒子的距离",
    "CAPTURE": "正在同步采集相机帧",
    "REMOTE_INFERENCE": "正在请求云端分割与抓取推理",
    "TF_TRANSFORM": "正在转换到手臂坐标系",
    "CANDIDATE_CHECK": "正在筛选抓取候选",
    "CANDIDATE_REJECT": "该候选被拒，尝试下一个",
    "REMOTE_IK": "正在做碰撞感知 IK",
    "REMOTE_PLAN": "正在做 MoveIt 纯规划",
    "REMOTE_EXECUTE": "正在执行抓取动作",
    "POSE_MOVE": "正在调整姿态",
    "PLACE_LOCATE": "正在定位放置目标",
    "TRANSPORT": "正在搬运到放置点",
    "PLACE": "正在放置",
}

TERMINAL_STAGES = frozenset({"SUCCESS", "FAILED", "DRY_RUN", "IDLE"})


def _package_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def static_dir() -> Path:
    here = _package_dir() / "static"
    if here.is_dir():
        return here
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("x5a_web_agent")) / "static"
    except Exception:
        return here


def cert_dir() -> Path:
    here = _package_dir() / "certs"
    here.mkdir(parents=True, exist_ok=True)
    return here


class TextIn(BaseModel):
    text: str = Field(..., min_length=0)


class TaskIn(BaseModel):
    target_color: str
    action: str = "pick_place"


class ConnectionHub:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._observers: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = asyncio.Lock()
        self.state: Dict[str, Any] = {
            "robot_state": "IDLE",
            "busy": False,
            "stage": "IDLE",
            "stage_zh": STAGE_ZH["IDLE"],
            "transcript": "",
            "parsed": None,
            "last_result": None,
            "server_ready": False,
            "queue_active": False,
            # 交互会话镜像：手里有没有东西（真相在执行器，这里跟着更新）
            "holding": False,
            "held_object": None,
        }

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        await ws.send_json({"type": "state", **self.state})

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)
        stale: List[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        if stale:
            async with self._lock:
                for ws in stale:
                    self._clients.discard(ws)
        self._notify_observers(payload)

    async def update(self, **kwargs: Any) -> None:
        self.state.update(kwargs)
        if "stage" in kwargs:
            self.state["stage_zh"] = STAGE_ZH.get(
                str(kwargs["stage"]), str(kwargs["stage"])
            )
        await self.broadcast({"type": "state", **self.state})

    def add_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册同步观察者（目前只有终端输入层），用于把阶段变化打到本地终端。"""
        self._observers.append(callback)

    def _notify_observers(self, payload: Dict[str, Any]) -> None:
        for callback in list(self._observers):
            try:
                callback(payload)
            except Exception as exc:
                print(f"[x5a_web_agent] observer failed: {exc}")


hub = ConnectionHub()
parser = RuleBasedCommandParser()
bridge: Optional[RosBridge] = None
# 多轮交互会话（内存态）：记住夹爪里拿着什么，让「放到白色方块上」这类
# 后续指令能接着上一步动作执行。真相在执行器（execute 返回 holding）。
session = Session()
_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None
asr_backend = create_asr(os.environ.get("X5A_ASR_BACKEND", "whisper"))


async def _sync_session_state() -> None:
    """把会话里的持物状态同步到 hub（手机页显示用）。"""
    await hub.update(**session.as_dict())


def _bridge() -> RosBridge:
    if bridge is None:
        raise RuntimeError("ROS bridge is not started")
    return bridge


class InProcessTerminalBackend(TerminalBackend):
    """终端层跑在 agent 进程内：直接调 handle_text_command / handle_structured_task。

    和手机语音进的是同一个函数，解析、白名单、数字人握手、Action 全一致；提交
    后立刻返回，阶段和结果靠 hub 广播打到终端（跟手机页看到的是同一串消息）。
    """

    name = "web_agent 进程内"

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _spawn(self, coro: Any, what: str) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _done(done: Any) -> None:
            exc = done.exception()
            if exc is not None:
                print(f"[x5a_web_agent] 终端{what}失败: {exc}", flush=True)

        future.add_done_callback(_done)

    def submit_text(self, text: str) -> Optional[Dict[str, Any]]:
        self._spawn(handle_text_command(text), "指令")
        return None

    def submit_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._spawn(handle_structured_task(task), "任务")
        return None

    def cancel(self) -> Optional[Dict[str, Any]]:
        return _request_cancel()

    def state(self) -> Dict[str, Any]:
        return dict(hub.state)


app = FastAPI(title="X5A Voice Pick-Place")
app.mount("/static", StaticFiles(directory=str(static_dir())), name="static")


@app.on_event("startup")
async def _startup() -> None:
    global bridge, _MAIN_LOOP
    if bridge is not None:
        return
    _MAIN_LOOP = asyncio.get_event_loop()
    dry_run = os.environ.get("X5A_WEB_DRY_RUN", "0") in {"1", "true", "TRUE", "yes"}
    loop = asyncio.get_event_loop()
    bridge = RosBridge(dry_run=dry_run)
    try:
        bridge.start()
        bridge.set_stage_callback(
            lambda stage: asyncio.run_coroutine_threadsafe(_apply_stage(stage), loop)
        )
    except Exception as exc:
        print(f"[x5a_web_agent] ROS bridge start failed: {exc}")
    if bridge is not None:
        hub.state["server_ready"] = bool(bridge.server_ready(0.2))
    start_mqtt_listener()
    if terminal_enabled():
        console = TerminalConsole(InProcessTerminalBackend(loop))
        hub.add_observer(console.on_payload)
        console.start()
    # 会话状态是 agent 进程内存里的镜像，重启会丢。执行器才是「手里有没有
    # 东西」的真相，所以启动时问一次，把状态对齐回来。
    threading.Thread(target=_resync_session, name="x5a-session-sync", daemon=True).start()
    if isinstance(asr_backend, WhisperSpeechToText):
        def _preload() -> None:
            try:
                print("[x5a_web_agent] loading faster-whisper small ...")
                asr_backend.preload()
                print("[x5a_web_agent] faster-whisper ready")
            except Exception as exc:
                asr_backend.error = str(exc)
                print(f"[x5a_web_agent] faster-whisper load failed: {exc}")

        threading.Thread(target=_preload, name="x5a-whisper", daemon=True).start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    cancel_mqtt_start_wait()
    stop_mqtt_listener()


def _resync_session() -> None:
    """启动时向执行器查询真实持物状态，对齐 agent 侧会话（阻塞式，跑在独立
    线程里）。查不到就保持「手里没有东西」的保守状态。"""
    time.sleep(2.0)
    if bridge is None:
        return
    state: Dict[str, Any] = {}
    for _ in range(6):
        try:
            state = bridge.executor_state(timeout=5.0)
        except Exception as exc:
            state = {}
            print(f"[x5a_web_agent] session resync attempt failed: {exc}")
        if state.get("success"):
            break
        time.sleep(1.0)
    holding = state.get("holding")
    if holding is True and not session.holding:
        session.hold(prompt="(重启前夹持的物体)")
    elif holding is False and session.holding:
        session.clear("executor says the gripper is empty")
    print(
        f"[x5a_web_agent] session resync: holding={holding} "
        f"stages={state.get('plan_stages')}"
    )
    if _MAIN_LOOP is not None:
        asyncio.run_coroutine_threadsafe(_sync_session_state(), _MAIN_LOOP)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir() / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    ready = False
    try:
        ready = _bridge().server_ready(0.3)
    except Exception:
        ready = False
    hub.state["server_ready"] = ready
    return {
        "ok": True,
        "asr": asr_backend.name,
        "asr_ready": bool(getattr(asr_backend, "ready", True)),
        "dry_run": bool(bridge.dry_run) if bridge else False,
        "task_server": ready,
        "state": hub.state,
    }


@app.get("/api/state")
async def api_state() -> Dict[str, Any]:
    return hub.state


@app.post("/api/parse")
async def api_parse(body: TextIn) -> Dict[str, Any]:
    result = parser.parse_command(body.text)
    return {"text": body.text, "parsed": result}


@app.post("/api/asr")
async def api_asr(audio: UploadFile = File(...)) -> JSONResponse:
    data = await audio.read()
    if not data:
        return JSONResponse({"ok": False, "text": "", "message": "没有收到音频"}, status_code=400)
    if len(data) > 8 * 1024 * 1024:
        return JSONResponse({"ok": False, "text": "", "message": "录音太长"}, status_code=400)
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(
            None,
            lambda: asr_backend.transcribe(
                data, audio.content_type or "audio/webm", "zh"
            ),
        )
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "text": "", "message": f"识别失败: {exc}"},
            status_code=500,
        )
    await hub.update(transcript=text)
    return {"ok": True, "text": text, "asr": asr_backend.name}


@app.post("/api/command")
async def api_command(body: TextIn) -> JSONResponse:
    payload = await handle_text_command(body.text)
    status = 200 if payload.get("ok", True) else 409
    return JSONResponse(payload, status_code=status)


@app.post("/api/task")
async def api_task(body: TaskIn) -> JSONResponse:
    payload = await handle_structured_task(
        {"action": body.action, "target_color": body.target_color}
    )
    status = 200 if payload.get("ok", True) else 409
    return JSONResponse(payload, status_code=status)


@app.post("/api/onebot")
@app.post("/api/qq")
async def api_onebot(request: Request) -> Dict[str, Any]:
    try:
        event = await request.json()
    except Exception:
        return {"status": "ignored"}
    asyncio.create_task(_handle_qq_event(event))
    return {"status": "ok"}


@app.post("/api/cancel")
async def api_cancel() -> Dict[str, Any]:
    return _request_cancel()


def _request_cancel() -> Dict[str, Any]:
    waiting = str(hub.state.get("stage") or "") == "WAITING_START"
    cancel_mqtt_start_wait()
    ok = _bridge().cancel()
    if waiting:
        return {"ok": True, "message": "已取消等待数字人开始标志"}
    return {"ok": ok, "message": "已请求停止当前任务" if ok else "当前没有可取消的任务"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            msg_type = str(data.get("type", "")).strip()
            if msg_type == "ping":
                await ws.send_json({"type": "pong"})
            elif msg_type in {"text", "text_command"}:
                await handle_text_command(str(data.get("text", "")))
            elif msg_type in {"task", "manual_task"}:
                await handle_structured_task(
                    {
                        "action": data.get("action", "pick_place"),
                        "target_color": data.get("target_color", ""),
                    }
                )
            elif msg_type == "cancel":
                payload = _request_cancel()
                await hub.broadcast(
                    {"type": "info", "message": payload.get("message")}
                )
            else:
                await ws.send_json({"type": "error", "message": f"未知消息类型: {msg_type}"})
    except WebSocketDisconnect:
        await hub.disconnect(ws)
    except Exception:
        await hub.disconnect(ws)


def _grasp_route() -> str:
    """X5A_GRASP_ROUTE=rule -> HSV cube pick_place chain; remote (default) ->
    SAM+GraspNet remote_grasp chain."""
    return os.environ.get("X5A_GRASP_ROUTE", "remote").strip().lower()


def _convert_to_remote_chain(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """All grasp intents go through the remote SAM+GraspNet chain. The HSV
    pick_place chain is retired from the voice/text flow: even「抓红色的」
    becomes a cloud-segmented GraspNet grasp (prompt "red cube").
    X5A_GRASP_ROUTE=rule restores the HSV cube pick_place chain."""
    if _grasp_route() == "rule":
        return parsed
    if parsed.get("action") != "pick_place":
        return parsed
    color = str(parsed.get("target_color") or "").strip().lower()
    if color in ("nearest", "farthest"):
        prompt = "cube"  # GraspNet ranks candidates by grasp score, not box distance
    elif color:
        prompt = f"{color} cube"
    else:
        prompt = "cube"
    return {
        "action": "remote_grasp",
        "prompt": prompt,
        "object": "cube",
        "color": color or None,
        "source": "rule",
    }


async def handle_text_command(text: str) -> Dict[str, Any]:
    transcript = (text or "").strip()
    await hub.update(transcript=transcript)
    await hub.broadcast({"type": "transcript", "text": transcript})
    parsed = await _cloud_parse_command(transcript)
    await hub.update(parsed=parsed)
    await hub.broadcast({"type": "parsed", "result": parsed})
    return await dispatch_parsed(parsed, source_text=transcript)


async def _cloud_parse_command(transcript: str) -> Dict[str, Any]:
    """All semantic parsing goes through the cloud LLM gateway. Local shortcuts
    are limited to words that must work offline (cancel/home/ready) plus the
    multi-turn follow-ups, which the stateless cloud gateway cannot resolve."""
    lowered = transcript.lower()
    if any(keyword in lowered for keyword in _CANCEL_KEYWORDS):
        return {"action": "cancel", "source": "local"}
    if any(keyword in lowered for keyword in _HOME_KEYWORDS):
        return {"action": "go_home", "source": "local"}
    if any(keyword in lowered for keyword in _READY_KEYWORDS):
        return {"action": "go_ready", "source": "local"}
    if not transcript:
        return {"action": "unknown", "message": "没有收到指令内容"}
    follow_up = follow_up_intent(transcript)
    if follow_up is not None:
        return await _resolve_follow_up(follow_up, transcript)
    if not cloud_parse_enabled():
        if _grasp_route() == "rule":
            return parse_command(transcript)
        return {"action": "unknown", "message": "云端解析已禁用（X5A_CLOUD_PARSE=0）"}
    loop = asyncio.get_running_loop()
    try:
        cloud = await loop.run_in_executor(None, lambda: cloud_parse_text(transcript))
    except Exception:
        cloud = None
    mapped = _map_cloud_result(cloud)
    if mapped is not None:
        return mapped
    if _grasp_route() == "rule":
        rule = parse_command(transcript)
        if rule.get("action") != "unknown":
            return rule
    return {
        "action": "unknown",
        "message": "云端解析不可用或无法理解该指令",
        "source": "cloud",
    }


async def _resolve_follow_up(
    follow_up: Dict[str, Any], transcript: str
) -> Dict[str, Any]:
    """把本地判定的后续意图补全成可执行的解析结果。

    放置意图需要放置目标的名称：优先用云端 /parse 做名词归一化（它会把
    「那个白色的方块」变成 SAM 能吃的 "white cube"），云端不可用时退回原话里
    截出来的名词短语。云端网关是无状态的，所以动作判定必须由本地规则给出。
    """
    action = follow_up.get("action")
    if action == "release":
        return {"action": "release", "source": "rule"}
    target = str(follow_up.get("target_text") or "").strip()
    relation = str(follow_up.get("relation") or "on")
    if cloud_parse_enabled():
        loop = asyncio.get_running_loop()
        try:
            cloud = await loop.run_in_executor(None, lambda: cloud_parse_text(transcript))
        except Exception:
            cloud = None
        if isinstance(cloud, dict):
            normalized = str(cloud.get("sam_prompt") or "").strip()
            if not normalized:
                color = str(cloud.get("color") or "").strip()
                obj = str(cloud.get("object") or "").strip()
                normalized = f"{color} {obj}".strip()
            if normalized:
                target = normalized
    if not target:
        return {
            "action": "unknown",
            "message": "没听清要放到哪里，请说「放到<目标物>上」",
            "source": "rule",
        }
    return {
        "action": "remote_place",
        "prompt": target,
        "relation": relation,
        "source": "rule",
    }


_CANCEL_KEYWORDS = ("取消", "停止", "停下", "stop", "cancel")
_HOME_KEYWORDS = ("归位", "归零", "回位", "复位", "回原点", "回家", "home")
_READY_KEYWORDS = ("准备", "预备", "就位", "ready")


def _map_cloud_result(cloud: Any) -> Optional[Dict[str, Any]]:
    """Map the cloud /parse result onto a plan-only remote_grasp task.

    Every grasp intent uses the remote SAM+GraspNet chain; only the object
    name/color determine the SAM prompt."""
    if not isinstance(cloud, dict):
        return None
    obj = str(cloud.get("object") or "").strip().lower()
    if not obj:
        return None
    color = str(cloud.get("color") or "").strip().lower() or None
    action = str(cloud.get("action") or "pick").strip().lower()
    if action in {"cancel", "stop"}:
        return {"action": "cancel", "source": "cloud"}
    if action in {"pick", "place", "grasp", "handover"}:
        # 规则（HSV 色块）链只认红/白/橙小方块和最近/最远
        if _grasp_route() == "rule":
            if color not in ALLOWED_TARGETS:
                return None
            return {
                "action": "pick_place",
                "target_color": color,
                "source": "cloud",
            }
        # 云端把关系指代拆成 目标短语(sam_prompt) + 参照物短语(reference_prompt)，
        # 用 "| ref:" 编码进 prompt，由抓取服务做两段式几何定位。
        sam = str(cloud.get("sam_prompt") or "").strip() or (
            f"{color} {obj}".strip() if color else obj
        )
        reference = str(cloud.get("reference_prompt") or "").strip()
        relation = str(cloud.get("relation") or "").strip()
        place = str(cloud.get("place_prompt") or "").strip()
        place_rel = str(cloud.get("place_rel") or "").strip() or "in"
        if reference:
            prompt = f"{sam} | ref: {reference}"
            if relation:
                prompt += f"; rel: {relation}"
        else:
            prompt = sam
        if place:
            prompt += f" | place: {place}; prel: {place_rel}"
        return {
            "action": "remote_grasp",
            "prompt": prompt,
            "object": obj,
            "color": color,
            "source": "cloud",
        }
    return None


async def _cloud_parse_fallback(text: str) -> Optional[Dict[str, Any]]:
    """Rule parser failed -> ask the cloud LLM gateway for open-vocabulary intent."""
    if not cloud_parse_enabled():
        return None
    loop = asyncio.get_running_loop()
    try:
        cloud = await loop.run_in_executor(None, lambda: cloud_parse_text(text))
    except Exception:
        return None
    return _map_cloud_result(cloud)


async def handle_structured_task(task: Dict[str, Any]) -> Dict[str, Any]:
    parsed = sanitize_task(task) or {
        "action": "unknown",
        "message": "暂时无法理解该指令",
    }
    parsed = _convert_to_remote_chain(parsed)
    await hub.update(parsed=parsed)
    await hub.broadcast({"type": "parsed", "result": parsed})
    return await dispatch_parsed(parsed)


def _needs_digital_human_handshake(source_text: str) -> bool:
    if not (source_text or "").strip():
        return False
    if bridge is not None and bridge.dry_run:
        return False
    return bool(text_forward_enabled())


async def _handshake_or_error(source_text: str) -> Optional[Dict[str, Any]]:
    """HTTP text then wait for MQTT true. None means the arm may start."""
    if not _needs_digital_human_handshake(source_text):
        return None
    prepare_for_start()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: forward_text(source_text))
    if not result.ok:
        return {
            "ok": False,
            "type": "error",
            "message": result.error or "数字人 HTTP 转发失败",
            "stage": "FAILED",
        }
    if not wait_start_enabled():
        return None
    await hub.update(robot_state="BUSY", busy=True, stage="WAITING_START")
    await hub.broadcast(
        {
            "type": "info",
            "message": "已通知数字人，等待播报「正在执行中」完成后开始",
            "stage": "WAITING_START",
            "stage_zh": STAGE_ZH["WAITING_START"],
        }
    )
    got_true = await loop.run_in_executor(None, wait_mqtt_true)
    if got_true:
        return None
    if wait_was_cancelled():
        return {
            "ok": False,
            "type": "error",
            "message": "已取消等待数字人开始标志",
            "stage": "FAILED",
        }
    return {
        "ok": False,
        "type": "error",
        "message": "等待数字人开始标志超时（未收到 MQTT true）",
        "stage": "FAILED",
    }


async def _fail_before_motion(payload: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    payload["parsed"] = parsed
    await hub.update(
        robot_state="IDLE",
        busy=False,
        stage="FAILED",
        last_result={
            "success": False,
            "message": payload.get("message"),
            "stage": "FAILED",
        },
        queue_active=False,
    )
    await hub.broadcast(payload)
    return payload


async def dispatch_parsed(
    parsed: Dict[str, Any],
    source_text: str = "",
) -> Dict[str, Any]:
    action = parsed.get("action")
    if action == "unknown":
        payload = {
            "ok": False,
            "type": "error",
            "message": parsed.get("message") or "暂时无法理解该指令",
            "parsed": parsed,
        }
        await hub.broadcast(payload)
        return payload
    if action == "cancel":
        waiting = str(hub.state.get("stage") or "") == "WAITING_START"
        cancel_mqtt_start_wait()
        ok = _bridge().cancel()
        if waiting:
            message = "已取消等待数字人开始标志"
            ok = True
        else:
            message = "已请求停止当前任务" if ok else "当前没有可取消的任务"
        payload = {
            "ok": ok,
            "type": "info",
            "message": message,
            "parsed": parsed,
        }
        await hub.broadcast(payload)
        return payload
    if action == "sequence":
        tasks = parsed.get("tasks") or []
        if not tasks:
            payload = {
                "ok": False,
                "type": "error",
                "message": "暂时无法理解该指令",
                "parsed": parsed,
            }
            await hub.broadcast(payload)
            return payload
        if _bridge().busy and hub.state.get("stage") not in TERMINAL_STAGES:
            payload = {
                "ok": False,
                "type": "error",
                "message": BUSY_MESSAGE,
                "parsed": parsed,
            }
            await hub.broadcast(payload)
            return payload
        handshake = _needs_digital_human_handshake(source_text)
        stage = "WAITING_START" if handshake else "ACCEPTED"
        await hub.update(
            robot_state="BUSY", busy=True, stage=stage, queue_active=True
        )
        asyncio.create_task(_run_linked_task_queue(tasks, parsed, source_text))
        tasks = [_convert_to_remote_chain(dict(item)) for item in tasks]
        names = "、".join(
            str(item.get("prompt") or item.get("target_color") or "?") for item in tasks
        )
        payload = {
            "ok": True,
            "type": "info",
            "message": (
                f"等待数字人播报完成后开始：{names}"
                if handshake
                else f"已加入{len(tasks)}个任务：{names}"
            ),
            "parsed": parsed,
            "stage": stage,
            "stage_zh": STAGE_ZH[stage],
        }
        await hub.broadcast(payload)
        return payload
    if action == "remote_grasp":
        prompt = str(parsed.get("prompt") or "").strip()
        if not prompt:
            payload = {
                "ok": False,
                "type": "error",
                "message": "暂时无法理解该指令",
                "parsed": parsed,
            }
            await hub.broadcast(payload)
            return payload
        if session.holding:
            # 手上还拿着东西时再发起抓取，执行器会先张爪把原来的东西扔掉。
            # 想接着放就说「放到…上」，想丢掉就说「放下」——不替用户做这个决定。
            payload = {
                "ok": False,
                "type": "error",
                "message": (
                    f"手上还拿着{session.held_label()}，请先说放到哪里（「放到…上」），"
                    "或说「放下」把它放掉"
                ),
                "parsed": parsed,
            }
            await hub.broadcast(payload)
            return payload
        if _bridge().busy and hub.state.get("stage") not in TERMINAL_STAGES:
            payload = {
                "ok": False,
                "type": "error",
                "message": BUSY_MESSAGE,
                "parsed": parsed,
            }
            await hub.broadcast(payload)
            return payload
        await hub.update(robot_state="BUSY", busy=True, stage="ACCEPTED")
        asyncio.create_task(_run_remote_grasp(prompt, parsed))
        payload = {
            "ok": True,
            "type": "info",
            "message": f"已提交远程抓取规划：{prompt}（仅规划与可视化，不执行运动）",
            "parsed": parsed,
            "stage": "ACCEPTED",
            "stage_zh": STAGE_ZH["ACCEPTED"],
        }
        await hub.broadcast(payload)
        return payload
    if action == "remote_place":
        if not session.holding:
            payload = {
                "ok": False,
                "type": "error",
                "message": "手上没有东西，先说要抓什么（例如「抓起红色方块」）",
                "parsed": parsed,
            }
            await hub.broadcast(payload)
            return payload
        prompt = str(parsed.get("prompt") or "").strip()
        if not _bridge().busy or hub.state.get("stage") in TERMINAL_STAGES:
            await hub.update(robot_state="BUSY", busy=True, stage="ACCEPTED")
            asyncio.create_task(_run_remote_place(prompt, parsed))
            payload = {
                "ok": True,
                "type": "info",
                "message": f"把{session.held_label()}放到「{prompt}」上",
                "parsed": parsed,
                "stage": "ACCEPTED",
                "stage_zh": STAGE_ZH["ACCEPTED"],
            }
        else:
            payload = {
                "ok": False,
                "type": "error",
                "message": BUSY_MESSAGE,
                "parsed": parsed,
            }
        await hub.broadcast(payload)
        return payload
    if action == "release":
        # Opening the gripper is valid even when this session missed holding
        # feedback or the gripper is closed without an object.
        if hub.state.get('busy') or (_bridge().busy and hub.state.get("stage") not in TERMINAL_STAGES):
            payload = {
                "ok": False,
                "type": "error",
                "message": BUSY_MESSAGE,
                "parsed": parsed,
            }
            await hub.broadcast(payload)
            return payload
        await hub.update(robot_state="BUSY", busy=True, stage="ACCEPTED", last_result=None)
        asyncio.create_task(_run_release(parsed))
        payload = {
            "ok": True,
            "type": "info",
            "message": "原地松开夹爪" + (f"，放下{session.held_label()}" if session.holding else ""),
            "parsed": parsed,
            "stage": "ACCEPTED",
            "stage_zh": STAGE_ZH["ACCEPTED"],
        }
        await hub.broadcast(payload)
        return payload
    if action in {"go_home", "go_ready"}:
        if _bridge().busy and hub.state.get("stage") not in TERMINAL_STAGES:
            payload = {
                "ok": False,
                "type": "error",
                "message": BUSY_MESSAGE,
                "parsed": parsed,
            }
            await hub.broadcast(payload)
            return payload
        await hub.update(robot_state="BUSY", busy=True, stage="ACCEPTED")
        asyncio.create_task(_run_pose_move(action, parsed))
        label = "归位 (Home)" if action == "go_home" else "预抓取就绪姿态"
        payload = {
            "ok": True,
            "type": "info",
            "message": f"开始调整到{label}",
            "parsed": parsed,
            "stage": "ACCEPTED",
            "stage_zh": STAGE_ZH["ACCEPTED"],
        }
        await hub.broadcast(payload)
        return payload
    if action != "pick_place":
        payload = {
            "ok": False,
            "type": "error",
            "message": "暂时无法理解该指令",
            "parsed": parsed,
        }
        await hub.broadcast(payload)
        return payload

    color = parsed.get("target_color")
    if color not in ALLOWED_TARGETS:
        payload = {
            "ok": False,
            "type": "error",
            "message": "暂时无法理解该指令",
            "parsed": parsed,
        }
        await hub.broadcast(payload)
        return payload

    if _bridge().busy and hub.state.get("stage") not in TERMINAL_STAGES:
        payload = {
            "ok": False,
            "type": "error",
            "message": BUSY_MESSAGE,
            "parsed": parsed,
        }
        await hub.broadcast(payload)
        return payload

    handshake = _needs_digital_human_handshake(source_text)
    stage = "WAITING_START" if handshake else "ACCEPTED"
    await hub.update(robot_state="BUSY", busy=True, stage=stage)
    asyncio.create_task(_run_linked_pick_place(str(color), parsed, source_text))
    payload = {
        "ok": True,
        "type": "info",
        "message": "等待数字人播报完成后开始" if handshake else "任务已提交",
        "parsed": parsed,
        "stage": stage,
        "stage_zh": STAGE_ZH[stage],
    }
    await hub.broadcast(payload)
    return payload


async def _announce_motion_start() -> None:
    await hub.update(robot_state="BUSY", busy=True, stage="ACCEPTED")
    await hub.broadcast(
        {
            "type": "info",
            "message": "数字人开始标志已收到，开始执行",
            "stage": "ACCEPTED",
            "stage_zh": STAGE_ZH["ACCEPTED"],
        }
    )


async def _run_pose_move(action: str, parsed: Dict[str, Any]) -> None:
    loop = asyncio.get_running_loop()
    fn = _bridge().pose_home if action == "go_home" else _bridge().pose_ready
    try:
        result = await loop.run_in_executor(None, fn)
    except Exception as exc:
        result = {"success": False, "message": f"ROS bridge exception: {exc}", "stage": "FAILED"}
    success = bool(result.get("success"))
    stage = "SUCCESS" if success else "FAILED"
    # 归位/就位不动夹爪：手里有东西时照样保持夹持（按用户要求带着物体归位）
    await hub.update(robot_state="IDLE", busy=False, stage=stage, last_result=result)
    await hub.broadcast({
        "ok": success, "type": "result", "success": success,
        "message": result.get("message"), "stage": stage,
        "stage_zh": STAGE_ZH.get(stage, stage), "parsed": parsed,
    })


async def _carry_to_ready() -> None:
    """抓住物体后移到 hold 姿态等下一句指令。

    相机是 eye-to-hand 固定安装的，手臂和手里的物体悬在原位会挡住后续
    /v1/locate 的视野，所以抓完先让开到 ready 姿态。"""
    if not carry_to_ready_enabled():
        return
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _bridge().pose_ready())
    except Exception as exc:
        result = {"success": False, "message": f"ROS bridge exception: {exc}"}
    if not result.get("success"):
        await hub.broadcast({
            "type": "info",
            "message": f"已抓住物体，但移到等待姿态失败：{result.get('message')}",
        })


def carry_to_ready_enabled() -> bool:
    """抓取成功后是否自动带着物体移到等待姿态（interaction.carry_to_ready）。"""
    return os.environ.get("X5A_CARRY_TO_READY", "1").strip().lower() not in {
        "0", "false", "no",
    }


async def _run_remote_grasp(
    prompt: str,
    parsed: Dict[str, Any],
    on_done=None,
) -> None:
    """Plan-only SAM + GraspNet task: cloud inference, TF, markers, optional
    MoveIt planning. The remote node never executes motion.

    执行结果以执行器返回的 holding 为准：合爪未张 = 手里有东西（抓取流），
    张爪放完 = 手里空了（一步抓放流）。会话状态据此更新。"""
    loop = asyncio.get_running_loop()

    def on_feedback(stage: str) -> None:
        asyncio.run_coroutine_threadsafe(_apply_stage(stage), loop)

    try:
        result = await loop.run_in_executor(
            None,
            lambda: _bridge().send_remote_grasp(
                prompt, on_feedback=on_feedback, plan_motion=True
            ),
        )
    except Exception as exc:
        result = {
            "success": False,
            "message": f"ROS bridge exception: {exc}",
            "stage": "FAILED",
        }
    executed = False
    if result.get("success") and auto_execute_enabled():
        # 规划成功 → 自动触发执行器（真机动作）。执行器侧仍有独立安全门：
        # enabled 开关、规划 180 秒新鲜度、/arm_cmd 发布者数量校验。
        await hub.update(robot_state="BUSY", busy=True, stage="REMOTE_EXECUTE")
        await hub.broadcast(
            {
                "type": "status",
                "stage": "REMOTE_EXECUTE",
                "stage_zh": STAGE_ZH.get("REMOTE_EXECUTE", "REMOTE_EXECUTE"),
                "busy": True,
            }
        )
        try:
            exec_result = await loop.run_in_executor(
                None, lambda: _bridge().execute_latest()
            )
        except Exception as exc:
            exec_result = {
                "success": False,
                "message": f"执行器调用异常: {exc}",
                "stage": "FAILED",
            }
        executed = True
        result = dict(result)
        result["success"] = bool(exec_result.get("success"))
        result["message"] = exec_result.get("message") or result.get("message")
        result["stage"] = exec_result.get("stage") or result.get("stage")
        result["holding"] = exec_result.get("holding")
    success = bool(result.get("success"))
    stage = "SUCCESS" if success else str(result.get("stage") or "FAILED")
    if stage == "BUSY":
        stage = "FAILED"
    if executed:
        session.observe_executor(
            result, prompt=prompt, object_name=str(parsed.get('object') or ''),
            color=str(parsed.get('color') or ''),
            width_m=float(result.get('gripper_width_m') or 0.0),
        )
        await _sync_session_state()
    await hub.update(
        robot_state="IDLE",
        busy=False,
        stage=stage,
        last_result=result,
    )
    await hub.broadcast(
        {
            "ok": success,
            "type": "result",
            "success": success,
            "message": result.get("message"),
            "stage": stage,
            "stage_zh": STAGE_ZH.get(stage, stage),
            "parsed": parsed,
            "holding": session.holding,
            "held_object": session.as_dict()["held_object"],
        }
    )
    if success and executed and session.holding:
        await _carry_to_ready()
    if on_done is not None:
        await on_done(
            {
                "ok": success,
                "message": result.get("message"),
                "stage": stage,
                "stage_zh": STAGE_ZH.get(stage, stage),
            }
        )


async def _run_remote_place(prompt: str, parsed: Dict[str, Any]) -> None:
    """放置流：物体已经在夹爪里，只定位放置目标并规划 TRANSPORT/PLACE/RETREAT
    三段（节点侧的纯放置模式，不重新抓取）。"""
    loop = asyncio.get_running_loop()

    def on_feedback(stage: str) -> None:
        asyncio.run_coroutine_threadsafe(_apply_stage(stage), loop)

    relation = str(parsed.get("relation") or "on")
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _bridge().send_remote_place(
                prompt, relation=relation, on_feedback=on_feedback
            ),
        )
    except Exception as exc:
        result = {"success": False, "message": f"ROS bridge exception: {exc}", "stage": "FAILED"}
    if result.get("success") and auto_execute_enabled():
        await hub.update(robot_state="BUSY", busy=True, stage="REMOTE_EXECUTE")
        await hub.broadcast({
            "type": "status", "stage": "REMOTE_EXECUTE",
            "stage_zh": STAGE_ZH.get("REMOTE_EXECUTE", "REMOTE_EXECUTE"), "busy": True,
        })
        try:
            exec_result = await loop.run_in_executor(
                None, lambda: _bridge().execute_latest()
            )
        except Exception as exc:
            exec_result = {"success": False, "message": f"执行器调用异常: {exc}", "stage": "FAILED"}
        result = dict(result)
        result["success"] = bool(exec_result.get("success"))
        result["message"] = exec_result.get("message") or result.get("message")
        result["stage"] = exec_result.get("stage") or result.get("stage")
        result["holding"] = exec_result.get("holding")
    success = bool(result.get("success"))
    stage = "SUCCESS" if success else str(result.get("stage") or "FAILED")
    if stage == "BUSY":
        stage = "FAILED"
    session.observe_executor(result)
    await _sync_session_state()
    await hub.update(robot_state="IDLE", busy=False, stage=stage, last_result=result)
    await hub.broadcast({
        "ok": success, "type": "result", "success": success,
        "message": result.get("message"), "stage": stage,
        "stage_zh": STAGE_ZH.get(stage, stage), "parsed": parsed,
        "holding": session.holding,
        "held_object": session.as_dict()["held_object"],
    })


async def _run_release(parsed: Dict[str, Any]) -> None:
    """放下：原地张开夹爪，不动手臂。失败时保留持物状态（物体还在手里）。"""
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _bridge().release())
    except Exception as exc:
        result = {"success": False, "message": f"ROS bridge exception: {exc}", "stage": "FAILED"}
    success = bool(result.get("success"))
    session.observe_executor(result)
    if success and result.get('holding') is None:
        session.clear("released")
    await _sync_session_state()
    stage = "SUCCESS" if success else "FAILED"
    await hub.update(robot_state="IDLE", busy=False, stage=stage, last_result=result)
    await hub.broadcast({
        "ok": success, "type": "result", "success": success,
        "message": result.get("message"), "stage": stage,
        "stage_zh": STAGE_ZH.get(stage, stage), "parsed": parsed,
        "holding": session.holding,
        "held_object": session.as_dict()["held_object"],
    })


async def _run_linked_pick_place(
    color: str,
    parsed: Dict[str, Any],
    source_text: str,
) -> None:
    err = await _handshake_or_error(source_text)
    if err:
        await _fail_before_motion(err, parsed)
        return
    if _needs_digital_human_handshake(source_text):
        await _announce_motion_start()
    await _run_pick_place(color, parsed, on_done=None)


async def _run_linked_task_queue(
    tasks: List[Dict[str, Any]],
    parsed: Dict[str, Any],
    source_text: str,
) -> None:
    err = await _handshake_or_error(source_text)
    if err:
        await _fail_before_motion(err, parsed)
        return
    if _needs_digital_human_handshake(source_text):
        await _announce_motion_start()
        await hub.update(queue_active=True)
    await _run_task_queue(tasks, parsed)


async def _apply_stage(stage: str) -> None:
    if str(hub.state.get("stage") or "") == "WAITING_START":
        return
    terminal = stage in TERMINAL_STAGES
    await hub.update(
        robot_state="IDLE" if terminal else "BUSY",
        busy=not terminal,
        stage=stage,
    )
    await hub.broadcast(
        {
            "type": "status",
            "stage": stage,
            "stage_zh": STAGE_ZH.get(stage, stage),
            "busy": not terminal,
        }
    )


async def _run_task_queue(tasks: List[Dict[str, Any]], parsed: Dict[str, Any]) -> None:
    last = {"success": False, "message": "队列为空", "stage": "FAILED"}
    for index, task in enumerate(tasks):
        label = str(task.get("prompt") or task.get("target_color") or "")
        await hub.broadcast(
            {
                "type": "info",
                "message": f"队列 {index + 1}/{len(tasks)}：抓取 {label}",
            }
        )
        last_item = index == len(tasks) - 1
        if task.get("action") == "remote_grasp":
            done_result: Dict[str, Any] = {}

            async def _collect(res: Dict[str, Any]) -> None:
                done_result.update(res)

            await _run_remote_grasp(str(task.get("prompt")), task, on_done=_collect)
            last = done_result or last
        else:
            await _run_pick_place(
                str(task.get("target_color") or ""),
                task,
                skip_return_home=not last_item,
                reuse_frozen_box=index > 0,
            )
            last = hub.state.get("last_result") or last
        if not bool((last or {}).get("success")):
            await hub.update(queue_active=False)
            await hub.broadcast(
                {
                    "type": "result",
                    "success": False,
                    "message": f"队列在第{index + 1}个任务失败，已停止",
                    "stage": "FAILED",
                    "stage_zh": STAGE_ZH["FAILED"],
                    "parsed": parsed,
                }
            )
            return
    await hub.update(
        robot_state="IDLE",
        busy=False,
        stage="SUCCESS",
        last_result=last,
        queue_active=False,
    )
    await hub.broadcast(
        {
            "type": "result",
            "success": True,
            "message": f"队列完成，共{len(tasks)}个任务",
            "stage": "SUCCESS",
            "stage_zh": STAGE_ZH["SUCCESS"],
            "parsed": parsed,
        }
    )


async def _run_pick_place(
    color: str,
    parsed: Dict[str, Any],
    on_done=None,
    skip_return_home: bool = False,
    reuse_frozen_box: bool = False,
) -> None:
    loop = asyncio.get_running_loop()

    def on_feedback(stage: str) -> None:
        asyncio.run_coroutine_threadsafe(_apply_stage(stage), loop)

    try:
        result = await loop.run_in_executor(
            None,
            lambda: _bridge().send_pick_place(
                color,
                on_feedback=on_feedback,
                skip_return_home=skip_return_home,
                reuse_frozen_box=reuse_frozen_box,
            ),
        )
    except Exception as exc:
        result = {
            "success": False,
            "message": f"ROS bridge exception: {exc}",
            "stage": "FAILED",
        }
    success = bool(result.get("success"))
    stage = "SUCCESS" if success else str(result.get("stage") or "FAILED")
    if stage == "BUSY":
        stage = "FAILED"
    await hub.update(
        robot_state="IDLE",
        busy=False,
        stage=stage,
        last_result=result,
    )
    await hub.broadcast(
        {
            "ok": success,
            "type": "result",
            "success": success,
            "message": result.get("message"),
            "stage": stage,
            "stage_zh": STAGE_ZH.get(stage, stage),
            "parsed": parsed,
        }
    )
    if on_done is not None:
        await on_done(
            {
                "ok": success,
                "message": result.get("message"),
                "stage": stage,
                "stage_zh": STAGE_ZH.get(stage, stage),
            }
        )


async def _handle_qq_event(event: Dict[str, Any]) -> None:
    cmd = extract_group_command(event if isinstance(event, dict) else {})
    if cmd is None:
        return
    group_id = int(cmd["group_id"])
    text = str(cmd.get("text") or "").strip()
    try:
        if not text and (cmd.get("record_file") or cmd.get("record_url")):
            audio, mime = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _qq_load_audio(cmd)
            )
            text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: asr_backend.transcribe(audio, mime, "zh")
            )
            await hub.update(transcript=text)
            if text:
                send_group_msg(group_id, f"听成：{text}")
            else:
                send_group_msg(group_id, "语音没有听清，请再说一次，或直接打字：抓红色")
                return
        if not text:
            return
        parsed = parser.parse_command(text)
        if parsed.get("action") == "unknown":
            return
        color_zh = COLOR_ZH.get(str(parsed.get("target_color") or ""), "")
        if parsed.get("action") == "pick_place" and color_zh:
            send_group_msg(group_id, f"收到，开始抓取{color_zh}方块")
        elif parsed.get("action") == "sequence":
            names = "、".join(
                COLOR_ZH.get(str(item.get("target_color")), str(item.get("target_color")))
                for item in (parsed.get("tasks") or [])
            )
            send_group_msg(group_id, f"收到，将依次抓取：{names}")
        elif parsed.get("action") == "cancel":
            send_group_msg(group_id, "收到停止请求")
        payload = await dispatch_parsed(parsed, source_text=text)
        if payload.get("type") == "info" and payload.get("stage") == "ACCEPTED":
            asyncio.create_task(_watch_and_reply_qq(group_id, payload))
        elif payload.get("message"):
            send_group_msg(group_id, str(payload.get("message")))
    except Exception as exc:
        try:
            send_group_msg(group_id, f"QQ 指令处理失败：{exc}")
        except Exception:
            pass


def _qq_load_audio(cmd: Dict[str, Any]) -> tuple:
    if cmd.get("record_file"):
        try:
            return fetch_record_wav(str(cmd["record_file"]))
        except Exception:
            pass
    url = str(cmd.get("record_url") or "")
    if url:
        from x5a_web_agent.qq_bot import download_bytes

        return download_bytes(url), "audio/webm"
    raise RuntimeError("群语音没有可用音频地址")


async def _watch_and_reply_qq(group_id: int, _payload: Dict[str, Any]) -> None:
    # Wait until hub leaves BUSY after this submission.
    for _ in range(400):
        await asyncio.sleep(0.5)
        stage = str(hub.state.get("stage") or "")
        if hub.state.get("queue_active"):
            continue
        if stage in TERMINAL_STAGES:
            last = hub.state.get("last_result") or {}
            ok = bool(last.get("success")) if last else stage == "SUCCESS"
            zh = hub.state.get("stage_zh") or last.get("message") or stage
            send_group_msg(group_id, f"{'完成' if ok else '失败'}：{zh}")
            return


def _is_phone_lan_ip(ip: str) -> bool:
    """Skip loopback and Clash/Meta fake-ip (198.18.0.0/15)."""
    if not ip or ip.startswith("127.") or ip.startswith("198.18.") or ip.startswith("198.19."):
        return False
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        first, second = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return first == 10 or (first == 192 and second == 168) or (
        first == 172 and 16 <= second <= 31
    )


def local_ip() -> str:
    found: List[str] = []
    try:
        found.extend(subprocess.check_output(["hostname", "-I"], text=True).split())
    except Exception:
        pass
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("192.168.0.1", 80))
        found.append(sock.getsockname()[0])
        sock.close()
    except Exception:
        pass
    for ip in found:
        if _is_phone_lan_ip(ip):
            return ip
    return next((ip for ip in found if ip), "127.0.0.1")


def ensure_self_signed_cert() -> tuple[Path, Path]:
    key = cert_dir() / "key.pem"
    cert = cert_dir() / "cert.pem"
    ip = local_ip()
    need = True
    if key.is_file() and cert.is_file():
        try:
            text = subprocess.check_output(
                ["openssl", "x509", "-in", str(cert), "-noout", "-text"],
                text=True,
            )
            if f"IP Address:{ip}" in text or f"IP:{ip}" in text:
                need = False
        except Exception:
            need = True
    if need:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-days",
                "365",
                "-nodes",
                "-subj",
                f"/CN={ip}",
                "-addext",
                f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
            ],
            check=True,
            capture_output=True,
        )
    return cert, key


def main(args: Optional[List[str]] = None) -> None:
    parser_cli = argparse.ArgumentParser(description="X5A phone voice web agent")
    parser_cli.add_argument("--host", default=os.environ.get("X5A_WEB_HOST", "0.0.0.0"))
    parser_cli.add_argument(
        "--port", type=int, default=int(os.environ.get("X5A_WEB_PORT", "8000"))
    )
    parser_cli.add_argument(
        "--https-port",
        type=int,
        default=int(os.environ.get("X5A_WEB_HTTPS_PORT", "8443")),
        help="HTTPS port for phone microphone. Default 8443.",
    )
    parser_cli.add_argument(
        "--no-https",
        action="store_true",
        help="Do not start the HTTPS voice port.",
    )
    parser_cli.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("X5A_WEB_DRY_RUN", "0") in {"1", "true", "TRUE", "yes"},
    )
    parser_cli.add_argument(
        "--terminal",
        action="store_true",
        help="在当前终端加一层文字输入（等同手机语音转写），方便测试。",
    )
    parsed = parser_cli.parse_args(args=args)
    if parsed.dry_run:
        os.environ["X5A_WEB_DRY_RUN"] = "1"
    if parsed.terminal:
        os.environ["X5A_WEB_TERMINAL"] = "1"

    ip = local_ip()
    print("========================================")
    print(" X5A Voice Web Agent")
    print("========================================")
    print(f"page:   http://{ip}:{parsed.port}")
    print(f"local:  http://127.0.0.1:{parsed.port}")
    print(f"ASR:    {asr_backend.name} (local faster-whisper small)")
    print(f"text->: {forward_url()}")
    mqtt_host, mqtt_port, mqtt_topic = mqtt_host_port_topic()
    print(
        f"mqtt:   {mqtt_host}:{mqtt_port} {mqtt_topic} "
        f"wait_start={int(wait_start_enabled())} timeout={mqtt_start_timeout():.0f}s"
    )
    print(f"dry_run:{os.environ.get('X5A_WEB_DRY_RUN', '0')}")
    if terminal_enabled():
        print("terminal: on（在当前终端直接打字，等同语音转写；/help 看命令）")

    import threading
    import uvicorn

    if not parsed.no_https:
        cert, key = ensure_self_signed_cert()
        print(f"voice:  https://{ip}:{parsed.https_port}")
        print("Voice needs HTTPS. On the phone, open the voice URL")
        print("and accept the certificate warning.")

        def _run_https() -> None:
            uvicorn.run(
                app,
                host=parsed.host,
                port=parsed.https_port,
                ssl_certfile=str(cert),
                ssl_keyfile=str(key),
                ws="wsproto",
                log_level="warning",
            )

        threading.Thread(target=_run_https, name="x5a-https", daemon=True).start()

    print("This process does not publish /arm_cmd.")
    print("========================================")

    uvicorn.run(
        app,
        host=parsed.host,
        port=parsed.port,
        ws="wsproto",
        log_level="info",
    )


if __name__ == "__main__":
    main()
