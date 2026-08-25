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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from x5a_web_agent.asr import WhisperSpeechToText, create_asr
from x5a_web_agent.command_parser import (
    ALLOWED_TARGETS,
    RuleBasedCommandParser,
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
from x5a_web_agent.text_forward import enabled as text_forward_enabled
from x5a_web_agent.text_forward import forward_text, forward_url

COLOR_ZH = {
    "red": "红色",
    "white": "白色",
    "orange": "橙色",
    "nearest": "离盒子最近的",
    "farthest": "离盒子最远的",
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

    async def update(self, **kwargs: Any) -> None:
        self.state.update(kwargs)
        if "stage" in kwargs:
            self.state["stage_zh"] = STAGE_ZH.get(
                str(kwargs["stage"]), str(kwargs["stage"])
            )
        await self.broadcast({"type": "state", **self.state})


hub = ConnectionHub()
parser = RuleBasedCommandParser()
bridge: Optional[RosBridge] = None
asr_backend = create_asr(os.environ.get("X5A_ASR_BACKEND", "whisper"))


def _bridge() -> RosBridge:
    if bridge is None:
        raise RuntimeError("ROS bridge is not started")
    return bridge


app = FastAPI(title="X5A Voice Pick-Place")
app.mount("/static", StaticFiles(directory=str(static_dir())), name="static")


@app.on_event("startup")
async def _startup() -> None:
    global bridge
    if bridge is not None:
        return
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
                waiting = str(hub.state.get("stage") or "") == "WAITING_START"
                cancel_mqtt_start_wait()
                ok = _bridge().cancel()
                if waiting:
                    message = "已取消等待数字人开始标志"
                else:
                    message = "已请求停止当前任务" if ok else "当前没有可取消的任务"
                await hub.broadcast({"type": "info", "message": message})
            else:
                await ws.send_json({"type": "error", "message": f"未知消息类型: {msg_type}"})
    except WebSocketDisconnect:
        await hub.disconnect(ws)
    except Exception:
        await hub.disconnect(ws)


async def handle_text_command(text: str) -> Dict[str, Any]:
    transcript = (text or "").strip()
    await hub.update(transcript=transcript)
    await hub.broadcast({"type": "transcript", "text": transcript})
    parsed = parser.parse_command(transcript)
    await hub.update(parsed=parsed)
    await hub.broadcast({"type": "parsed", "result": parsed})
    return await dispatch_parsed(parsed, source_text=transcript)


async def handle_structured_task(task: Dict[str, Any]) -> Dict[str, Any]:
    parsed = sanitize_task(task) or {
        "action": "unknown",
        "message": "暂时无法理解该指令",
    }
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
        names = "、".join(
            COLOR_ZH.get(str(item.get("target_color")), str(item.get("target_color")))
            for item in tasks
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
        color = str(task.get("target_color") or "")
        label = COLOR_ZH.get(color, color)
        await hub.broadcast(
            {
                "type": "info",
                "message": f"队列 {index + 1}/{len(tasks)}：抓取{label}",
            }
        )
        last_item = index == len(tasks) - 1
        await _run_pick_place(
            color,
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
    parsed = parser_cli.parse_args(args=args)
    if parsed.dry_run:
        os.environ["X5A_WEB_DRY_RUN"] = "1"

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
