"""Wait for the digital-human MQTT start flag before the arm moves.

After HTTP text is accepted, the digital human plays “正在执行中”. Only when
that clip actually finishes does it publish:

    Topic:   robot_arm/task_busy
    Payload: true          (string)
    QoS:     1
    Retain:  false

It then sets waiting_finish and waits for our later ``false``.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

_HOST_DEFAULT = "118.253.179.80"
_PORT_DEFAULT = 1883
_TOPIC_DEFAULT = "robot_arm/task_busy"
_TIMEOUT_DEFAULT = 30.0

_lock = threading.Lock()
_client = None
_seen_true = threading.Event()
_cancel = threading.Event()
_connected = threading.Event()


def enabled() -> bool:
    return os.environ.get("X5A_MQTT_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def wait_start_enabled() -> bool:
    if not enabled():
        return False
    return os.environ.get("X5A_MQTT_WAIT_START", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def start_timeout() -> float:
    try:
        return max(1.0, float(os.environ.get("X5A_MQTT_START_TIMEOUT", str(_TIMEOUT_DEFAULT))))
    except (TypeError, ValueError):
        return _TIMEOUT_DEFAULT


def host_port_topic() -> tuple[str, int, str]:
    host = str(os.environ.get("X5A_MQTT_HOST", _HOST_DEFAULT) or "").strip()
    port = int(os.environ.get("X5A_MQTT_PORT", str(_PORT_DEFAULT)))
    topic = str(os.environ.get("X5A_MQTT_TOPIC", _TOPIC_DEFAULT) or "").strip()
    return host, port, topic


def is_true_payload(payload: bytes) -> bool:
    text = payload.decode("utf-8", errors="replace").strip().strip('"').strip("'")
    return text.lower() == "true"


def start_listener() -> None:
    """Subscribe once at process start so a later true is not missed."""
    if not wait_start_enabled():
        return
    with _lock:
        _ensure_client_locked()


def stop_listener() -> None:
    global _client
    with _lock:
        client = _client
        _client = None
    if client is None:
        return
    try:
        client.loop_stop()
    except Exception:
        pass
    try:
        client.disconnect()
    except Exception:
        pass


def prepare_for_start() -> None:
    """Clear the previous true and (re)subscribe immediately before HTTP POST."""
    _cancel.clear()
    _seen_true.clear()
    if not wait_start_enabled():
        return
    with _lock:
        _ensure_client_locked()


def cancel_wait() -> None:
    _cancel.set()


def wait_was_cancelled() -> bool:
    return _cancel.is_set()


def wait_for_true(timeout: Optional[float] = None) -> bool:
    """Block until payload ``true`` arrives, cancel, or timeout. True means start."""
    if not wait_start_enabled():
        return True
    if not _connected.is_set():
        _connected.wait(timeout=3.0)
    deadline = time.monotonic() + (start_timeout() if timeout is None else float(timeout))
    while time.monotonic() < deadline:
        if _cancel.is_set():
            print("[x5a_web_agent] MQTT start wait cancelled")
            return False
        if _seen_true.is_set():
            print("[x5a_web_agent] MQTT start flag true received")
            return True
        remaining = deadline - time.monotonic()
        _seen_true.wait(timeout=min(0.1, max(0.0, remaining)))
    print("[x5a_web_agent] MQTT start wait timed out (no true)")
    return False


def _ensure_client_locked() -> None:
    global _client
    if _client is not None:
        return
    import paho.mqtt.client as mqtt

    host, port, topic = host_port_topic()
    username = str(os.environ.get("X5A_MQTT_USERNAME", "") or "").strip()
    password = str(os.environ.get("X5A_MQTT_PASSWORD", "") or "")
    client_id = f"x5a-web-agent-{os.getpid()}"
    kwargs = {"client_id": client_id, "clean_session": True}
    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, **kwargs)
    else:
        client = mqtt.Client(**kwargs)
    if username:
        client.username_pw_set(username, password)

    def _on_connect(_cli, _userdata, _flags, rc) -> None:
        if int(rc) != 0:
            print(f"[x5a_web_agent] MQTT start listener connect rc={rc}")
            return
        _cli.subscribe(topic, qos=1)
        _connected.set()
        print(f"[x5a_web_agent] MQTT start listener subscribed {topic} at {host}:{port}")

    def _on_message(_cli, _userdata, msg) -> None:
        if is_true_payload(getattr(msg, "payload", b"") or b""):
            _seen_true.set()

    def _on_disconnect(_cli, _userdata, rc) -> None:
        _connected.clear()
        if int(rc) != 0:
            print(f"[x5a_web_agent] MQTT start listener disconnected rc={rc}")

    client.on_connect = _on_connect
    client.on_message = _on_message
    client.on_disconnect = _on_disconnect
    try:
        client.connect_async(host, port, keepalive=30)
        client.loop_start()
        _client = client
    except Exception as exc:
        print(f"[x5a_web_agent] MQTT start listener connect failed: {exc}")
        try:
            client.loop_stop()
        except Exception:
            pass
