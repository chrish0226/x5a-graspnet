"""Publish the robot-arm completion flag to the digital-human MQTT broker."""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

_HOST_DEFAULT = "118.253.179.80"
_PORT_DEFAULT = 1883
_TOPIC_DEFAULT = "robot_arm/task_busy"


def enabled() -> bool:
    return os.environ.get("X5A_MQTT_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def publish_task_finished(logger: Optional[Any] = None) -> bool:
    """Publish string ``false`` with QoS 1 and Retain disabled."""
    if not enabled():
        return False
    thread = threading.Thread(
        target=_publish_with_retry,
        args=(logger,),
        name="x5a-mqtt-task-finished",
        daemon=True,
    )
    thread.start()
    return True


def _publish_with_retry(logger: Optional[Any] = None) -> None:
    from paho.mqtt import publish

    host = str(os.environ.get("X5A_MQTT_HOST", _HOST_DEFAULT) or "").strip()
    port = int(os.environ.get("X5A_MQTT_PORT", str(_PORT_DEFAULT)))
    topic = str(os.environ.get("X5A_MQTT_TOPIC", _TOPIC_DEFAULT) or "").strip()
    username = str(os.environ.get("X5A_MQTT_USERNAME", "") or "").strip()
    password = str(os.environ.get("X5A_MQTT_PASSWORD", "") or "")
    auth = {"username": username, "password": password} if username else None

    for attempt in range(1, 4):
        try:
            publish.single(
                topic,
                payload="false",
                qos=1,
                retain=False,
                hostname=host,
                port=port,
                keepalive=10,
                auth=auth,
            )
            _log(logger, "info", f"MQTT published {topic}=false to {host}:{port}")
            return
        except Exception as exc:
            _log(
                logger,
                "error" if attempt == 3 else "warn",
                f"MQTT publish attempt {attempt}/3 failed: {exc}",
            )
            if attempt < 3:
                time.sleep(float(attempt))


def _log(logger: Optional[Any], level: str, message: str) -> None:
    if logger is not None:
        getattr(logger, level)(message)
        return
    print(f"[x5a_task_server] {message}")
