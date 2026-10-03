"""Query the cloud LLM gateway (/parse) for open-vocabulary command parsing.

The rule-based parser only knows red/white/orange cubes. This module asks the
cloud gateway to map arbitrary Chinese commands to a structured result:
{"object": ..., "color": ..., "position": ..., "action": ...}.
Failures return None so callers can fall back to the plain "unknown" flow.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

_URL_DEFAULT = "http://118.253.179.80:8300/parse"


def parse_url() -> str:
    return str(os.environ.get("X5A_CLOUD_PARSE_URL", _URL_DEFAULT) or "").strip()


def enabled() -> bool:
    return os.environ.get("X5A_CLOUD_PARSE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def parse_text(text: str, timeout: float = 6.0):
    """Return the cloud parse result dict, or None on any failure."""
    payload_text = (text or "").strip()
    if not payload_text or not enabled():
        return None
    body = json.dumps({"text": payload_text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        parse_url(),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None
