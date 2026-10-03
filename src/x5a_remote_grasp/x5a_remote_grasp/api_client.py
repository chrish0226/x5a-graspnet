"""Small standard-library client for the authenticated grasp API.

The API key is read from an environment variable for each request. It is
never accepted as a ROS parameter, included in an exception, or logged.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit


KEY_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class RemoteAPIError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


def status_message(status: int) -> str:
    messages = {
        0: "cannot connect to remote inference service",
        401: "authentication failed (HTTP 401)",
        404: "remote endpoint not found (HTTP 404)",
        422: "remote request validation failed (HTTP 422)",
        500: "remote inference failed (HTTP 500)",
        503: "remote models are not ready (HTTP 503)",
    }
    return messages.get(int(status), f"remote service returned HTTP {int(status)}")


@dataclass(frozen=True)
class HTTPResult:
    status: int
    payload: object


class GraspAPIClient:
    def __init__(
        self,
        base_url: str,
        api_key_env: str = "SAM_GRASPNET_API_KEY",
        connect_timeout: float = 3.0,
        inference_timeout: float = 120.0,
    ) -> None:
        parsed = urlsplit(str(base_url).rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("remote.base_url must be an http(s) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("remote.base_url must not contain query or fragment")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.base_path = parsed.path.rstrip("/")
        self.api_key_env = str(api_key_env)
        self.connect_timeout = max(0.1, float(connect_timeout))
        self.inference_timeout = max(self.connect_timeout, float(inference_timeout))

    def api_key(self) -> str:
        key = str(os.environ.get(self.api_key_env, "") or "").strip()
        if not key:
            raise RemoteAPIError(401, f"environment variable {self.api_key_env} is not set")
        if not KEY_PATTERN.fullmatch(key):
            raise RemoteAPIError(401, f"environment variable {self.api_key_env} is not a 64-digit hex key")
        return key

    def health_status(self, authenticated: bool = True) -> HTTPResult:
        return self._json_request("GET", "/health", authenticated=authenticated)

    def health(self) -> Mapping[str, object]:
        result = self.health_status(authenticated=True)
        if result.status != 200:
            raise RemoteAPIError(result.status, status_message(result.status))
        if not isinstance(result.payload, dict):
            raise RemoteAPIError(500, "health response is not a JSON object")
        return result.payload

    def grasp(
        self,
        rgb_png: bytes,
        depth_png: bytes,
        fields: Mapping[str, object],
    ) -> Mapping[str, object]:
        body, content_type = encode_multipart(
            fields,
            {
                "rgb": ("color.png", "image/png", rgb_png),
                "depth": ("depth.png", "image/png", depth_png),
            },
        )
        result = self._json_request(
            "POST",
            "/v1/grasp",
            body=body,
            headers={"Content-Type": content_type},
            authenticated=True,
        )
        if result.status != 200:
            detail = _safe_detail(result.payload)
            suffix = f": {detail}" if detail else ""
            raise RemoteAPIError(result.status, status_message(result.status) + suffix)
        if not isinstance(result.payload, dict):
            raise RemoteAPIError(500, "grasp response is not a JSON object")
        return result.payload

    def locate(
        self,
        rgb_png: bytes,
        depth_png: bytes,
        fields: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Segment-only call (/v1/locate): returns the target's 3D center and
        top-surface point in the camera frame, without GraspNet inference."""
        body, content_type = encode_multipart(
            fields,
            {
                "rgb": ("color.png", "image/png", rgb_png),
                "depth": ("depth.png", "image/png", depth_png),
            },
        )
        result = self._json_request(
            "POST",
            "/v1/locate",
            body=body,
            headers={"Content-Type": content_type},
            authenticated=True,
        )
        if result.status != 200:
            detail = _safe_detail(result.payload)
            suffix = f": {detail}" if detail else ""
            raise RemoteAPIError(result.status, status_message(result.status) + suffix)
        if not isinstance(result.payload, dict):
            raise RemoteAPIError(500, "locate response is not a JSON object")
        return result.payload

    def _json_request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
        authenticated: bool = True,
    ) -> HTTPResult:
        request_headers: Dict[str, str] = {
            "Accept": "application/json",
            "Connection": "close",
        }
        request_headers.update(dict(headers or {}))
        if authenticated:
            request_headers["X-API-Key"] = self.api_key()
        connection_type = (
            http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(self.host, self.port, timeout=self.connect_timeout)
        try:
            connection.connect()
            if connection.sock is not None:
                connection.sock.settimeout(self.inference_timeout)
            connection.request(method, self.base_path + path, body=body, headers=request_headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RemoteAPIError(response.status, "remote JSON response is too large")
        except RemoteAPIError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise RemoteAPIError(0, f"{status_message(0)}: {type(exc).__name__}") from exc
        finally:
            connection.close()
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteAPIError(response.status, "remote response is not valid UTF-8 JSON") from exc
        return HTTPResult(status=int(response.status), payload=payload)


def encode_multipart(
    fields: Mapping[str, object],
    files: Mapping[str, Tuple[str, str, bytes]],
) -> Tuple[bytes, str]:
    boundary = "x5a-" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, (filename, mime, data) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {mime}\r\n\r\n".encode(),
                bytes(data),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _safe_detail(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail[:300]
    if isinstance(detail, (list, dict)):
        return json.dumps(detail, ensure_ascii=False)[:300]
    return ""
