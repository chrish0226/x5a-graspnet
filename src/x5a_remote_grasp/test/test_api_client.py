import pytest

from x5a_remote_grasp.api_client import GraspAPIClient, RemoteAPIError, encode_multipart


def test_key_must_be_64_hex(monkeypatch):
    monkeypatch.delenv("SAM_GRASPNET_API_KEY", raising=False)
    client = GraspAPIClient("http://192.168.0.129:8765")
    with pytest.raises(RemoteAPIError, match="is not set"):
        client.api_key()
    monkeypatch.setenv("SAM_GRASPNET_API_KEY", "not-a-key")
    with pytest.raises(RemoteAPIError, match="64-digit hex"):
        client.api_key()
    monkeypatch.setenv("SAM_GRASPNET_API_KEY", "a" * 64)
    assert client.api_key() == "a" * 64


def test_multipart_contains_fields_and_exact_binary():
    binary = b"\x89PNG\r\n\x1a\n\x00\xff"
    body, content_type = encode_multipart(
        {"prompt": "banana", "depth_scale": 1000},
        {"depth": ("depth.png", "image/png", binary)},
    )
    assert content_type.startswith("multipart/form-data; boundary=x5a-")
    assert b'name="prompt"' in body
    assert b"banana" in body
    assert b'name="depth_scale"' in body
    assert binary in body


def test_connection_error_does_not_contain_key(monkeypatch):
    secret = "b" * 64
    monkeypatch.setenv("SAM_GRASPNET_API_KEY", secret)
    client = GraspAPIClient("http://127.0.0.1:1", connect_timeout=0.05)
    with pytest.raises(RemoteAPIError) as caught:
        client.health()
    assert secret not in str(caught.value)
