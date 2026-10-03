"""Command-line health check that never prints the API key."""
from __future__ import annotations

import argparse
import json
import sys

from x5a_remote_grasp.api_client import GraspAPIClient, RemoteAPIError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.0.129:8765")
    parser.add_argument("--api-key-env", default="SAM_GRASPNET_API_KEY")
    parser.add_argument("--connect-timeout", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--skip-unauthorized-check",
        action="store_true",
        help="do not verify that an unauthenticated request is rejected",
    )
    args = parser.parse_args(argv)
    client = GraspAPIClient(
        args.base_url,
        api_key_env=args.api_key_env,
        connect_timeout=args.connect_timeout,
        inference_timeout=args.timeout,
    )
    try:
        if not args.skip_unauthorized_check:
            unauthorized = client.health_status(authenticated=False)
            if unauthorized.status != 401:
                print(
                    json.dumps(
                        {"ok": False, "unauthenticated_http": unauthorized.status},
                        ensure_ascii=False,
                    )
                )
                return 2
        payload = client.health()
    except RemoteAPIError as exc:
        print(json.dumps({"ok": False, "status": exc.status, "error": str(exc)}))
        return 1
    sam = payload.get("sam") if isinstance(payload.get("sam"), dict) else {}
    graspnet = (
        payload.get("graspnet") if isinstance(payload.get("graspnet"), dict) else {}
    )
    ready = (
        str(payload.get("status", "")).lower() == "ready"
        and sam.get("ready") is True
        and graspnet.get("ready") is True
    )
    summary = {
        "ok": ready,
        "unauthenticated_http": None if args.skip_unauthorized_check else 401,
        "status": payload.get("status"),
        "sam_ready": sam.get("ready"),
        "graspnet_ready": graspnet.get("ready"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if ready else 3


if __name__ == "__main__":
    sys.exit(main())
