"""Call the grasp API from a snapshot previously saved by the ROS node."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from x5a_remote_grasp.api_client import GraspAPIClient, RemoteAPIError
from x5a_remote_grasp.geometry import parse_api_grasps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-url", default="http://192.168.0.129:8765")
    parser.add_argument("--api-key-env", default="SAM_GRASPNET_API_KEY")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--mask-dilation-px", type=int, default=8)
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--target-distance-thresh", type=float, default=0.06)
    parser.add_argument("--max-gripper-width", type=float, default=0.088)
    parser.add_argument("--connect-timeout", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    snapshot = args.snapshot.resolve()
    try:
        metadata = json.loads((snapshot / "camera_info.json").read_text())
        rgb_png = (snapshot / "color.png").read_bytes()
        depth_png = (snapshot / "depth.png").read_bytes()
        depth_scale = float(metadata["depth_scale_divisor"])
        fields = {
            "prompt": args.prompt,
            "fx": float(metadata["fx"]),
            "fy": float(metadata["fy"]),
            "cx": float(metadata["cx"]),
            "cy": float(metadata["cy"]),
            "depth_scale": depth_scale,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "threshold": args.threshold,
            "mask_dilation_px": args.mask_dilation_px,
            "num_point": args.num_point,
            "top_k": args.top_k,
            "collision_thresh": args.collision_thresh,
            "voxel_size": args.voxel_size,
            "target_distance_thresh": args.target_distance_thresh,
            "max_gripper_width": args.max_gripper_width,
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": f"invalid snapshot: {exc}"}))
        return 2
    client = GraspAPIClient(
        args.base_url,
        api_key_env=args.api_key_env,
        connect_timeout=args.connect_timeout,
        inference_timeout=args.timeout,
    )
    try:
        response = client.grasp(rgb_png, depth_png, fields)
        grasps = parse_api_grasps(response)
    except (RemoteAPIError, ValueError) as exc:
        status = exc.status if isinstance(exc, RemoteAPIError) else 0
        print(json.dumps({"ok": False, "status": status, "error": str(exc)}))
        return 1
    output = (args.output or (snapshot / "grasp_response.json")).resolve()
    output.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n")
    first = grasps[0] if grasps else None
    summary = {
        "ok": True,
        "source_frame": metadata.get("source_frame"),
        "api_frame": response.get("frame"),
        "candidate_count": len(grasps),
        "top_score": None if first is None else first.score,
        "top_width_m": None if first is None else first.width_m,
        "top_translation_m": (
            None
            if first is None
            else first.transform_camera_grasp[:3, 3].round(6).tolist()
        ),
        "response_file": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
