#!/usr/bin/env python3
"""实时预览云端抓取姿态（SAM 3.1 + GraspNet）的推理结果。

只做三件事：取一帧对齐 RGB-D → 调云端 /v1/grasp → 把返回的候选夹爪位姿画回图像。
不规划、不发 Action、不碰 /arm_cmd，机械臂一动不动；真机跑着也可以随时开。

画法（相机光学系：x 右 / y 下 / z 前，GraspNet 约定 rot 第 1 列 = approach，
第 2 列 = 两指连线方向）：

    圆圈  抓取点 translation_m 投影
    箭头  接近方向 approach（长度 --axis-len，默认 6cm）
    短线  两指开合宽度 width_m（沿第 2 列 ±width/2）
    文字  #序号 score 宽度 深度，排名第一的加粗

窗口里按 q 退出、s 存图；没有显示器（$DISPLAY 为空）时自动只发 ROS 话题 + 打印。
RViz 里加一个 Image display 指向 --image-topic（默认 /x5a_grasp_preview/image）即可。
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from x5a_remote_grasp.api_client import GraspAPIClient, RemoteAPIError
from x5a_remote_grasp.geometry import parse_api_grasps

DEFAULT_COLOR_TOPIC = "/camera/camera/color/image_raw"
DEFAULT_DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
DEFAULT_INFO_TOPIC = "/camera/camera/color/camera_info"
DEFAULT_CAMERA_NODE = "/camera/camera"
DEFAULT_IMAGE_TOPIC = "/x5a_grasp_preview/image"
DEFAULT_BASE_URL = "http://118.253.179.80:8300"
DEFAULT_SAVE_DIR = "~/.local/share/x5a_remote_grasp/preview"


class GraspPreview(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("x5a_grasp_preview")
        self.args = args
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame: Optional[Tuple[Any, Any, Any]] = None
        self._frame_stamp: float = 0.0
        self._depth_scale: Optional[float] = None
        self._scale_logged = False
        self.client = GraspAPIClient(
            args.base_url,
            api_key_env=args.api_key_env,
            connect_timeout=args.connect_timeout,
            inference_timeout=args.timeout,
        )

        self._color_sub = message_filters.Subscriber(
            self, Image, args.color_topic, qos_profile=qos_profile_sensor_data
        )
        self._depth_sub = message_filters.Subscriber(
            self, Image, args.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self._info_sub = message_filters.Subscriber(
            self, CameraInfo, args.info_topic, qos_profile=qos_profile_sensor_data
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub, self._info_sub],
            queue_size=10,
            slop=0.06,
        )
        self._sync.registerCallback(self._on_frame)

        self._param_client = self.create_client(
            GetParameters, args.camera_node.rstrip("/") + "/get_parameters"
        )

        image_qos = QoSProfile(depth=1)
        image_qos.reliability = ReliabilityPolicy.RELIABLE
        self._image_pub = self.create_publisher(Image, args.image_topic, image_qos)
        self.get_logger().info(
            f"grasp preview: {args.color_topic} + {args.depth_topic} -> {args.base_url}/v1/grasp"
        )

    # ---------------------------------------------------------------- 取帧
    def _on_frame(self, color: Any, depth: Any, info: Any) -> None:
        with self._lock:
            self._frame = (color, depth, info)
            self._frame_stamp = time.time()

    def take_frame(self) -> Optional[Tuple[Any, Any, Any]]:
        with self._lock:
            frame = self._frame
            age = time.time() - self._frame_stamp
        if frame is None or age > self.args.max_frame_age:
            return None
        return frame

    # ------------------------------------------------------- 深度比例/编码
    def depth_scale(self) -> float:
        """跟 remote_grasp_node 一样：先问相机节点 depth_module.depth_units。"""
        if self._depth_scale is not None:
            return self._depth_scale
        scale = self.args.depth_scale
        if scale <= 0.0 and self._param_client.wait_for_service(timeout_sec=1.0):
            request = GetParameters.Request()
            request.names = [self.args.depth_units_parameter]
            future = self._param_client.call_async(request)
            deadline = time.time() + 1.5
            while not future.done() and time.time() < deadline:
                time.sleep(0.02)
            if future.done() and future.result() is not None and len(future.result().values) == 1:
                value = future.result().values[0]
                if value.type == ParameterType.PARAMETER_DOUBLE:
                    scale = float(value.double_value)
                elif value.type == ParameterType.PARAMETER_INTEGER:
                    scale = float(value.integer_value)
                elif value.type == ParameterType.PARAMETER_STRING:
                    try:
                        scale = float(value.string_value)
                    except ValueError:
                        scale = 0.0
        if not (1.0e-6 <= scale <= 0.01):
            scale = 0.001  # D435i 默认 depth_units
        self._depth_scale = float(scale)
        return self._depth_scale

    def encode(self, frame: Tuple[Any, Any, Any]) -> Tuple[bytes, bytes, float, float, float, float]:
        color_msg, depth_msg, info_msg = frame
        bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))
        encoding = str(depth_msg.encoding or "").upper()
        scale = self.depth_scale()
        if encoding in {"16UC1", "MONO16"}:
            if depth.dtype != np.uint16:
                raise RuntimeError(f"{encoding} 解出 {depth.dtype}，期望 uint16")
            depth_for_png = depth
        elif encoding == "32FC1":
            depth_for_png = np.rint(np.nan_to_num(depth, nan=0.0) / scale).astype(np.uint16)
        else:
            raise RuntimeError(f"不支持的深度编码 {depth_msg.encoding!r}")
        if not self._scale_logged:
            self.get_logger().info(
                f"depth encoding={encoding} scale={scale:.9g} m/unit "
                f"api_depth_scale={1.0 / scale:.9g}"
            )
            self._scale_logged = True
        ok_rgb, rgb_png = cv2.imencode(".png", bgr)
        ok_depth, depth_png = cv2.imencode(".png", depth_for_png)
        if not ok_rgb or not ok_depth:
            raise RuntimeError("PNG 编码失败")
        k = list(info_msg.k)
        return (
            rgb_png.tobytes(),
            depth_png.tobytes(),
            float(k[0]),
            float(k[4]),
            float(k[2]),
            float(k[5]),
        )

    # ---------------------------------------------------------------- 推理
    def infer(self, frame: Tuple[Any, Any, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        rgb_png, depth_png, fx, fy, cx, cy = self.encode(frame)
        fields = {
            "prompt": self.args.prompt,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "depth_scale": 1.0 / self.depth_scale(),
            "min_depth": self.args.min_depth,
            "max_depth": self.args.max_depth,
            "threshold": self.args.threshold,
            "mask_dilation_px": self.args.mask_dilation_px,
            "num_point": self.args.num_point,
            "top_k": self.args.top_k,
            "collision_thresh": self.args.collision_thresh,
            "voxel_size": self.args.voxel_size,
            "target_distance_thresh": self.args.target_distance_thresh,
            "max_gripper_width": self.args.max_gripper_width,
        }
        payload = self.client.grasp(rgb_png, depth_png, fields)
        return parse_api_grasps(payload), dict(payload)

    def segment_overlay(self, frame: Tuple[Any, Any, Any]) -> Optional[np.ndarray]:
        """可选：让 SAM 单独再切一次，拿它的叠加图（看模型选中的是哪个物体）。"""
        color_msg, _depth_msg, _info_msg = frame
        bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        ok, png = cv2.imencode(".png", bgr)
        if not ok:
            return None
        # GraspAPIClient 没有暴露 /segment_text，这里复用它的事务（含鉴权与超时处理）
        from x5a_remote_grasp.api_client import encode_multipart

        body, content_type = encode_multipart(
            {
                "prompt": self.args.prompt,
                "threshold": self.args.mask_threshold,
                "viz": "true",
            },
            {"file": ("color.png", "image/png", png.tobytes())},
        )
        result = self.client._json_request(
            "POST",
            "/segment_text",
            body=body,
            headers={"Content-Type": content_type},
            authenticated=True,
        )
        if result.status != 200 or not isinstance(result.payload, dict):
            return None
        viz_b64 = result.payload.get("viz_jpeg_b64")
        if not viz_b64:
            return None
        raw = np.frombuffer(base64.b64decode(viz_b64), dtype=np.uint8)
        overlay = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if overlay is None:
            return None
        if overlay.shape[:2] != bgr.shape[:2]:
            overlay = cv2.resize(overlay, (bgr.shape[1], bgr.shape[0]))
        return cv2.addWeighted(bgr, 0.45, overlay, 0.55, 0.0)

    # ---------------------------------------------------------------- 画图
    def draw(
        self,
        canvas: np.ndarray,
        grasps: List[Any],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        header: str,
    ) -> np.ndarray:
        def project(point: np.ndarray) -> Optional[Tuple[int, int]]:
            if point[2] <= 1.0e-6:
                return None
            u = int(round(fx * point[0] / point[2] + cx))
            v = int(round(fy * point[1] / point[2] + cy))
            return u, v

        for rank, grasp in enumerate(grasps):
            transform = np.asarray(grasp.transform_camera_grasp, dtype=float)
            point = transform[:3, 3]
            approach = transform[:3, 0]
            jaws = transform[:3, 1]
            center = project(point)
            if center is None:
                continue
            tip = project(point + approach * self.args.axis_len)
            half = self.args.jaw_len_scale * float(grasp.width_m) / 2.0
            left = project(point + jaws * half)
            right = project(point - jaws * half)
            if rank == 0:
                color, thickness, text_color = (0, 0, 255), 3, (0, 0, 255)
            else:
                color, thickness, text_color = (0, 190, 0), 1, (0, 190, 0)
            if tip is not None:
                cv2.arrowedLine(canvas, center, tip, color, thickness, tipLength=0.25)
            if left is not None and right is not None:
                cv2.line(canvas, left, right, color, thickness)
            cv2.circle(canvas, center, 3 if rank == 0 else 2, color, thickness)
            if self.args.labels and (rank < self.args.label_top or rank == 0):
                req = getattr(grasp, "required_opening_m", None)
                req_text = f" req={req * 1000:.0f}mm" if req is not None else ""
                flag = "!" if req is not None and req - grasp.width_m > 0.005 else ""
                label = (
                    f"{flag}#{rank} {grasp.score:.2f} "
                    f"{grasp.width_m * 1000:.0f}mm{req_text} z={point[2] * 100:.0f}cm"
                )
                origin = (center[0] + 6, center[1] - 6 - 12 * (rank % 3))
                cv2.putText(
                    canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 if rank == 0 else 0.38, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 if rank == 0 else 0.38, text_color, 1, cv2.LINE_AA,
                )

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (32, 32, 32), -1)
        cv2.putText(
            canvas, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA
        )
        return canvas

    def publish(self, canvas: np.ndarray, stamp: Any) -> None:
        msg = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.frame_id
        self._image_pub.publish(msg)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实时预览云端抓取姿态推理结果（只推理，不运动）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", default="cube", help="SAM 文本提示，例如 'cube' / 'black pen'")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="SAM_GRASPNET_API_KEY")
    parser.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--info-topic", default=DEFAULT_INFO_TOPIC)
    parser.add_argument("--camera-node", default=DEFAULT_CAMERA_NODE)
    parser.add_argument("--depth-units-parameter", default="depth_module.depth_units")
    parser.add_argument("--depth-scale", type=float, default=0.0, help="米/单位；0=问相机节点")
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC, help="标注图发布话题；空字符串=不发")
    parser.add_argument("--frame-id", default="camera_color_optical_frame")
    parser.add_argument("--interval", type=float, default=0.0, help="两次推理之间额外等待秒数")
    parser.add_argument("--once", action="store_true", help="只推理一次就退出（快速自检）")
    parser.add_argument("--max-frame-age", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--mask-dilation-px", type=int, default=8)
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--target-distance-thresh", type=float, default=0.06)
    parser.add_argument("--max-gripper-width", type=float, default=0.088)
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=1.50)
    parser.add_argument("--mask", dest="mask", action="store_true", default=True,
                        help="额外叠加 SAM 分割图（多花 2~3 秒）")
    parser.add_argument("--no-mask", dest="mask", action="store_false")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--axis-len", type=float, default=0.06, help="接近箭头长度，米")
    parser.add_argument("--jaw-len-scale", type=float, default=1.0)
    parser.add_argument("--labels", dest="labels", action="store_true", default=True)
    parser.add_argument("--no-labels", dest="labels", action="store_false")
    parser.add_argument("--label-top", type=int, default=3, help="标注前 N 个候选")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="按 s 存图目录；空字符串=不存")
    parser.add_argument("--window", dest="window", action="store_true", default=None,
                        help="强制开窗口（默认：有 $DISPLAY 就开）")
    parser.add_argument("--no-window", dest="window", action="store_false")
    parser.add_argument("--connect-timeout", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not os.environ.get(args.api_key_env):
        key_env = Path.home() / ".config/x5a_remote_grasp/key.env"
        if key_env.is_file():
            for line in key_env.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    name, _, value = line.partition("=")
                    os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    if not os.environ.get(args.api_key_env):
        print(f"环境变量 {args.api_key_env} 没有设置（先 source scripts/stack_env.sh）", file=sys.stderr)
        return 2

    use_window = args.window if args.window is not None else bool(os.environ.get("DISPLAY"))
    if use_window:
        try:
            cv2.namedWindow("x5a grasp preview", cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            print(f"OpenCV 窗口打不开（{exc}），改为只发话题/打印", file=sys.stderr)
            use_window = False

    rclpy.init(args=None)
    node = GraspPreview(args)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("=" * 60)
    print(" X5A 抓取姿态实时预览（只推理，不规划、不运动）")
    print(f" prompt : {args.prompt!r}   top_k={args.top_k} threshold={args.threshold}")
    print(f" remote : {args.base_url}/v1/grasp")
    print(f" image  : {args.image_topic or '(不发)'}")
    print(f" window : {'开（q 退出 / s 存图）' if use_window else '关'}")
    print("=" * 60)

    save_dir = Path(os.path.expanduser(args.save_dir)) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    cycle = 0
    last_frame_warn = 0.0
    try:
        while rclpy.ok():
            frame = node.take_frame()
            if frame is None:
                if time.time() - last_frame_warn > 5.0:
                    print("等待同步的 RGB-D 帧……（相机是否在跑？ros2 topic hz /camera/camera/color/image_raw）")
                    last_frame_warn = time.time()
                time.sleep(0.5)
                continue
            cycle += 1
            started = time.time()
            try:
                grasps, payload = node.infer(frame)
            except RemoteAPIError as exc:
                print(f"[{time.strftime('%H:%M:%S')}] 推理失败：{exc}（HTTP {exc.status}）")
                time.sleep(2.0)
                continue
            except Exception as exc:
                print(f"[{time.strftime('%H:%M:%S')}] 推理失败：{exc}")
                time.sleep(2.0)
                continue
            latency = time.time() - started

            color_msg, _depth_msg, info_msg = frame
            bgr = node.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            canvas = bgr
            if args.mask:
                overlay = node.segment_overlay(frame)
                if overlay is not None:
                    canvas = overlay
            k = list(info_msg.k)
            instance = payload.get("instance") or {}
            header = (
                f"{time.strftime('%H:%M:%S')} '{args.prompt}' n={len(grasps)} "
                f"sam={instance.get('sam_score')} {latency:.1f}s"
            )
            canvas = node.draw(canvas, grasps, float(k[0]), float(k[4]), float(k[2]), float(k[5]), header)
            node.publish(canvas, color_msg.header.stamp)

            if grasps:
                top = grasps[0]
                xyz = np.asarray(top.transform_camera_grasp, dtype=float)[:3, 3]
                print(
                    f"[{time.strftime('%H:%M:%S')}] 第 {cycle} 轮：{len(grasps)} 个候选，"
                    f"top score={top.score:.3f} 宽={top.width_m * 1000:.1f}mm "
                    f"抓取点=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})m "
                    f"sam_score={instance.get('sam_score')} 云端 {payload.get('latency_s')}s 本机 {latency:.1f}s"
                )
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 第 {cycle} 轮：0 个候选（提示词没切出目标？）")

            if save_dir is not None and args.once:
                path = save_dir / f"preview_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), canvas)
                print(f"已存 {path}")

            if use_window:
                cv2.imshow("x5a grasp preview", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and save_dir is not None:
                    path = save_dir / f"preview_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(str(path), canvas)
                    print(f"已存 {path}")
            if args.once:
                break
            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，退出")
    finally:
        if use_window:
            cv2.destroyAllWindows()
        # 顺序要紧：先让 executor.spin() 退出，再销毁节点/关闭 rclpy，
        # 否则边 spin 边 shutdown 会在退出时 terminate。
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
