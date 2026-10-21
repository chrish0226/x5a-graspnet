#!/usr/bin/env python3
"""Multi-color RGB-D cube localization and movable-box localization."""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from x5a_vision.top_plane import deproject, fit_horizontal_top, height_mask, plane_rectangle_center, rgbd_dimensions_match
from x5a_vision.box_rim import locate_rim, rim_candidates, adaptive_rim_candidates, rim_color_mask
from x5a_vision.box_track import BoxTrack
from x5a_vision.rgbd_pairing import RgbdPairBuffer
from x5a_vision.clearance import DEFAULT_BUFFER_FILE, read_buffer, buffer_policy, buffer_candidates, base_cloud, blockers_from_sides, buffer_surface, BufferSurfaceIndex


def wrap_half_pi(yaw: float) -> float:
    """Wrap yaw into [0, pi/2) — a cube has 90-degree face symmetry."""
    period = 0.5 * math.pi
    wrapped = math.fmod(float(yaw), period)
    if wrapped < 0.0:
        wrapped += period
    return wrapped


COLORS = ("red", "white", "orange")
DRAW_COLORS = {
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "orange": (0, 140, 255),
}


class CubeDetector(Node):
    def __init__(self) -> None:
        super().__init__("cube_detector")
        self._declare_parameters()
        self._load_parameters()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.color_msg: Optional[Image] = None
        self.info: Optional[CameraInfo] = None
        self.histories = {name: deque(maxlen=self.window) for name in COLORS}
        self.yaw_histories = {name: deque(maxlen=self.window) for name in COLORS}
        self.cube_last_seen = {name: 0.0 for name in COLORS}
        self.box_track = BoxTrack(self.box_window, self.box_max_std_xy, self.box_reset_distance, self.box_miss_grace)
        self.box_history = self.box_track.history
        self.box_yaws = deque(maxlen=self.box_window)
        self.last_box_rim = None
        self.last_box_rim_time = 0.0
        self._normal_bounds = dict(self.bounds)
        self._normal_expected_xy = self.expected_xy.copy()
        self.observation_side = 'current'
        self.cover_history = {name: deque(maxlen=4) for name in COLORS}
        self.declare_parameter('clearing.buffer_file', DEFAULT_BUFFER_FILE)
        self.buffer_request = None
        self.create_subscription(String, '/wrist_pick/buffer_request', self.on_buffer_request,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, '/wrist_pick/observation_side', self.on_observation_side,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.color_frames = deque(maxlen=6)
        self.rgbd_pairs = RgbdPairBuffer(self.max_rgbd_gap)
        self._frame_transform = None
        self.last_log_ns = 0
        self.depth_encoding_logged = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_pubs = {
            name: self.create_publisher(
                PoseStamped, f"/x5a_vision/{name}_cube_pose", 10
            )
            for name in COLORS
        }
        self.stable_pubs = {
            name: self.create_publisher(
                Bool, f"/x5a_vision/{name}_cube_stable", 10
            )
            for name in COLORS
        }
        self.box_pose_pub = self.create_publisher(
            PoseStamped, "/x5a_vision/box_pose", 10
        )
        self.box_stable_pub = self.create_publisher(
            Bool, "/x5a_vision/box_stable", 10
        )
        # Backward-compatible aliases for the previously verified red-cube path.
        self.point_pub = self.create_publisher(
            PointStamped, "/x5a_vision/object_point_camera", 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, "/x5a_vision/object_pose", 10
        )
        self.stable_pub = self.create_publisher(
            Bool, "/x5a_vision/detection_stable", 10
        )
        self.debug_pub = self.create_publisher(
            Image, "/x5a_vision/debug_image", 10
        )
        self.details_pub = self.create_publisher(String, '/x5a_vision/detection_details', 10)
        # Processing HD RGB-D can take longer than a camera frame interval.
        # Keep only the newest pending image so detection does not work through
        # an old queue while the robot waits for a fresh stop-and-look result.
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        # The camera offers reliable images. HD RGB exceeds the default SHM
        # segment; best-effort reception loses fragmented samples on this host.
        # Keep depth-one reliable RGB intake independent of frame processing.
        color_qos = QoSProfile(depth=1, reliability=self.color_reliability)
        self.intake_group = ReentrantCallbackGroup()
        self.create_subscription(Image, self.color_topic, self.on_color, color_qos,
                                 callback_group=self.intake_group)
        self.create_subscription(Image, self.depth_topic, self.on_depth_received, image_qos,
                                 callback_group=self.intake_group)
        self.create_subscription(
            CameraInfo, self.info_topic, self.on_info, qos_profile_sensor_data,
            callback_group=self.intake_group
        )
        self.create_timer(.02, self.process_pending_pair)
        self.get_logger().info(
            f"multi-color RGB-D detector color={self.color_topic} "
            f"depth={self.depth_topic} {self.optical_frame}->{self.base_frame}; "
            "box=dark interior ray/table-plane intersection"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('calibration_label', '')
        defaults = {
            "color_topic": "/camera/camera/color/image_raw",
            "color_reliability": "best_effort",
            "aligned_depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "optical_frame": "camera_color_optical_frame",
            "base_frame": "base_link",
            "detection.min_area_px": 500.0,
            "detection.max_area_px": 5000.0,
            "detection.min_rectangularity": 0.50,
            "detection.max_aspect_ratio": 1.8,
            "detection.open_iterations": 1,
            "detection.close_iterations": 2,
            "detection.inner_erode_px": 5,
            "detection.top_face_mode": "depth_percentile",
            "detection.face_size_tolerance_m": .012,
            "detection.max_rgbd_gap_s": .04,
            "depth.scale_16uc1": 0.001,
            "depth.min_m": 0.20,
            "depth.max_m": 0.80,
            "depth.near_percentile": 55.0,
            "object.size_z": 0.03,
            "detection.max_cube_layers": 1,
            "table.z": -0.006,
            "table.top_tolerance": 0.03,
            "workspace.x_min": 0.08,
            "workspace.x_max": 0.41,
            "workspace.y_min": 0.08,
            "workspace.y_max": 0.48,
            "workspace.r_max": 0.54,
            "workspace.z_min": -0.015,
            "workspace.z_max": 0.08,
            "workspace.expected_x": 0.20,
            "workspace.expected_y": 0.33,
            'workspace.allow_mirrored_side': False,
            "filter.window": 15,
            "filter.max_std_x": 0.005,
            "filter.max_std_y": 0.005,
            "filter.max_std_z": 0.008,
            "filter.reset_distance_m": 0.03,
            "filter.miss_grace_s": 2.0,
            "box.hsv_lower": [0, 0, 0],
            "box.hsv_upper": [179, 255, 45],
            "box.min_area_px": 12000.0,
            "box.max_area_px": 60000.0,
            "box.min_aspect_ratio": 1.15,
            "box.max_aspect_ratio": 2.50,
            "box.min_rectangularity": 0.80,
            "box.border_margin_px": 5,
            "box.open_iterations": 1,
            "box.close_iterations": 2,
            "box.filter_window": 15,
            "box.max_std_xy": 0.010,
            "box.reset_distance_m": 0.05,
            "box.rim_height_m": 0.0,
            "box.rim_tolerance_m": 0.005,
            "box.adaptive_height_range_m": 0.0,
            "box.plane_tolerance_m": .003,
            "box.max_rim_tilt_deg": 8.0,
            "box.inner_size_x": 0.14,
            "box.inner_size_y": 0.09,
            'box.miss_grace_s': 2.0,
            'box.min_supported_sides': 4,
            'box.size_tolerance_m': .018,
            'box.edge_support_fraction': .35,
            'box.gray_rim_enabled': False,
            'box.gray_rim_max_s': 100,
            'box.gray_rim_max_v': 165,
        }
        color_defaults = {
            "red": ([0, 80, 55], [7, 255, 255], True, [170, 80, 55], [179, 255, 255]),
            "white": ([0, 0, 170], [179, 90, 255], False, [0, 0, 0], [0, 0, 0]),
            "orange": ([8, 90, 70], [28, 255, 255], False, [0, 0, 0], [0, 0, 0]),
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        for name, (lower, upper, second, lower2, upper2) in color_defaults.items():
            self.declare_parameter(f"colors.{name}.hsv_lower", lower)
            self.declare_parameter(f"colors.{name}.hsv_upper", upper)
            self.declare_parameter(f"colors.{name}.use_second_hsv", second)
            self.declare_parameter(f"colors.{name}.hsv_lower_2", lower2)
            self.declare_parameter(f"colors.{name}.hsv_upper_2", upper2)

    def _load_parameters(self) -> None:
        g = lambda name: self.get_parameter(name).value
        self.color_topic = str(g("color_topic"))
        reliability = str(g('color_reliability'))
        if reliability not in ('reliable', 'best_effort'):
            raise ValueError('color_reliability must be reliable or best_effort')
        self.color_reliability = (ReliabilityPolicy.RELIABLE if reliability == 'reliable'
                                  else ReliabilityPolicy.BEST_EFFORT)
        self.depth_topic = str(g("aligned_depth_topic"))
        self.info_topic = str(g("camera_info_topic"))
        self.optical_frame = str(g("optical_frame"))
        self.base_frame = str(g("base_frame"))
        self.color_ranges = {}
        for name in COLORS:
            self.color_ranges[name] = {
                "lower": np.array(g(f"colors.{name}.hsv_lower"), np.uint8),
                "upper": np.array(g(f"colors.{name}.hsv_upper"), np.uint8),
                "second": bool(g(f"colors.{name}.use_second_hsv")),
                "lower2": np.array(g(f"colors.{name}.hsv_lower_2"), np.uint8),
                "upper2": np.array(g(f"colors.{name}.hsv_upper_2"), np.uint8),
            }
        self.min_area = float(g("detection.min_area_px"))
        self.max_area = float(g("detection.max_area_px"))
        self.min_rectangularity = float(g("detection.min_rectangularity"))
        self.max_aspect = float(g("detection.max_aspect_ratio"))
        self.open_iterations = int(g("detection.open_iterations"))
        self.close_iterations = int(g("detection.close_iterations"))
        self.inner_erode_px = int(g("detection.inner_erode_px"))
        self.top_face_mode = str(g('detection.top_face_mode'))
        self.face_size_tolerance = float(g('detection.face_size_tolerance_m'))
        self.max_rgbd_gap = float(g('detection.max_rgbd_gap_s'))
        if not 0 < self.face_size_tolerance <= .02 or not 0 < self.max_rgbd_gap <= .10:
            raise ValueError('invalid face size tolerance or RGB-D synchronization tolerance')
        if self.top_face_mode not in ('depth_percentile', 'base_plane'):
            raise ValueError('unsupported detection.top_face_mode')
        self.depth_scale_16u = float(g("depth.scale_16uc1"))
        self.depth_min = float(g("depth.min_m"))
        self.depth_max = float(g("depth.max_m"))
        self.near_percentile = float(g("depth.near_percentile"))
        self.object_height = float(g("object.size_z"))
        self.max_cube_layers = int(g('detection.max_cube_layers'))
        if not 1 <= self.max_cube_layers <= 3:
            raise ValueError('one to three cube layers are supported')
        self.cube_miss_grace = float(g('filter.miss_grace_s'))
        self.table_z = float(g("table.z"))
        self.table_tol = float(g("table.top_tolerance"))
        self.bounds = {
            key: float(g(f"workspace.{key}"))
            for key in ("x_min", "x_max", "y_min", "y_max", "r_max", "z_min", "z_max")
        }
        self.expected_xy = np.array(
            [float(g("workspace.expected_x")), float(g("workspace.expected_y"))]
        )
        self.allow_mirrored_side = bool(g('workspace.allow_mirrored_side'))
        self.window = int(g("filter.window"))
        self.max_std = np.array(
            [
                float(g("filter.max_std_x")),
                float(g("filter.max_std_y")),
                float(g("filter.max_std_z")),
            ]
        )
        self.reset_distance = float(g("filter.reset_distance_m"))
        self.box_hsv = (
            np.array(g("box.hsv_lower"), np.uint8),
            np.array(g("box.hsv_upper"), np.uint8),
        )
        self.box_min_area = float(g("box.min_area_px"))
        self.box_max_area = float(g("box.max_area_px"))
        self.box_min_aspect = float(g("box.min_aspect_ratio"))
        self.box_max_aspect = float(g("box.max_aspect_ratio"))
        self.box_min_rectangularity = float(g("box.min_rectangularity"))
        self.box_border_margin = int(g("box.border_margin_px"))
        self.box_open_iterations = int(g("box.open_iterations"))
        self.box_close_iterations = int(g("box.close_iterations"))
        self.box_window = int(g("box.filter_window"))
        self.box_max_std_xy = float(g("box.max_std_xy"))
        self.box_reset_distance = float(g("box.reset_distance_m"))
        self.box_rim_height = float(g("box.rim_height_m"))
        self.box_rim_tolerance = float(g("box.rim_tolerance_m"))
        self.box_adaptive_range = float(g('box.adaptive_height_range_m'))
        self.box_plane_tolerance = float(g('box.plane_tolerance_m'))
        self.box_max_tilt = float(g('box.max_rim_tilt_deg'))
        self.box_gray_rim = bool(g('box.gray_rim_enabled'))
        self.box_gray_max_s = int(g('box.gray_rim_max_s'))
        self.box_gray_max_v = int(g('box.gray_rim_max_v'))
        if self.box_gray_rim:
            rim_color_mask(np.zeros((1,1,3), np.uint8), np.zeros((1,1), bool),
                           self.box_gray_max_s, self.box_gray_max_v)
        if (not 0 <= self.box_adaptive_range <= .02 or not .001 <= self.box_plane_tolerance <= .005
                or not 0 < self.box_max_tilt <= 10):
            raise ValueError('invalid adaptive box depth parameters')
        self.box_inner_size = (float(g("box.inner_size_x")), float(g("box.inner_size_y")))
        self.box_miss_grace = float(g('box.miss_grace_s'))
        self.rim_options = dict(min_supported_sides=int(g('box.min_supported_sides')),
                               size_tolerance=float(g('box.size_tolerance_m')),
                               edge_support_fraction=float(g('box.edge_support_fraction')))
        if (not 0 <= self.box_miss_grace <= 3 or self.rim_options['min_supported_sides'] not in (3,4)
                or not 0 < self.rim_options['size_tolerance'] <= .03
                or not .1 <= self.rim_options['edge_support_fraction'] <= .5):
            raise ValueError('invalid box rim/tracking parameters')

    @staticmethod
    def stamp_sec(msg) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def on_color(self, msg: Image) -> None:
        with self.lock:
            self.color_msg = msg
            self.color_frames.append(msg)
            self.rgbd_pairs.colors.append(msg)

    def on_depth_received(self, msg: Image) -> None:
        with self.lock:
            self.rgbd_pairs.depths.append(msg)

    def process_pending_pair(self) -> None:
        with self.lock:
            pair = self.rgbd_pairs.take_latest(self.get_clock().now().nanoseconds)
        if pair is not None:
            color, depth = pair
            self.on_depth(depth, color)

    def on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    def depth_meters(self, image: np.ndarray, encoding: str) -> np.ndarray:
        enc = encoding.upper()
        if enc in ("16UC1", "MONO16"):
            scale = self.depth_scale_16u
        elif enc == "32FC1":
            scale = 1.0
        else:
            raise ValueError(f"unsupported depth encoding {encoding}")
        if not self.depth_encoding_logged:
            self.get_logger().info(f"depth encoding={encoding} scale_to_m={scale}")
            self.depth_encoding_logged = True
        return np.asarray(image, dtype=np.float32) * scale

    def clean_mask(self, mask: np.ndarray, open_count: int, close_count: int) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        if open_count:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_count)
        if close_count:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_count)
        return mask

    def mask_for_color(self, hsv: np.ndarray, name: str) -> np.ndarray:
        spec = self.color_ranges[name]
        mask = cv2.inRange(hsv, spec["lower"], spec["upper"])
        if spec["second"]:
            mask |= cv2.inRange(hsv, spec["lower2"], spec["upper2"])
        return self.clean_mask(mask, self.open_iterations, self.close_iterations)

    def lookup_transform(self, stamp):
        return self.tf_buffer.lookup_transform(
            self.base_frame,
            self.optical_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=0.2),
        )

    def transform_matrix(self, stamp):
        key = (stamp.sec, stamp.nanosec)
        cached = getattr(self, '_frame_transform', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        tf = self.lookup_transform(stamp).transform
        q = np.array([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])
        q /= np.linalg.norm(q)
        x, y, z, w = q
        T = np.eye(4)
        T[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        T[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
        self._frame_transform = (key, T)
        return T

    def valid_cube_top(self, z):
        return any(abs(z - (self.table_z + layer*self.object_height)) <= self.table_tol
                   for layer in range(1, self.max_cube_layers+1))

    def plane_measurement(self, inner, depth, camera_info, stamp):
        try:
            T = self.transform_matrix(stamp)
        except TransformException as exc:
            self.get_logger().warn(f'TF unavailable: {exc}', throttle_duration_sec=2.)
            return None
        K = np.asarray(camera_info.k).reshape(3, 3)
        ys, xs = np.nonzero(inner & np.isfinite(depth)
                            & (depth >= self.depth_min) & (depth <= self.depth_max))
        camera = deproject(xs, ys, depth[ys, xs], K)
        base = camera @ T[:3, :3].T+T[:3, 3]
        base = base[base[:, 2] >= self.table_z+.5*self.object_height]
        top = fit_horizontal_top(base)
        if top is None:
            return None
        p, rect = plane_rectangle_center(top)
        pc = T[:3, :3].T @ (p-T[:3, 3])
        if pc[2] <= 0:
            return None
        if any(abs(side-self.object_height) > self.face_size_tolerance for side in rect[1]):
            return None
        corners = cv2.boxPoints(rect)
        edge = corners[1]-corners[0]
        yaw = wrap_half_pi(math.atan2(float(edge[1]), float(edge[0])))
        return p, pc, (K[0, 0]*pc[0]/pc[2]+K[0, 2], K[1, 1]*pc[1]/pc[2]+K[1, 2]), yaw

    def contour_measurement(
        self, contour, depth: np.ndarray, camera_info: CameraInfo, stamp
    ) -> Optional[Dict]:
        area = float(cv2.contourArea(contour))
        if area < self.min_area or area > self.max_area:
            return None
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) <= 1.0:
            return None
        aspect = max(rw, rh) / min(rw, rh)
        rectangularity = area / (rw * rh)
        if aspect > self.max_aspect or rectangularity < self.min_rectangularity:
            return None
        h, w = depth.shape[:2]
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bx < 2 or by < 2 or bx+bw > w-2 or by+bh > h-2:
            return None  # A cropped face cannot provide the cube's full center.
        mask = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(mask, [contour-np.array([bx, by])], -1, 255, -1)
        k = max(1, self.inner_erode_px | 1)
        inner = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1,
                          borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        cropped_depth = depth[by:by+bh, bx:bx+bw]
        if self.top_face_mode == 'base_plane':
            cropped_k = np.asarray(camera_info.k).copy()
            cropped_k[2] -= bx
            cropped_k[5] -= by
            measurement = self.plane_measurement(inner, cropped_depth, SimpleNamespace(k=cropped_k), stamp)
            if measurement is None:
                return None
            p, pc, pixel, yaw = measurement
            pixel = (pixel[0]+bx, pixel[1]+by)
            point_camera = PointStamped()
            point_camera.header.stamp, point_camera.header.frame_id = stamp, self.optical_frame
            point_camera.point.x, point_camera.point.y, point_camera.point.z = map(float, pc)
            b = self.bounds
            in_workspace = (b['x_min'] <= p[0] <= b['x_max'] and b['y_min'] <= p[1] <= b['y_max']
                            and b['z_min'] <= p[2] <= b['z_max'] and np.hypot(*p[:2]) <= b['r_max'])
            on_table = self.valid_cube_top(p[2])
            return dict(area=area, pixel=pixel, depth=float(pc[2]), camera=point_camera,
                        base=p, bbox=cv2.boundingRect(contour), contour=contour, rect=rect, yaw=yaw,
                        reject='WS' if not in_workspace else ('Z' if not on_table else None),
                        score=float(np.linalg.norm(p[:2]-self.expected_xy)))
        top = self.select_top_face(inner, cropped_depth)
        if top is None:
            return None
        u, v, z = top
        u, v = u+bx, v+by
        if not (1 <= u < w - 1 and 1 <= v < h - 1):
            return None
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None
        point_camera = PointStamped()
        point_camera.header.stamp = stamp
        point_camera.header.frame_id = self.optical_frame
        point_camera.point.x = (u - cx) * z / fx
        point_camera.point.y = (v - cy) * z / fy
        point_camera.point.z = z
        try:
            point_base = do_transform_point(point_camera, self.lookup_transform(stamp))
        except TransformException as exc:
            self.get_logger().warn(
                f"TF unavailable: {exc}", throttle_duration_sec=2.0
            )
            return None
        p = np.array(
            [point_base.point.x, point_base.point.y, point_base.point.z], dtype=float
        )
        b = self.bounds
        in_workspace = (
            b["x_min"] <= p[0] <= b["x_max"]
            and b["y_min"] <= p[1] <= b["y_max"]
            and b["z_min"] <= p[2] <= b["z_max"]
            and float(np.hypot(p[0], p[1])) <= b["r_max"]
        )
        on_table = self.valid_cube_top(p[2])
        x, y, wb, hb = cv2.boundingRect(contour)
        yaw = self.cube_yaw_base(rect, z, camera_info, stamp)
        reject = None
        if not in_workspace:
            reject = "WS"
        elif not on_table:
            reject = "Z"
        return {
            "area": area,
            "pixel": (u, v),
            "depth": z,
            "camera": point_camera,
            "base": p,
            "bbox": (x, y, wb, hb),
            "contour": contour,
            "rect": rect,
            "yaw": yaw,
            "reject": reject,
            "score": float(np.linalg.norm(p[:2] - self.expected_xy)),
        }

    def select_top_face(
        self, inner: np.ndarray, depth: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Return (u, v, z) of the cube top face, not the camera-facing side.

        A slanted view makes red/orange sides light up. Those side pixels are
        nearer than the top. If the depth span is large, keep the farther
        cluster (top). If the blob is almost flat, keep the nearer cluster
        (top-down view, same as the old white-cube path).
        """
        ys, xs = np.nonzero(inner)
        if xs.size < 20:
            return None
        zvals = depth[ys, xs]
        good = (
            np.isfinite(zvals)
            & (zvals >= self.depth_min)
            & (zvals <= self.depth_max)
        )
        xs, ys, zvals = xs[good], ys[good], zvals[good]
        if xs.size < 10:
            return None
        span = float(zvals.max() - zvals.min())
        if span >= 0.012:
            keep = zvals >= float(np.percentile(zvals, 55.0))
        else:
            keep = zvals <= float(np.percentile(zvals, self.near_percentile))
        xs, ys, zvals = xs[keep], ys[keep], zvals[keep]
        if xs.size < 8:
            return None
        return float(np.mean(xs)), float(np.mean(ys)), float(np.median(zvals))

    def cube_yaw_base(self, rect, depth_z: float, camera_info: CameraInfo, stamp) -> Optional[float]:
        """Yaw of one cube edge in base_link, wrapped to [0, pi/2)."""
        corners = cv2.boxPoints(rect)
        d01 = float(np.linalg.norm(corners[1] - corners[0]))
        d12 = float(np.linalg.norm(corners[2] - corners[1]))
        a, b = (corners[0], corners[1]) if d01 >= d12 else (corners[1], corners[2])
        pa = self.pixel_to_base(float(a[0]), float(a[1]), depth_z, camera_info, stamp)
        pb = self.pixel_to_base(float(b[0]), float(b[1]), depth_z, camera_info, stamp)
        if pa is None or pb is None:
            return None
        yaw = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
        return wrap_half_pi(yaw)

    def pixel_to_base(
        self, u: float, v: float, z: float, camera_info: CameraInfo, stamp
    ) -> Optional[np.ndarray]:
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0 or z <= 1e-6:
            return None
        point_camera = PointStamped()
        point_camera.header.stamp = stamp
        point_camera.header.frame_id = self.optical_frame
        point_camera.point.x = (u - cx) * z / fx
        point_camera.point.y = (v - cy) * z / fy
        point_camera.point.z = float(z)
        try:
            point_base = do_transform_point(point_camera, self.lookup_transform(stamp))
        except TransformException:
            return None
        return np.array(
            [point_base.point.x, point_base.point.y, point_base.point.z], dtype=float
        )

    def ray_table_intersection(
        self, u: float, v: float, camera_info: CameraInfo, stamp
    ) -> Optional[np.ndarray]:
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None
        origin = PointStamped()
        origin.header.stamp = stamp
        origin.header.frame_id = self.optical_frame
        ray = PointStamped()
        ray.header = origin.header
        ray.point.x = (u - cx) / fx
        ray.point.y = (v - cy) / fy
        ray.point.z = 1.0
        try:
            transform = self.lookup_transform(stamp)
            origin_base = do_transform_point(origin, transform)
            ray_base = do_transform_point(ray, transform)
        except TransformException as exc:
            self.get_logger().warn(
                f"box TF unavailable: {exc}", throttle_duration_sec=2.0
            )
            return None
        o = np.array(
            [origin_base.point.x, origin_base.point.y, origin_base.point.z], dtype=float
        )
        r = np.array(
            [ray_base.point.x, ray_base.point.y, ray_base.point.z], dtype=float
        )
        direction = r - o
        if abs(direction[2]) < 1e-9:
            return None
        scale = (self.table_z - o[2]) / direction[2]
        if scale <= 0.0:
            return None
        p = o + scale * direction
        p[2] = self.table_z
        return p

    def in_cube_workspace_xy(self, x: float, y: float) -> bool:
        b = self.bounds
        return (
            b["x_min"] <= x <= b["x_max"]
            and b["y_min"] <= y <= b["y_max"]
            and float(math.hypot(x, y)) <= b["r_max"]
        )

    def box_measurement(
        self, contour, camera_info: CameraInfo, stamp, image_shape, depth=None
    ) -> Optional[Dict]:
        area = float(cv2.contourArea(contour))
        if area < self.box_min_area or area > self.box_max_area:
            return None
        (u, v), (rw, rh), angle = cv2.minAreaRect(contour)
        if min(rw, rh) <= 1.0:
            return None
        aspect = max(rw, rh) / min(rw, rh)
        rectangularity = area / (rw * rh)
        if not (
            self.box_min_aspect <= aspect <= self.box_max_aspect
            and rectangularity >= self.box_min_rectangularity
        ):
            return None
        corners = cv2.boxPoints(((u, v), (rw, rh), angle))
        if self.box_rim_height > 0:
            if depth is None:
                return None
            mask = np.zeros(depth.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            try:
                T = self.transform_matrix(stamp)
            except TransformException:
                return None
            K = np.asarray(camera_info.k).reshape(3, 3)
            rim = locate_rim(depth, mask.astype(bool), K, T,
                             self.table_z+self.box_rim_height, self.box_rim_tolerance,
                             self.box_inner_size)
            if rim is None:
                return None
            camera_corners = (rim['corners']-T[:3, 3]) @ T[:3, :3]
            projected = camera_corners @ K.T
            corners = projected[:, :2]/projected[:, 2, None]
            center_camera = T[:3, :3].T @ (rim['center']-T[:3, 3])
            pixel = K @ center_camera
            return dict(pixel=tuple(pixel[:2]/pixel[2]), base=rim['center'],
                        contour=contour, corners=corners, area=area, yaw=rim['yaw'])
        # The black sponge is intentionally used as a simple drop target.  Its
        # centre remains usable when the large sponge contour touches an image
        # edge, so do not reject an otherwise valid candidate solely for that.
        base = self.ray_table_intersection(float(u), float(v), camera_info, stamp)
        if base is None:
            return None
        # The drop box can sit slightly outside the cube grasp envelope.
        if float(math.hypot(base[0], base[1])) > 0.70:
            return None
        return {
            "pixel": (float(u), float(v)),
            "base": base,
            "contour": contour,
            "corners": corners,
            "area": area,
            "aspect": aspect,
            "rectangularity": rectangularity,
        }

    @staticmethod
    def update_history(history, point: np.ndarray, reset_distance: float):
        if history:
            center = np.median(np.asarray(history), axis=0)
            if float(np.linalg.norm(point - center)) > reset_distance:
                history.clear()
        history.append(point)
        values = np.asarray(history)
        return np.median(values, axis=0), np.std(values, axis=0)

    def publish_pose(
        self, publisher, stamp, xyz: np.ndarray, yaw: Optional[float] = None
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.base_frame
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        if yaw is None:
            pose.pose.orientation.w = 1.0
        else:
            half = 0.5 * float(yaw)
            pose.pose.orientation.z = math.sin(half)
            pose.pose.orientation.w = math.cos(half)
        publisher.publish(pose)
        return pose

    @staticmethod
    def median_yaw_90(yaws) -> Optional[float]:
        if not yaws:
            return None
        angs = np.asarray(yaws, dtype=float) * 4.0
        yaw = 0.25 * math.atan2(float(np.mean(np.sin(angs))), float(np.mean(np.cos(angs))))
        return wrap_half_pi(yaw)

    def on_buffer_request(self, msg):
        try:
            data = json.loads(msg.data)
            xy = data.get('xy_m')
            if (data.get('base_frame') == self.base_frame and data.get('observation_side') == self.observation_side
                    and isinstance(data.get('token'), str) and 0 < len(data['token']) <= 64
                    and isinstance(xy, list) and len(xy) == 2
                    and all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in xy)):
                self.buffer_request = data
        except (ValueError, TypeError, AttributeError):
            pass

    def on_observation_side(self, msg):
        if not self.allow_mirrored_side or msg.data not in ('current', 'mirror') or msg.data == self.observation_side:
            return
        self.observation_side = msg.data
        self.buffer_request = None
        self.bounds = dict(self._normal_bounds)
        self.expected_xy = self._normal_expected_xy.copy()
        if msg.data == 'mirror':
            self.bounds['y_min'], self.bounds['y_max'] = -self._normal_bounds['y_max'], -self._normal_bounds['y_min']
            self.expected_xy[1] *= -1
        for name in COLORS:
            self.histories[name].clear()
            self.yaw_histories[name].clear()
            self.cover_history[name].clear()
            self.cube_last_seen[name] = 0.0
            self.stable_pubs[name].publish(Bool(data=False))
        self.box_track.clear()
        self.box_yaws.clear()
        self.last_box_rim = None
        self.box_stable_pub.publish(Bool(data=False))

    def on_depth(self, msg: Image, color_msg: Optional[Image] = None) -> None:
        processing_started = time.perf_counter()
        if color_msg is None:
            with self.lock:
                color_msg = min(self.color_frames, key=lambda c: abs(self.stamp_sec(c)-self.stamp_sec(msg))) if self.color_frames else None
        info = self.info
        if (
            color_msg is None
            or info is None
            or abs(self.stamp_sec(color_msg) - self.stamp_sec(msg)) > self.max_rgbd_gap
        ):
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth = self.depth_meters(
                self.bridge.imgmsg_to_cv2(msg, "passthrough"), msg.encoding
            )
        except Exception as exc:
            self.get_logger().error(f"image conversion failed: {exc}")
            return
        if not rgbd_dimensions_match(bgr.shape, depth.shape, info.width, info.height):
            self.get_logger().warn(f'discarding unaligned image sizes: color={bgr.shape[:2]} depth={depth.shape}',
                                   throttle_duration_sec=2.)
            for name in COLORS:
                self.histories[name].clear()
                self.yaw_histories[name].clear()
                self.stable_pubs[name].publish(Bool(data=False))
            self.box_track.clear()
            self.box_yaws.clear()
            self.box_stable_pub.publish(Bool(data=False))
            return
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        debug = bgr.copy()
        status_parts = []
        display_rows = []
        details = {}
        visible_tops = {}
        color_masks = {}

        height_gate = None
        if self.top_face_mode == 'base_plane':
            try:
                height_gate = height_mask(depth, info.k, self.transform_matrix(msg.header.stamp),
                                          self.table_z+.5*self.object_height,
                                          self.table_z+self.max_cube_layers*self.object_height+self.table_tol,
                                          self.depth_min, self.depth_max)
            except TransformException as exc:
                self.get_logger().warn(f'TF unavailable: {exc}', throttle_duration_sec=2.)
                height_gate = np.zeros(depth.shape, dtype=bool)

        for name in COLORS:
            mask = self.mask_for_color(hsv, name)
            color_masks[name] = mask
            color_visible = np.count_nonzero(mask) >= self.min_area
            if height_gate is not None:
                mask = mask.copy()
                mask[~height_gate] = 0
                mask = self.clean_mask(mask, self.open_iterations, self.close_iterations)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            measured = [
                result
                for result in (
                    self.contour_measurement(c, depth, info, msg.header.stamp)
                    for c in contours
                )
                if result is not None
            ]
            candidates = [item for item in measured if item.get("reject") is None]
            rejected = [item for item in measured if item.get("reject")]
            color = DRAW_COLORS[name]
            for item in rejected:
                u, v = item["pixel"]
                cv2.drawContours(debug, [item["contour"]], -1, color, 1)
                cv2.putText(
                    debug,
                    f"{name.upper()} {item['reject']}",
                    (round(u) + 8, round(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
            if not candidates:
                if time.monotonic()-self.cube_last_seen[name] > self.cube_miss_grace:
                    self.histories[name].clear()
                    self.yaw_histories[name].clear()
                self.stable_pubs[name].publish(Bool(data=False))
                if name == "red":
                    self.stable_pub.publish(Bool(data=False))
                status_parts.append(f"{name}=NO")
                reason = 'missing'
                if color_visible:
                    reason = ('out_of_workspace' if any(item['reject'] == 'WS' for item in rejected)
                              else 'height_out_of_range' if any(item['reject'] == 'Z' for item in rejected)
                              else 'no_top_face')
                details[name] = reason
                display_rows.append((f'{name.upper()}: ' + ('NO VALID TOP FACE' if color_visible else 'NO DETECTION'), color))
                continue
            best = min(candidates, key=lambda item: item["score"])
            now = time.monotonic()
            if now-self.cube_last_seen[name] > max(3.0, 2*self.cube_miss_grace):
                self.histories[name].clear()
                self.yaw_histories[name].clear()
            self.cube_last_seen[name] = now
            median, std = self.update_history(
                self.histories[name], best["base"], self.reset_distance
            )
            if len(self.histories[name]) <= 1:
                self.yaw_histories[name].clear()
            if best.get("yaw") is not None:
                self.yaw_histories[name].append(float(best["yaw"]))
            yaw = self.median_yaw_90(self.yaw_histories[name])
            stable = len(self.histories[name]) >= self.window and bool(
                np.all(std <= self.max_std)
            )
            details[name] = 'stable' if stable else 'tracking'
            if stable:
                visible_tops[name] = median.copy()
            pose = self.publish_pose(
                self.pose_pubs[name], msg.header.stamp, median, yaw
            )
            self.stable_pubs[name].publish(Bool(data=stable))
            if name == "red":
                self.pose_pub.publish(pose)
                self.point_pub.publish(best["camera"])
                self.stable_pub.publish(Bool(data=stable))
            color = DRAW_COLORS[name]
            cv2.drawContours(debug, [best["contour"]], -1, color, 2)
            u, v = best["pixel"]
            cv2.circle(debug, (round(u), round(v)), 5, color, -1)
            cv2.putText(
                debug,
                f"{name.upper()} {'STABLE' if stable else 'TRACKING'}",
                (round(u) + 8, round(v) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
            )
            status_parts.append(
                f"{name}=({median[0]:.3f},{median[1]:.3f},{median[2]:.3f})/{stable}"
            )
            x_mm, y_mm, z_mm = median*1000.
            display_rows.append((f"{name.upper()} X={x_mm:.1f} Y={y_mm:.1f} Z={z_mm:.1f} mm "
                                 f"{'STABLE' if stable else 'TRACKING'}", color))

        # Side colors plus measured upper-layer height establish support; no
        # stale hidden-cube coordinates enter this calculation.
        try:
            points, xs, ys = base_cloud(depth, info.k, self.transform_matrix(msg.header.stamp),
                                        self.depth_min, self.depth_max)
            pixel_colors = np.full(len(points), '', dtype='<U6')
            for name in COLORS:
                pixel_colors[color_masks[name][ys, xs] != 0] = name
            relations = blockers_from_sides(points, pixel_colors, visible_tops,
                [name for name in COLORS if details[name] == 'no_top_face'],
                self.table_z, self.object_height, max_layers=self.max_cube_layers)
            confirmed = {}
            for name in COLORS:
                self.cover_history[name].append(relations.get(name))
                history = self.cover_history[name]
                if len(history) == 4 and history[-1] is not None and len(set(history)) == 1:
                    confirmed[name] = history[-1]
            clearing = {'blocked_by': confirmed,
                        'stamp_ns': msg.header.stamp.sec*1_000_000_000 + msg.header.stamp.nanosec}
            try:
                filename = str(self.get_parameter('clearing.buffer_file').value)
                mode, gap = buffer_policy(filename)
                clearing['mode'] = mode
                if mode == 'auto':
                    upper_tops = {color: top for color, top in visible_tops.items()
                                  if top[2] > self.table_z+1.5*self.object_height and self.valid_cube_top(top[2])}
                    surface_index = BufferSurfaceIndex(points, self.table_z, self.bounds) if upper_tops else None
                    clearing['buffer_search'] = {color: {} for color in upper_tops}
                    clearing['buffer_candidates'] = {color: buffer_candidates(points, top[:2],
                        self.object_height, self.table_z, self.bounds, gap, surface_index,
                        clearing['buffer_search'][color]) for color, top in upper_tops.items()}
                else:
                    xy = read_buffer(filename, self.base_frame, self.observation_side)
                    clearing['buffer'] = {'xy_m': xy, 'base_frame': self.base_frame,
                                          'observation_side': self.observation_side,
                                          **buffer_surface(points, xy, self.table_z)}
            except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
                clearing['buffer'] = {'empty': False, 'reason': str(exc)}
            requested = self.buffer_request
            if (requested and requested.get('base_frame') == self.base_frame
                    and requested.get('observation_side') == self.observation_side):
                clearing['selected_buffer'] = {**requested,
                    **buffer_surface(points, requested['xy_m'], self.table_z)}
            details['clearing'] = clearing
        except (TransformException, ValueError) as exc:
            for history in self.cover_history.values():
                history.clear()
            details['clearing'] = {'blocked_by': {}, 'buffer': {'empty': False, 'reason': str(exc)}}

        box_mask = cv2.inRange(hsv, self.box_hsv[0], self.box_hsv[1])
        box_mask = self.clean_mask(
            box_mask, self.box_open_iterations, self.box_close_iterations
        )
        if self.box_rim_height > 0:
            box_candidates = []
            support_mask = (rim_color_mask(hsv, box_mask, self.box_gray_max_s, self.box_gray_max_v)
                            if self.box_gray_rim else box_mask.astype(bool))
            try:
                T = self.transform_matrix(msg.header.stamp)
                K = np.asarray(info.k).reshape(3,3)
                locate = adaptive_rim_candidates if self.box_adaptive_range > 0 else rim_candidates
                options = dict(self.rim_options)
                if self.box_adaptive_range > 0:
                    options.update(height_range=self.box_adaptive_range,
                                   plane_tolerance=self.box_plane_tolerance, max_tilt_deg=self.box_max_tilt)
                    if self.last_box_rim is not None and time.monotonic()-self.last_box_rim_time <= self.box_miss_grace:
                        options['hint']=self.last_box_rim
                located=locate(depth, support_mask, K, T,
                    self.table_z+self.box_rim_height, self.box_rim_tolerance, self.box_inner_size, **options)
                if not located and options.pop('hint',None) is not None:
                    located=locate(depth, support_mask, K, T,
                        self.table_z+self.box_rim_height, self.box_rim_tolerance, self.box_inner_size, **options)
                for rim, contour in located:
                    camera_corners = (rim['corners']-T[:3,3]) @ T[:3,:3]
                    projected = camera_corners @ K.T
                    if np.any(projected[:,2] <= 0):
                        continue
                    corners = projected[:,:2]/projected[:,2,None]
                    pixel = K @ (T[:3,:3].T @ (rim['center']-T[:3,3]))
                    if not self.in_cube_workspace_xy(*rim['center'][:2]):
                        continue
                    box_candidates.append(dict(pixel=tuple(pixel[:2]/pixel[2]), base=rim['center'],
                         contour=contour, corners=corners, area=cv2.contourArea(contour), yaw=rim['yaw'],
                         rim_z_m=float(rim['center'][2]), tilt_deg=rim.get('tilt_deg',0.),
                         rim_fit=rim,
                         plane_residual_m=rim.get('plane_residual_m')))
            except TransformException:
                pass
        else:
            box_contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            box_candidates = [result for result in (self.box_measurement(c, info, msg.header.stamp, bgr.shape, depth)
                              for c in box_contours) if result is not None]
        if box_candidates:
            best_box = max(box_candidates, key=lambda item: item["area"])
            if best_box.get('rim_fit') is not None:
                self.last_box_rim=best_box['rim_fit']
                self.last_box_rim_time=time.monotonic()
            median, stable = self.box_track.update(best_box['base'][:2], time.monotonic())
            # Only box-center XY is used. Z is the known table plane and is not
            # interpreted as box height or reflective depth.
            box_xyz = np.array([median[0], median[1], self.table_z], dtype=float)
            if len(self.box_history) <= 1:
                self.box_yaws.clear()
            if best_box.get('yaw') is not None:
                self.box_yaws.append(best_box['yaw'])
            yaw = (.5*math.atan2(np.mean(np.sin(np.asarray(self.box_yaws)*2)),
                                np.mean(np.cos(np.asarray(self.box_yaws)*2)))) if self.box_yaws else None
            details['box'] = {'status': 'stable' if stable else 'tracking',
                              'rim_z_m': best_box.get('rim_z_m'),
                              'tilt_deg': best_box.get('tilt_deg'),
                              'plane_residual_m': best_box.get('plane_residual_m')}
            self.publish_pose(self.box_pose_pub, msg.header.stamp, box_xyz, yaw)
            self.box_stable_pub.publish(Bool(data=stable))
            corners = np.int32(np.round(best_box["corners"]))
            cv2.polylines(debug, [corners], True, (255, 255, 0), 2)
            u, v = best_box["pixel"]
            cv2.circle(debug, (round(u), round(v)), 6, (255, 255, 0), -1)
            cv2.putText(
                debug,
                f"BOX CENTER {'STABLE' if stable else 'TRACKING'}",
                (round(u) - 70, round(v)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 0),
                2,
            )
            status_parts.append(
                f"box=({median[0]:.3f},{median[1]:.3f})/{stable} rim_z={best_box.get('rim_z_m',0.):.3f}"
            )
            display_rows.append((f"BOX X={median[0]*1000:.1f} Y={median[1]*1000:.1f} mm "
                                 f"RIM Z={best_box.get('rim_z_m',0.)*1000:.1f} "
                                 f"{'STABLE' if stable else 'TRACKING'}", (255,255,0)))
        else:
            stable = self.box_track.missing(time.monotonic())
            details['box'] = {'status': 'short_gap' if stable else 'no_valid_depth'}
            if not self.box_history:
                self.box_yaws.clear()
                self.last_box_rim=None
            # Do not publish another pose or stamp: the last actual measurement
            # keeps its age and expires at the task server even if the camera stops.
            self.box_stable_pub.publish(Bool(data=stable))
            status_parts.append('box=SHORT_GAP' if stable else 'box=NO')
            display_rows.append(('BOX: LAST POSITION (SHORT GAP)' if stable else 'BOX: NO VALID RIM', (255,255,0)))

        cv2.rectangle(debug, (0, 0), (min(debug.shape[1]-1, 640), 137), (25, 25, 25), -1)
        cv2.putText(debug, 'Object position - robot base coordinates', (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (230, 230, 230), 1)
        for i, (line, color) in enumerate(display_rows):
            cv2.putText(debug, line, (10, 44+27*i), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1)
        calibration_label = str(self.get_parameter('calibration_label').value)
        if calibration_label:
            cv2.rectangle(debug, (0, 137), (min(debug.shape[1]-1, 640), 165), (25, 25, 25), -1)
            cv2.putText(debug, calibration_label, (10, 157), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (0, 200, 255), 1)
        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)
        details['performance'] = {
            'processing_ms': round((time.perf_counter()-processing_started)*1000, 1),
            'image_age_ms': round((self.get_clock().now().nanoseconds
                                  -msg.header.stamp.sec*1_000_000_000-msg.header.stamp.nanosec)/1e6, 1),
            'rgbd_gap_ms': round(abs(self.stamp_sec(color_msg)-self.stamp_sec(msg))*1000, 2),
        }
        self.details_pub.publish(String(data=json.dumps(details)))
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_log_ns > 1_000_000_000:
            self.last_log_ns = now_ns
            self.get_logger().debug("MULTI DETECTION " + " ".join(status_parts))


def detection_executor(node):
    # TransformListener uses a separate reentrant callback group. A second
    # worker lets TF arrive while an image callback waits for its timestamp;
    # Paired-frame processing remains serialized; raw RGB/depth/info intake
    # continues independently, including RGB that arrives after its depth.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    return executor


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CubeDetector()
    executor = detection_executor(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
