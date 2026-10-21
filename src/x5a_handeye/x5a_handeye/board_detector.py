#!/usr/bin/env python3
"""Detect ChArUco with camera intrinsics, quality checks and raw observations."""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from x5a_handeye.transforms import R_to_quat, rvec_tvec_to_T
from x5a_handeye.charuco_detection import (
    DETECTOR_VERSION, detect_charuco, pose_quality, quality_passes)


def load_board(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    spec = cfg["board"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    size = (int(spec["squares_x"]), int(spec["squares_y"]))
    square = float(spec["square_length_m"])
    marker = float(spec["marker_length_m"])
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(size, square, marker, dictionary)
    else:
        board = cv2.aruco.CharucoBoard_create(size[0], size[1], square, marker, dictionary)
    return dictionary, board, cfg


class BoardDetector(Node):
    def __init__(self) -> None:
        super().__init__("x5a_board_detector")
        self.declare_parameter("board_config", "")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("min_charuco_corners", 6)
        self.declare_parameter('max_reprojection_error_px', 2.0)
        self.declare_parameter('max_board_tilt_deg', 75.0)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("board_frame", "calibration_board")

        board_config = str(self.get_parameter("board_config").value)
        if not board_config:
            from ament_index_python.packages import get_package_share_directory

            board_config = str(
                Path(get_package_share_directory("x5a_handeye")) / "config" / "board.yaml"
            )
        self.dictionary, self.board, self.cfg = load_board(board_config)
        self.min_corners = int(self.get_parameter("min_charuco_corners").value)
        self.max_reprojection = float(self.get_parameter('max_reprojection_error_px').value)
        self.max_tilt = float(self.get_parameter('max_board_tilt_deg').value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.board_frame = str(self.get_parameter("board_frame").value)
        self.params = (
            cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "DetectorParameters")
            else cv2.aruco.DetectorParameters_create()
        )
        self.bridge = CvBridge()
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.camera_frame = ""
        self.info_size = None
        self.last_T: Optional[np.ndarray] = None
        self.last_quality = 0
        self.tf_broadcaster = TransformBroadcaster(self)

        image_topic = str(self.get_parameter("image_topic").value)
        info_topic = str(self.get_parameter("camera_info_topic").value)
        self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info_topic, self.on_info, qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, "/calibration_board/pose", 10)
        self.observation_pub = self.create_publisher(String, '/calibration_board/observation', 10)
        self.debug_pub = self.create_publisher(Image, "/x5a_handeye/debug_image", 10)
        self.get_logger().info(
            f"board detector ready image={image_topic} info={info_topic} config={board_config}"
        )

    def on_info(self, msg: CameraInfo) -> None:
        K = np.array(msg.k, dtype=float).reshape(3, 3)
        D = np.array(msg.d, dtype=float).reshape(-1)
        if (not np.isfinite(K).all() or not np.isfinite(D).all()
                or K[0, 0] <= 0 or K[1, 1] <= 0 or not msg.header.frame_id):
            self.K = None
            return
        self.K, self.D = K, D
        self.camera_frame = msg.header.frame_id
        self.info_size = (msg.width, msg.height)

    def detect(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return detect_charuco(gray, self.dictionary, self.board, self.params, self.K, self.D)

    def estimate_pose(self, charuco_corners, charuco_ids) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_ids) < self.min_corners
            or self.K is None
        ):
            return False, None, None
        if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.board, self.K, self.D, None, None
            )
            return bool(ok), rvec, tvec
        if hasattr(self.board, "getChessboardCorners"):
            all_obj = self.board.getChessboardCorners()
        else:
            all_obj = self.board.chessboardCorners
        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        obj = np.asarray(all_obj, dtype=np.float32)[ids]
        img = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        return bool(ok), rvec, tvec

    def on_image(self, msg: Image) -> None:
        if self.K is None:
            return
        if (msg.header.frame_id != self.camera_frame
                or (msg.width, msg.height) != self.info_size):
            self.get_logger().warn('image and CameraInfo frame/size disagree', throttle_duration_sec=5.0)
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        marker_corners, marker_ids, charuco_corners, charuco_ids = self.detect(frame)
        debug = frame.copy()
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
        n_corners = 0 if charuco_ids is None else int(len(charuco_ids))
        try:
            ok, rvec, tvec = self.estimate_pose(charuco_corners, charuco_ids)
        except cv2.error:
            ok, rvec, tvec = False, None, None
        quality = None
        if ok:
            quality = pose_quality(self.board, charuco_corners, charuco_ids, self.K, self.D, rvec, tvec)
            ok = quality_passes(quality, self.min_corners, self.max_reprojection, self.max_tilt)
        if ok:
            if charuco_corners is not None and charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)
            cv2.drawFrameAxes(debug, self.K, self.D, rvec, tvec, 0.05)
            T = rvec_tvec_to_T(rvec, tvec)
            self.last_T = T
            self.last_quality = n_corners
            xyz = T[:3, 3]
            qx, qy, qz, qw = R_to_quat(T[:3, :3])
            pose = PoseStamped()
            pose.header = msg.header
            if not pose.header.frame_id:
                pose.header.frame_id = self.camera_frame
            pose.pose.position.x = float(xyz[0])
            pose.pose.position.y = float(xyz[1])
            pose.pose.position.z = float(xyz[2])
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            self.pose_pub.publish(pose)
            observation = {
                'stamp_ns': int(msg.header.stamp.sec)*1000000000 + int(msg.header.stamp.nanosec),
                'camera_frame': self.camera_frame,
                'image_size': [int(msg.width), int(msg.height)],
                'K': self.K.tolist(), 'D': self.D.tolist(),
                'board': self.cfg['board'], 'detector_version': DETECTOR_VERSION,
                'opencv_version': cv2.__version__,
                'quality': quality,
                'marker_ids': np.asarray(marker_ids).reshape(-1).tolist(),
                'marker_corners_px': [np.asarray(c).reshape(-1, 2).tolist() for c in marker_corners],
                'charuco_ids': np.asarray(charuco_ids).reshape(-1).tolist(),
                'charuco_corners_px': np.asarray(charuco_corners).reshape(-1, 2).tolist(),
            }
            self.observation_pub.publish(String(data=json.dumps(observation, allow_nan=False)))
            if self.publish_tf:
                tf = TransformStamped()
                tf.header = pose.header
                tf.child_frame_id = self.board_frame
                tf.transform.translation.x = float(xyz[0])
                tf.transform.translation.y = float(xyz[1])
                tf.transform.translation.z = float(xyz[2])
                tf.transform.rotation.x = qx
                tf.transform.rotation.y = qy
                tf.transform.rotation.z = qz
                tf.transform.rotation.w = qw
                self.tf_broadcaster.sendTransform(tf)
            cv2.putText(
                debug,
                f"corners={n_corners} rms={quality['reprojection_rms_px']:.2f}px tilt={quality['tilt_deg']:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            self.last_T = None
            self.last_quality = n_corners
            cv2.putText(
                debug,
                (f"REJECT rms={quality['reprojection_rms_px']:.2f}px tilt={quality['tilt_deg']:.1f}"
                 if quality else f"NO POSE corners={n_corners}"),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BoardDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
