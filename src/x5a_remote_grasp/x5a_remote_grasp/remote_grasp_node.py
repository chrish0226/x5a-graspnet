#!/usr/bin/env python3
"""Synchronized RGB-D to remote GraspNet, TF, RViz, IK and plan-only MoveIt."""
from __future__ import annotations

import json
import base64
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.msg import CollisionObject, DisplayTrajectory, PlanningScene
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from x5a_task_interfaces.action import PlanRemoteGrasp

from x5a_remote_grasp.api_client import GraspAPIClient, RemoteAPIError
from x5a_remote_grasp.geometry import (
    RemoteGrasp,
    approach_from_below,
    approach_yaw,
    candidate_waypoints,
    is_steep_topdown,
    matrix_to_quaternion,
    parallel_jaw_rotations,
    parse_api_grasps,
    rpy_rotation,
    transform_from_xyz_quat,
    validate_homogeneous,
    validate_rotation_matrix,
)
from x5a_remote_grasp.moveit_plan import PlanOnlyMoveIt, SequencePlan
from x5a_remote_grasp.reachability import ReachabilityTable
from x5a_remote_grasp.grasp_quality import (
    GripperGeometry, PreparedGraspScene, assess_tool_grasp, depth_cloud, height_statistics,
    graspnet_tool_transform, recenter_tool, voxel_cloud,
)
from x5a_remote_grasp.parallel_geometry import ordered_geometry_map
from x5a_remote_grasp.place_geometry import (
    object_in_tool, placement_tool, support_surface, supported_center, preserves_upright, support_prism,
)


@dataclass(frozen=True)
class SynchronizedFrame:
    color: Image
    depth: Image
    info: CameraInfo
    received_monotonic: float


@dataclass(frozen=True)
class EncodedFrame:
    rgb_png: bytes
    depth_png: bytes
    source_frame: str
    stamp: object
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m_per_unit: float
    depth_scale_divisor: float
    encoding: str
    width: int
    height: int
    stamp_delta_s: float


class RemoteGraspNode(Node):
    def __init__(self) -> None:
        super().__init__("x5a_remote_grasp")
        self._declare_parameters()
        self._load_parameters()
        if not self.safety_plan_only:
            raise RuntimeError("safety.plan_only must remain true; this node cannot execute motion")
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        self.frame_lock = threading.Lock()
        self.goal_lock = threading.Lock()
        self.latest_frame: Optional[SynchronizedFrame] = None
        self.latest_joints: Optional[List[float]] = None
        self.last_successful_seed: Optional[List[float]] = None
        self.target_cloud = None
        self.scene_cloud = None
        self.last_plan_validation = None
        self.place_support_hull = None
        self.place_support_zrange = None
        self.place_relation = None
        self.place_object_cloud_tool = None
        self.place_grasp_rotation = None
        self.goal_active = False
        self.depth_scale_logged = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.color_sub = message_filters.Subscriber(
            self, Image, self.color_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, self.info_topic, qos_profile=qos_profile_sensor_data
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.synchronizer.registerCallback(self._synchronized_callback)

        parameter_service = self.camera_node.rstrip("/") + "/get_parameters"
        self.camera_parameter_client = self.create_client(
            GetParameters, parameter_service, callback_group=self.callback_group
        )
        marker_qos = QoSProfile(depth=1)
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.marker_publisher = self.create_publisher(
            MarkerArray, self.marker_topic, marker_qos
        )
        # VOLATILE matches MoveIt and the executor; TRANSIENT_LOCAL here
        # made /display_planned_path undeliverable after stack restarts.
        path_qos = QoSProfile(depth=1)
        path_qos.reliability = ReliabilityPolicy.RELIABLE
        path_qos.durability = DurabilityPolicy.VOLATILE
        self.display_publisher = self.create_publisher(
            DisplayTrajectory, str(self.get_parameter('topics.display_path').value), path_qos
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )
        self.gripper_plan_publisher = self.create_publisher(
            JointTrajectory, "/x5a_remote_grasp/planned_gripper_close", marker_qos
        )
        self.health_server = self.create_service(
            Trigger,
            self.health_service,
            self._health_callback,
            callback_group=self.callback_group,
        )
        self.snapshot_server = self.create_service(
            Trigger,
            self.save_snapshot_service,
            self._save_snapshot_callback,
            callback_group=self.callback_group,
        )
        self.validation_server = self.create_service(
            Trigger, str(self.get_parameter('validation_service').value), self._validate_latest_callback,
            callback_group=self.callback_group,
        )
        self.executor_state_client = self.create_client(
            Trigger, '/x5a_remote_grasp/executor_state', callback_group=self.callback_group,
        )
        self.action_server = ActionServer(
            self,
            PlanRemoteGrasp,
            self.action_name,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group,
        )
        self.planner = PlanOnlyMoveIt(
            self,
            self.callback_group,
            group_name=self.planning_group,
            tcp_frame=self.tcp_frame,
            move_action=self.move_action,
            ik_service=self.ik_service,
            validity_service=self.validity_service,
            apply_scene_service=self.apply_scene_service,
            allowed_time=self.planning_time,
            result_timeout=self.planning_result_timeout,
            ik_timeout=self.ik_timeout,
            attempts=self.planning_attempts,
            velocity_scaling=self.velocity_scaling,
            acceleration_scaling=self.acceleration_scaling,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            cartesian_step=self.cartesian_step,
            min_cartesian_fraction=self.min_cartesian_fraction,
            parallel_ik_service=str(self.get_parameter('planning.parallel_ik_service').value),
            parallel_ik_workers=int(self.get_parameter('planning.parallel_ik_workers').value),
            parallel_ik_fallback_serial=bool(self.get_parameter('planning.parallel_ik_fallback_serial').value),
        )
        self.planner.expected_place_open_position = self.max_gripper_width/2
        self.get_logger().info(
            "remote grasp ready in STRICT PLAN-ONLY mode; "
            f"rgb={self.color_topic} depth={self.depth_topic} info={self.info_topic}; "
            f"orientation_mode={self.orientation_mode} "
            f"geometry_workers={self.geometry_workers} "
            f"parallel_ik_workers={int(self.get_parameter('planning.parallel_ik_workers').value)} "
            f"tcp_mapping_confirmed={int(self.tcp_mapping_confirmed)}"
        )
        if not self.tcp_mapping_confirmed:
            self.get_logger().warn(
                "GraspNet grasp-frame to tool0 mapping is UNCONFIRMED; API/TF/RViz "
                "validation is allowed, but IK and planning are gated off"
            )

    def _declare_parameters(self) -> None:
        defaults = {
            "remote.base_url": "http://192.168.0.129:8765",
            "remote.api_key_env": "SAM_GRASPNET_API_KEY",
            "remote.connect_timeout_s": 3.0,
            "remote.inference_timeout_s": 120.0,
            "remote.threshold": 0.45,
            "remote.mask_dilation_px": 8,
            "remote.num_point": 20000,
            "remote.top_k_default": 10,
            "remote.top_k_max": 40,
            "remote.collision_thresh": 0.01,
            "remote.voxel_size": 0.01,
            "remote.target_distance_thresh": 0.06,
            "topics.color": "/camera/camera/color/image_raw",
            "topics.aligned_depth": "/camera/camera/aligned_depth_to_color/image_raw",
            "topics.camera_info": "/camera/camera/color/camera_info",
            "sync.queue_size": 10,
            "sync.slop_s": 0.06,
            "sync.max_stamp_delta_s": 0.08,
            "sync.max_frame_age_s": 1.0,
            "depth.camera_node": "/camera/camera",
            "depth.units_parameter": "depth_module.depth_units",
            "depth.scale_m_per_unit_override": 0.0,
            "depth.allow_32fc1_reencode": False,
            "depth.min_m_default": 0.10,
            "depth.max_m_default": 1.50,
            "frames.base": "base_link",
            "frames.tf_timeout_s": 0.5,
            "frames.max_tf_stamp_delta_s": 0.10,
            "workspace.x_min": 0.08,
            "workspace.x_max": 0.41,
            "workspace.y_min": 0.08,
            "workspace.y_max": 0.48,
            "workspace.z_min": -0.005,
            "workspace.z_max": 0.50,
            "workspace.r_max": 0.54,
            "workspace.table_clearance_m": 0.005,
            "gripper.max_width_m": 0.088,
            "gripper.width_margin_m": 0.006,
            "gripper.width_to_finger_ratio": 0.5,
            "gripper.joint_names": ["joint7", "joint8"],
            "grasp.approach_axis": "x",
            "grasp.pregrasp_distance_m": 0.05,
            "grasp.lift_distance_m": 0.08,
            "grasp.retreat_distance_m": 0.08,
            "grasp.max_pregrasp_z": 0.11,
            "grasp.orientation_mode": "graspnet",
            "grasp.template_rpy": [0.0, 1.45, 0.0],
            "grasp.tcp_backoff_m": 0.05,
            "grasp.template_pregrasp_distance_m": 0.04,
            "grasp.min_tool0_height_m": 0.05,
            "grasp.aperture_audit_enabled": True,
            "grasp.aperture_tolerance_m": 0.005,
            "grasp.center_band_enabled": True,
            "grasp.center_band_margin_m": 0.007,
            "grasp.finger_length_m": 0.10,
            "grasp.finger_thickness_m": 0.01,
            "grasp.finger_height_m": 0.02,
            "grasp.pad_length_m": 0.03,
            "grasp.tip_offset_m": 0.015,
            "grasp.native_budget_fraction": 0.45,
            "grasp.max_target_motion_m": 0.01,
            "grasp.max_capture_to_execute_s": 60.0,
            "grasp.approach_offset_m": 0.0,
            "grasp.max_approach_z": 0.35,
            "grasp.allow_reachable_fallback": True,
            "poses.ready_joints": [0.72, 0.85, 0.95, -0.55, 0.0, 0.0],
            "grasp.place_pitches": [1.45, 1.15, 0.90, 0.75],
            "grasp.place_release_clearances_m": [0.010, 0.015, 0.018, 0.020],
            "grasp.transit_pitches": [1.15, 0.90],
            # 可解性阶梯：姿态按可达性表/经验规律排序后逐个降级尝试
            "grasp.pitch_ladder": [1.45, 1.15, 0.90, 0.75],
            "grasp.yaw_ladder_deg": [0, 45, -45, 180],
            "grasp.min_score": 0.0,
            "grasp.template_fallback": True,
            "grasp.place_offset_search_m": 0.03,
            "grasp.place_offset_step_m": 0.015,
            "grasp.place_offset_max_points": 9,
            "planning.total_budget_s": 30.0,
            "planning.geometry_workers": 2,
            "planning.parallel_ik_service": "/x5a_parallel_ik/solve",
            "planning.parallel_ik_workers": 4,
            "planning.parallel_ik_fallback_serial": True,
            "planning.reachability_table": "/home/quella/.local/share/x5a_remote_grasp/reachability.json",
            "grasp.transform_grasp_tool_translation": [0.0, 0.0, 0.0],
            "grasp.transform_grasp_tool_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "safety.plan_only": True,
            "safety.tcp_mapping_confirmed": True,
            "safety.require_table_collision": True,
            "safety.require_camera_collision": True,
            "planning.enabled": True,
            "planning.group": "arm",
            "planning.tcp_frame": "tool0",
            "planning.move_group_action": "/move_action",
            "planning.compute_ik_service": "/compute_ik",
            "planning.state_validity_service": "/check_state_validity",
            "planning.apply_scene_service": "/apply_planning_scene",
            "planning.allowed_time_s": 5.0,
            "planning.result_timeout_s": 30.0,
            "planning.ik_timeout_s": 0.50,
            "planning.attempts": 4,
            "planning.velocity_scaling": 0.10,
            "planning.acceleration_scaling": 0.10,
            "planning.position_tolerance_m": 0.005,
            "planning.orientation_tolerance_rad": 0.05,
            "planning.cartesian_step": 0.01,
            "planning.min_cartesian_fraction": 0.95,
            "scene.table_center": [0.25, 0.30, -0.026],
            "scene.table_size": [0.52, 0.72, 0.04],
            "scene.camera_size": [0.10, 0.05, 0.05],
            "scene.camera_center_in_camera_link": [0.0, 0.0, 0.0],
            "scene.camera_frame": "camera_link",
            "markers.topic": "/x5a_remote_grasp/candidates",
            "markers.axis_length_m": 0.06,
            "markers.axis_diameter_m": 0.006,
            "markers.lifetime_s": 60.0,
            "capture.directory": "/home/quella/.local/share/x5a_remote_grasp/captures",
            "action_name": "/x5a_remote_grasp/plan",
            "topics.display_path": "/display_planned_path",
            "validation_service": "/x5a_remote_grasp/validate_latest",
            "health_service": "/x5a_remote_grasp/health",
            "save_snapshot_service": "/x5a_remote_grasp/save_snapshot",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        value = lambda name: self.get_parameter(name).value
        self.base_url = str(value("remote.base_url"))
        self.api_key_env = str(value("remote.api_key_env"))
        self.connect_timeout = float(value("remote.connect_timeout_s"))
        self.inference_timeout = float(value("remote.inference_timeout_s"))
        self.threshold = float(value("remote.threshold"))
        self.mask_dilation_px = int(value("remote.mask_dilation_px"))
        self.num_point = int(value("remote.num_point"))
        self.top_k_default = int(value("remote.top_k_default"))
        self.top_k_max = int(value("remote.top_k_max"))
        self.collision_thresh = float(value("remote.collision_thresh"))
        self.voxel_size = float(value("remote.voxel_size"))
        self.target_distance_thresh = float(value("remote.target_distance_thresh"))
        self.color_topic = str(value("topics.color"))
        self.depth_topic = str(value("topics.aligned_depth"))
        self.info_topic = str(value("topics.camera_info"))
        self.sync_queue_size = int(value("sync.queue_size"))
        self.sync_slop = float(value("sync.slop_s"))
        self.max_stamp_delta = float(value("sync.max_stamp_delta_s"))
        self.max_frame_age = float(value("sync.max_frame_age_s"))
        self.camera_node = str(value("depth.camera_node"))
        self.depth_units_parameter = str(value("depth.units_parameter"))
        self.depth_scale_override = float(value("depth.scale_m_per_unit_override"))
        self.allow_32f = bool(value("depth.allow_32fc1_reencode"))
        self.min_depth_default = float(value("depth.min_m_default"))
        self.max_depth_default = float(value("depth.max_m_default"))
        self.base_frame = self._frame_id(str(value("frames.base")))
        self.tf_timeout = float(value("frames.tf_timeout_s"))
        self.max_tf_stamp_delta = float(value("frames.max_tf_stamp_delta_s"))
        self.workspace = {
            key: float(value("workspace." + key))
            for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "r_max")
        }
        self.table_clearance = float(value("workspace.table_clearance_m"))
        self.max_gripper_width = float(value("gripper.max_width_m"))
        self.width_margin = float(value("gripper.width_margin_m"))
        self.width_to_finger = float(value("gripper.width_to_finger_ratio"))
        self.gripper_joints = [str(item) for item in value("gripper.joint_names")]
        if str(value("grasp.approach_axis")).lower() != "x":
            raise RuntimeError("only the GraspNet official +X approach axis is supported")
        self.pregrasp_distance = float(value("grasp.pregrasp_distance_m"))
        self.lift_distance = float(value("grasp.lift_distance_m"))
        self.retreat_distance = float(value("grasp.retreat_distance_m"))
        self.max_pregrasp_z = float(value("grasp.max_pregrasp_z"))
        self.orientation_mode = str(value("grasp.orientation_mode")).strip().lower()
        if self.orientation_mode not in {"template", "graspnet"}:
            raise RuntimeError("grasp.orientation_mode must be 'template' or 'graspnet'")
        self.template_rpy = [float(item) for item in value("grasp.template_rpy")]
        self.tcp_backoff_m = float(value("grasp.tcp_backoff_m"))
        self.template_pregrasp_distance_m = float(
            value("grasp.template_pregrasp_distance_m")
        )
        self.min_tool0_height_m = float(value("grasp.min_tool0_height_m"))
        # 候选开度审计：服务端实测的需要开度 vs 回归宽度。回归宽度只是模型
        # 输出，可能与物体几何矛盾（实测盒案例：报 87.9mm、实际需要 96mm）。
        # 测量缺失（旧服务端）时跳过审计，fail-open 不挡链路。
        self.aperture_audit_enabled = bool(value("grasp.aperture_audit_enabled"))
        self.aperture_tolerance_m = float(value("grasp.aperture_tolerance_m"))
        # 中心带检查：服务端从原始物体点云返回 z 波段（p5/p95，相机系两点），
        # 节点转到 base 系后要求抓取中心落在波段内。GraspNet 对小物体的中心
        # 回归常贴顶面/桌面（2026-10-02 方块案例：5 个候选中心 0.3/0.3/2.4/
        # 2.3/3.1cm，物体真实带 0.6~3.3cm），波段外的抓取只能擦边。
        self.center_band_enabled = bool(value("grasp.center_band_enabled"))
        self.center_band_margin_m = float(value("grasp.center_band_margin_m"))
        self.gripper_geometry = GripperGeometry(
            tip_offset=float(value('grasp.tip_offset_m')),
            finger_length=float(value('grasp.finger_length_m')),
            finger_thickness=float(value('grasp.finger_thickness_m')),
            finger_height=float(value('grasp.finger_height_m')),
            pad_length=float(value('grasp.pad_length_m')),
            max_width=self.max_gripper_width, opening_margin=self.width_margin,
        )
        self.native_budget_fraction = float(value('grasp.native_budget_fraction'))
        self.max_target_motion = float(value('grasp.max_target_motion_m'))
        self.max_capture_to_execute = float(value('grasp.max_capture_to_execute_s'))
        self.approach_offset_m = float(value("grasp.approach_offset_m"))
        self.max_approach_z = float(value("grasp.max_approach_z"))
        self.allow_reachable_fallback = bool(value("grasp.allow_reachable_fallback"))
        self.ready_joints = [float(v) for v in value("poses.ready_joints")]
        self.place_pitches = [float(item) for item in value("grasp.place_pitches")]
        self.place_release_clearances = sorted(set(float(v) for v in value('grasp.place_release_clearances_m')))
        if not self.place_release_clearances or not all(0.010 <= v <= 0.020 for v in self.place_release_clearances):
            raise RuntimeError('release clearances must stay within 10..20mm')
        self.transit_pitches = [float(item) for item in value("grasp.transit_pitches")] or [1.15, 0.90]
        self.pitch_ladder = [float(item) for item in value("grasp.pitch_ladder")] or [1.45, 1.15, 0.90]
        self.yaw_ladder_deg = [int(item) for item in value("grasp.yaw_ladder_deg")] or [0]
        self.min_score = float(value("grasp.min_score"))
        self.template_fallback = bool(value("grasp.template_fallback"))
        self.place_offset_search_m = float(value("grasp.place_offset_search_m"))
        self.place_offset_step_m = max(1.0e-3, float(value("grasp.place_offset_step_m")))
        self.place_offset_max_points = int(value("grasp.place_offset_max_points"))
        self.plan_budget_s = float(value("planning.total_budget_s"))
        self.geometry_workers = max(1, min(4, int(value("planning.geometry_workers"))))
        self.reachability_path = Path(str(value("planning.reachability_table")))
        self.reachability = ReachabilityTable.load(self.reachability_path)
        self._reachability_mtime = self._reachability_mtime_now()
        self._log_reachability(initial=True)
        translation = np.asarray(value("grasp.transform_grasp_tool_translation"), dtype=float)
        rotation = np.asarray(value("grasp.transform_grasp_tool_rotation"), dtype=float).reshape(3, 3)
        validate_rotation_matrix(rotation)
        self.transform_grasp_tool = np.eye(4)
        self.transform_grasp_tool[:3, :3] = rotation
        self.transform_grasp_tool[:3, 3] = translation + np.array(
            [-self.approach_offset_m, 0.0, 0.0]
        )
        validate_homogeneous(self.transform_grasp_tool)
        self.safety_plan_only = bool(value("safety.plan_only"))
        self.tcp_mapping_confirmed = bool(value("safety.tcp_mapping_confirmed"))
        self.require_table = bool(value("safety.require_table_collision"))
        self.require_camera = bool(value("safety.require_camera_collision"))
        self.planning_enabled = bool(value("planning.enabled"))
        self.planning_group = str(value("planning.group"))
        self.tcp_frame = self._frame_id(str(value("planning.tcp_frame")))
        self.move_action = str(value("planning.move_group_action"))
        self.ik_service = str(value("planning.compute_ik_service"))
        self.validity_service = str(value("planning.state_validity_service"))
        self.apply_scene_service = str(value("planning.apply_scene_service"))
        self.planning_time = float(value("planning.allowed_time_s"))
        self.planning_result_timeout = float(value("planning.result_timeout_s"))
        self.ik_timeout = float(value("planning.ik_timeout_s"))
        self.planning_attempts = int(value("planning.attempts"))
        self.velocity_scaling = float(value("planning.velocity_scaling"))
        self.acceleration_scaling = float(value("planning.acceleration_scaling"))
        self.position_tolerance = float(value("planning.position_tolerance_m"))
        self.orientation_tolerance = float(value("planning.orientation_tolerance_rad"))
        self.cartesian_step = float(value("planning.cartesian_step"))
        self.min_cartesian_fraction = float(value("planning.min_cartesian_fraction"))
        self.table_center = np.asarray(value("scene.table_center"), dtype=float)
        self.table_size = np.asarray(value("scene.table_size"), dtype=float)
        self.camera_size = np.asarray(value("scene.camera_size"), dtype=float)
        self.camera_center = np.asarray(value("scene.camera_center_in_camera_link"), dtype=float)
        self.camera_frame = self._frame_id(str(value("scene.camera_frame")))
        self.marker_topic = str(value("markers.topic"))
        self.marker_axis_length = float(value("markers.axis_length_m"))
        self.marker_axis_diameter = float(value("markers.axis_diameter_m"))
        self.marker_lifetime = float(value("markers.lifetime_s"))
        self.capture_directory = Path(os.path.expanduser(str(value("capture.directory"))))
        self.action_name = str(value("action_name"))
        self.health_service = str(value("health_service"))
        self.save_snapshot_service = str(value("save_snapshot_service"))

    @staticmethod
    def _frame_id(value: str) -> str:
        return str(value or "").strip().lstrip("/")

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    def _synchronized_callback(self, color: Image, depth: Image, info: CameraInfo) -> None:
        with self.frame_lock:
            self.latest_frame = SynchronizedFrame(color, depth, info, time.monotonic())

    def _joint_state_cb(self, msg: JointState) -> None:
        names = list(msg.name)
        positions = list(msg.position)
        try:
            self.latest_joints = [
                float(positions[names.index(f"joint{i}")]) for i in range(1, 7)
            ]
        except ValueError:
            return

    def _ik_joint_seeds(self, x: float, y: float) -> List[List[float]]:
        """IK 种子：6 自由度臂的 IK 对种子极敏感，这里尽量给足候选。

        顺序大致为「当前姿态 → 上一次成功规划的姿态 → 朝向目标的 ready 家族 →
        base 关节扰动」，先试最可能命中的，命中即返回。
        """
        seeds: List[List[float]] = []
        if self.latest_joints is not None:
            seeds.append(list(self.latest_joints))
        if self.last_successful_seed is not None:
            seeds.append(list(self.last_successful_seed))
        yaw = math.atan2(float(y), float(x))
        seeds.append([yaw, 0.85, 0.95, -0.55, 0.0, 0.0])
        if self.ready_joints:
            seeds.append(list(self.ready_joints))
        # base 关节扰动：同一个末端位姿常因 joint1 落在不同的解分支而求解失败
        for base in (yaw + 0.4, yaw - 0.4, yaw + math.pi):
            seeds.append([base, 0.85, 0.95, -0.55, 0.0, 0.0])
        unique: List[List[float]] = []
        for seed in seeds:
            if any(
                all(abs(a - b) < 1.0e-3 for a, b in zip(seed, existing))
                for existing in unique
            ):
                continue
            unique.append(seed)
        return unique[:8]

    def _reachable_orientation_variants(
        self, xyz: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """X5A-reachable pitch/yaw family at a GraspNet position.

        pitch/yaw 阶梯来自配置并按实测可达性表排序：竖直工具(pitch1.45)只在
        低处可解，z 越高越需要倾斜，所以先把最可能解出来的姿态排前面。
        """
        x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        yaw_face = math.atan2(y, x)
        transit_cap = max(self.transit_pitches) if self.transit_pitches else 1.15
        variants: List[Tuple[np.ndarray, np.ndarray]] = []
        for pitch in self.reachability.order_pitches(self.pitch_ladder, x, y, z):
            transit_pitch = min(pitch, transit_cap)
            for yaw_rel in self.reachability.order_yaws(self.yaw_ladder_deg, x, y, z, pitch):
                yaw = yaw_face + math.radians(float(yaw_rel))
                variants.append((
                    rpy_rotation(0.0, pitch, yaw),
                    rpy_rotation(0.0, transit_pitch, yaw),
                ))
        return variants

    def _goal_callback(self, request: PlanRemoteGrasp.Goal) -> GoalResponse:
        prompt = str(request.prompt or "").strip()
        if not prompt or len(prompt) > 160 or any(ord(char) < 32 for char in prompt):
            return GoalResponse.REJECT
        with self.goal_lock:
            if self.goal_active:
                return GoalResponse.REJECT
            self.goal_active = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel_callback(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _feedback(self, goal_handle, stage: str, index: int = 0, detail: str = "") -> None:
        feedback = PlanRemoteGrasp.Feedback()
        feedback.stage = str(stage)
        feedback.candidate_index = int(index)
        feedback.detail = str(detail)
        goal_handle.publish_feedback(feedback)

    def _new_result(self) -> PlanRemoteGrasp.Result:
        result = PlanRemoteGrasp.Result()
        result.success = False
        result.ik_valid = False
        result.collision_free = False
        result.plan_valid = False
        return result

    def _finish_error(self, goal_handle, result, message: str):
        result.success = False
        result.error_message = str(message)
        goal_handle.abort()
        return result

    def _health_callback(self, _request, response):
        client = self._api_client()
        try:
            payload = client.health()
            sam = payload.get("sam") if isinstance(payload.get("sam"), dict) else {}
            graspnet = payload.get("graspnet") if isinstance(payload.get("graspnet"), dict) else {}
            ready = (
                str(payload.get("status", "")).lower() == "ready"
                and sam.get("ready") is True
                and graspnet.get("ready") is True
            )
            response.success = bool(ready)
            response.message = json.dumps(
                {
                    "status": payload.get("status"),
                    "sam_ready": sam.get("ready"),
                    "graspnet_ready": graspnet.get("ready"),
                },
                ensure_ascii=False,
            )
        except RemoteAPIError as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _save_snapshot_callback(self, _request, response):
        try:
            encoded = self._encode_latest_frame()
            directory = self._save_capture(encoded)
            response.success = True
            response.message = str(directory)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _api_client(self) -> GraspAPIClient:
        return GraspAPIClient(
            self.base_url,
            api_key_env=self.api_key_env,
            connect_timeout=self.connect_timeout,
            inference_timeout=self.inference_timeout,
        )

    def _encode_latest_frame(self) -> EncodedFrame:
        with self.frame_lock:
            frame = self.latest_frame
        if frame is None:
            raise RuntimeError("no synchronized RGB/depth/CameraInfo frame received")
        color_frame = self._frame_id(frame.color.header.frame_id)
        depth_frame = self._frame_id(frame.depth.header.frame_id)
        info_frame = self._frame_id(frame.info.header.frame_id)
        if not color_frame or not depth_frame or not info_frame:
            raise RuntimeError("RGB/depth/CameraInfo frame_id must not be empty")
        if len({color_frame, depth_frame, info_frame}) != 1:
            raise RuntimeError(
                "aligned RGB/depth/CameraInfo frame mismatch: "
                f"rgb={color_frame} depth={depth_frame} info={info_frame}"
            )
        stamps = [
            self._stamp_seconds(frame.color.header.stamp),
            self._stamp_seconds(frame.depth.header.stamp),
            self._stamp_seconds(frame.info.header.stamp),
        ]
        delta = max(stamps) - min(stamps)
        if delta > self.max_stamp_delta:
            raise RuntimeError(f"synchronized frame stamp spread {delta:.3f}s exceeds limit")
        age = self.get_clock().now().nanoseconds * 1.0e-9 - stamps[1]
        if age > self.max_frame_age:
            raise RuntimeError(f"latest synchronized frame is stale ({age:.3f}s)")
        if frame.color.width != frame.depth.width or frame.color.height != frame.depth.height:
            raise RuntimeError("depth is not pixel-aligned to RGB dimensions")
        if frame.info.width != frame.color.width or frame.info.height != frame.color.height:
            raise RuntimeError("CameraInfo dimensions do not match RGB/depth")
        fx, fy = float(frame.info.k[0]), float(frame.info.k[4])
        cx, cy = float(frame.info.k[2]), float(frame.info.k[5])
        if not all(math.isfinite(item) for item in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0:
            raise RuntimeError("CameraInfo intrinsics are invalid")
        bgr = self.bridge.imgmsg_to_cv2(frame.color, desired_encoding="bgr8")
        depth = np.asarray(self.bridge.imgmsg_to_cv2(frame.depth, desired_encoding="passthrough"))
        encoding = str(frame.depth.encoding or "").upper()
        if encoding in {"16UC1", "MONO16"}:
            if depth.dtype != np.uint16:
                raise RuntimeError(f"{encoding} depth decoded as {depth.dtype}, expected uint16")
            scale = self._runtime_depth_scale()
            depth_for_png = depth
        elif encoding == "32FC1":
            if not self.allow_32f:
                raise RuntimeError(
                    "32FC1 re-encoding is disabled; use aligned Z16/16UC1 to preserve raw depth"
                )
            scale = self.depth_scale_override
            if not (0.0 < scale < 0.01):
                raise RuntimeError("32FC1 re-encoding requires a valid depth scale override")
            if np.any(~np.isfinite(depth)) or float(np.nanmax(depth)) / scale > 65535.0:
                raise RuntimeError("32FC1 depth cannot be represented losslessly as uint16 PNG")
            depth_for_png = np.rint(depth / scale).astype(np.uint16)
        else:
            raise RuntimeError(f"unsupported depth encoding {frame.depth.encoding!r}")
        rgb_ok, rgb_encoded = cv2.imencode(".png", bgr)
        depth_ok, depth_encoded = cv2.imencode(".png", depth_for_png)
        if not rgb_ok or not depth_ok:
            raise RuntimeError("PNG encoding failed")
        if not self.depth_scale_logged:
            self.get_logger().info(
                f"depth encoding={encoding} runtime_scale_m_per_unit={scale:.9g} "
                f"API_depth_scale_divisor={1.0 / scale:.9g}"
            )
            self.depth_scale_logged = True
        return EncodedFrame(
            rgb_png=rgb_encoded.tobytes(),
            depth_png=depth_encoded.tobytes(),
            source_frame=depth_frame,
            stamp=frame.depth.header.stamp,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale_m_per_unit=scale,
            depth_scale_divisor=1.0 / scale,
            encoding=encoding,
            width=int(frame.depth.width),
            height=int(frame.depth.height),
            stamp_delta_s=delta,
        )

    def _runtime_depth_scale(self) -> float:
        if self.depth_scale_override > 0.0:
            scale = self.depth_scale_override
        else:
            scale = self._depth_scale_from_ros_parameter()
            if scale is None:
                scale = self._depth_scale_from_librealsense()
        if not math.isfinite(scale) or not (1.0e-6 <= scale <= 0.01):
            raise RuntimeError(f"RealSense runtime depth scale is implausible ({scale!r})")
        return float(scale)

    def _depth_scale_from_ros_parameter(self) -> Optional[float]:
        if not self.camera_parameter_client.wait_for_service(timeout_sec=1.0):
            return None
        request = GetParameters.Request()
        request.names = [self.depth_units_parameter]
        future = self.camera_parameter_client.call_async(request)
        if not PlanOnlyMoveIt.wait_future(future, 1.5):
            return None
        response = future.result()
        if response is None or len(response.values) != 1:
            return None
        parameter = response.values[0]
        if parameter.type == ParameterType.PARAMETER_DOUBLE:
            return float(parameter.double_value)
        if parameter.type == ParameterType.PARAMETER_INTEGER:
            return float(parameter.integer_value)
        if parameter.type == ParameterType.PARAMETER_STRING:
            try:
                return float(parameter.string_value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _depth_scale_from_librealsense() -> float:
        try:
            completed = subprocess.run(
                ["rs-enumerate-devices", "-o"],
                capture_output=True,
                text=True,
                timeout=8.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("cannot query librealsense Depth Units") from exc
        for line in completed.stdout.splitlines():
            if "Depth Units" not in line:
                continue
            match = re.search(r"([0-9.eE+-]+)\s*$", line)
            if match:
                return float(match.group(1))
        raise RuntimeError("librealsense did not report current Depth Units")

    def _save_capture(self, frame: EncodedFrame) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = self.capture_directory / stamp
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "color.png").write_bytes(frame.rgb_png)
        (directory / "depth.png").write_bytes(frame.depth_png)
        metadata = {
            "source_frame": frame.source_frame,
            "stamp": {"sec": int(frame.stamp.sec), "nanosec": int(frame.stamp.nanosec)},
            "width": frame.width,
            "height": frame.height,
            "depth_encoding": frame.encoding,
            "depth_scale_m_per_unit": frame.depth_scale_m_per_unit,
            "depth_scale_divisor": frame.depth_scale_divisor,
            "fx": frame.fx,
            "fy": frame.fy,
            "cx": frame.cx,
            "cy": frame.cy,
            "synchronized_stamp_delta_s": frame.stamp_delta_s,
        }
        (directory / "camera_info.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        self.get_logger().info(f"saved synchronized RGB-D snapshot at {directory}")
        return directory

    def _quality_clouds(self, encoded, payload, transform):
        raw_mask = payload.get('target_mask_png_b64')
        if payload.get('quality_schema') != 2 or not isinstance(raw_mask, str):
            raise RuntimeError('cloud service lacks quality_schema=2/original target mask; update GraspNet service')
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(raw_mask, validate=True), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imdecode(np.frombuffer(encoded.depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None or depth is None or mask.shape != depth.shape:
            raise RuntimeError('target mask does not match the captured aligned depth')
        depth_m = depth.astype(float)*encoded.depth_scale_m_per_unit
        cloud = depth_cloud(depth_m, encoded.fx, encoded.fy, encoded.cx, encoded.cy)
        valid = np.isfinite(depth_m) & (depth_m >= self.min_depth_default) & (depth_m <= self.max_depth_default)
        target_camera = cloud[(mask > 127) & valid]
        target, center, low, high = height_statistics(target_camera, transform)
        full_scene = cloud[valid] @ transform[:3, :3].T + transform[:3, 3]
        lo, hi = np.min(target, axis=0)-0.22, np.max(target, axis=0)+0.22
        scene = full_scene[np.all((full_scene >= lo) & (full_scene <= hi), axis=1)]
        return target, voxel_cloud(scene), center, (low, high)

    def _validate_latest_callback(self, _request, response):
        """Perception-only pre-execution check; never sends controller goals."""
        try:
            expected = self.last_plan_validation
            if expected is None or self.goal_active:
                raise RuntimeError('没有已验证的新规划，或正在重新规划')
            if time.monotonic()-expected['captured_monotonic'] > self.max_capture_to_execute:
                raise RuntimeError('规划使用的图像已过期，请重新规划')
            encoded = self._encode_latest_frame()
            located = self._api_client().locate(encoded.rgb_png, encoded.depth_png, {
                'prompt': expected['prompt'], 'fx': encoded.fx, 'fy': encoded.fy,
                'cx': encoded.cx, 'cy': encoded.cy, 'depth_scale': encoded.depth_scale_divisor,
                'threshold': self.threshold, 'min_depth': self.min_depth_default,
                'max_depth': self.max_depth_default, 'mask_dilation_px': 0,
            })
            transform = self._lookup_matrix(self.base_frame, encoded.source_frame, encoded.stamp)
            target, scene, center, _ = self._quality_clouds(encoded, located, transform)
            shift = float(np.linalg.norm(center-expected['center']))
            if shift > self.max_target_motion:
                raise RuntimeError(f'目标移动 {shift*1000:.1f}mm，请重新规划')
            if expected.get('place_surface') is not None:
                support = target
                if expected.get('place_prompt'):
                    support_payload = self._api_client().locate(encoded.rgb_png, encoded.depth_png, {
                        'prompt': expected['place_prompt'], 'fx': encoded.fx, 'fy': encoded.fy,
                        'cx': encoded.cx, 'cy': encoded.cy, 'depth_scale': encoded.depth_scale_divisor,
                        'threshold': self.threshold, 'min_depth': self.min_depth_default,
                        'max_depth': self.max_depth_default, 'mask_dilation_px': 0,
                    })
                    support, _, _, _ = self._quality_clouds(encoded, support_payload, transform)
                surface, _ = support_surface(support)
                surface_shift = float(np.linalg.norm(surface-expected['place_surface']))
                if surface_shift > 0.003:
                    raise RuntimeError(f'放置支撑面移动 {surface_shift*1000:.1f}mm，请重新规划')
            if expected.get('tool') is not None:
                assessment = assess_tool_grasp(target, scene, expected['tool'], self.gripper_geometry, self.pregrasp_distance)
                if not assessment['valid']:
                    raise RuntimeError('执行前抓取几何已变化: '+assessment['reason'])
            response.success = True
            response.message = json.dumps({'plan_fingerprint': expected['fingerprint'], 'motion_m': shift,
                'tool': None if expected.get('tool') is None else expected['tool'].tolist(),
                'object_cloud_base': None if expected.get('tool') is None else voxel_cloud(target, 0.001).tolist(),
                'place_tool': None if expected.get('place_tool') is None else expected['place_tool'].tolist()})
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _execute_callback(self, goal_handle):
        result = self._new_result()
        capture_directory: Optional[Path] = None
        try:
            self.last_plan_validation = None
            self.place_support_hull = None
            self.place_support_zrange = None
            self.place_relation = None
            self.place_object_cloud_tool = None
            self.place_grasp_rotation = None
            captured_monotonic = time.monotonic()
            self._feedback(goal_handle, "CAPTURE")
            self._refresh_reachability()
            encoded = self._encode_latest_frame()
            result.source_frame = encoded.source_frame
            capture_directory = self._save_capture(encoded)
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.error_message = "request canceled before remote inference"
                return result
            request = goal_handle.request
            min_depth = float(request.min_depth) if request.min_depth > 0.0 else self.min_depth_default
            max_depth = float(request.max_depth) if request.max_depth > 0.0 else self.max_depth_default
            if not (0.0 < min_depth < max_depth <= 10.0):
                return self._finish_error(goal_handle, result, "invalid min_depth/max_depth")
            top_k = int(request.top_k) if request.top_k > 0 else self.top_k_default
            top_k = min(max(1, top_k), self.top_k_max)
            grasp_prompt, ref_prompt, ref_rel, place_prompt, place_rel = (
                self._parse_planning_prompt(request.prompt)
            )
            if grasp_prompt is None:
                # 纯放置：手里已经拿着东西，只把放置目标定位出来并按当前位姿规划
                return self._execute_place_only(
                    goal_handle, result, encoded, capture_directory,
                    place_prompt, place_rel, min_depth, max_depth,
                )
            api_prompt = grasp_prompt
            if ref_prompt and place_prompt is None:
                # 仅抓取自身的参照（如"红色方块旁边的方块"）才走云端两段式定位；
                # 放置意图里的参照属于放置目标，不参与抓取定位。
                api_prompt += " | ref: " + ref_prompt
                if ref_rel:
                    api_prompt += "; rel: " + ref_rel
            fields = {
                "prompt": api_prompt,
                "fx": encoded.fx,
                "fy": encoded.fy,
                "cx": encoded.cx,
                "cy": encoded.cy,
                "depth_scale": encoded.depth_scale_divisor,
                "min_depth": min_depth,
                "max_depth": max_depth,
                "threshold": self.threshold,
                "mask_dilation_px": self.mask_dilation_px,
                "num_point": self.num_point,
                "top_k": top_k,
                "collision_thresh": self.collision_thresh,
                "voxel_size": self.voxel_size,
                "target_distance_thresh": self.target_distance_thresh,
                "max_gripper_width": self.max_gripper_width,
            }
            self._feedback(goal_handle, "REMOTE_INFERENCE")
            payload = self._api_client().grasp(encoded.rgb_png, encoded.depth_png, fields)
            (capture_directory / "grasp_response.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            )
            grasps = parse_api_grasps(payload)
            result.candidate_count = len(grasps)
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.error_message = "request canceled after remote inference"
                return result
            self._feedback(goal_handle, "TF_TRANSFORM")
            transform_base_camera = self._lookup_matrix(
                self.base_frame, encoded.source_frame, encoded.stamp
            )
            base_grasps = [
                transform_base_camera @ grasp.transform_camera_grasp for grasp in grasps
            ]
            for transform in base_grasps:
                validate_homogeneous(transform)
            self._publish_grasp_markers(grasps, base_grasps, encoded.stamp)
            top = max(range(len(grasps)), key=lambda index: grasps[index].score) if grasps else None
            if top is not None:
                self.get_logger().info(
                "API/TF summary: "
                f"api_frame={payload.get('frame')!r} source_frame={encoded.source_frame} "
                f"candidates={len(grasps)} top_camera_xyz="
                f"{grasps[top].transform_camera_grasp[:3, 3].round(5).tolist()} "
                f"top_base_xyz={base_grasps[top][:3, 3].round(5).tolist()}"
            )
            self.target_cloud, self.scene_cloud, geometric_center, center_band = self._quality_clouds(encoded, payload, transform_base_camera)
            (capture_directory/'target_cloud_base.npy').write_bytes(self._numpy_bytes(self.target_cloud))

            summary = {
                "api_frame": payload.get("frame"),
                "source_frame": encoded.source_frame,
                "base_frame": self.base_frame,
                "object_center_band_base": (
                    list(center_band) if center_band is not None else None
                ),
                "transform_base_camera": transform_base_camera.tolist(),
                "candidates": [
                    {
                        "index": grasp.index,
                        "score": grasp.score,
                        "width_m": grasp.width_m,
                        "required_opening_m": grasp.required_opening_m,
                        "transform_camera_grasp": grasp.transform_camera_grasp.tolist(),
                        "transform_base_grasp": base.tolist(),
                    }
                    for grasp, base in zip(grasps, base_grasps)
                ],
            }
            (capture_directory / "tf_validation.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            )
            place_target = None
            if place_prompt:
                place_target = self._locate_place_target(
                    goal_handle,
                    encoded,
                    transform_base_camera,
                    place_prompt,
                    place_rel,
                    min_depth,
                    max_depth,
                    capture_directory,
                )
                if place_target is None:
                    return self._finish_error(
                        goal_handle, result,
                        f"放置目标「{place_prompt}」没有定位到（/v1/locate 无结果）",
                    )
            if not self.tcp_mapping_confirmed:
                return self._finish_error(
                    goal_handle,
                    result,
                    "GraspNet grasp-frame to tool0 mapping is unconfirmed; "
                    "candidates and TF were published for RViz inspection, IK was not run",
                )
            report: List[dict] = []

            def centroid_locate():
                """保底抓取用：/v1/locate 返回物体质心（base 系）。"""
                return geometric_center.copy()

            selected = self._select_candidate(
                goal_handle,
                grasps,
                base_grasps,
                encoded.stamp,
                plan_motion=bool(request.plan_motion),
                capture_directory=capture_directory,
                place_target=place_target,
                report=report,
                centroid_locate=centroid_locate,
                center_band=center_band,
            )
            if selected is None:
                top_position = None
                if base_grasps:
                    best = max(range(len(base_grasps)), key=lambda index: grasps[index].score)
                    top_position = base_grasps[best][:3, 3]
                return self._finish_error(
                    goal_handle,
                    result,
                    self._planning_failure_message(
                        report, top_position, capture_directory, place_prompt
                    ),
                )
            grasp, waypoints, plan = selected
            if grasp is not None:
                self.get_logger().info(
                    f"selected {self.selection_kind} pose: "
                    f"#{grasp.index} score={grasp.score:.3f} width={grasp.width_m:.3f} "
                    f"mode={self.orientation_mode} "
                    f"approach={waypoints['grasp'][:3, 0].round(3).tolist()} "
                    f"tool0={waypoints['grasp'][:3, 3].round(3).tolist()}"
                )
                result.selected_index = int(grasp.index)
                result.gripper_width_m = float(grasp.width_m)
                result.score = float(grasp.score)
                self._publish_gripper_plan(grasp.width_m)
            else:
                result.selected_index = 0
                result.gripper_width_m = 0.0
                result.score = 0.0
            result.pregrasp_pose = self._pose_stamped(waypoints["pregrasp"], encoded.stamp)
            result.selected_pose = self._pose_stamped(waypoints["grasp"], encoded.stamp)
            result.lift_pose = self._pose_stamped(waypoints["lift"], encoded.stamp)
            result.retreat_pose = self._pose_stamped(waypoints["retreat"], encoded.stamp)
            result.ik_valid = plan is not None
            result.collision_free = plan is not None
            result.plan_valid = plan is not None
            result.planned_stages = [] if plan is None else list(plan.stages) + ["CLOSE_GRIPPER_DIAGNOSTIC_ONLY"]
            if plan is not None:
                from x5a_remote_grasp.grasp_quality import trajectory_fingerprint
                self.last_plan_validation = {
                    'captured_monotonic': captured_monotonic, 'prompt': api_prompt,
                    'center': geometric_center, 'tool': waypoints['grasp'],
                    'place_tool': waypoints.get('place'),
                    'place_surface': (None if place_target is None or place_target[1] != 'on'
                                      else place_target[0].copy()),
                    'place_prompt': place_prompt,
                    'fingerprint': trajectory_fingerprint(plan.trajectories),
                }
                self.display_publisher.publish(self.planner.display_message(plan))
            result.success = True
            result.error_message = ""
            goal_handle.succeed()
            return result
        except (RemoteAPIError, TransformException, RuntimeError, ValueError, OSError) as exc:
            return self._finish_error(goal_handle, result, str(exc))
        finally:
            self.planner.deadline = None
            with self.goal_lock:
                self.goal_active = False

    @staticmethod
    def _numpy_bytes(array):
        import io
        buffer = io.BytesIO()
        np.save(buffer, array)
        return buffer.getvalue()

    def _lookup_matrix(self, target: str, source: str, stamp) -> np.ndarray:
        transform = self.tf_buffer.lookup_transform(
            target,
            source,
            Time.from_msg(stamp),
            timeout=Duration(seconds=self.tf_timeout),
        )
        transform_stamp = self._stamp_seconds(transform.header.stamp)
        request_stamp = self._stamp_seconds(stamp)
        if transform_stamp > 0.0 and abs(transform_stamp - request_stamp) > self.max_tf_stamp_delta:
            raise RuntimeError(
                f"TF {source}->{target} timestamp differs by "
                f"{abs(transform_stamp - request_stamp):.3f}s"
            )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return transform_from_xyz_quat(
            [translation.x, translation.y, translation.z],
            [rotation.x, rotation.y, rotation.z, rotation.w],
        )

    def _template_waypoints(
        self,
        grasp_pos: np.ndarray,
        grasp_R: np.ndarray,
        transit_R: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Waypoints for template mode, mirroring the proven HSV flow:
        PREGRASP and GRASP share the near-vertical grasp orientation so the
        final approach is a straight vertical descent (pregrasp sits directly
        above the grasp point). LIFT and RETREAT tilt to the transit
        orientation because the arm cannot hold a steep tool at height
        (verified by /compute_ik scan: vertical tool fails above z~0.13)."""
        def make(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = position
            return matrix

        grasp = make(grasp_R, grasp_pos - grasp_R[:, 0] * self.tcp_backoff_m)
        # 法兰下探硬下限（IK 实测 z<0.04 竖直工具无解）：平躺物体（笔、胶带）
        # 的抓取中心贴近桌面，靠钳制抬回法兰下限，剩余深度由手指长度补足。
        if grasp[2, 3] < self.min_tool0_height_m:
            grasp[2, 3] = self.min_tool0_height_m
        pregrasp = make(
            grasp_R,
            grasp[:3, 3] - grasp_R[:, 0] * self.template_pregrasp_distance_m,
        )
        lift = make(transit_R, grasp[:3, 3] + np.array([0.0, 0.0, self.lift_distance]))
        retreat = make(
            transit_R, lift[:3, 3] - transit_R[:, 0] * self.retreat_distance
        )
        return {
            "pregrasp": pregrasp,
            "grasp": grasp,
            "lift": lift,
            "retreat": retreat,
        }

    @staticmethod
    def _make_tf(rotation: np.ndarray, position) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = position
        return matrix

    def _locate_place_target(
        self, goal_handle, encoded, transform_base_camera, place_prompt, place_rel,
        min_depth, max_depth, capture_directory,
    ):
        """调 /v1/locate 定位放置目标，返回 (base 系放置点, 关系)。"""
        self._feedback(goal_handle, "PLACE_LOCATE")
        located = self._api_client().locate(encoded.rgb_png, encoded.depth_png, {
            "prompt": place_prompt, "fx": encoded.fx, "fy": encoded.fy,
            "cx": encoded.cx, "cy": encoded.cy,
            "depth_scale": encoded.depth_scale_divisor,
            "threshold": self.threshold,
            "mask_dilation_px": self.mask_dilation_px,
            "min_depth": min_depth, "max_depth": max_depth,
        })
        if not isinstance(located, dict) or "center_cam" not in located or "top_cam" not in located:
            return None
        support, _, center_b, band = self._quality_clouds(encoded, located, transform_base_camera)
        self.last_place_center = center_b.copy()
        self.place_relation = place_rel or 'in'
        z_top = band[1]
        if self.place_relation == 'on':
            surface, self.place_support_hull = support_surface(support)
            self.place_support_zrange = band
        else:
            surface = np.r_[center_b[:2], z_top]
        place_target = (
            surface, self.place_relation,
        )
        if capture_directory is not None:
            np.save(capture_directory/'place_support_cloud_base.npy', support)
            (capture_directory/'place_response.json').write_text(json.dumps(located)+'\n')
            (capture_directory / "place_locate.json").write_text(json.dumps({
                "place_prompt": place_prompt, "place_rel": place_target[1],
                "center_cam": located["center_cam"], "top_cam": located["top_cam"],
                "center_base": center_b.round(5).tolist(),
                "z_top_base": round(float(z_top), 5),
                "place_point_base": place_target[0].round(5).tolist(),
                "point_semantics": "object_bottom_support_surface",
            }, indent=1) + "\n")
        self.get_logger().info(
            "place point (base): "
            f"{place_target[0].round(3).tolist()} rel={place_target[1]}"
        )
        return place_target

    def _execute_place_only(
        self, goal_handle, result, encoded, capture_directory,
        place_prompt, place_rel, min_depth, max_depth,
    ):
        """纯放置模式：不调 /v1/grasp、不选候选，直接按当前位姿规划放上去。

        用于交互式的第二句指令（"放到白色方块上"）：物体已经在夹爪里，
        相机是 eye-to-hand 固定安装，所以手臂移动后放置目标照样能定位。
        """
        if not place_prompt:
            return self._finish_error(
                goal_handle, result, "纯放置请求缺少 place: 目标（例：place: white cube; prel: on）"
            )
        if not self.planning_enabled:
            return self._finish_error(goal_handle, result, "纯放置需要 planning.enabled=true")
        if not self.executor_state_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError('执行器持物几何服务未就绪')
        future = self.executor_state_client.call_async(Trigger.Request())
        if not PlanOnlyMoveIt.wait_future(future, 3.0):
            raise RuntimeError('读取持物几何超时')
        state_response = future.result()
        if state_response is None or not state_response.success:
            raise RuntimeError('无法读取实际持物状态')
        state = json.loads(state_response.message)
        held = state.get('held_geometry')
        if not state.get('holding') or state.get('holding_state') != 'held' or not held:
            raise RuntimeError('缺少已确认持物的抓取几何，请先重新抓取后再放置')
        self.place_object_cloud_tool = np.asarray(held['cloud_tool'], dtype=float)
        self.place_grasp_rotation = np.asarray(held['grasp_rotation'], dtype=float)
        transform_base_camera = self._lookup_matrix(
            self.base_frame, encoded.source_frame, encoded.stamp
        )
        place_target = self._locate_place_target(
            goal_handle, encoded, transform_base_camera, place_prompt, place_rel,
            min_depth, max_depth, capture_directory,
        )
        if place_target is None:
            return self._finish_error(
                goal_handle, result, f"放置目标「{place_prompt}」没有定位到（/v1/locate 无结果）"
            )
        self._feedback(goal_handle, 'REMOTE_PLAN')
        ready, reason = self.planner.ready()
        if not ready:
            return self._finish_error(goal_handle, result, reason)
        applied, reason = self.planner.apply_scene(self._safety_scene(encoded.stamp))
        if not applied:
            return self._finish_error(goal_handle, result, reason)
        report: List[dict] = []
        deadline = time.monotonic() + max(1.0, self.plan_budget_s)
        selected = self._plan_place_only(
            goal_handle, place_target, encoded.stamp, deadline, report, capture_directory
        )
        if selected is None:
            return self._finish_error(
                goal_handle, result,
                '放置目标已定位，但未找到物体底部距顶面 1～2cm 的完整无碰撞路径。'
                +self._planning_failure_message(
                    report, np.asarray(place_target[0], dtype=float),
                    capture_directory, place_prompt,
                ),
            )
        waypoints, plan = selected
        result.source_frame = encoded.source_frame
        result.planned_stages = list(plan.stages) + ["OPEN_GRIPPER_AFTER_PLACE"]
        result.pregrasp_pose = self._pose_stamped(waypoints["transport"], encoded.stamp)
        result.selected_pose = self._pose_stamped(waypoints["place"], encoded.stamp)
        result.lift_pose = self._pose_stamped(waypoints["transport"], encoded.stamp)
        result.retreat_pose = self._pose_stamped(waypoints["retreat"], encoded.stamp)
        result.ik_valid = True
        result.collision_free = True
        result.plan_valid = True
        result.gripper_width_m = 0.0
        result.score = 0.0
        result.selected_index = 0
        from x5a_remote_grasp.grasp_quality import trajectory_fingerprint
        self.last_plan_validation = {
            'captured_monotonic': deadline-self.plan_budget_s,
            'prompt': place_prompt, 'center': self.last_place_center,
            'tool': None, 'fingerprint': trajectory_fingerprint(plan.trajectories),
            'place_tool': waypoints['place'],
            'place_surface': None if place_target[1] != 'on' else place_target[0].copy(),
        }
        (capture_directory/'selected_place.json').write_text(json.dumps({
            'tool0': waypoints['place'].tolist(), 'held_geometry_source': 'actual_grasp_tcp',
            'grasp_fingerprint': held.get('grasp_fingerprint'),
            'observed_bottom_clearance_m': self._place_clearance(waypoints['place'], place_target[0], self.place_object_cloud_tool),
        }, indent=2)+'\n')
        self.display_publisher.publish(self.planner.display_message(plan))
        result.success = True
        result.error_message = ""
        goal_handle.succeed()
        return result

    def _plan_place_only(self, goal_handle, place_target, stamp, deadline,
                         report=None, capture_directory=None):
        self.planner.deadline = deadline
        for place_xyz in self._place_point_candidates(place_target):
            yaw = math.atan2(place_xyz[1], place_xyz[0])
            for pitch in self.transit_pitches:
                transit = rpy_rotation(0, pitch, yaw)
                for poses, waypoints in self._place_pose_variants(float(place_xyz[2]), transit, place_xyz, stamp):
                    if goal_handle.is_cancel_requested:
                        return None
                    if self._budget_exhausted(deadline):
                        self._record_reject(report, goal_handle, None, 'place_budget',
                                            'PLACE: planning budget exhausted', feedback=False)
                        return None
                    self.planner.default_joint_seeds = self._ik_joint_seeds(*place_xyz[:2])
                    self.planner.retreat_clearance_pose = self._pose_stamped(waypoints['retreat_clearance'], stamp)
                    plan, reason = self.planner.plan_sequence(poses)
                    if plan is not None:
                        self.get_logger().info(f'place-only plan: stages={plan.stages} target={place_xyz.tolist()}')
                        return waypoints, plan
                    self._record_reject(report, goal_handle, None, 'place_plan', reason, feedback=False)
        return None

    def _select_candidate(
        self, goal_handle, grasps, base_grasps, stamp, plan_motion,
        capture_directory=None, place_target=None, report=None,
        centroid_locate=None, center_band=None,
    ):
        if self.target_cloud is None or self.scene_cloud is None:
            raise RuntimeError('no original-target/scene geometry for grasp validation')
        if plan_motion:
            ready, reason = self.planner.ready()
            if not self.planning_enabled or not ready:
                raise RuntimeError(reason or 'planning is disabled')
            applied, reason = self.planner.apply_scene(self._safety_scene(stamp))
            if not applied:
                raise RuntimeError(reason)
        started = time.monotonic()
        deadline = started + self.plan_budget_s
        native_deadline = started + self.plan_budget_s*self.native_budget_fraction
        indexed = sorted(zip(grasps, base_grasps), key=lambda pair: pair[0].score, reverse=True)
        # Phase 1 is exclusively model poses + the equivalent jaw swap. No
        # modified pose from candidate #0 can outrank candidate #1's native pose.
        native = []
        for grasp, transform in indexed:
            if grasp.score < self.min_score or approach_from_below(transform, self.max_approach_z):
                self._record_reject(report, goal_handle, grasp, 'score_or_approach', 'native candidate rejected')
                continue
            for grasp_R, transit_R, kind in self._orientation_variants(transform):
                if kind != 'graspnet':
                    continue
                native.append((grasp, transform, grasp_R, transit_R, 'native'))
        derived = []
        if self.allow_reachable_fallback or self.orientation_mode == 'template' or self.template_fallback:
            center = np.asarray(centroid_locate(), dtype=float) if centroid_locate is not None else np.median(self.target_cloud, axis=0)
            if center_band is not None:
                lo, hi = center_band
                table_top = float(self.table_center[2] + self.table_size[2]/2)
                center[2] = max(table_top+0.006, lo+0.35*(hi-lo))
            variants = self._template_orientation_variants(center)
            # Add orientations along the target's observed principal axes so a
            # long object can be grasped across its short side, independently
            # of the arm's radial direction. Every pose is evaluated again.
            xy = self.target_cloud[:, :2]
            _, axes = np.linalg.eigh(np.cov(xy.T))
            yaw = math.atan2(axes[1, 0], axes[0, 0])-math.pi/2
            for pitch in self.reachability.order_pitches(self.pitch_ladder, *center):
                for dy in (0.0, math.pi, math.radians(15), -math.radians(15)):
                    variants.append((rpy_rotation(0, pitch, yaw+dy),
                                     rpy_rotation(0, min(pitch, max(self.transit_pitches)), yaw+dy), 'geometric'))
            reference = indexed[0][0] if indexed else RemoteGrasp(
                index=4294967295, score=0.0, width_m=0.0,
                height_m=self.gripper_geometry.finger_height, depth_m=0.01,
                transform_camera_grasp=np.eye(4),
            )
            centers = [center]
            if center_band is not None:
                # Low grasps wrap deeply but can be below this arm's IK limit.
                # Try modestly higher contact planes, each independently checked
                # against the SAME target and physical jaw geometry.
                for fraction in (0.55, 0.75):
                    raised = center.copy()
                    raised[2] = max(table_top+0.006, lo+fraction*(hi-lo))
                    if np.linalg.norm(raised-center) > 0.002:
                        centers.append(raised)
            derived = [(reference, self._make_tf(np.eye(3), position), a, b, 'geometric')
                       for a, b, _ in variants for position in centers]
        quality_reports = []
        # One immutable scene per captured frame. Each candidate used to repeat
        # the identical voxel reduction of tens of thousands of scene points.
        prepared_scene = PreparedGraspScene(self.scene_cloud)
        target = np.array(self.target_cloud, copy=True)
        target.setflags(write=False)
        def check_geometry(proposal):
            grasp, transform, grasp_R, transit_R, kind = proposal
            if kind == 'native':
                matrix = transform.copy()
                matrix[:3, :3] = grasp_R
                tool = graspnet_tool_transform(matrix, grasp.depth_m, self.transform_grasp_tool)
            else:
                tool = self._make_tf(grasp_R, transform[:3, 3]-grasp_R[:, 0]*self.gripper_geometry.tip_offset)
            tool = recenter_tool(target, tool, self.gripper_geometry,
                                 max_shift=0.008 if kind == 'native' else 0.02)
            measure = assess_tool_grasp(target, prepared_scene, tool,
                                       self.gripper_geometry, self.pregrasp_distance)
            return tool, measure
        try:
            for phase, phase_deadline, proposals in (
                ('native', native_deadline if derived else deadline, native),
                ('geometric', deadline, derived),
            ):
                if goal_handle.is_cancel_requested:
                    return None
                evaluated = []
                checks = ordered_geometry_map(check_geometry, proposals,
                    workers=getattr(self, 'geometry_workers', 1),
                    stop=lambda: goal_handle.is_cancel_requested or time.monotonic() >= deadline)
                for proposal_index, checked in enumerate(checks):
                    proposal = proposals[proposal_index]
                    grasp, transform, grasp_R, transit_R, kind = proposal
                    tool, measure = checked
                    if not measure['valid']:
                        quality_reports.append({'index': grasp.index, 'kind': kind, 'tool': tool.tolist(), **measure})
                        continue
                    # A template is independently scored by geometry; never
                    # carry the old neural score over to a different rotation.
                    score = grasp.score if kind == 'native' else 0.0
                    updated = replace(grasp, score=score,
                                      width_m=measure['required_opening_m'],
                                      required_opening_m=measure['required_opening_m'])
                    tip = tool[:3, 3] + tool[:3, 0]*self.gripper_geometry.tip_offset
                    table_top = float(self.table_center[2]+self.table_size[2]/2)
                    if tip[2] < table_top+0.003:
                        quality_reports.append({'index': grasp.index, 'kind': kind, 'valid': False, 'reason': 'tip_below_table'})
                        continue
                    # Construct approach waypoints around the already validated
                    # ACTUAL tool pose, not around the discarded prediction.
                    waypoints = candidate_waypoints(
                        tool, np.eye(4), self.pregrasp_distance, self.lift_distance,
                        self.retreat_distance, transit_rotation=transit_R,
                        max_pregrasp_z=self.max_pregrasp_z,
                    )
                    if any(not self._waypoint_in_bounds(stage, matrix[:3, 3]) for stage, matrix in waypoints.items()):
                        continue
                    # Small opening, central contact, then reachability order.
                    # Native proposals remain in model score order.
                    rank = (-score, measure['required_opening_m']) if kind == 'native' else (
                        math.ceil(measure['required_opening_m']/0.01), proposal_index)
                    evaluated.append((rank, updated, waypoints, measure))
                if phase == 'geometric':
                    evaluated.sort(key=lambda entry: entry[0])
                for _, grasp, waypoints, measure in evaluated:
                    if goal_handle.is_cancel_requested or time.monotonic() >= phase_deadline:
                        break
                    self.planner.deadline = phase_deadline
                    seeds = self._ik_joint_seeds(*waypoints['grasp'][:2, 3])
                    self.planner.default_joint_seeds = seeds
                    self.planner.expected_grasp_finger_position = grasp.width_m/2
                    places = self._place_point_candidates(place_target) if place_target is not None else [None]
                    for place_xyz in places:
                        if time.monotonic() >= phase_deadline:
                            break
                        all_waypoints = dict(waypoints)
                        poses = [(s.upper(), self._pose_stamped(waypoints[s], stamp))
                                 for s in ('pregrasp', 'grasp', 'lift')]
                        if place_xyz is not None:
                            # Each place pitch is tried through IK AND planning;
                            # previously the first in-bounds pitch hid the rest.
                            place_variants = self._place_pose_variants(
                                waypoints['lift'][2, 3], waypoints['lift'][:3, :3], place_xyz, stamp,
                                object_in_tool(self.target_cloud, waypoints['grasp']), waypoints['grasp'][:3, :3])
                        else:
                            place_variants = [([('RETREAT', self._pose_stamped(waypoints['retreat'], stamp))], {})]
                        for extra_poses, extra_waypoints in place_variants:
                            if time.monotonic() >= phase_deadline:
                                break
                            plan_poses = poses + extra_poses
                            all_waypoints.update(extra_waypoints)
                            self.planner.retreat_clearance_pose = (None if 'retreat_clearance' not in extra_waypoints
                                else self._pose_stamped(extra_waypoints['retreat_clearance'], stamp))
                            plan, reason = self.planner.plan_sequence(plan_poses) if plan_motion else (None, '')
                            if plan_motion and plan is None:
                                self._record_reject(report, goal_handle, grasp, phase+'_plan', reason)
                                continue
                            self.selection_kind = phase
                            selected_report = {'kind': phase, 'index': grasp.index, 'score': grasp.score,
                                               'tool0': waypoints['grasp'].tolist(), **measure}
                            if capture_directory is not None:
                                (capture_directory/'selected_grasp.json').write_text(json.dumps(selected_report, indent=2)+'\n')
                            self._feedback(goal_handle, 'SELECTED_'+phase.upper(), grasp.index,
                                           f"opening={grasp.width_m*1000:.1f}mm contacts={measure['contact_points']}")
                            if self.planner.last_solution_seed is not None:
                                self.last_successful_seed = list(self.planner.last_solution_seed)
                            return grasp, all_waypoints, plan
            self._record_reject(report, goal_handle, None, 'geometry_or_budget',
                                f'{len(quality_reports)} poses rejected by target/scene geometry; planning budget {self.plan_budget_s}s')
            return None
        finally:
            self.planner.deadline = None
            self.planner.expected_grasp_finger_position = None
            if capture_directory is not None:
                (capture_directory/'grasp_quality.json').write_text(json.dumps(quality_reports, indent=2)+'\n')


    def _budget_exhausted(self, deadline: float) -> bool:
        return time.monotonic() > deadline

    def _log_reachability(self, initial: bool = False) -> None:
        if self.reachability:
            caps = self.reachability.pitch_capabilities()
            measured = ", ".join(
                f"{pitch}: z<={height:.2f}" for pitch, height in caps.items()
            )
            self.get_logger().info(
                f"reachability table loaded ({len(self.reachability.results)} cells): "
                + (measured or "no pitch summary")
            )
        elif initial:
            self.get_logger().warn(
                f"no reachability table at {self.reachability_path}; "
                "pitch ladder falls back to the empirical z rule "
                "(run ./scripts/probe_reachability.sh to measure)"
            )

    def _reachability_mtime_now(self) -> Optional[float]:
        try:
            return self.reachability_path.stat().st_mtime
        except OSError:
            return None

    def _refresh_reachability(self) -> None:
        """扫描工具会在节点运行期间持续写表，文件一变就重新加载（每次 goal 一次
        stat()，代价可忽略），省得为了用上新表重启节点。"""
        mtime = self._reachability_mtime_now()
        if mtime is None:
            return
        if self._reachability_mtime is not None and mtime == self._reachability_mtime:
            return
        self._reachability_mtime = mtime
        table = ReachabilityTable.load(self.reachability_path)
        if table:
            self.reachability = table
            self.get_logger().info(f"reachability table refreshed from {self.reachability_path}")
            self._log_reachability()

    def _planning_failure_message(
        self,
        report: Optional[List[dict]],
        target_position: Optional[np.ndarray],
        capture_directory: Optional[Path],
        place_prompt: Optional[str] = None,
    ) -> str:
        """把结构化失败原因翻成一句能照着动手的说明，并落盘 plan_failures.json。

        以前这里只有一句 "no candidate passed ..."，排查全靠猜；现在把每个候选/
        姿态的失败原因（哪一段、IK 还是规划、是否工作空间）统计后一起报出来。
        """
        entries = [entry for entry in (report or []) if isinstance(entry, dict)]
        stage_counts: Dict[str, int] = {}
        reason_counts: Dict[str, int] = {}
        budget_hit = False
        for entry in entries:
            reason = str(entry.get("reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            text = str(entry.get("detail") or "")
            if "budget" in text.lower():
                budget_hit = True
            for stage in re.findall(
                r"(PREGRASP|GRASP|LIFT|TRANSPORT|PLACE|RETREAT)", text
            ):
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if capture_directory is not None:
            try:
                (capture_directory / "plan_failures.json").write_text(
                    json.dumps(
                        {
                            "attempts": len(entries),
                            "reason_counts": reason_counts,
                            "stage_counts": stage_counts,
                            "budget_exhausted": budget_hit,
                            "target_position": None
                            if target_position is None
                            else [round(float(v), 4) for v in target_position],
                            "reachability_verdict": None
                            if target_position is None or not self.reachability
                            else self.reachability.verdict(*[float(v) for v in target_position]),
                            "entries": entries[:200],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
            except OSError:
                pass
        parts = [f"规划失败：{len(entries)} 次候选/姿态尝试都没通过检查"]
        if stage_counts:
            ranked = sorted(stage_counts.items(), key=lambda item: -item[1])[:3]
            parts.append("集中在 " + "、".join(f"{stage}×{count}" for stage, count in ranked))
        if budget_hit:
            parts.append(f"{self.plan_budget_s:.0f}s 规划预算已用尽")
        if target_position is not None and self.reachability:
            parts.append(self.reachability.verdict(*[float(v) for v in target_position]))
        if place_prompt and stage_counts.get("PLACE", 0) >= max(
            1, sum(stage_counts.values()) // 2
        ):
            parts.append(
                f"放置点「{place_prompt}」周围 ±{self.place_offset_search_m * 100:.0f}cm "
                "内也没找到可解位姿，换个参照物或挪动盒子再试"
            )
        return "；".join(parts)

    def _record_reject(
        self, report: Optional[List[dict]], goal_handle, grasp, reason: str, detail: str,
        occupied: str = "", feedback: bool = True,
    ) -> None:
        """记录并上报一次候选/姿态拒绝：feedback 给实时反馈，report 给最终错误信息。"""
        index = int(grasp.index) if grasp is not None else 4294967295
        if feedback:
            self._feedback(goal_handle, "CANDIDATE_REJECT", index, f"{reason}: {detail}")
        if report is None:
            return
        entry = {
            "index": index,
            "score": None if grasp is None else round(float(grasp.score), 4),
            "reason": reason,
            "detail": detail,
        }
        if occupied:
            entry["stage"] = occupied
        report.append(entry)

    def _orientation_variants(
        self, transform_base_grasp: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """(抓取姿态, 过渡姿态, 来源) 降级阶梯：GraspNet 原生 → 可达家族 → 模板。"""
        approach = transform_base_grasp[:3, 0]
        if abs(approach[0]) + abs(approach[1]) > 1.0e-6:
            yaw0 = math.atan2(approach[1], approach[0])
        else:
            yaw0 = math.atan2(transform_base_grasp[1, 3], transform_base_grasp[0, 3])
        variants: List[Tuple[np.ndarray, np.ndarray, str]] = []
        if self.orientation_mode == "graspnet":
            steep = is_steep_topdown(transform_base_grasp)
            yaw = approach_yaw(transform_base_grasp[:3, :3])
            for grasp_R in parallel_jaw_rotations(transform_base_grasp[:3, :3]):
                if steep:
                    for pitch in self.transit_pitches:
                        variants.append(
                            (grasp_R, rpy_rotation(0.0, pitch, yaw), "graspnet")
                        )
                else:
                    variants.append((grasp_R, grasp_R, "graspnet"))
            if self.allow_reachable_fallback:
                for grasp_R, transit_R in self._reachable_orientation_variants(
                    transform_base_grasp[:3, 3]
                ):
                    variants.append((grasp_R, transit_R, "reachable"))
        else:
            variants.extend(self._template_orientation_variants(transform_base_grasp[:3, 3], yaw0))
        return variants

    def _template_orientation_variants(
        self, position: np.ndarray, yaw0: Optional[float] = None
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """模板姿态家族：pitch/yaw 阶梯按实测可达性排序。"""
        x, y, z = (float(item) for item in position)
        base_yaw = math.atan2(y, x) if yaw0 is None else float(yaw0)
        variants: List[Tuple[np.ndarray, np.ndarray, str]] = []
        for pitch in self.reachability.order_pitches(self.pitch_ladder, x, y, z):
            transit_pitch = min(pitch, max(self.transit_pitches) if self.transit_pitches else 1.15)
            for yaw_rel in self.reachability.order_yaws(self.yaw_ladder_deg, x, y, z, pitch):
                yaw = base_yaw + math.radians(float(yaw_rel))
                variants.append((
                    self._rpy_matrix(0.0, pitch, yaw),
                    self._rpy_matrix(0.0, transit_pitch, yaw),
                    "template",
                ))
        return variants

    def _place_point_candidates(self, place_target) -> List[np.ndarray]:
        """放置点阶梯：先精确分割中心，再按同心网格就近偏移。"""
        center = np.asarray(place_target[0], dtype=float)
        points = [center]
        hull = getattr(self, 'place_support_hull', None)
        if place_target[1] == 'on' and hull is not None and not supported_center(center[:2], hull):
            raise RuntimeError('支撑面中心距边缘不足 4mm，不能稳定放置')
        search = max(0.0, self.place_offset_search_m)
        if search <= 0.0:
            return points
        step = self.place_offset_step_m
        rings = int(round(search / step))
        offsets = []
        for ix in range(-rings, rings + 1):
            for iy in range(-rings, rings + 1):
                if ix == 0 and iy == 0:
                    continue
                dx, dy = ix * step, iy * step
                distance = math.hypot(dx, dy)
                if distance <= search + 1.0e-9:
                    offsets.append((distance, dx, dy))
        offsets.sort()
        for _, dx, dy in offsets[: max(0, self.place_offset_max_points)]:
            shifted = center.copy()
            shifted[0] += dx
            shifted[1] += dy
            if place_target[1] == 'on' and hull is not None and not supported_center(shifted[:2], hull):
                continue
            points.append(shifted)
        return points

    def _place_poses(
        self,
        lift_z: float,
        transit_R: np.ndarray,
        place_xyz: np.ndarray,
        stamp,
        failures: List[str],
    ) -> Tuple[Optional[List[Tuple[str, PoseStamped]]], Dict[str, np.ndarray]]:
        """TRANSPORT/PLACE/RETREAT 三段位姿；放置姿态也走可达性排序的 pitch 阶梯。"""
        for built, updates in self._place_pose_variants(lift_z, transit_R, place_xyz, stamp):
            return built, updates
        return None, {}

    def _place_pose_variants(self, lift_z, transit_R, place_xyz, stamp,
                             object_cloud_tool=None, grasp_rotation=None):
        px, py, pz = (float(v) for v in place_xyz)
        yaw_p = math.atan2(py, px)
        cloud = self.place_object_cloud_tool if object_cloud_tool is None else object_cloud_tool
        grasp_R = self.place_grasp_rotation if grasp_rotation is None else grasp_rotation
        if cloud is None or grasp_R is None:
            raise RuntimeError('放置规划缺少物体相对于夹爪的位置')
        radial_delta = yaw_p-math.atan2(grasp_R[1, 0], grasp_R[0, 0])
        rotations = [grasp_R, self._rpy_matrix(0, 0, radial_delta) @ grasp_R]
        rotations.extend(self._rpy_matrix(0.0, pitch, yaw_p)
                         for pitch in self.reachability.order_pitches(self.place_pitches, px, py, pz))
        # Try all upright orientations at the closest release height first.
        # Release the object bottom 10..20 mm above the support, at the lowest
        # collision-free height, rather than descending into the support.
        clearances = self.place_release_clearances if self.place_relation == 'on' else [0.002]
        for clearance in clearances:
            for place_R in rotations:
                if self.place_relation == 'on' and not preserves_upright(grasp_R, place_R):
                    continue
                place_tool = placement_tool(cloud, place_R, place_xyz, clearance)
                transport_z = max(lift_z, place_tool[2, 3] + 0.12)
                waypoints = {
                    "transport": self._make_tf(transit_R, [*place_tool[:2, 3], transport_z]),
                    "place": place_tool,
                    "retreat_clearance": self._make_tf(place_R, place_tool[:3, 3]+[0, 0, min(0.03, self.lift_distance)]),
                    "retreat": self._make_tf(transit_R, place_tool[:3, 3]+[0, 0, self.lift_distance]),
                }
                if any(not self._in_workspace_place(matrix[:3, 3]) for matrix in waypoints.values()):
                    continue
                poses = [(name.upper(), self._pose_stamped(matrix, stamp))
                         for name, matrix in waypoints.items() if name != 'retreat_clearance']
                yield poses, waypoints

    @staticmethod
    def _place_clearance(tool, surface, cloud_tool):
        bottom = np.percentile(np.asarray(cloud_tool) @ tool[:3, :3].T + tool[:3, 3], 2, axis=0)[2]
        return float(bottom-surface[2])

    @staticmethod
    def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ROS fixed-axis RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])

    def _in_workspace(self, xyz: np.ndarray) -> bool:
        x, y, z = (float(item) for item in xyz)
        bounds = self.workspace
        table_top = float(self.table_center[2] + 0.5 * self.table_size[2])
        return (
            bounds["x_min"] <= x <= bounds["x_max"]
            and bounds["y_min"] <= y <= bounds["y_max"]
            and bounds["z_min"] <= z <= bounds["z_max"]
            and math.hypot(x, y) <= bounds["r_max"]
            and z >= table_top + self.table_clearance
        )

    def _waypoint_in_bounds(self, _stage: str, xyz: np.ndarray) -> bool:
        """Cube-tuned XY box is only for the old template path.

        GraspNet 6D waypoints (side grasps, bottles) are filtered by arm
        reach, table clearance and height; IK/collision still reject the rest.
        """
        if self.orientation_mode != "graspnet":
            return self._in_workspace(xyz)
        return self._in_workspace_place(xyz)

    def _pose_stamped(self, transform: np.ndarray, stamp) -> PoseStamped:
        validate_homogeneous(transform)
        quaternion = matrix_to_quaternion(transform[:3, :3])
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = stamp
        pose.pose.position.x = float(transform[0, 3])
        pose.pose.position.y = float(transform[1, 3])
        pose.pose.position.z = float(transform[2, 3])
        pose.pose.orientation.x = float(quaternion[0])
        pose.pose.orientation.y = float(quaternion[1])
        pose.pose.orientation.z = float(quaternion[2])
        pose.pose.orientation.w = float(quaternion[3])
        norm = math.sqrt(
            pose.pose.orientation.x ** 2
            + pose.pose.orientation.y ** 2
            + pose.pose.orientation.z ** 2
            + pose.pose.orientation.w ** 2
        )
        if abs(norm - 1.0) > 1.0e-6:
            raise RuntimeError("generated quaternion is not normalized")
        return pose

    def _in_workspace_place(self, xyz: np.ndarray) -> bool:
        """放置段的宽松工作区：x/y 矩形（为 HSV 抓取调的）不适用用户指定的
        放置位置，只保留臂展半径、桌面净空和高度上限，可行性交给 IK/碰撞。"""
        x, y, z = (float(item) for item in xyz)
        bounds = self.workspace
        table_top = float(self.table_center[2] + 0.5 * self.table_size[2])
        return (
            math.hypot(x, y) <= bounds["r_max"]
            and z >= table_top + self.table_clearance
            and z <= bounds["z_max"]
        )

    def _safety_scene(self, stamp) -> PlanningScene:
        objects: List[CollisionObject] = []
        if self.require_table:
            objects.append(
                self._box_object("remote_grasp_table", self.table_center, self.table_size)
            )
        if self.require_camera:
            transform_base_camera = self._lookup_matrix(
                self.base_frame, self.camera_frame, stamp
            )
            center = transform_base_camera @ np.asarray(
                [self.camera_center[0], self.camera_center[1], self.camera_center[2], 1.0]
            )
            objects.append(
                self._box_object("remote_grasp_camera", center[:3], self.camera_size)
            )
        # The support must collide with the real URDF fingers, even though the
        # carried object's bottom is intentionally allowed to approach its top.
        support = CollisionObject()
        support.header.frame_id = self.base_frame
        support.id = 'remote_grasp_place_support'
        if self.place_support_hull is not None:
            vertices, triangles = support_prism(self.place_support_hull, *self.place_support_zrange)
            mesh = Mesh()
            mesh.vertices = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in vertices]
            mesh.triangles = [MeshTriangle(vertex_indices=list(t)) for t in triangles]
            support.meshes = [mesh]
            support.mesh_poses = [Pose()]
            support.mesh_poses[0].orientation.w = 1.0
            support.operation = CollisionObject.ADD
        else:
            # A previous placement's obstacle must not block a new grasp of it.
            support.operation = CollisionObject.REMOVE
        objects.append(support)
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objects
        return scene

    def _box_object(self, name: str, center: np.ndarray, size: np.ndarray) -> CollisionObject:
        if len(center) != 3 or len(size) != 3 or np.any(np.asarray(size) <= 0.0):
            raise RuntimeError(f"invalid collision box {name}")
        collision = CollisionObject()
        collision.header.frame_id = self.base_frame
        collision.id = name
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(item) for item in size]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(item) for item in center)
        pose.orientation.w = 1.0
        collision.primitives = [primitive]
        collision.primitive_poses = [pose]
        collision.operation = CollisionObject.ADD
        return collision

    def _parse_planning_prompt(self, prompt: str):
        """解析组合 prompt："cube | ref: red cube | place: black box; prel: in"

        首段直接是 "place: ..." 时表示**纯放置**（手里已经拿着东西，不再抓取），
        此时抓取目标返回 None。返回 (抓取目标, 抓取参照物, 参照关系, 放置目标, 放置关系)。
        """
        parts = [p.strip() for p in str(prompt).split(" | ") if p.strip()]
        grasp_prompt: Optional[str] = parts[0] if parts else ""
        ref_prompt = ref_rel = None
        place_prompt = place_rel = None
        segments = parts[1:]
        if grasp_prompt and grasp_prompt.lower().startswith("place:"):
            place_prompt, place_rel = self._split_place_segment(grasp_prompt)
            grasp_prompt = None
        for seg in segments:
            if seg.startswith("ref: "):
                body = seg[5:]
                if "; rel: " in body:
                    ref_prompt, ref_rel = body.split("; rel: ", 1)
                else:
                    ref_prompt = body
                ref_prompt = ref_prompt.strip() or None
            elif seg.startswith("place: "):
                place_prompt, place_rel = self._split_place_segment(seg)
        return grasp_prompt, ref_prompt, ref_rel, place_prompt, place_rel

    @staticmethod
    def _split_place_segment(segment: str) -> Tuple[Optional[str], str]:
        """从 "place: black box; prel: in" 拆出 (目标, 关系)。"""
        body = segment[7:] if segment.startswith("place: ") else segment
        if "; prel: " in body:
            target, relation = body.split("; prel: ", 1)
        else:
            target, relation = body, ""
        return target.strip() or None, (relation.strip() or "in")

    def _publish_grasp_markers(
        self,
        grasps: Sequence[RemoteGrasp],
        transforms: Sequence[np.ndarray],
        stamp,
    ) -> None:
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        marker_id = 1
        colors = [(1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.3, 1.0)]
        for grasp, transform in zip(grasps, transforms):
            origin = transform[:3, 3]
            for axis in range(3):
                marker = Marker()
                marker.header.frame_id = self.base_frame
                marker.header.stamp = stamp
                marker.ns = "remote_grasp_axes"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                end = origin + transform[:3, axis] * self.marker_axis_length
                marker.points = [
                    Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
                    Point(x=float(end[0]), y=float(end[1]), z=float(end[2])),
                ]
                marker.scale.x = self.marker_axis_diameter
                marker.scale.y = self.marker_axis_diameter * 1.8
                marker.scale.z = self.marker_axis_diameter * 2.5
                marker.color.r, marker.color.g, marker.color.b = colors[axis]
                marker.color.a = 0.9
                marker.lifetime = self._duration_message(self.marker_lifetime)
                array.markers.append(marker)
            text = Marker()
            text.header.frame_id = self.base_frame
            text.header.stamp = stamp
            text.ns = "remote_grasp_scores"
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(origin[0])
            text.pose.position.y = float(origin[1])
            text.pose.position.z = float(origin[2] + 0.025)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.018
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"#{grasp.index} s={grasp.score:.3f} w={grasp.width_m:.3f}m"
            text.lifetime = self._duration_message(self.marker_lifetime)
            array.markers.append(text)
        self.marker_publisher.publish(array)

    @staticmethod
    def _duration_message(seconds: float) -> DurationMsg:
        message = DurationMsg()
        seconds = max(0.0, float(seconds))
        message.sec = int(seconds)
        message.nanosec = int((seconds - int(seconds)) * 1.0e9)
        return message

    def _publish_gripper_plan(self, object_width: float) -> None:
        trajectory = JointTrajectory()
        trajectory.header.frame_id = self.base_frame
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = list(self.gripper_joints)
        open_finger = min(
            0.044,
            max(0.0, (float(object_width) + self.width_margin) * self.width_to_finger),
        )
        start = JointTrajectoryPoint()
        start.positions = [open_finger] * len(self.gripper_joints)
        start.time_from_start = self._duration_message(0.0)
        closed = JointTrajectoryPoint()
        closed.positions = [0.0] * len(self.gripper_joints)
        closed.time_from_start = self._duration_message(1.0)
        trajectory.points = [start, closed]
        self.gripper_plan_publisher.publish(trajectory)

    def destroy_node(self):
        self.action_server.destroy()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RemoteGraspNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
