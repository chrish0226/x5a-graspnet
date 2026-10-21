#!/usr/bin/env python3
"""Publish calibrated base/tool -> camera root, preserving driver optical TF."""
from __future__ import annotations

from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

from x5a_handeye.transforms import pose_to_T, T_to_pose
from x5a_handeye.calibrated_tf import calibrated_transform, camera_root_transform, prefixed_robot_frame, require_failed_preview_context
from x5a_handeye.robot_model import require_matching_model


class HandeyeTF(Node):
    def __init__(self) -> None:
        super().__init__("x5a_handeye_tf")
        self.declare_parameter("result_yaml", "")
        self.declare_parameter('allow_model_candidate', False)
        self.declare_parameter('allow_failed_candidate', False)
        self.declare_parameter('robot_description_topic', '/robot_description')
        self.declare_parameter('robot_frame_prefix', '')
        path = str(self.get_parameter("result_yaml").value)
        if not path:
            from ament_index_python.packages import get_package_share_directory

            path = str(Path(get_package_share_directory("x5a_handeye")) / "config" / "handeye_result.yaml")
        self.result = yaml.safe_load(Path(path).read_text())
        allow_failed = bool(self.get_parameter('allow_failed_candidate').value)
        require_failed_preview_context(allow_failed,
            str(self.get_parameter('robot_frame_prefix').value),
            str(self.get_parameter('robot_description_topic').value))
        self.parent, self.child, self.optical, self.T_parent_optical = calibrated_transform(
            self.result, bool(self.get_parameter('allow_model_candidate').value), allow_failed)
        self.parent = prefixed_robot_frame(self.parent, str(self.get_parameter('robot_frame_prefix').value))
        self.expected_model = self.result.get('robot_calibration', {}).get('expected_model')
        self.model_matches = not bool(self.expected_model)
        self.model_topic = str(self.get_parameter('robot_description_topic').value)
        if self.expected_model:
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
            self.model_sub = self.create_subscription(String, self.model_topic, self.on_model, qos)
        self.static_br = StaticTransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.try_publish)
        self.published = False
        self.get_logger().info(f"loaded handeye result from {path}")
        if self.result.get('verdict') == 'FAIL':
            self.get_logger().warn('Using a user-accepted experimental candidate in the isolated preview; original validation verdict remains FAIL')

    def on_model(self, msg):
        try:
            require_matching_model(msg.data, self.expected_model)
            self.model_matches = True
        except Exception as exc:
            self.model_matches = False
            self.get_logger().error(f'refusing calibrated camera TF: {exc}', throttle_duration_sec=5.0)

    def try_publish(self):
        if self.published:
            return
        if self.expected_model and (not self.model_matches or self.count_publishers(self.model_topic) != 1):
            self.get_logger().warn('waiting for exactly one matching calibrated robot_description publisher',
                                   throttle_duration_sec=5.0)
            return
        optical = self.optical
        # RealSense frames often: camera_link -> camera_color_frame -> camera_color_optical_frame
        # We calibrated base->optical. Convert to base->camera_link if possible.
        parent, child = self.parent, self.child
        frames = yaml.safe_load(self.tf_buffer.all_frames_as_yaml()) or {}
        if frames.get(child, {}).get('parent'):
            self.get_logger().error(
                f'{child} already has a TF parent; stop the existing camera extrinsic publisher first',
                throttle_duration_sec=5.0,
            )
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                child, optical, Time(), timeout=Duration(seconds=0.5)
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            T_link_optical = pose_to_T([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])
            T_base_link = camera_root_transform(self.T_parent_optical, T_link_optical)
            xyz, quat = T_to_pose(T_base_link)
            self.get_logger().info(f"publishing {parent} -> {child} using optical calibration")
        except Exception as e:
            # The calibrated transform is base->optical, while RealSense owns
            # the optical frame. Never create a second optical-frame parent or
            # a substitute frame: wait until the camera's link->optical chain
            # exists, then publish the one allowed transform base->camera_link.
            self.get_logger().warn(
                f"camera root TF unavailable ({e}); waiting for camera driver TF",
                throttle_duration_sec=5.0,
            )
            return
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(xyz[0])
        msg.transform.translation.y = float(xyz[1])
        msg.transform.translation.z = float(xyz[2])
        msg.transform.rotation.x = float(quat[0])
        msg.transform.rotation.y = float(quat[1])
        msg.transform.rotation.z = float(quat[2])
        msg.transform.rotation.w = float(quat[3])
        self.static_br.sendTransform(msg)
        self.published = True
        self.get_logger().info(
            f"static TF {parent}->{child} xyz={[round(v,4) for v in xyz]} quat={[round(v,4) for v in quat]}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = HandeyeTF()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
