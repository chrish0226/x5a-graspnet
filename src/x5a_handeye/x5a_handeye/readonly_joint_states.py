"""RobotStatus -> JointState display bridge, with no command/action interfaces."""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from arx5_arm_msg.msg import RobotStatus


class ReadOnlyJointStates(Node):
    def __init__(self):
        super().__init__('x5a_readonly_joint_states')
        self.declare_parameter('status_topic', '/arm_status')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.topic = str(self.get_parameter('joint_state_topic').value)
        self.publisher = self.create_publisher(JointState, self.topic, 10)
        self.subscription = self.create_subscription(
            RobotStatus, str(self.get_parameter('status_topic').value), self.on_status, 10)
        self.last_publish = -math.inf
        self.last_stamp = None

    def on_status(self, msg):
        # Decimate the vendor's 1 kHz feedback; preserve the measured timestamp.
        now = time.monotonic()
        if now - self.last_publish < 1./30.:
            return
        if self.count_publishers(self.topic) != 1:
            self.get_logger().warn('another JointState publisher exists; read-only bridge will wait',
                                   throttle_duration_sec=5.)
            return
        stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        age = (self.get_clock().now().nanoseconds - stamp)*1e-9
        if stamp <= 0 or stamp == self.last_stamp or not -.1 <= age <= .5:
            return
        q = [float(v) for v in msg.joint_pos]
        if len(q) < 6 or not all(math.isfinite(v) for v in q):
            return
        out = JointState()
        out.header.stamp = msg.header.stamp
        out.name = [f'joint{i}' for i in range(1, 7)]
        out.position = q[:6]  # raw encoder values, including with a calibrated URDF
        if len(q) >= 7:
            finger = .044 * max(0., min(1., q[6]/5.))
            out.name.extend(['joint7', 'joint8'])
            out.position.extend([finger, finger])
        self.publisher.publish(out)
        self.last_publish, self.last_stamp = now, stamp


def main(args=None):
    rclpy.init(args=args)
    node = ReadOnlyJointStates()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
