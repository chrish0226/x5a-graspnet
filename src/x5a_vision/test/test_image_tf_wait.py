import threading
import time

import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from tf2_ros import TransformBroadcaster

from x5a_vision.cube_detector import CubeDetector, detection_executor


def test_image_callback_receives_tf_that_arrives_while_waiting():
    # A separate DDS domain prevents test topics reaching any live robot code.
    rclpy.init(domain_id=173, args=['--ros-args', '-p', 'base_frame:=test_base',
                                 '-p', 'optical_frame:=test_camera'])
    node = CubeDetector()
    broadcaster = TransformBroadcaster(node)
    executor = detection_executor(node)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    waiting, done = threading.Event(), threading.Event()
    result = {}
    timer = None
    try:
        stamp = node.get_clock().now()-Duration(seconds=.1)

        def tf_at(delta_s, x):
            m = TransformStamped()
            m.header.frame_id, m.child_frame_id = 'test_base', 'test_camera'
            m.header.stamp = (stamp+Duration(seconds=delta_s)).to_msg()
            m.transform.translation.x = x
            m.transform.rotation.w = 1.
            return m

        before = tf_at(-.02, 0.)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            broadcaster.sendTransform(before)
            if node.tf_buffer.can_transform('test_base', 'test_camera', stamp-Duration(seconds=.02)):
                break
            time.sleep(.02)
        else:
            pytest.fail('initial TF was not received')

        def image_callback():
            timer.cancel()
            waiting.set()
            try:
                result['tf'] = node.lookup_transform(stamp.to_msg())
            except Exception as exc:
                result['error'] = exc
            finally:
                done.set()

        timer = node.create_timer(.05, image_callback)
        assert waiting.wait(2)
        time.sleep(.03)
        broadcaster.sendTransform(tf_at(.02, 1.))
        assert done.wait(2)
        assert 'error' not in result, repr(result.get('error'))
        assert result['tf'].transform.translation.x == pytest.approx(.5)
    finally:
        executor.shutdown()
        thread.join(timeout=2)
        node.destroy_node()
        rclpy.try_shutdown()
