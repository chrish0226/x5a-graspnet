import math
from types import SimpleNamespace as NS
import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('arx5_arm_msg')
from geometry_msgs.msg import PoseStamped
from x5a_handeye.readonly_handeye_sampler import rotation_std_deg, ReadOnlyHandeyeSampler


def test_symmetric_oscillation_cannot_look_stationary():
    q, w = math.sin(math.radians(2)/2), math.cos(math.radians(2)/2)
    poses = [{'xyz': [0, 0, 0], 'quat_xyzw': [0, 0, q, w]},
             {'xyz': [0, 0, 0], 'quat_xyzw': [0, 0, -q, w]}]
    assert rotation_std_deg(poses) == pytest.approx(2.0)
    assert rotation_std_deg([poses[0], poses[0]]) < 1e-4


def test_board_stale_and_repeated_messages_are_not_new_samples():
    state = NS(last_board_stamp=None, max_board_age=.5, have_board=False,
               capturing=True, board_records=[],
               get_clock=lambda: NS(now=lambda: NS(nanoseconds=10_000_000_000)))
    msg = PoseStamped(); msg.header.frame_id='wrist_camera_color_optical_frame'
    msg.pose.orientation.w=1.0; msg.header.stamp.sec=8
    ReadOnlyHandeyeSampler.on_board(state, msg)
    assert not state.have_board and not state.board_records
    msg.header.stamp.sec=10
    ReadOnlyHandeyeSampler.on_board(state, msg)
    ReadOnlyHandeyeSampler.on_board(state, msg)
    assert state.have_board and len(state.board_records)==1
