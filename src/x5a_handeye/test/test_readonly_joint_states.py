from types import SimpleNamespace

from arx5_arm_msg.msg import RobotStatus
from x5a_handeye.readonly_joint_states import ReadOnlyJointStates


def test_raw_feedback_bridge_preserves_encoders_timestamp_and_rejects_conflicts_and_stale_data():
    published = []
    bridge = SimpleNamespace(
        last_publish=-float('inf'), last_stamp=None, topic='/joint_states',
        count_publishers=lambda _: 1,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        publisher=SimpleNamespace(publish=published.append),
        get_logger=lambda: SimpleNamespace(warn=lambda *args, **kwargs: None),
    )
    msg = RobotStatus()
    msg.header.stamp.sec = 1
    msg.joint_pos = [.1, -.2, .3, -.4, .5, -.6, 2.5]
    ReadOnlyJointStates.on_status(bridge, msg)
    assert len(published) == 1
    assert list(published[0].position[:6]) == list(msg.joint_pos[:6])
    assert list(published[0].position[6:]) == [.022, .022]
    assert published[0].header.stamp == msg.header.stamp
    bridge.last_publish = -float('inf')
    bridge.count_publishers = lambda _: 2
    ReadOnlyJointStates.on_status(bridge, msg)
    assert len(published) == 1
    bridge.count_publishers = lambda _: 1
    msg.header.stamp.sec = 0
    msg.header.stamp.nanosec = 1
    ReadOnlyJointStates.on_status(bridge, msg)
    assert len(published) == 1
