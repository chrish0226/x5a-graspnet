import pytest
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from x5a_pick_place.retiming import scale_cartesian_timing


def trajectory():
    t = RobotTrajectory()
    t.joint_trajectory.joint_names = ['joint1']
    for index in range(2):
        p = JointTrajectoryPoint()
        p.positions, p.velocities, p.accelerations = [float(index)], [1.], [2.]
        p.time_from_start.sec = index
        t.joint_trajectory.points.append(p)
    return t


def test_precision_scaling_keeps_path_and_applies_velocity_and_acceleration():
    original = trajectory()
    scaled = scale_cartesian_timing(original, .33, .19)
    end = scaled.joint_trajectory.points[-1]
    assert list(end.positions) == [1.]
    assert end.time_from_start.sec+end.time_from_start.nanosec*1e-9 == pytest.approx(1/.33)
    assert end.velocities == pytest.approx([.33])
    assert end.accelerations == pytest.approx([2*.33**2])
    assert original.joint_trajectory.points[-1].time_from_start.sec == 1


def test_acceleration_limit_can_require_more_time_and_malformed_timing_is_rejected():
    scaled = scale_cartesian_timing(trajectory(), .82, .16)
    assert scaled.joint_trajectory.points[-1].velocities == pytest.approx([.4])
    bad = trajectory()
    bad.joint_trajectory.points[-1].time_from_start.sec = 0
    with pytest.raises(ValueError):
        scale_cartesian_timing(bad, .33, .19)
    with pytest.raises(ValueError):
        scale_cartesian_timing(trajectory(), 0., .19)
