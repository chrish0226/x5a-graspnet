from types import SimpleNamespace
from action_msgs.msg import GoalStatus
from x5a_pick_place.pick_place_node import gripper_result_ok


def test_cube_stopping_fingers_before_zero_is_a_successful_close():
    r = SimpleNamespace(position=0.015, reached_goal=False, stalled=True)
    assert gripper_result_ok(False,r,GoalStatus.STATUS_SUCCEEDED,.004)
    assert not gripper_result_ok(True,r,GoalStatus.STATUS_SUCCEEDED,.004)


def test_empty_close_and_aborted_stall_cannot_claim_a_held_cube():
    r = SimpleNamespace(position=0.,reached_goal=True,stalled=False)
    assert not gripper_result_ok(False,r,GoalStatus.STATUS_SUCCEEDED,.004)
    r.position=.015;r.stalled=True
    assert not gripper_result_ok(False,r,GoalStatus.STATUS_ABORTED,.004)
