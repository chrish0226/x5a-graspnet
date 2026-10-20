from types import SimpleNamespace
import time

from x5a_pick_place.pick_place_node import PickPlaceNode


def fixture(q):
    n = SimpleNamespace(q=q, joint_state_arrival=time.monotonic(),
                        servo_lower=[-3.14,-.05,-.1,-1.28,-1.57,-2.],
                        servo_upper=[2.618,3.5,3.2,1.55,1.57,2.])
    n.servo_bounds_fault = lambda q: PickPlaceNode.servo_bounds_fault(n,q)
    return n


def test_passive_wrist_pose_cannot_be_used_as_a_servo_start():
    n = fixture([.8,1.14,1.5,-1.64,-.03,-.07])
    assert 'joint4' in PickPlaceNode.servo_readiness_fault(n)
    n.q[3] = -1.2
    assert not PickPlaceNode.servo_readiness_fault(n)


def test_missing_stale_and_nonfinite_feedback_block_motion():
    n = fixture([0.]*6)
    n.joint_state_arrival -= 1.
    assert 'stale' in PickPlaceNode.servo_readiness_fault(n)
    assert n.servo_bounds_fault(None)
    assert n.servo_bounds_fault([0.,0.,0.,float('nan'),0.,0.])
