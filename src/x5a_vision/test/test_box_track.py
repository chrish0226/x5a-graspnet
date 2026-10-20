import numpy as np
from x5a_vision.box_track import BoxTrack


def track():
    return BoxTrack(window=5,max_std=.008,reset_distance=.02,grace=2.)


def test_intermittent_depth_does_not_reset_box_stability():
    box=track()
    for i in range(5):
        center,stable=box.update(np.array([.30+.002*(i%2),.20]),i*.3)
        box.missing(i*.3+.15)
    assert stable and box.missing(2.0)
    assert box.last_seen==1.2
    assert len(box.history)==5
    assert not box.missing(3.3)
    assert not box.history


def test_moving_box_immediately_invalidates_previous_stable_center():
    box=track()
    for i in range(5):box.update(np.array([.30,.20]),i*.3)
    assert box.stable
    center,stable=box.update(np.array([.36,.20]),1.5)
    assert not stable and len(box.history)==1
    np.testing.assert_allclose(center,[.36,.20])


def test_first_miss_does_not_create_a_box_pose():
    box=track()
    assert not box.missing(0.) and box.last_seen is None


def test_slow_hd_measurements_and_one_missed_frame_preserve_window():
    box=track()
    for i in range(5):box.update(np.array([.30,.20]),i*2.5)
    assert box.stable
    assert box.missing(11.5)
    _,stable=box.update(np.array([.30,.20]),13.)
    assert stable
    _,stable=box.update(np.array([.30,.20]),18.)
    assert not stable and len(box.history)==1
