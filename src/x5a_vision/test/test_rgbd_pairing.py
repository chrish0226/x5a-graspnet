from types import SimpleNamespace

from x5a_vision.rgbd_pairing import RgbdPairBuffer


def frame(seconds):
    ns = round(seconds*1e9)
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=ns//1_000_000_000,
                                                                       nanosec=ns%1_000_000_000)))


def test_delayed_color_matches_retained_depth_instead_of_newest_depth():
    queue = RgbdPairBuffer()
    matching, latest = frame(10.), frame(10.4)
    queue.depths.extend((matching, latest))
    assert queue.take_latest(10_500_000_000) is None
    color = frame(10.0002)
    queue.colors.append(color)
    assert queue.take_latest(10_500_000_000) == (color, matching)
    # Reusing a pair must never fake five stable observations.
    assert queue.take_latest(10_600_000_000) is None


def test_newest_matching_pair_wins_even_if_depth_arrives_out_of_order():
    queue = RgbdPairBuffer()
    old, new = frame(10.), frame(10.2)
    queue.depths.extend((new, old))
    queue.colors.extend((frame(10.001), frame(10.201)))
    assert queue.take_latest(10_300_000_000)[1] is new
    assert queue.take_latest(10_400_000_000) is None


def test_mismatched_expired_and_future_exposures_are_rejected():
    queue = RgbdPairBuffer()
    queue.depths.append(frame(10.))
    queue.colors.append(frame(10.041))
    assert queue.take_latest(10_100_000_000) is None
    queue.colors.append(frame(10.))
    assert queue.take_latest(11_600_000_000) is None
    assert queue.take_latest(9_900_000_000) is None
    assert queue.take_latest(10_100_000_000) is not None
