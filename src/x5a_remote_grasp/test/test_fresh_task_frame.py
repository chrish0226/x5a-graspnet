"""A dependent task must not reuse a pre-placement RGB-D frame."""
from types import SimpleNamespace as NS
import threading
import time

import pytest

from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode


def frame(stamps):
    return NS(**{name: NS(header=NS(stamp=NS(sec=sec, nanosec=0)))
                 for name, sec in zip(('color', 'depth', 'info'), stamps)})


def test_new_task_waits_until_all_three_images_are_after_request():
    node = NS(frame_lock=threading.Lock(), latest_frame=frame([1, 1, 1]))
    def update():
        time.sleep(.02)
        with node.frame_lock:
            node.latest_frame = frame([3, 3, 3])
    worker = threading.Thread(target=update)
    worker.start()
    try:
        RemoteGraspNode._wait_for_new_frame(node, 2_000_000_000, timeout=.5)
    finally:
        worker.join()
    assert node.latest_frame.depth.header.stamp.sec == 3


@pytest.mark.parametrize('stamps', [None, [1, 1, 1], [3, 1, 3]])
def test_old_or_partially_old_frames_do_not_start_new_scene_planning(stamps):
    node = NS(frame_lock=threading.Lock(), latest_frame=None if stamps is None else frame(stamps))
    with pytest.raises(RuntimeError, match='新同步 RGB-D 图像超时'):
        RemoteGraspNode._wait_for_new_frame(node, 2_000_000_000, timeout=.01)
