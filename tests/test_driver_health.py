import json
import threading
import time
from types import SimpleNamespace

from std_msgs.msg import String
from x5a_moveit_official_adapter.official_trajectory_adapter import OfficialTrajectoryAdapter


def adapter():
    n = SimpleNamespace(lock=threading.Lock(), driver_health=None,
                        driver_health_time=0.0, status_timeout=.5,
                        require_driver_health=True)
    n.driver_health_fault = lambda: OfficialTrajectoryAdapter.driver_health_fault(n)
    return n


def test_required_health_cannot_be_replaced_with_fresh_robot_status():
    n = adapter()
    assert n.driver_health_fault() == 'no motor feedback health'
    OfficialTrajectoryAdapter._driver_health_cb(n, String(data=json.dumps(
        {'online': True, 'protected': True, 'ready': False})))
    assert 'protection latched' in n.driver_health_fault()
    OfficialTrajectoryAdapter._driver_health_cb(n, String(data=json.dumps(
        {'online': True, 'protected': False, 'ready': True})))
    assert n.driver_health_fault() == ''
    n.driver_health_time = time.monotonic() - 1
    assert n.driver_health_fault() == 'motor feedback health is stale'


def test_invalid_health_cannot_authorize_motion():
    n = adapter()
    for text in ('[]', '{', '{"online": "true"}', '{}'):
        OfficialTrajectoryAdapter._driver_health_cb(n, String(data=text))
        assert n.driver_health_fault()


def test_original_driver_compatibility_is_explicit():
    n = adapter()
    n.require_driver_health = False
    assert n.driver_health_fault() == ''
