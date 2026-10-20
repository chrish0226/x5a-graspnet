from types import SimpleNamespace

import pytest
from std_srvs.srv import Trigger

from x5a_task_server.observation_profiles import mirrored_seed, crosses_home_plane
from x5a_web_agent.wrist_commands import parse_wrist_command


@pytest.mark.parametrize('text,side', [
    ('镜像侧准备', 'mirror'), ('镜像准备', 'mirror'), ('右侧准备', 'mirror'),
    ('请准备右边', 'mirror'), ('当前侧准备', 'current'), ('左侧准备', 'current'),
])
def test_prepare_side_voice(text, side):
    assert parse_wrist_command(text) == {'action': 'prepare', 'observation_side': side}


def test_home_plane_crossing_requires_folded_waypoint():
    q = [.8, .7, .8, -1., .05, .06]
    home = [0.]*6
    mirrored = mirrored_seed(q, home)
    assert mirrored == [-.8, .7, .8, -1., -.05, -.06]
    assert crosses_home_plane(q, mirrored, home)
    assert not crosses_home_plane(home, mirrored, home)
    assert not crosses_home_plane(mirrored, mirrored, home)
    assert mirrored_seed(mirrored, home) == q
