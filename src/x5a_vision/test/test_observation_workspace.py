from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from x5a_vision.cube_detector import CubeDetector, COLORS


def test_side_switch_reflects_workspace_and_clears_old_tracks():
    normal={'x_min':-.08,'x_max':.48,'y_min':.02,'y_max':.58,'r_max':.65}
    n=SimpleNamespace(
        allow_mirrored_side=True, observation_side='current', _normal_bounds=normal,
        _normal_expected_xy=np.array([.2,.33]), bounds=dict(normal), expected_xy=np.array([.2,.33]),
        histories={c:deque([[1,2,3]]) for c in COLORS},
        cover_history={c:deque(['orange']) for c in COLORS},
        yaw_histories={c:deque([.1]) for c in COLORS},cube_last_seen={c:10. for c in COLORS},
        stable_pubs={c:Mock() for c in COLORS},box_track=Mock(),box_yaws=deque([.1]),box_stable_pub=Mock(),
    )
    CubeDetector.on_observation_side(n, SimpleNamespace(data='mirror'))
    assert n.bounds['y_min']==-.58 and n.bounds['y_max']==-.02
    assert n.bounds['x_min']==normal['x_min'] and n.bounds['r_max']==normal['r_max']
    np.testing.assert_allclose(n.expected_xy,[.2,-.33])
    assert not any(n.histories[c] for c in COLORS)
    assert not n.box_yaws
    n.box_track.clear.assert_called_once()
    CubeDetector.on_observation_side(n, SimpleNamespace(data='current'))
    assert n.bounds==normal
    np.testing.assert_allclose(n.expected_xy,[.2,.33])
