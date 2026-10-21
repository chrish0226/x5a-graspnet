from types import SimpleNamespace
import numpy as np
from x5a_vision.cube_detector import CubeDetector


def test_physical_cube_size_accepts_top_and_rejects_large_patch():
    info = SimpleNamespace(k=[1000.,0.,100.,0.,1000.,100.,0.,0.,1.])
    node = SimpleNamespace(transform_matrix=lambda _:np.eye(4), depth_min=.2,depth_max=.8,
                           table_z=.47,object_height=.03,face_size_tolerance=.012)
    depth = np.full((200,200),.5)
    inner = np.zeros(depth.shape,bool)
    inner[70:130,70:130] = True
    assert CubeDetector.plane_measurement(node,inner,depth,info,None) is not None
    inner[50:150,50:150] = True
    assert CubeDetector.plane_measurement(node,inner,depth,info,None) is None
