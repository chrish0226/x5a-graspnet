from types import SimpleNamespace
from types import MethodType
import cv2
import numpy as np
from x5a_vision.cube_detector import CubeDetector


def test_cube_cut_by_image_edge_cannot_become_a_stable_center():
    detector = SimpleNamespace(min_area=0., max_area=10000.,
                               max_aspect=3., min_rectangularity=.5)
    depth = np.full((100,100), .3)
    contour = np.array([[[30,75]], [[60,75]], [[60,99]], [[30,99]]], np.int32)
    # Reject before attempting TF, top-plane fitting, or depth center estimation.
    assert CubeDetector.contour_measurement(detector, contour, depth, None, None) is None


def test_contour_roi_preserves_full_image_erosion_intrinsics_and_top_pose():
    detector = SimpleNamespace(min_area=0., max_area=10000., max_aspect=3., min_rectangularity=.5,
        inner_erode_px=5, top_face_mode='base_plane', depth_min=.2, depth_max=.8,
        table_z=.27, object_height=.03, face_size_tolerance=.012, optical_frame='camera',
        bounds=dict(x_min=-.5, x_max=.5, y_min=-.5, y_max=.5, z_min=.2, z_max=.4, r_max=.65),
        expected_xy=np.array([0., 0.]), transform_matrix=lambda stamp: np.eye(4), valid_cube_top=lambda z: True)
    detector.plane_measurement = MethodType(CubeDetector.plane_measurement, detector)
    info = SimpleNamespace(k=[200., 0., 60., 0., 200., 50., 0., 0., 1.])
    depth = np.full((100, 120), .3)
    contour = np.round(cv2.boxPoints(((48., 37.), (24., 24.), 17.))).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros(depth.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    from builtin_interfaces.msg import Time
    stamp = Time()
    original = detector.plane_measurement(inner, depth, info, stamp)
    optimized = CubeDetector.contour_measurement(detector, contour, depth, info, stamp)
    assert original is not None and optimized is not None
    np.testing.assert_allclose(optimized['base'], original[0], atol=1e-8)
    np.testing.assert_allclose(optimized['pixel'], original[2], atol=1e-8)
    assert abs(optimized['yaw']-original[3]) < 1e-8
