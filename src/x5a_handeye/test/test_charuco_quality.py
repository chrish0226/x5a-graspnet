import cv2
import numpy as np
import pytest

from x5a_handeye.charuco_detection import detect_charuco, pose_quality, quality_passes, chessboard_points
from x5a_handeye.observations import observation_geometry, pair_observations


def make_board():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    board = (cv2.aruco.CharucoBoard((5, 7), .030, .022, d)
             if hasattr(cv2.aruco, 'CharucoBoard') else cv2.aruco.CharucoBoard_create(5, 7, .030, .022, d))
    return d, board


def test_real_charuco_detection_uses_intrinsics_and_recovers_projected_pose():
    d, board = make_board()
    pattern = board.generateImage((600, 840)) if hasattr(board, 'generateImage') else board.draw((600, 840))
    K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
    D = np.zeros(5)
    rv = np.array([.20, -.15, .1]); tv = np.array([-.07, -.1, .55])
    corners3d = np.array([[0, 0, 0], [.15, 0, 0], [.15, .21, 0], [0, .21, 0]], np.float32)
    dst, _ = cv2.projectPoints(corners3d, rv, tv, K, D)
    H = cv2.getPerspectiveTransform(np.array([[0, 0], [600, 0], [600, 840], [0, 840]], np.float32), dst.reshape(-1, 2))
    gray = cv2.warpPerspective(pattern, H, (640, 480), borderValue=255)
    params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'DetectorParameters') else cv2.aruco.DetectorParameters_create()
    mc, mi, cc, ci = detect_charuco(gray, d, board, params, K, D)
    assert len(mi) == 17 and len(ci) == 24
    obj = chessboard_points(board)[ci.flatten()]
    ok, actual_r, actual_t = cv2.solvePnP(obj, cc, K, D)
    assert ok
    expected_R = cv2.Rodrigues(rv)[0]
    expected_t = tv.copy()
    # OpenCV 4.5's printed board origin is bottom-left, +Y up. Later
    # versions use top-left, +Y down. The renderer above uses image axes.
    if not hasattr(board, 'getChessboardCorners'):
        expected_t += expected_R @ np.array([0, .21, 0])
        expected_R = expected_R @ np.diag([1., -1., -1.])
    np.testing.assert_allclose(actual_t.flatten(), expected_t, atol=.002)
    rotation_difference = cv2.Rodrigues(cv2.Rodrigues(actual_r)[0].T @ expected_R)[0]
    assert np.linalg.norm(rotation_difference) < np.deg2rad(1)
    quality = pose_quality(board, cc, ci, K, D, actual_r, actual_t)
    assert quality_passes(quality, 12, 1., 60.)


def test_reprojection_gate_rejects_bad_corner_fit_and_edge_on_or_negative_depth():
    _, board = make_board()
    ids = np.arange(24, dtype=np.int32).reshape(-1, 1)
    K = np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1.]])
    rv, tv = np.zeros(3), np.array([-.06, -.09, .4])
    pixels, _ = cv2.projectPoints(chessboard_points(board), rv, tv, K, None)
    good = pose_quality(board, pixels, ids, K, None, rv, tv)
    assert quality_passes(good, 12, 1., 60.)
    bad_pixels = pixels.copy(); bad_pixels[0] += [30, 0]
    assert not quality_passes(pose_quality(board, bad_pixels, ids, K, None, rv, tv), 12, 1., 60.)
    for key, value in [('tilt_deg', 70), ('min_corner_depth_m', -1), ('n_corners', 6), ('reprojection_rms_px', float('nan'))]:
        assert not quality_passes(dict(good, **{key: value}), 12, 1., 60.)


def test_pairing_rejects_stale_missing_or_other_camera_observations():
    pose = {'xyz': [0, 0, 0], 'quat_xyzw': [0, 0, 0, 1]}
    boards = [{'stamp_ns': 1_000_000_000, 'frame_id': 'optical', 'pose': pose}]
    robots = [{'stamp_ns': 1_010_000_000, 'pose': pose, 'joints': [0]*6}]
    observations = {1_000_000_000: {'camera_frame': 'optical'}}
    pairs = pair_observations(boards, robots, observations, .04)
    assert len(pairs) == 1 and pairs[0]['pair_delta_s'] == pytest.approx(.01)
    assert pairs[0]['T_base_tool'] == pose and pairs[0]['vision'] == observations[1_000_000_000]
    assert not pair_observations(boards, robots, observations, .005)
    assert not pair_observations(boards, robots, {}, .04)
    assert not pair_observations(boards, robots, {1_000_000_000: {'camera_frame': 'another'}}, .04)


def test_geometry_signature_changes_when_board_size_or_camera_model_changes():
    observation = {'board': {'marker_length_m': .022}, 'detector_version': 'v2',
                   'opencv_version': cv2.__version__,
                   'camera_frame': 'optical', 'image_size': [640, 480],
                   'K': [[400, 0, 320], [0, 400, 240], [0, 0, 1]], 'D': [0]*5}
    geometry = observation_geometry(observation)
    assert geometry != observation_geometry(dict(observation, board={'marker_length_m': .020}))
    assert geometry != observation_geometry(dict(observation, detector_version='v1'))
    with pytest.raises(ValueError): observation_geometry(dict(observation, K=[[0]*3]*3))
