"""ChArUco detection and geometric quality checks, independent of ROS."""
import cv2
import numpy as np

DETECTOR_VERSION = 'intrinsics_quality_v2'


def detect_charuco(gray, dictionary, board, params, K, D):
    if hasattr(cv2.aruco, 'CharucoDetector'):
        options = cv2.aruco.CharucoParameters()
        options.cameraMatrix = K
        options.distCoeffs = D
        detector = cv2.aruco.CharucoDetector(board, options, params)
        cc, ci, mc, mi = detector.detectBoard(gray)
    else:
        mc, mi, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        cc = ci = None
        if mi is not None and len(mi):
            _, cc, ci = cv2.aruco.interpolateCornersCharuco(
                mc, mi, gray, board, cameraMatrix=K, distCoeffs=D)
    return mc, mi, cc, ci


def chessboard_points(board):
    return np.asarray(board.getChessboardCorners() if hasattr(board, 'getChessboardCorners')
                      else board.chessboardCorners, dtype=np.float32)


def pose_quality(board, corners, ids, K, D, rvec, tvec):
    obj = chessboard_points(board)[np.asarray(ids).reshape(-1)]
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    delta = projected.reshape(-1, 2) - np.asarray(corners).reshape(-1, 2)
    errors = np.linalg.norm(delta, axis=1)
    R = cv2.Rodrigues(rvec)[0]
    depth = (obj @ R.T + np.asarray(tvec).reshape(3))[:, 2]
    return {
        'n_corners': int(len(obj)),
        'reprojection_rms_px': float(np.sqrt(np.mean(errors**2))),
        'reprojection_max_px': float(np.max(errors)),
        'tilt_deg': float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1)))),
        'min_corner_depth_m': float(np.min(depth)),
    }


def quality_passes(quality, min_corners, max_rms_px, max_tilt_deg):
    return (all(np.isfinite(v) for v in quality.values())
            and quality['n_corners'] >= min_corners
            and quality['reprojection_rms_px'] <= max_rms_px
            and quality['tilt_deg'] <= max_tilt_deg
            and quality['min_corner_depth_m'] > 0)
