"""Frame conventions shared by the offline solver and ROS TF publisher."""
import numpy as np

from x5a_handeye.transforms import inv_T


def require_failed_preview_context(enabled, robot_prefix, model_topic):
    if enabled and (robot_prefix != 'wrist_preview/' or model_topic != '/wrist_preview/robot_description'):
        raise ValueError('failed candidate override requires the isolated wrist_preview robot/model topics')


def calibrated_transform(result, allow_model_candidate=False, allow_failed_candidate=False):
    mode = result.get('calibration_type', 'eye_to_hand')
    if mode not in ('eye_in_hand', 'eye_to_hand'):
        raise ValueError(f'unsupported calibration type: {mode}')
    if result.get('verdict') == 'FAIL':
        if not (allow_failed_candidate and allow_model_candidate
                and result.get('validation_status') == 'offline_model_candidate'):
            raise ValueError('calibration failed validation; refusing to publish TF')
    if result.get('validation_status') == 'offline_model_candidate':
        if not allow_model_candidate:
            raise ValueError('offline model candidate requires allow_model_candidate:=true for preview')
        if not result.get('robot_calibration', {}).get('expected_model'):
            raise ValueError('model candidate is missing its matching robot description signature')
    parent = result.get('tool_frame', 'tool0') if mode == 'eye_in_hand' else result.get('base_frame', 'base_link')
    if result.get('parent_frame', parent) != parent:
        raise ValueError('parent_frame conflicts with calibration type')
    optical = result.get('camera_frame', 'camera_color_optical_frame')
    root = result.get('camera_root_frame', 'camera_link')
    if not all(isinstance(f, str) and f and not f.startswith('/') for f in (parent, optical, root)):
        raise ValueError('frame names must be nonempty and have no leading slash')
    if parent in (root, optical):
        raise ValueError('robot parent cannot equal a camera frame')
    key = 'T_tool_camera' if mode == 'eye_in_hand' else 'T_base_camera'
    T = np.asarray(result[key], dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-5)):
        raise ValueError('calibration matrix is not a finite rigid transform')
    return parent, root, optical, T


def camera_root_transform(T_parent_optical, T_root_optical):
    return T_parent_optical @ inv_T(T_root_optical)


def prefixed_robot_frame(frame, prefix):
    """Place a preview robot in its own TF tree while retaining camera frames."""
    if not prefix:
        return frame
    if (not isinstance(prefix, str) or prefix.startswith('/')
            or any(c.isspace() for c in prefix)
            or any(not part for part in prefix.rstrip('/').split('/'))):
        raise ValueError('robot frame prefix must have no leading slash, whitespace or empty segment')
    return prefix.rstrip('/') + '/' + frame
