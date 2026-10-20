"""Pair immutable board observations with robot feedback at image time."""
import numpy as np


def observation_geometry(observation):
    K, D = np.asarray(observation['K'], float), np.asarray(observation['D'], float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(D).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError('invalid observation camera model')
    return {k: observation[k] for k in
            ('board', 'detector_version', 'opencv_version', 'camera_frame', 'image_size', 'K', 'D')}


def pair_observations(board_records, robot_records, observations, max_delta_s):
    if not robot_records:
        return []
    pairs = []
    for board in board_records:
        stamp = board['stamp_ns']
        observation = observations.get(stamp)
        if observation is None or observation['camera_frame'] != board['frame_id']:
            continue
        robot = min(robot_records, key=lambda r: abs(r['stamp_ns'] - stamp))
        delta = abs(robot['stamp_ns'] - stamp)*1e-9
        if delta > max_delta_s:
            continue
        pairs.append({'vision': observation, 'robot_stamp_ns': robot['stamp_ns'],
                      'pair_delta_s': delta, 'joints': robot['joints'],
                      'T_base_tool': robot['pose'], 'T_camera_board': board['pose']})
    return pairs
