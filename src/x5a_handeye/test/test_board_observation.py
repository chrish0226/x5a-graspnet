import json
from types import SimpleNamespace as NS
import cv2
import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('arx5_arm_msg')
from cv_bridge import CvBridge
from x5a_handeye.board_detector import BoardDetector
from x5a_handeye.readonly_handeye_sampler import ReadOnlyHandeyeSampler
from x5a_handeye.observations import pair_observations


def test_detector_publishes_pose_and_replayable_observation_with_same_image_time():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    board = (cv2.aruco.CharucoBoard((5, 7), .030, .022, dictionary)
             if hasattr(cv2.aruco, 'CharucoBoard') else cv2.aruco.CharucoBoard_create(5, 7, .030, .022, dictionary))
    pattern = board.generateImage((300, 420)) if hasattr(board, 'generateImage') else board.draw((300, 420))
    gray = np.full((480, 640), 255, np.uint8)
    gray[30:450, 170:470] = pattern
    detector = object.__new__(BoardDetector)
    detector.dictionary, detector.board = dictionary, board
    detector.params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'DetectorParameters') else cv2.aruco.DetectorParameters_create()
    detector.K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
    detector.D = np.zeros(5); detector.camera_frame = 'wrist_camera_color_optical_frame'
    detector.info_size = (640, 480); detector.min_corners = 12
    detector.max_reprojection = 1.; detector.max_tilt = 60.; detector.publish_tf = False
    detector.cfg = {'board': {'squares_x': 5, 'squares_y': 7, 'square_length_m': .030,
                              'marker_length_m': .022, 'dictionary': 'DICT_5X5_50'}}
    detector.bridge = CvBridge()
    poses, observations, debug = [], [], []
    detector.pose_pub = NS(publish=poses.append)
    detector.observation_pub = NS(publish=observations.append)
    detector.debug_pub = NS(publish=debug.append)
    image = detector.bridge.cv2_to_imgmsg(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'bgr8')
    image.header.frame_id = detector.camera_frame; image.header.stamp.sec = 10
    detector.on_image(image)
    assert len(poses) == len(observations) == 1
    observation = json.loads(observations[0].data)
    assert observation['quality']['n_corners'] == 24
    assert observation['stamp_ns'] == 10_000_000_000
    assert observation['board']['marker_length_m'] == .022
    assert len(observation['marker_corners_px']) == 17
    assert len(observation['charuco_corners_px']) == 24
    assert poses[0].header == debug[0].header == image.header
    sampler = NS(observations={}, max_board_age=.5,
                 get_clock=lambda: NS(now=lambda: NS(nanoseconds=10_000_000_000)))
    ReadOnlyHandeyeSampler.on_observation(sampler, observations[0])
    assert sampler.observations[observation['stamp_ns']] == observation
    # The image observation pairs with robot feedback, independently of
    # callback arrival order. No ROS node or robot command client is created.
    boards = [{'stamp_ns': observation['stamp_ns'], 'frame_id': observation['camera_frame'], 'pose': {}}]
    robots = [{'stamp_ns': observation['stamp_ns']+1_000_000, 'pose': {}, 'joints': [0]*6}]
    pairs = pair_observations(boards, robots, sampler.observations, .04)
    assert len(pairs) == 1 and pairs[0]['vision']['quality']['reprojection_rms_px'] < 1.
