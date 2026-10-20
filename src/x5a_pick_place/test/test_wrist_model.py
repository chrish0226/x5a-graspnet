from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from x5a_handeye.robot_model import export_candidate_urdf, model_signature, nominal_signature
from x5a_pick_place.wrist_model import prefixed_models, PREFIX


def test_private_model_preserves_calibration_raw_joint_names_limits_and_mass():
    source = Path(__file__).resolve().parents[2]/'x5a_moveit_config/config/X5A.urdf.xacro'
    urdf = export_candidate_urdf(source.read_text(), np.deg2rad([0., 1., -0.5, 2., 0.7, 0.]))
    expected = model_signature(urdf)
    srdf = (source.parent/'X5A.srdf').read_text()
    xml, semantic = prefixed_models(urdf, srdf, expected, np.eye(4))
    actual = model_signature(xml)
    for joint in expected:
        assert actual[joint]['parent'] == PREFIX + expected[joint]['parent']
        assert actual[joint]['child'] == PREFIX + expected[joint]['child']
        np.testing.assert_array_equal(actual[joint]['rotation'], expected[joint]['rotation'])
    original, clone = ET.fromstring(urdf), ET.fromstring(xml)
    for i in range(1, 7):
        assert clone.find(f"joint[@name='joint{i}']/limit").attrib == original.find(f"joint[@name='joint{i}']/limit").attrib
        assert clone.find(f"link[@name='{PREFIX}link{i}']/inertial/mass").attrib == original.find(f"link[@name='link{i}']/inertial/mass").attrib
    assert clone.find(f"link[@name='{PREFIX}camera_payload']/collision") is not None
    assert clone.find(f"link[@name='{PREFIX}camera_payload']/inertial") is None
    assert ET.fromstring(semantic).find("group[@name='arm']/chain").attrib == {'base_link': PREFIX+'base_link', 'tip_link': PREFIX+'tool0'}


def test_wrong_candidate_cannot_start_private_planner():
    source = Path(__file__).resolve().parents[2]/'x5a_moveit_config/config/X5A.urdf.xacro'
    changed = export_candidate_urdf(source.read_text(), np.deg2rad([0., 1., 0., 0., 0., 0.]))
    with pytest.raises(ValueError):
        prefixed_models(changed, '<robot/>', nominal_signature(), np.eye(4))
