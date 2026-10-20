import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from launch import LaunchContext

from x5a_handeye.calibrated_tf import prefixed_robot_frame
from x5a_handeye.robot_model import export_candidate_urdf, model_signature


@pytest.mark.parametrize('prefix,expected', [('', 'tool0'), ('wrist_preview', 'wrist_preview/tool0'),
                                           ('wrist_preview/', 'wrist_preview/tool0')])
def test_prefix_preserves_real_frames_or_selects_preview_tree(prefix, expected):
    assert prefixed_robot_frame('tool0', prefix) == expected


@pytest.mark.parametrize('prefix', ['/preview/', 'preview//nested/', 'preview /', 'preview\n/'])
def test_prefix_rejects_ambiguous_frames(prefix):
    with pytest.raises(ValueError):
        prefixed_robot_frame('tool0', prefix)


def launch_fixture(tmp_path):
    src = Path(__file__).resolve().parents[2]
    xml = export_candidate_urdf((src/'x5a_moveit_config/config/X5A.urdf.xacro').read_text(),
                                np.deg2rad([0., 1., -1., 2., 1., 0.]))
    model = tmp_path/'candidate.urdf'
    model.write_text(xml)
    candidate = tmp_path/'candidate.yaml'
    result = dict(calibration_type='eye_in_hand', verdict='PASS',
                  validation_status='offline_model_candidate', T_tool_camera=np.eye(4).tolist(),
                  camera_frame='wrist_camera_color_optical_frame', camera_root_frame='wrist_camera_link',
                  robot_calibration={'candidate_urdf': str(model), 'expected_model': model_signature(xml)})
    candidate.write_text(yaml.safe_dump(result))
    path = src/'x5a_vision/launch/wrist_localization_preview.launch.py'
    spec = importlib.util.spec_from_file_location('wrist_preview_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update(candidate_result=str(candidate),
                                          vision_config=str(src/'x5a_vision/config/vision_wrist.yaml'))
    return module, context, model


def test_localization_preview_isolates_feedback_model_and_object_topics(tmp_path, monkeypatch):
    module, context, _ = launch_fixture(tmp_path)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    nodes = module.preview_nodes(context)
    executables = {node['executable'] for node in nodes}
    assert not executables & {'X5Controller', 'KeyBoard', 'official_trajectory_adapter', 'move_group'}
    by_exec = {node['executable']: node for node in nodes}
    rsp = by_exec['robot_state_publisher']
    assert rsp['parameters'][0]['frame_prefix'] == 'wrist_preview/'
    assert ('/robot_description', '/wrist_preview/robot_description') in rsp['remappings']
    assert ('/joint_states', '/wrist_preview/joint_states') in rsp['remappings']
    assert by_exec['readonly_joint_states']['parameters'][0]['joint_state_topic'] == '/wrist_preview/joint_states'
    tf = by_exec['publish_handeye_tf']['parameters'][0]
    assert tf['robot_frame_prefix'] == rsp['parameters'][0]['frame_prefix']
    assert tf['robot_description_topic'] == '/wrist_preview/robot_description'
    detector = by_exec['cube_detector']
    assert detector['parameters'][1]['base_frame'] == 'wrist_preview/base_link'
    for name in ('white_cube_pose', 'red_cube_pose', 'orange_cube_pose', 'debug_image', 'detection_stable'):
        assert (f'/x5a_vision/{name}', f'/wrist_preview/{name}') in detector['remappings']


def test_preview_refuses_model_file_that_does_not_match_extrinsic(tmp_path, monkeypatch):
    module, context, model = launch_fixture(tmp_path)
    xml = model.read_text().replace('xyz="0.11277 0 0"', 'xyz="0.2 0 0"')
    model.write_text(xml)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    with pytest.raises(ValueError, match='differs'):
        module.preview_nodes(context)


def test_failed_candidate_can_only_be_explicitly_selected_for_isolated_preview(tmp_path,monkeypatch):
    module,context,_=launch_fixture(tmp_path)
    path=Path(context.launch_configurations['candidate_result'])
    result=yaml.safe_load(path.read_text());result['verdict']='FAIL';path.write_text(yaml.safe_dump(result))
    monkeypatch.setattr(module,'Node',lambda **kwargs:kwargs)
    with pytest.raises(ValueError,match='failed validation'):module.preview_nodes(context)
    context.launch_configurations['allow_failed_candidate']='true'
    nodes=module.preview_nodes(context)
    tf=next(n for n in nodes if n['executable']=='publish_handeye_tf')
    assert tf['parameters'][0]['allow_failed_candidate'] is True
    detector=next(n for n in nodes if n['executable']=='cube_detector')
    assert 'FAIL' in detector['parameters'][1]['calibration_label']
