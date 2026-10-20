"""Exercise real launch construction; never execute actions or start hardware."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT/'config/calibration/handeye_model_candidate.yaml'
CONFIG = ROOT/'config/task_execute.yaml'


def launch_module(filename):
    path = Path(get_package_share_directory('x5a_pick_place'))/'launch'/filename
    spec = importlib.util.spec_from_file_location(filename.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(**values):
    result = LaunchContext()
    result.launch_configurations.update({name: str(value) for name, value in values.items()})
    return result


def test_active_stack_constructs_with_box_collision_disabled():
    settings = yaml.safe_load(CONFIG.read_text())['/**']['ros__parameters']
    assert not settings['box']['collision']['enabled']
    assert not settings['wrist']['scene_objects_collision_enabled']
    stack = launch_module('wrist_stack.launch.py')
    actions = stack.stack(context(workspace=ROOT, start_driver='true',
                                  start_adapter='true', start_camera='true'))
    # Construction must evaluate the task's OpaqueFunction work, not merely
    # list argument declarations (ros2 launch --show-args misses that failure).
    assert len(actions) == 6
    assert sum(isinstance(action, Node) for action in actions) == 5


def test_standalone_task_launch_uses_same_collision_optional_factory():
    task = launch_module('wrist_pick.launch.py')
    actions = task.nodes(context(candidate_result=CANDIDATE, task_config=CONFIG,
                                 experimental_model_accepted='true', plan_only='false'))
    assert len(actions) == 2 and all(isinstance(action, Node) for action in actions)


def test_bad_servo_configuration_fails_before_hardware_actions(tmp_path, monkeypatch):
    config_dir = tmp_path/'config'
    (config_dir/'calibration').mkdir(parents=True)
    (config_dir/'calibration/handeye_model_candidate.yaml').write_text(CANDIDATE.read_text())
    settings = yaml.safe_load(CONFIG.read_text())
    settings['/**']['ros__parameters']['execution']['require_servo_bounds'] = False
    (config_dir/'task_execute.yaml').write_text(yaml.safe_dump(settings))
    stack = launch_module('wrist_stack.launch.py')
    hardware_node = Mock(side_effect=AssertionError('hardware action constructed before validation'))
    monkeypatch.setattr(stack, 'Node', hardware_node)
    with pytest.raises(ValueError, match='active servo bounds checks'):
        stack.stack(context(workspace=tmp_path))
    hardware_node.assert_not_called()


def test_invalid_calibration_fails_before_hardware_actions(tmp_path, monkeypatch):
    config_dir = tmp_path/'config/calibration'
    config_dir.mkdir(parents=True)
    (config_dir/'handeye_model_candidate.yaml').write_text('{}')
    stack = launch_module('wrist_stack.launch.py')
    hardware_node = Mock(side_effect=AssertionError('hardware action constructed before validation'))
    monkeypatch.setattr(stack, 'Node', hardware_node)
    with pytest.raises((ValueError, KeyError)):
        stack.stack(context(workspace=tmp_path))
    hardware_node.assert_not_called()


@pytest.mark.parametrize('solver', ['kdl', 'trac_ik'])
def test_selected_ik_and_planning_pipelines_reach_move_group_parameters(monkeypatch, solver):
    import x5a_pick_place.wrist_launch as factory
    from ament_index_python.packages import get_package_share_directory as original_share
    monkeypatch.setattr(factory, 'Node', lambda **kwargs: kwargs)
    monkeypatch.setattr(factory, 'get_package_share_directory',
                        lambda name: '/tmp/optional-plugin' if name == 'trac_ik_kinematics_plugin' else original_share(name))
    nodes = factory.build_task_nodes(CANDIDATE, CONFIG, accepted=True,
        include_task_server=False, ik_solver=solver,
        planning_pipelines=['ompl', 'pilz_industrial_motion_planner', 'chomp'])
    params = nodes[0]['parameters'][0]
    plugin = params['robot_description_kinematics']['arm']
    assert plugin['kinematics_solver'] == ({
        'kdl': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
        'trac_ik': 'trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin'}[solver])
    assert params['robot_description_planning']['joint_limits']['joint4']['min_position'] == -1.22
    assert params['pilz_industrial_motion_planner']['planning_plugin'].endswith('/CommandPlanner')
    assert params['chomp']['planning_plugin'] == 'chomp_interface/CHOMPPlanner'
    if solver == 'trac_ik':
        assert plugin['epsilon'] == 1.e-5 and plugin['solve_type'] == 'Speed'
        assert not plugin['position_only_ik']
