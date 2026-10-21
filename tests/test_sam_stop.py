"""Exercise shutdown only against an isolated fake ros2 process."""
import importlib.util
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('stop_sam', ROOT/'scripts/stop_sam.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_launch_matching_excludes_other_workspaces_and_shell_strings(tmp_path):
    stop = module()
    argv = ['/usr/bin/python3', '/opt/ros/humble/bin/ros2', 'launch',
            'x5a_pick_place', 'wrist_sam.launch.py', f'workspace:={ROOT}']
    assert stop.launch_command(argv)
    assert not stop.launch_command(argv[:-1]+[f'workspace:={tmp_path}'])
    assert not stop.launch_command(['/bin/bash', '-c', ' '.join(argv)])
    assert not stop.launch_command(['python3', '-c', 'ros2 launch x5a_pick_place wrist_sam.launch.py'])


def test_check_mode_never_sends_signals(monkeypatch):
    stop = module()
    monkeypatch.setattr(sys, 'argv', ['stop_sam.py', '--check'])
    monkeypatch.setattr(stop, 'launch_pids', lambda: [123])
    monkeypatch.setattr(stop.os, 'kill', lambda *args: pytest.fail('check mode sent signal'))
    assert stop.main() == 0


def test_shutdown_reaches_fake_launch_via_sigint_and_waits_for_exit(tmp_path, monkeypatch):
    stop = module()
    executable = tmp_path/'ros2'
    executable.write_text('import signal,sys,time\n'
        'signal.signal(signal.SIGINT, lambda *args: sys.exit(0))\n'
        'print("ready",flush=True)\ntime.sleep(60)\n')
    child = subprocess.Popen([sys.executable, str(executable), 'launch', 'x5a_pick_place',
        'wrist_sam.launch.py', f'workspace:={ROOT}'], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        assert stop.launch_command(stop.command(child.pid))
        monkeypatch.setattr(sys, 'argv', ['stop_sam.py'])
        monkeypatch.setattr(stop, 'launch_pids', lambda: [child.pid])
        notified = []
        monkeypatch.setattr(stop, 'stop_managed_replacements', lambda pids: notified.extend(pids))
        assert stop.main() == 0
        assert child.wait(timeout=3) == 0
        assert notified == [child.pid]
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            child.wait(timeout=3)


def test_pid_reuse_does_not_signal_replacement(monkeypatch):
    stop = module()
    monkeypatch.setattr(sys, 'argv', ['stop_sam.py'])
    monkeypatch.setattr(stop, 'launch_pids', lambda: [123])
    values = iter(['original', 'replacement', 'replacement'])
    monkeypatch.setattr(stop, 'identity', lambda _: next(values))
    monkeypatch.setattr(stop.os, 'kill', lambda *args: pytest.fail('reused PID was signalled'))
    monkeypatch.setattr(stop, 'stop_managed_replacements', lambda _: None)
    assert stop.main() == 0


def test_managed_moveit_is_limited_to_wrist_namespace():
    stop=module()
    binary='/opt/ros/humble/lib/moveit_ros_move_group/move_group'
    assert stop.managed_component([binary,'--ros-args','-r','__ns:=/wrist_pick'])
    assert not stop.managed_component([binary,'--ros-args','-r','__ns:=/other_arm'])
    assert not stop.managed_component(['/tmp/foreign_move_group','__ns:=/wrist_pick'])


def test_managed_adapter_is_limited_to_this_workspace():
    stop = module()
    executable = ROOT/'install/x5a_moveit_official_adapter/lib/x5a_moveit_official_adapter/official_trajectory_adapter'
    assert stop.managed_component([str(executable)])
    assert not stop.managed_component(['/tmp/other_workspace/official_trajectory_adapter'])


def test_managed_sam_web_replacement_belongs_to_this_workspace():
    stop = module()
    assert stop.managed_component([str(stop.ROOT/'install/x5a_web_agent/lib/x5a_web_agent/sam_web_agent')])
    assert not stop.managed_component(['/tmp/other_project/sam_web_agent'])
