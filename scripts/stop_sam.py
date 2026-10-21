#!/usr/bin/env python3
"""Gracefully stop this workspace's SAM launch and its managed replacements."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


ROOT = Path(__file__).resolve().parents[1]


def command(pid, proc=Path('/proc')):
    try:
        if (proc/str(pid)).stat().st_uid != os.getuid():
            return []
        return (proc/str(pid)/'cmdline').read_bytes().decode().strip('\0').split('\0')
    except (OSError, UnicodeError):
        return []


def launch_command(argv, root=ROOT):
    for index in range(1, len(argv)-2):
        if (Path(argv[index-1]).name == 'ros2'
                and argv[index:index+3] == ['launch', 'x5a_pick_place', 'wrist_sam.launch.py']):
            workspaces = [arg.split(':=', 1)[1] for arg in argv[index+3:]
                          if arg.startswith('workspace:=')]
            return len(workspaces) == 1 and Path(workspaces[0]).resolve() == root.resolve()
    return False


def identity(pid, proc=Path('/proc')):
    try:
        fields = (proc/str(pid)/'stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def launch_pids(proc=Path('/proc'), root=ROOT):
    return [int(entry.name) for entry in proc.iterdir()
            if entry.name.isdigit() and launch_command(command(int(entry.name), proc), root)]


def managed_component(argv, root=ROOT):
    executables = [root/'install/x5a_remote_grasp/lib/x5a_remote_grasp'/name
                   for name in ('remote_grasp_node', 'remote_grasp_executor')]
    executables.append(root/'install/x5a_parallel_ik/lib/x5a_parallel_ik/parallel_ik_server')
    executables.append(root/'install/x5a_web_agent/lib/x5a_web_agent/sam_web_agent')
    executables.append(root/'install/x5a_moveit_official_adapter/lib/x5a_moveit_official_adapter/official_trajectory_adapter')
    return (any(str(path) in argv for path in executables)
            or ('/opt/ros/humble/lib/moveit_ros_move_group/move_group' in argv
                and '__ns:=/wrist_pick' in argv))


def stop_managed_replacements(owners):
    try:
        report = json.loads((ROOT/'reports/sam-integration/sam_active_processes.json').read_text())
    except (OSError, ValueError):
        return
    if report.get('owner_launch_pid') not in owners:
        return
    supervisor = report.get('supervisor_pid')
    argv = command(supervisor)
    if (not argv or not any(arg.startswith('/tmp/wrist_sam_reload_') and arg.endswith('.py')
                            for arg in argv)):
        return
    # Verify every still-running child belongs to that supervisor and workspace.
    for child in report.get('child_pids', []):
        if identity(child) is None:
            continue
        child_args = command(child)
        try:
            parent = int(Path(f'/proc/{child}/stat').read_text().split(') ', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return
        if parent != supervisor or not managed_component(child_args):
            return
    try:
        os.kill(supervisor, signal.SIGINT)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='只显示将停止的启动进程，不发送信号')
    args = parser.parse_args()
    pids = launch_pids()
    if not pids:
        print('没有找到本工作区运行中的 SAM 程序。')
        return 0
    print('SAM 启动进程：'+', '.join(map(str, pids)), flush=True)
    if args.check:
        return 0
    identities = {pid: identity(pid) for pid in pids}
    for pid in pids:
        if (identities[pid] is not None and identity(pid) == identities[pid]
                and launch_command(command(pid))):
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic()+20.
    while any(identity(pid) == original for pid, original in identities.items()
              if original is not None):
        if time.monotonic() >= deadline:
            print('SAM 仍在退出，请查看启动终端；未发送强制终止信号。')
            return 1
        time.sleep(.2)
    stop_managed_replacements(pids)
    print('已结束 SAM 启动程序，并通知托管的规划/执行服务退出。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
