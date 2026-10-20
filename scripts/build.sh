#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/env.bash"
cd "$WRIST_WS_ROOT"
colcon build --base-paths src --symlink-install --parallel-workers 2 \
  --packages-select X5A x5a_handeye x5a_moveit_config x5a_moveit_official_adapter \
  x5a_pick_place x5a_task_interfaces x5a_task_server x5a_vision x5a_web_agent x5a_wrist_driver \
  x5a_parallel_ik x5a_remote_grasp
