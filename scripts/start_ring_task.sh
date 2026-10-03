#!/usr/bin/env bash
# One-shot bringup for the ring pick-and-place task:
#   1. arm driver  (vendor open_single_arm.launch.py)
#   2. MoveIt + official adapter (x5a_real_moveit.launch.py, driver disabled)
#   3. ring_pick_place.py task in the foreground
# Ctrl-C shuts everything down.  Camera must be free (no other preview/task).
#
# Usage:
#   start_ring_task.sh [cycles]          # default 2
#   start_ring_task.sh 5
# NOTE: no "set -u" here — ROS setup.bash scripts break under nounset.

log() { echo -e "\033[32m[start_ring_task]\033[0m $*"; }

CYCLES="${1:-2}"
PLACE_GOOD="${PLACE_GOOD:-$HOME/arx/arm/logs/poses/place_good.yaml}"
PLACE_BAD="${PLACE_BAD:-$HOME/arx/arm/logs/poses/place_bad.yaml}"
PY=/usr/bin/python3
TASK_SCRIPT="$HOME/arx/arm/scripts/ring_pick_place.py"
PIDS=()

# node process patterns this script owns; leftovers from a previous run
# must die or the adapter's "only one /arm_cmd publisher" check fails
NODE_PATTERNS=(
  "x5a_official_trajectory_adapter"
  "ros2 launch x5a_moveit_config"
  "ros2 launch arx_x5_controller"
  "move_group"
  "X5Controller"
  "robot_state_publisher"
  "static_transform_publisher"
  "official_trajectory_adapter"
)

kill_nodes() {
  local sig="$1"
  for pat in "${NODE_PATTERNS[@]}"; do
    pkill "$sig" -f "$pat" 2>/dev/null
    # fallback: kill each matched PID individually
    for pid in $(pgrep -f "$pat" 2>/dev/null); do
      kill "$sig" "$pid" 2>/dev/null
    done
  done
}

cleanup() {
  log "shutting down all launches and nodes..."
  for pid in ${PIDS[@]:-}; do kill "$pid" 2>/dev/null; done
  sleep 2
  kill_nodes -TERM
  sleep 1
  kill_nodes -KILL
  exit 0
}
trap cleanup INT TERM

# ---- preflight ----
for f in "$PLACE_GOOD" "$PLACE_BAD" "$TASK_SCRIPT"; do
  if [ ! -f "$f" ]; then log "ERROR: missing $f"; exit 1; fi
done
if pgrep -af "usb_circle_detect|ring_pick_place" | grep -v grep >/dev/null 2>&1; then
  log "ERROR: another detection/task process holds the camera; close it first"
  exit 1
fi
# auto-clean orphaned nodes from a previous run (kill -9 on the launch
# process leaves its children alive)
if pgrep -f "x5a_official_trajectory_adapter|X5Controller|move_group " >/dev/null 2>&1; then
  log "killing leftover nodes from a previous run..."
  kill_nodes -TERM
  sleep 2
  kill_nodes -KILL
  sleep 1
fi

# ---- sources (vendor ws provides arx5_arm_msg + arx_x5_controller) ----
source /opt/ros/humble/setup.bash
source "$HOME/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash"
source "$HOME/arx/arm/install/setup.bash"

# ---- 1. arm driver ----
log "starting arm driver (open_single_arm.launch.py)..."
ros2 launch arx_x5_controller open_single_arm.launch.py &
PIDS+=($!)

log "waiting for /arm_status data..."
ok=0
for i in $(seq 1 30); do
  if timeout 2 ros2 topic echo /arm_status --once >/dev/null 2>&1; then ok=1; break; fi
  sleep 1
done
if [ "$ok" != 1 ]; then
  log "ERROR: /arm_status has no data after 30s (arm powered? CAN cable?)"
  cleanup
fi
log "driver online"

# ---- 2. MoveIt + official adapter ----
log "starting MoveIt + official adapter..."
ros2 launch x5a_moveit_config x5a_real_moveit.launch.py \
  start_driver:=false start_adapter:=true use_rviz:=false arm_can_id:=can1 &
PIDS+=($!)

log "waiting for /move_action..."
ok=0
for i in $(seq 1 30); do
  if timeout 2 ros2 action list 2>/dev/null | grep -qx "/move_action"; then ok=1; break; fi
  sleep 1
done
if [ "$ok" != 1 ]; then
  log "ERROR: move_group did not come up after 30s"
  cleanup
fi
# give the adapter time to receive stable joint state (it rejects
# execution for 2s after the first status packet)
log "waiting for /joint_states data..."
ok=0
for i in $(seq 1 15); do
  if timeout 2 ros2 topic echo /joint_states --once >/dev/null 2>&1; then ok=1; break; fi
  sleep 1
done
if [ "$ok" != 1 ]; then
  log "ERROR: /joint_states has no data after 15s"
  cleanup
fi
sleep 3
log "MoveIt online"

# ---- 3. the task (foreground) ----
log "starting ring pick-place task (cycles=$CYCLES)"
$PY -u "$TASK_SCRIPT" \
  --cycles "$CYCLES" \
  --place-good "$PLACE_GOOD" \
  --place-bad "$PLACE_BAD"
RC=$?

log "task exited rc=$RC"
cleanup
