#!/usr/bin/env bash
# Standard eye-to-hand pipeline (OpenCV calibrateHandEye / easy_handeye style)
#
# Setup (once, hardware):
#   1) Fix RealSense so it sees the workspace (camera NOT on arm)
#   2) Clamp ChArUco board rigidly on the gripper (no slip)
#   3) Printed square size == 20.0 mm (board.yaml)
#
# Usage:
#   bash standard_handeye_pipeline.sh start_prereq
#   bash standard_handeye_pipeline.sh sample       # read-only manual sampling UI
#   bash standard_handeye_pipeline.sh solve
#   bash standard_handeye_pipeline.sh publish_tf
#
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="${WS:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
SRC="$WS/src/x5a_handeye"
DATA="$SRC/data"
CFG="$SRC/config"
SAMPLES="${SAMPLES:-$DATA/manual_samples_readonly.json}"
RESULT="${RESULT:-$CFG/handeye_result.yaml}"
BOARD_CFG="${BOARD_CFG:-$CFG/board.yaml}"

# ROS setup.bash uses unbound vars; must disable nounset around source.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u

export LD_LIBRARY_PATH="${WS}/install/arx_x5_controller/lib:/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SRC}:${PYTHONPATH:-}"

cmd="${1:-help}"

start_prereq() {
  mkdir -p "$DATA"
  if ! pgrep -x X5Controller >/dev/null; then
    echo "ERROR: official X5Controller is not running."
    echo "First run: bash ~/arx/ARX_X5/00-sh/ROS2/04single_arm.sh"
    return 2
  fi
  if pgrep -f 'x5a_control_bridge|control_bridge' >/dev/null; then
    echo "ERROR: stop x5a_control_bridge before read-only calibration."
    return 3
  fi
  if ! pgrep -f realsense2_camera_node >/dev/null; then
    echo "[prereq] start RealSense"
    nohup ros2 run realsense2_camera realsense2_camera_node --ros-args \
      -r __ns:=/camera -r __node:=camera \
      -p serial_no:=_342522073696 \
      -p enable_color:=true -p enable_depth:=false \
      -p enable_infra1:=false -p enable_infra2:=false \
      -p enable_gyro:=false -p enable_accel:=false \
      -p rgb_camera.color_profile:=640x480x15 \
      >"$DATA/realsense.log" 2>&1 &
    echo $! >"$DATA/realsense.pid"
    sleep 3
  fi
  if ! pgrep -f board_detector >/dev/null; then
    echo "[prereq] start board_detector"
    nohup ros2 run x5a_handeye board_detector --ros-args \
      -p board_config:="$BOARD_CFG" \
      -p image_topic:=/camera/camera/color/image_raw \
      -p camera_info_topic:=/camera/camera/color/camera_info \
      >"$DATA/board.log" 2>&1 &
    sleep 1
  fi
  echo "[prereq] ready"
  echo "  next: bash $0 sample"
}

gravity() {
  echo "ERROR: gravity command helper is disabled for hardware safety."
  echo "Set gravity mode once through the official controller/rqt before sampling."
  return 2
}

sample() {
  echo "=== Read-only manual sampling ==="
  echo "Saving samples to: $SAMPLES"
  echo "Keys: s=sample  q=quit"
  echo "IMPORTANT: board must be rigidly clamped; stay still before s"
  SAMPLES="$SAMPLES" exec "$SRC/scripts/readonly_handeye_sample.sh"
}

solve() {
  echo "=== OpenCV standard eye-to-hand solve ==="
  # /usr/bin/python3 has cv2; conda base python3 does not
  /usr/bin/python3 -m x5a_handeye.solve_handeye \
    --samples "$SAMPLES" \
    --output "$RESULT" \
    --max-reject 5 \
    --holdout 0 \
    --pass-mm 10
  echo "result: $RESULT"
}

publish_tf() {
  echo "=== publish static TF from $RESULT ==="
  ros2 run x5a_handeye publish_handeye_tf --ros-args \
    -p result_yaml:="$RESULT"
}

status() {
  echo "SAMPLES=$SAMPLES"
  if [[ -f "$SAMPLES" ]]; then
    python3 - <<PY
import json
from pathlib import Path
p=Path(r"$SAMPLES")
d=json.loads(p.read_text())
ss=d.get("samples",[])
print(f"samples: {len(ss)}")
PY
  else
    echo "no samples file yet"
  fi
  if [[ -f "$RESULT" ]]; then
    python3 - <<PY
import yaml
from pathlib import Path
r=yaml.safe_load(Path(r"$RESULT").read_text())
print("verdict:", r.get("verdict"))
e=r.get("error") or {}
print("train mean mm:", (e.get("translation_mean_m") or 0)*1000)
PY
  fi
  echo "--- processes ---"
  pgrep -af 'X5Controller|bridge_node|board_detector|gravity|realsense2_camera_node' || true
}

case "$cmd" in
  start_prereq|prereq) start_prereq ;;
  gravity|g) gravity ;;
  sample) sample ;;
  solve) solve ;;
  publish_tf|tf) publish_tf ;;
  status) status ;;
  help|*)
    cat <<EOF
Usage: $0 <command>

  start_prereq   Start CAN/driver/bridge/camera/board_detector
  gravity        Disabled: use official controller/rqt
  sample         Read-only manual sampler (no arm commands)
  solve          OpenCV multi-solver eye-to-hand
  publish_tf     Publish base_link->camera_link static TF
  status         Show sample count / processes

Typical:
  $0 start_prereq
  # Put the arm in gravity mode using the official controller/rqt first.
  $0 sample           # drag, stop, press s (15~25 poses)
  $0 solve
EOF
    ;;
esac
