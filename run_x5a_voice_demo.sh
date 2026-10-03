#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${SCRIPT_DIR}"
ROS_SETUP="/opt/ros/humble/setup.bash"

HTTPS=false
DRY_RUN=false
PLAN_ONLY=false
TERMINAL=false
PORT=8000
while (( $# > 0 )); do
  case "$1" in
    --https) HTTPS=true ;;
    --dry-run) DRY_RUN=true ;;
    --plan-only) PLAN_ONLY=true ;;
    --terminal) TERMINAL=true ;;
    --port) shift; PORT="${1}" ;;
    --help|-h)
      cat <<'EOF'
Usage:
  ./run_x5a_voice_demo.sh
  ./run_x5a_voice_demo.sh --plan-only
  ./run_x5a_voice_demo.sh --dry-run
  ./run_x5a_voice_demo.sh --https --port 8000
  ./run_x5a_voice_demo.sh --terminal

Starts only:
  - x5a_task_server   (/x5a/pick_place)
  - x5a_web_agent     (phone page)

Does not start X5Controller, MoveIt, vision, or pick_place_node.
Bring those up first, or reuse an already running stack.
Never start this together with x5a_pick_place/pick_place_node.

--terminal adds a text input layer to this terminal: typed text goes through
the same parse/dispatch path as phone speech. /help lists its commands.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

discover_vendor_setup() {
  if [[ -n "${X5A_VENDOR_SETUP:-}" && -f "${X5A_VENDOR_SETUP}" ]]; then
    printf '%s\n' "${X5A_VENDOR_SETUP}"
    return 0
  fi
  local candidate
  for candidate in \
    "${HOME}/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash" \
    "${HOME}/arx/ARX_X5/ROS2/X5_ws/install/setup.bash"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
}

[[ -f "${ROS_SETUP}" ]] || { echo "ROS 2 Humble not found: ${ROS_SETUP}" >&2; exit 1; }
VENDOR_SETUP="$(discover_vendor_setup || true)"

set +u
source "${ROS_SETUP}"
if [[ -n "${VENDOR_SETUP}" ]]; then
  source "${VENDOR_SETUP}"
fi
if [[ -f "${WS_DIR}/install/setup.bash" ]]; then
  source "${WS_DIR}/install/setup.bash"
fi
set -u

if ! ros2 pkg prefix x5a_task_server >/dev/null 2>&1 || \
   ! ros2 pkg prefix x5a_web_agent >/dev/null 2>&1; then
  echo "Building voice packages..."
  (
    set +u
    source "${ROS_SETUP}"
    [[ -n "${VENDOR_SETUP}" ]] && source "${VENDOR_SETUP}"
    set -u
    cd "${WS_DIR}"
    PATH="/usr/bin:/bin:${PATH}" /usr/bin/colcon build --symlink-install \
      --packages-select x5a_task_interfaces x5a_pick_place x5a_task_server x5a_web_agent \
      --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
  )
  set +u
  source "${WS_DIR}/install/setup.bash"
  set -u
fi

if pgrep -f '/pick_place_node([[:space:]]|$)' >/dev/null 2>&1; then
  echo "ERROR: x5a_pick_place is already running." >&2
  echo "Stop the one-shot pick_place node before starting the voice task server." >&2
  exit 1
fi

if pgrep -f '/x5a_mtc_task_server([[:space:]]|$)' >/dev/null 2>&1; then
  echo "ERROR: experimental MTC task server is running. Stop it first." >&2
  exit 1
fi

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-127.0.0.1}"
SCHEME="http"
[[ "${HTTPS}" == "true" ]] && SCHEME="https"

echo "========================================"
echo " X5A VOICE DEMO"
echo "========================================"
echo "task_server: /x5a/pick_place"
echo "web:         ${SCHEME}://${HOST_IP}:${PORT}"
echo "plan_only:   ${PLAN_ONLY}"
echo "web dry-run: ${DRY_RUN}"
echo "terminal:    ${TERMINAL}"
echo "This script does not publish /arm_cmd."
echo

TASK_PID=""
WEB_PID=""
cleanup() {
  trap - EXIT INT TERM
  [[ -n "${WEB_PID}" ]] && kill -INT "${WEB_PID}" 2>/dev/null || true
  [[ -n "${TASK_PID}" ]] && kill -INT "${TASK_PID}" 2>/dev/null || true
  wait || true
}
trap cleanup EXIT INT TERM

ros2 launch x5a_task_server task_server.launch.py \
  plan_only:="${PLAN_ONLY}" vision_enabled:=true &
TASK_PID=$!

sleep 1
EXTRA=()
[[ "${DRY_RUN}" == "true" ]] && EXTRA+=(--dry-run)
[[ "${HTTPS}" == "true" ]] && EXTRA+=(--https)
[[ "${TERMINAL}" == "true" ]] && EXTRA+=(--terminal)
ros2 run x5a_web_agent web_agent --host 0.0.0.0 --port "${PORT}" "${EXTRA[@]}" &
WEB_PID=$!

echo "Phone browser:"
echo "  ${SCHEME}://${HOST_IP}:${PORT}"
if [[ "${TERMINAL}" == "true" ]]; then
  echo "Terminal input layer is on: type text here (same path as phone speech),"
  echo "  /help 看命令，Ctrl-D 或 /quit 只关掉输入层，Ctrl-C 停整栈。"
fi
echo
wait
