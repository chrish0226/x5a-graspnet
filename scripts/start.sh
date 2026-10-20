#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/env.bash"
if [[ ! -f "$WRIST_WS_ROOT/install/local_setup.bash" ]]; then
  echo "请先执行 $WRIST_WS_ROOT/scripts/build.sh" >&2
  exit 2
fi
case "${1-}" in
  --dry-run)
    exec ros2 run x5a_web_agent wrist_web_agent --dry-run --terminal ;;
  --help|-h)
    echo 'start.sh：启动腕部颜色抓放，等待“准备”和抓取指令'
    echo 'start.sh --dry-run：只测试手机/文字/语音交互，不启动机械臂或相机'
    exit 0 ;;
  '') ;;
  *) echo '未知参数，请用 --help' >&2; exit 2 ;;
esac
exec 9>"/tmp/x5a-wrist-stack-$UID.lock"
flock -n 9 || { echo '腕部工作区已启动，请使用已有页面' >&2; exit 2; }
wrist_runtime="$(/usr/bin/python3 "$WRIST_WS_ROOT/scripts/preflight.py")"
read -r wrist_driver wrist_adapter wrist_camera <<< "$wrist_runtime"
echo '启动后先说“准备”，到观察位后再说“抓红色”。'
echo '网页：http://192.168.0.50:8001  语音：https://192.168.0.50:8444'
echo '原手机 App：http://192.168.0.50:8000 / https://192.168.0.50:8443'
exec ros2 launch x5a_pick_place wrist_stack.launch.py \
  workspace:="$WRIST_WS_ROOT" start_driver:="$wrist_driver" \
  start_adapter:="$wrist_adapter" start_camera:="$wrist_camera"
