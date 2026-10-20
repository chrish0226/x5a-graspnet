#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/env.bash"
wrist_sam_execute=true
wrist_sam_verbose=false
for wrist_sam_option in "$@"; do
case "$wrist_sam_option" in
  --plan-only) wrist_sam_execute=false ;;
  --verbose) wrist_sam_verbose=true ;;
  --help|-h)
    echo 'start_sam.sh：腕部相机 → SAM + GraspNet → MoveIt，等待手机/文字抓放指令'
    echo 'start_sam.sh --plan-only：推理和规划；准备、抓取和松爪均不执行'
    echo 'start_sam.sh --verbose：同时在终端显示完整底层规划日志'
    exit 0 ;;
  '') ;;
  *) echo '未知参数，请用 --help' >&2; exit 2 ;;
esac
done
if [[ ! -f "$WRIST_WS_ROOT/install/x5a_remote_grasp/share/x5a_remote_grasp/package.xml" ]]; then
  echo "请先执行 $WRIST_WS_ROOT/scripts/build.sh" >&2; exit 2
fi
if [[ -f /home/quella/.config/x5a_remote_grasp/key.env ]]; then
  source /home/quella/.config/x5a_remote_grasp/key.env
fi
if [[ -z "${SAM_GRASPNET_API_KEY:-}" ]]; then
  echo '未配置原 SAM/GraspNet 服务密钥 SAM_GRASPNET_API_KEY' >&2; exit 2
fi
export SAM_GRASPNET_API_KEY
exec 9>"/tmp/x5a-wrist-stack-$UID.lock"
flock -n 9 || { echo '腕部抓取服务正在运行，请先在其启动终端结束任务并停止服务' >&2; exit 2; }
wrist_runtime="$(/usr/bin/python3 "$WRIST_WS_ROOT/scripts/preflight.py")"
read -r wrist_driver wrist_adapter wrist_camera <<< "$wrist_runtime"
echo 'SAM 腕部模式：手机 http://192.168.0.50:8000 / https://192.168.0.50:8443'
echo '先说“准备”，再下达抓取或连续抓放指令。'
echo '例如：把白色方块放到橙色方块上，然后把红色方块放到白色方块上'
echo '终端显示步骤、阶段和耗时；底层详细日志保存在 ROS 日志目录（--verbose 可展开）。'
exec ros2 launch x5a_pick_place wrist_sam.launch.py workspace:="$WRIST_WS_ROOT" \
  verbose_logs:="$wrist_sam_verbose" \
  execution_enabled:="$wrist_sam_execute" start_driver:="$wrist_driver" \
  start_adapter:="$wrist_adapter" start_camera:="$wrist_camera"
