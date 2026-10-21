#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.bash"
set -u
case "${1-}" in
  --check-only) wrist_check_only=1 ;;
  '') wrist_check_only=0 ;;
  *) echo '用法：start_driver.sh [--check-only]；检查模式不会启动硬件' >&2; exit 2 ;;
esac
if [[ ! -x "$WRIST_WS_ROOT/install/x5a_wrist_driver/lib/x5a_wrist_driver/wrist_controller" ]]; then
  echo '请先运行腕部工作区 scripts/build.sh' >&2
  exit 2
fi
"$WRIST_WS_ROOT/install/x5a_wrist_driver/lib/x5a_wrist_driver/wrist_controller" --check-sdk
# Read-only duplicate check. This script does NOT kill an existing driver.
if pgrep -x X5Controller >/dev/null || pgrep -x wrist_controlle >/dev/null; then
  echo '已有机械臂驱动运行。请先放稳机械臂，再关闭旧驱动终端；禁止同时启动两个驱动。' >&2
  exit 2
fi
if [[ "$wrist_check_only" == 1 ]]; then
  echo 'SDK校验通过，未发现重复驱动；检查模式未启动硬件'
  exit 0
fi
exec 8>"/tmp/x5a-wrist-driver-$UID.lock"
flock -n 8 || { echo '腕部驱动已启动' >&2; exit 2; }
exec ros2 run x5a_wrist_driver wrist_controller --ros-args \
  --params-file "$WRIST_WS_ROOT/config/driver.yaml"
