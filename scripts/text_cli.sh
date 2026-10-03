#!/bin/bash
# 终端文字输入层：接入已经在跑的 web_agent，用键盘打字代替语音（不用麦克风/手机）
# 前置：web_agent 已在运行（scripts/stack_start.sh 或 run_x5a_voice_demo.sh），
#       脚本只读 /api 接口，不重启、不打断正在跑的服务。
#   用法:  ./scripts/text_cli.sh                          # 默认 http://127.0.0.1:8000
#          ./scripts/text_cli.sh --url http://192.168.0.50:8000
#          ./scripts/text_cli.sh --help                   # 命令列表见层内 /help
# 注意：连的是真 agent，打进去的指令会真的动机械臂；只想看解析用层里的 /parse。
# （不用 set -u：ROS 的 setup.bash 在 nounset 下会报未绑定变量，照 controller.sh 的老写法）
source /home/quella/arx/arm/scripts/stack_env.sh

if ! ros2 pkg prefix x5a_web_agent >/dev/null 2>&1; then
  echo "找不到 x5a_web_agent：先 cd ~/arx/arm && colcon build --symlink-install --packages-select x5a_web_agent" >&2
  exit 1
fi
exec ros2 run x5a_web_agent text_cli "$@"
