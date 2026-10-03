#!/bin/bash
# 规则抓取小方块链一键启动（HSV 色块：红/白/橙 -> x5a_task_server /x5a/pick_place）
# 与 stack_start.sh 的区别：不起 remote_grasp/executor，起 task_server，
# 且 web_agent 以 X5A_GRASP_ROUTE=rule 运行（语音/文字意图走规则色块抓取）。
# 前置：厂商控制器请先用 scripts/controller.sh 或原命令在前台启动。
# 重复执行安全：已启动的服务自动跳过。
#   用法:  ./scripts/stack_start_rule.sh
#   验证:  ./scripts/stack_status.sh   手机访问 http://<本机IP>:8000
set -u
BASE=/home/quella/arx/arm
LOGDIR=$BASE/logs/stack
ENV=$BASE/scripts/stack_env.sh
export X5A_GRASP_ROUTE=rule
mkdir -p "$LOGDIR"

launch() {  # launch <名称> <识别模式> <启动后等待秒> <命令...>
  local name=$1 pattern=$2 delay=$3
  shift 3
  if pgrep -u "$(whoami)" -f "$pattern" >/dev/null 2>&1; then
    echo "[跳过] $name 已在运行"
    return 0
  fi
  echo "[启动] $name"
  nohup bash -c "source '$ENV'; exec $*" >> "$LOGDIR/$name.log" 2>&1 &
  echo $! > "$LOGDIR/$name.pid"
  sleep "$delay"
}

echo "=== 1/4 MoveIt + official adapter（前置：控制器已前台启动）==="
launch moveit "x5a_real_moveit.launch.py" 4 \
  "ros2 launch x5a_moveit_config x5a_real_moveit.launch.py start_driver:=false start_adapter:=true use_rviz:=false arm_can_id:=can1"

echo "=== 2/4 RealSense + 手眼 TF + HSV 视觉 ==="
launch vision "vision.launch.py" 3 \
  ros2 launch x5a_vision vision.launch.py start_camera:=true

echo "=== 3/4 规则抓取任务服务 (/x5a/pick_place) ==="
launch task_server "task_server_node" 2 \
  "ros2 launch x5a_task_server task_server.launch.py plan_only:=false vision_enabled:=true"

echo "=== 4/4 网页/语音后端 (端口 8000, 规则路由) ==="
launch web_agent "web_agent --host" 2 \
  ros2 run x5a_web_agent web_agent --host 0.0.0.0 --port 8000

echo
echo "=== 就绪检查 ==="
sleep 3
if curl -s --max-time 3 http://127.0.0.1:8000/api/health | grep -q '"ok":true'; then
  echo "web_agent: OK (http://<本机IP>:8000)  抓取路由: rule (HSV 色块)"
else
  echo "web_agent: 未就绪（可能还在加载，10 秒后再试 scripts/stack_status.sh）"
fi
echo "说「抓红色的方块」即可触发规则抓取。日志目录: $LOGDIR"
echo "RViz: ./scripts/rviz.sh   全部停止: ./scripts/stack_stop.sh"
