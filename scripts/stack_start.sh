#!/bin/bash
# 一键启动除厂商控制器外的全链路（后台运行，日志见 ~/arx/arm/logs/stack/）
# 前置：厂商控制器请先用 scripts/controller.sh 或你的原命令在前台启动。
# 重复执行安全：已启动的服务自动跳过。
#   用法:  ./scripts/stack_start.sh
#   RViz 另开:  ./scripts/rviz.sh
set -u
BASE=/home/quella/arx/arm
LOGDIR=$BASE/logs/stack
ENV=$BASE/scripts/stack_env.sh
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

echo "=== 1/5 MoveIt + official adapter（前置：控制器已前台启动）==="
launch moveit "x5a_real_moveit.launch.py" 4 \
  "ros2 launch x5a_moveit_config x5a_real_moveit.launch.py start_driver:=false start_adapter:=true use_rviz:=false arm_can_id:=can1"

echo "=== 2/5 RealSense + 手眼 TF + 视觉 ==="
launch vision "vision.launch.py" 3 \
  ros2 launch x5a_vision vision.launch.py start_camera:=true

echo "=== 3/5 远程抓取节点 (SAM+GraspNet) ==="
launch remote_grasp "remote_grasp.launch.py" 2 \
  "ros2 launch x5a_remote_grasp remote_grasp.launch.py planning_enabled:=true tcp_mapping_confirmed:=true"

echo "=== 4/5 抓取执行器 ==="
launch executor "remote_grasp_executor" 1 \
  "ros2 run x5a_remote_grasp remote_grasp_executor --ros-args -p execution.enabled:=true"

echo "=== 5/5 网页/语音后端 (端口 8000) ==="
launch web_agent "web_agent --host" 2 \
  ros2 run x5a_web_agent web_agent --host 0.0.0.0 --port 8000

echo
echo "=== 就绪检查 ==="
sleep 3
if curl -s --max-time 3 http://127.0.0.1:8000/api/health | grep -q '"ok":true'; then
  echo "web_agent: OK (http://<本机IP>:8000)"
else
  echo "web_agent: 未就绪（可能还在加载，10 秒后再试 scripts/stack_status.sh）"
fi
echo "日志目录: $LOGDIR   查看某服务: tail -f $LOGDIR/<名称>.log"
echo "RViz: ./scripts/rviz.sh   全部停止: ./scripts/stack_stop.sh"
echo "云端模型服务（服务器上）: ~/model_start.sh / ~/model_stop.sh / ~/model_status.sh"
