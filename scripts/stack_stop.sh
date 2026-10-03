#!/bin/bash
# 按逆序停止脚本管理的五个服务（厂商控制器用 Ctrl+C 自行停止）
set -u
PATTERNS=(
  "web_agent --host"
  "task_server_node"
  "remote_grasp_executor"
  "remote_grasp_node"
  "parallel_ik_server"
  "remote_grasp.launch.py"
  "official_trajectory_adapter"
  "realsense2_camera_node"
  "vision.launch.py"
  "move_group"
  "x5a_real_moveit.launch.py"
)
NAMES=(网页后端 规则任务服务 抓取执行器 远程抓取节点 并行IK服务 远程抓取launch adapter子进程 相机节点 视觉/相机 MoveGroup MoveIt)

for i in "${!PATTERNS[@]}"; do
  p="${PATTERNS[$i]}"; n="${NAMES[$i]}"
  if pgrep -u "$(whoami)" -f "$p" >/dev/null 2>&1; then
    pkill -u "$(whoami)" -f "$p"
    echo "[停止] $n"
    sleep 2
  else
    echo "[无] $n 未在运行"
  fi
done
echo "完成。"
