#!/bin/bash
# 查看各服务运行状态 + 最近日志 + 网页后端健康
set -u
LOGDIR=/home/quella/arx/arm/logs/stack
check() {  # check <名称> <识别模式> <日志名>
  local n=$1 p=$2 log=$LOGDIR/$3.log
  if pgrep -u "$(whoami)" -f "$p" >/dev/null 2>&1; then
    printf "[运行中] %-22s 最近日志: %s\n" "$n" "$(tail -1 "$log" 2>/dev/null | cut -c1-90)"
  else
    printf "[未运行] %-22s\n" "$n"
  fi
}
check 厂商控制器 open_single_arm.launch.py controller
check MoveIt+adapter x5a_real_moveit.launch.py moveit
check 视觉/相机 vision.launch.py vision
check 规则任务服务 task_server_node task_server
check 远程抓取 remote_grasp.launch.py remote_grasp
check 并行IK服务 parallel_ik_server remote_grasp
check 抓取执行器 remote_grasp_executor executor
check 网页后端 "web_agent --host" web_agent
echo
curl -s --max-time 3 http://127.0.0.1:8000/api/health | grep -q '"ok":true' \
  && echo "web_agent HTTP: OK" || echo "web_agent HTTP: 不可达"
