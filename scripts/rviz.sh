#!/bin/bash
# 打开 RViz（远程抓取候选/规划可视化配置）
source /home/quella/arx/arm/scripts/stack_env.sh
exec rviz2 -d /home/quella/arx/arm/src/x5a_remote_grasp/config/remote_grasp.rviz
