#!/bin/bash
# 全链路共用环境：屏蔽 conda、加载 ROS + 厂商工作空间 + 项目 + 密钥
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH
source /opt/ros/humble/setup.bash
source /home/quella/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash
source /home/quella/arx/arm/install/setup.bash
[ -f /home/quella/.config/x5a_remote_grasp/key.env ] && source /home/quella/.config/x5a_remote_grasp/key.env
# 语音指令规划成功后自动触发真机执行；置 0 退回「只规划+RViz」两步模式
export X5A_AUTO_EXECUTE=${X5A_AUTO_EXECUTE:-1}
# 抓取路由：remote=SAM+GraspNet 云端链（默认）；rule=HSV 色块规则链（x5a_task_server）
export X5A_GRASP_ROUTE=${X5A_GRASP_ROUTE:-remote}
# 数字人联动：1=语音指令先转发数字人、等 MQTT 开始标志再动；0=不等直接执行
export X5A_TEXT_FORWARD=${X5A_TEXT_FORWARD:-1}
