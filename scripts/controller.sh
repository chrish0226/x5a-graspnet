#!/bin/bash
# 厂商控制器（终端1，前台运行，保持你原来的启动方式）
# CAN 就绪前置: sudo ~/arx/arm/scripts/setup_can_x5a.sh（或手动 slcand）
source /home/quella/arx/arm/scripts/stack_env.sh
exec ros2 launch arx_x5_controller open_single_arm.launch.py
