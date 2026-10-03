#!/bin/bash
# 实时预览云端抓取姿态推理结果（SAM 分割 + GraspNet 候选夹爪位姿）
# 只推理和显示：不规划、不发 Action、不碰 /arm_cmd，机械臂不动。
#   用法:  ./scripts/grasp_preview.sh --prompt "cube"
#          ./scripts/grasp_preview.sh --prompt "black pen" --once      # 只跑一次
#          ./scripts/grasp_preview.sh --prompt "cube" --no-mask        # 不叠加 SAM 分割
# 窗口里 q 退出、s 存图；没显示器时自动只发 /x5a_grasp_preview/image + 打印。
source /home/quella/arx/arm/scripts/stack_env.sh
exec /usr/bin/python3 /home/quella/arx/arm/scripts/grasp_preview.py "$@"
