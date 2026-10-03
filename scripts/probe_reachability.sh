#!/bin/bash
# 可达性实测（只读）：扫 /compute_ik 生成 ~/.local/share/x5a_remote_grasp/reachability.json
# 需要 MoveIt + 适配器在运行。参数直传：./scripts/probe_reachability.sh --step 0.05 --max-minutes 3
source /home/quella/arx/arm/scripts/stack_env.sh
exec python3 /home/quella/arx/arm/scripts/probe_reachability.py "$@"
