# 2026-10-03 独立求解器并行 IK

## 结果与运行状态

已新增并接入 x5a_parallel_ik C++ 服务，4 个独立 RobotModelLoader/RobotModel、运动学求解器及碰撞场景实例。默认随 remote_grasp.launch.py 启动；正常启动命令不变。没有新增运动执行接口。

服务复制正在运行的 /move_group 的 URDF、SRDF、robot_description_kinematics 和 robot_description_planning 参数，启动时检查四个 arm 求解器指针互不相同。客户端以 SHA256(URDF + NUL + SRDF) 校验模型身份；不能用另一个 xacro 或静态场景悄悄替代当前模型。

部署在用户任务及执行器空闲时进行。仅更新规划器与并行 IK 服务；执行器 PID 2215596、持物记录、MoveIt、相机和厂商控制器均保留。测试脚本没有发送控制器/夹爪命令，也没有发布供执行器使用的轨迹。

## 实测

8 个相同不可达目标的不同种子，单种子预算 0.15 s，两种模式使用同一服务实现及同一场景快照。三轮中位数：1 个工作实例约 1.204 s，4 个实例约 0.303 s，接近 4 倍。可达测试目标约 3–4 ms。

完整记录画面的抓取规划回放（从候选选择到四段轨迹全部完成，含点云几何检查、IK、MoveIt、FK，排除云端视觉及机械臂动作）：

|画面|旧串行 IK|独立并行 IK|规划耗时减少|
|---|---:|---:|---:|
|20261003_170234_987941|0.397 s|0.433 s|-9.1%|
|20261003_170536_818755|17.139 s|13.001 s|24.1%|

两份画面的选中 tool0 抓取位姿和所需夹爪开度逐项相同，四段轨迹均通过原有完整路径/FK 检查。底层 IK 随机搜索可能产生不同关节路径，验证不要求轨迹逐点相同。

该表仅是两次离线回放，不是线上分位数或成功率统计。顺畅场景增加约数十毫秒，因为新增场景快照/服务传输；慢场景节省约 4.1 s，约 24%。不能把微基准 4 倍直接当成全流程 4 倍。

两份历史放置回放使用记录的实际物体相对夹爪点云、模拟 READY 与指关节 0.015 m。串行/并行都生成 TRANSPORT/PLACE/RETREAT，释放 tool0 位姿相同，保留 8 mm 的观察物体底面释放间隙。耗时约 1.4 s，基本没有加速，因为这些回放的主要时间在释放高度的笛卡尔路径碰撞检查。该夹持状态是模拟回放，未实际抓放。

原始数据：service_validation.json、planning_replay.json、placement_replay.json。规划及放置脚本临时导入历史安全场景，finally 恢复原完整场景并保留实时关节状态，均已确认恢复成功。

## 并行方式与安全检查

- ROS 服务边界按批次串行处理；批次内部最多 4 个 C++ 线程，各自访问独立求解器/碰撞场景，避免共用 KDL 内部状态。
- 客户端先单独试优先种子，成功即返回，避免正常可达目标等待无用的备用搜索；优先种子失败后，再并行试剩余种子。
- 每次 collision_aware_ik 调用重新取得完整 MoveIt 场景，包括关节、附着物、障碍物、Octomap、ACM、固定变换、padding/scaling。两个种子批次绑定同一快照；不跨捕获画面缓存安全场景。
- 场景的碰撞机器人/网格 BVH 每个工作实例只构造一次，完整场景消息逐批覆盖。测试确认前一快照的障碍物不会残留到下一批。
- GRASP 后计划闭爪、PLACE 后计划开爪；所有备用种子（包括 current-state 种子）均保留该阶段的计划手指状态。场景仍从实时完整快照获取，只覆盖明确提供的计划指关节。
- 只接收完整场景、匹配模型、有限数值、arm 组、有效末端和单位四元数，要求 avoid_collisions=true。时间限制包含纳秒部分，并受单种子与整批预算约束。
- 服务在搜索过程中检查关节范围、碰撞与请求约束，结果再次检查 FK <=3 mm/0.05 rad。客户端还需通过真实 /check_state_validity，随后交回原 MoveIt 路径规划和每段终点验证。
- 结果按原种子优先级消费，不能把最先完成的工作线程当成最佳抓姿；候选的神经网络/几何排序和总规划预算保留。
- 模型摘要不匹配、不完整场景、协议异常或失败响应直接停止，不能默认为可抓。只有服务缺失时，配置允许回退旧串行 IK。

实现参考 [MoveIt Humble 官方运动学服务](https://github.com/moveit/moveit2/blob/humble/moveit_ros/move_group/src/default_capabilities/kinematics_service_capability.cpp) 的 RobotState、约束和碰撞回调方式；另加独立求解实例、完整场景身份校验、纳秒超时和 FK 复核。

## 验证

- 84 项抓取、规划、执行与交互单元测试通过。
- 真实服务验证：可达/不可达目标；返回解的真实 MoveIt 碰撞/FK；闭爪/开爪状态；快照内临时障碍物；旧障碍清理；错误模型、不完整场景、缺失手指数据拒绝。
- 两份真实抓取画面的完整规划回放成功，目标位姿/开度相同。
- 两份历史放置回放成功，目标位姿/释放间隙相同，模拟指关节明确记录。
- x5a_task_interfaces、x5a_parallel_ik、x5a_remote_grasp 构建通过；最终 C++ 服务为 Release 构建。
- 运行参数 planning.parallel_ik_workers=4，服务健康信息 independent_workers=4，模型摘要匹配当前 /move_group。

## 启动与配置

```bash
source scripts/stack_env.sh
colcon build --packages-select x5a_task_interfaces x5a_parallel_ik x5a_remote_grasp --symlink-install
./scripts/stack_start.sh
```

默认 remote_grasp.launch.py 自动带起服务。已有独立服务时传 start_ik_server:=false，实例数可用 ik_workers:=2 调整（1–4）。planning.parallel_ik_service 留空可明确关闭并行，回到原串行路径；planning.parallel_ik_fallback_serial 控制服务缺失时是否允许回退。

复现测试：

```bash
source scripts/stack_env.sh
/usr/bin/python3 scripts/validate_parallel_ik.py
PYTHONPATH="src/x5a_remote_grasp:$PYTHONPATH" /usr/bin/python3 scripts/validate_parallel_planning.py
PYTHONPATH="src/x5a_remote_grasp:$PYTHONPATH" /usr/bin/python3 scripts/validate_parallel_place.py
```

后两个脚本需要任务流水线空闲。它们只生成轨迹、临时导入并恢复规划场景，不发送动作。实际抓取成功率及线上耗时仍需要用户新任务日志和观察确认，本次没有把离线可解性当成真机抓取成功。
