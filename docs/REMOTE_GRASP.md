# Remote SAM 3.1 + GraspNet

`x5a_remote_grasp` 是独立节点，不修改现有 HSV 检测链。它同步对齐 RGB、
原始 16 位深度和 CameraInfo，调用远端服务，校验并转换所有 6D 候选，在
RViz 显示坐标轴，并按 **GraspNet 原始姿态** 做碰撞感知 IK 与 MoveIt 规划。

规划节点本身不发 `/arm_cmd`（`safety.plan_only=true`）。真机执行走
`remote_grasp_executor`，沿用 official adapter 的 FollowJointTrajectory /
GripperCommand。

默认 `orientation_mode: graspnet`。不再把瓶子等物体压成小方块那套近竖直模板。
`template` 只作为覆盖选项留给旧的立方块流程。

## 编译和密钥

```bash
cd /home/quella/arx/arm
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select x5a_task_interfaces x5a_parallel_ik x5a_remote_grasp
source install/setup.bash
source /home/quella/.config/x5a_remote_grasp/key.env
```

Key 只从 `SAM_GRASPNET_API_KEY` 读取，不接受 ROS 参数且不会写日志。

默认启动本机 `x5a_parallel_ik` C++ 服务：4 个独立机器人模型、KDL 求解器和
碰撞场景实例，并行处理备用 IK 种子。服务从正在运行的 `/move_group` 复制
URDF/SRDF 与求解器参数，每次请求使用完整场景快照，结果再由 MoveIt 检查。
已有单独的并行服务时可传 `start_ik_server:=false`；`ik_workers:=2` 可调整实例数。
正常启动仍使用 `scripts/stack_start.sh`，不需要另开 IK 终端。
实测与验证见 [并行 IK 审查](reviews/parallel_ik_20261003/FIXES.md)。

独立放置指令（例如“放到蓝色盒子上”）使用执行器保存的实际持物几何，
将物体底部移到目标顶面上方 1～2cm、物体中心对齐后松爪。
默认依次尝试 10/15/18/20mm，选择最低的完整无碰撞路径。
这里的高度指物体底部到目标顶面的距离；工具 TCP 高度随夹持位置计算。
放置终点仍检查实际 TCP，松爪后撤退。验证见
[本次放置修复](reviews/place_failure_20261003/FIXES.md)。

## 实时预览推理结果（不动机器人）

`scripts/grasp_preview.py` 单独取一帧对齐 RGB-D，按与节点**完全相同**的契约调
`/v1/grasp`，把候选夹爪位姿投影画回图像：圆圈=抓取点、箭头=接近方向
（rot 第 1 列）、短线=两指开合宽度（第 2 列 ±width/2）、文字=序号/score/宽度/深度，
排名第一的加粗。叠加 SAM 分割（`--mask`，默认开）可同时看到模型选中的是哪个物体。

```bash
./scripts/grasp_preview.sh --prompt "cube"                # 连续预览（约 1 秒一轮）
./scripts/grasp_preview.sh --prompt "red cube" --once     # 只跑一次，存一张标注图
./scripts/grasp_preview.sh --prompt "cube" --no-mask      # 不叠加分割，更快
```

- 只推理和显示：不规划、不发 Action、不碰 `/arm_cmd`，机械臂一动不动，真机跑着也能开。
- 有 `$DISPLAY` 时自动开窗口（`q` 退出 / `s` 存图），没有则只发话题 + 打印；
  自动存图在 `~/.local/share/x5a_remote_grasp/preview/`（`--once` 一定会存）。
- 标注图同时发布到 `/x5a_grasp_preview/image`（`sensor_msgs/Image`），RViz 里加一个
  Image display 指过去即可。
- 模型热起来后一轮约 1 秒（云端 0.4 s + 本机取帧绘图 0.6 s）；冷启动第一轮要等
  SAM 加载（约 25 s）。
- 提示词越具体越准：`"cube"` 会把桌上三个方块都切出来（分数都 0.94+），链路按
  **SAM 分数最高**的那个选（实测选了红色方块）。要指定就写 `"red cube"` /
  `"green cube"` / `"white cube"`，或用节点支持的两段式 `"目标 | ref: 参照; rel: on"`。

## 候选开度审计（选用逻辑）

GraspNet 回归的 `width_m` 是**模型输出**，不保证等于物体在闭合轴上的真实跨度。
实测案例（2026-10-02，6.6×12.2cm 盒子）：top-1 报 87.9mm、按其闭合轴走廊实测需要
96mm（整面投影口径）/ 83.4mm（走廊口径），另有 4 个候选报 49~87mm、实际需要
113~137mm——物理上不可能，却因碰撞检查用预测宽度+1cm 容忍而存活。

参照成熟管线（GPG 的 antipodal 几何核查、Contact-GraspNet 的场景点云碰撞过滤、
GraspNet 自带 MFCD）补上「几何实测」一环：

1. **服务端**（`grasp_server.py`，2026-10-02 起）：`_run_graspnet` 对每个存活候选
   计算实测需要开度——掩码点云沿闭合轴（rot 第 2 列）在接近走廊（±`approach_span`
   =5cm，与 MFCD 的 approach_dist 一致）内的跨度，走廊点数 <30 时为 None
   （fail-open）。`/v1/grasp` 响应新增 `required_opening_m` 字段（纯新增，旧节点兼容）。
2. **节点**（`remote_grasp_node.py`）：`_select_candidate` 在 width/orientation 检查
   之后剔除：
   - 实测开度 > `gripper.max_width_m` − 2mm（堵转爪没有接触余量，会闭到头夹空）；
   - 实测开度 > 回归宽度 + `grasp.aperture_tolerance_m`（默认 5mm，吸收深度噪声；
     超出即下探途中撞物体两侧）。
   拒绝原因记为 `aperture`，进 action feedback、`plan_failures.json`；测量值随
   `tf_validation.json` 快照落盘。`grasp.aperture_audit_enabled:=false` 可整体关闭。

排序不变（仍按 GraspNet 分数），审计只剔除几何不可能的候选。走廊口径比整面
投影口径更保守且更物理正确：手指只扫掠走廊内的区域，近边缘抓取不再被整面
跨度误判（盒案例 #0 整面 96mm / 走廊 83.4mm，实为可行）。

## 验证顺序

1. 健康检查（同时确认无 Key 返回 401）：

```bash
ros2 run x5a_remote_grasp remote_grasp_health
```

2. 启动现有 D435i 与手眼 TF，再启动新节点并保存同步帧：

```bash
ros2 launch x5a_remote_grasp remote_grasp.launch.py \
  planning_enabled:=false tcp_mapping_confirmed:=false
ros2 service call /x5a_remote_grasp/save_snapshot std_srvs/srv/Trigger '{}'
```

保存目录位于 `/home/quella/.local/share/x5a_remote_grasp/captures/`，包含
`color.png`、未降位的 `depth.png` 和 `camera_info.json`。depth scale 从运行中
RealSense 的 `depth_module.depth_units` 获取，默认不会假设 1000。

3. 用保存数据离线调用 API：

```bash
ros2 run x5a_remote_grasp remote_grasp_offline \
  /home/quella/.local/share/x5a_remote_grasp/captures/时间戳目录 \
  --prompt banana
```

4. 请求 API、TF 并显示所有候选（不做 IK）：

```bash
ros2 action send_goal /x5a_remote_grasp/plan \
  x5a_task_interfaces/action/PlanRemoteGrasp \
  "{prompt: banana, min_depth: 0.1, max_depth: 1.5, top_k: 10, plan_motion: false}" \
  --feedback
```

在 RViz 添加 MarkerArray：`/x5a_remote_grasp/candidates`。红、绿、蓝分别是
GraspNet +X 接近轴、+Y 两夹爪方向、+Z 正交轴。结果目录中的
`tf_validation.json` 记录相机光学系与 base_link 中的变换前后姿态。

`T_grasp_tool0` 默认单位旋转：URDF 里 `tool0` 相对 `link6` 只有 `+X 0.11277 m`
平移，轴向与 GraspNet 一致（`+X` 接近 / 指向手指，`+Y` 夹爪开合）。若实机
手指总差一截才包住瓶子，调 `grasp.approach_offset_m`（沿 GraspNet `-X` 往掌心
回退），不要再改回 `orientation_mode: template`。

5. 用 GraspNet 6D 做 IK 和规划：

```bash
ros2 launch x5a_remote_grasp remote_grasp.launch.py \
  planning_enabled:=true tcp_mapping_confirmed:=true
ros2 action send_goal /x5a_remote_grasp/plan \
  x5a_task_interfaces/action/PlanRemoteGrasp \
  "{prompt: bottle, min_depth: 0.1, max_depth: 1.5, top_k: 10, plan_motion: true}" \
  --feedback
```

节点按分数依次检查，而不是直接用 top1：夹爪宽度、从桌下伸上来的接近方向、
工作空间、桌面净空、碰撞感知 IK、`/check_state_validity`。每个候选还会试
绕接近轴旋转 180°（平行爪等价姿态）。

路点沿 GraspNet `+X` 接近、世界 `+Z` 抬起、再沿 `-X` 撤离。`GRASP`/`PLACE`
优先笛卡尔直线，其余阶段先 IK 再关节空间规划。结果发布到
`/display_planned_path`。夹爪闭合仅发布到诊断话题
`/x5a_remote_grasp/planned_gripper_close`，不会发给控制器。

## ROS 接口

- Action：`/x5a_remote_grasp/plan`
- 类型：`x5a_task_interfaces/action/PlanRemoteGrasp`
- Service：`/x5a_remote_grasp/health`（`std_srvs/Trigger`）
- Service：`/x5a_remote_grasp/save_snapshot`（`std_srvs/Trigger`）

Action 输入：prompt、可选 min/max depth、top_k、是否纯规划。

Action 输出：base_link 中选中/预抓取/抬升/撤离 PoseStamped，宽度、分数、
候选编号和 IK/碰撞/规划标志。

主要参数在 `config/remote_grasp.yaml`：

- `remote.*`：地址、超时及推理字段。
- `topics.*`、`sync.*`：D435i 对齐话题与时间同步。
- `depth.*`：运行时深度尺度来源。
- `workspace.*`、`scene.*`：工作空间、桌面和相机碰撞体。
- `grasp.*`：接近、抬升、撤离距离及 TCP 映射。
- `planning.*`：`arm`、`tool0`、IK/规划服务和容差。

## 当前风险

- 当前手眼标定 hold-out 平移误差均值约 4.18 mm、最大约 5.47 mm；相机移动需重标。
- 相机碰撞盒是保守近似，线缆和支架未完整建模。
- 远端没有返回物体 mesh；本地场景覆盖机器人、桌面、相机，目标与全场景点云
  碰撞仍依赖服务端过滤。
- X5A 工作空间和 `joint4` 限位会丢掉一部分 GraspNet 6D 候选；这是筛候选，
  不是改回竖直模板。
- `approach_offset_m` 默认 0。若夹爪总是提前碰到或包不住，按实机只调这一项。
