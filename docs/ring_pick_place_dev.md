# 圆环抓放分拣任务 — 开发思路与文件索引

> X5A 机械臂 + USB 相机 + 传送带的好坏工件分拣。本文记录整体设计、每个关键决策的原因、
> 踩过的坑，以及所有相关代码文件路径。生成于 2026-08-29。

---

## 1. 任务定义

1. 传送带把圆环运到相机检测区（画面中心 ±25%）；圆**停稳 0.65s** 后启动 **6s 倒计时**
   ——倒计时的物理含义是传送带把圆环从检测区运到机械臂抓取点的时间，倒计时期间圆
   离开检测区属正常现象，**不取消**。
2. 倒计时结束，机械臂在固定**预抓取位**（圆环正上方）执行抓取。
3. **好坏分拣**：按出现次序，奇数次 = 好工件 → `place_good` 点；偶数次 = 坏工件 →
   `place_bad` 点。判定逻辑集中在 `classify_workpiece()` 一个函数，未来可换成真视觉质检。
4. 每个循环完成后回预抓取位等下一个；跑完指定循环数后**回 home（关节零位）并闭合夹爪**。

## 2. 系统架构与数据流

```
USB 相机 /dev/video0 (HDMI 采集卡, 1280x720 MJPG)
  │
  ├─ RingMonitor 线程                    [ring_pick_place.py]
  │    HoughCircles param2 阶梯(60→18, 严格优先)
  │    → 同心去重 dedupe_concentric (内外圈只留外圈)
  │    → 中心锁定跟踪器 CircleTracker (0.3s 容错, 锁定圆消失自动解锁)
  │    → 检测区门控 + arming (清空0.3s武装 / 3s强制武装)
  │    状态机: watch →(停稳)→ countdown 6s → TRIGGER
  │    看门狗: 2s 无帧 → 报错退出 (防 V4L2 取流阻塞导致无声挂起)
  │
  ├─ RingTaskNode (继承 MoveToPoseNode)  [ring_pick_place.py]
  │    循环: 开爪 → APPROACH(笛卡尔↓grasp_depth) → 合爪 → LIFT(关节原路返回)
  │           → 移动到放置点上方8cm → DESCEND(笛卡尔↓) → 松爪
  │           → RETREAT(关节原路返回) → 回预抓取位
  │    收尾: RETURN_HOME(零位) → FINAL_CLOSE_GRIPPER
  │
  ├─ MoveToPoseNode                      [move_to_pose.py]
  │    IK 预解算 (/compute_ik, 位姿快照的 joint_pos_rad 作首选种子)
  │    → 关节空间 MoveGroup 目标 (plan_only=True 规划)
  │    → ExecuteTrajectory 动作执行; 失败回退 Pose 目标约束 (球2cm+姿态0.25rad)
  │    逻辑状态 last_joints: plan-only 链式预演; 执行失败回滚到真实关节角
  │
  ├─ x5a_official_trajectory_adapter     [x5a_moveit_official_adapter]
  │    FollowJointTrajectory / GripperCommand → /arm_cmd (RobotCmd, mode=5)
  │    /arm_status (RobotStatus) → /joint_states
  │    安全检查: /arm_cmd 仅 1 个发布者; 首个状态包后 2s 内拒绝执行; 限速 1.0 rad/s
  │
  └─ X5Controller (厂商驱动, CAN can1)   [~/repos/arx/ARX_X5/ROS2/X5_ws]
```

## 3. 关键设计决策与踩坑记录

| # | 决策 | 原因 / 踩的坑 |
|---|------|--------------|
| 1 | IK 预解算 → 关节空间规划，位姿目标只做回退 | 这台臂上 Pose 目标直接规划不可靠（沿用 x5a_pick_place 的实战经验） |
| 2 | 位姿快照的 `joint_pos_rad` 作 IK 首选种子 | 快照关节角是物理验证过的精确解；KDL 对放置点姿态否则解不出，进而拖垮 OMPL(99999) |
| 3 | 逻辑状态 `last_joints` + 失败回滚 | plan-only 预演需要链式规划；**执行失败必须回滚到真实关节角**，否则重试从幻影位姿出发，move_group 报 `start point deviates >0.1` ABORT |
| 4 | APPROACH/DESCEND 笛卡尔直线，LIFT/RETREAT 关节原路返回 | 下降段精度关键必须直线；预抓取构型贴近关节限位，垂直线**反向**笛卡尔最后 8mm 解不收敛（实测下行 fraction=1.0、上行 0.8） |
| 5 | 圆检测四层防抖 | ①param2 阶梯严格优先 ②同心去重（内外圈横跳）③中心锁定跟踪器（跨帧横跳，锁定后半径固定+引导式重试降漏检）④0.3s 容错（运动模糊不打断稳定计时） |
| 6 | 监控看门狗 | V4L2 读取会偶发阻塞在不可中断 ioctl（进程连 SIGTERM 都杀不掉，需 kill -9），曾导致"执行一次后永远等待"。2s 无帧即报错退出 |
| 7 | arming 有界化 | 检测区清空 0.3s 即武装（正常传送带节奏）；3s 不清空就地对现有物体武装（任务启动时环已在位也能触发） |
| 8 | /arm_cmd 单发布者互斥 | 适配器硬性安全检查。rqt 的 ARX 插件、上一轮残留的适配器节点都会挡执行 → 启动脚本 preflight 自动清理残留节点 |
| 9 | 速度缩放 0.70/0.25（仅关节段） | 关节限位 1.0/1.2 rad/s + 适配器 1.0 rad/s 硬限，0.70 留安全裕量；抓取下降是笛卡尔段不受影响 |
| 10 | 分拣策略单函数化 | `classify_workpiece()` 按出现次序奇好偶坏；换真视觉质检只改这一个函数 |

## 4. 文件路径

### 自研脚本 `~/arx/arm/scripts/`
| 文件 | 作用 |
|------|------|
| `ring_pick_place.py` | **主任务节点**：检测状态机 + 抓放序列 + 分拣 + 收尾 |
| `move_to_pose.py` | MoveIt 规划/执行封装（IK→关节规划→执行，Pose 回退，逻辑状态跟踪） |
| `usb_circle_detect.py` | 圆检测库（检测+去重+跟踪器）+ 独立单拍/预览工具 |
| `record_pose.py` | 位姿示教快照（/joint_states → FK → yaml） |
| `start_ring_task.sh` | 一键启动：驱动 → MoveIt+适配器 → 任务；Ctrl-C 全关；自动清残留 |

### 位姿数据 `~/arx/arm/logs/`
| 文件 | 内容 |
|------|------|
| `end_pose_20260828_173732.yaml` | 预抓取位（抓取点正上方）+ 当时关节角 |
| `poses/place_good.yaml` | A 点（好工件放置位） |
| `poses/place_bad.yaml` | B 点（坏工件放置位） |

yaml 统一格式：`joint_pos_rad`（IK 种子）+ `base_link_to_tool0.position_m/rpy_rad/quaternion_xyzw`

### 依赖的项目内代码 `~/arx/arm/src/`
| 路径 | 作用 |
|------|------|
| `x5a_pick_place/x5a_pick_place/pick_place_node.py` | 规划套路的参考来源（原视觉抓放） |
| `x5a_moveit_official_adapter/x5a_moveit_official_adapter/official_trajectory_adapter.py` | 官方适配器（拒绝条件、限速、跟踪容差全在这） |
| `x5a_moveit_official_adapter/config/adapter.yaml` | 适配器参数（`max_joint_speed_rad_s: 1.0` 等） |
| `x5a_moveit_config/launch/x5a_real_moveit.launch.py` | 真机 MoveIt 启动 |
| `x5a_moveit_config/config/joint_limits.yaml` | 关节限位（joint4 实测有效下限 -1.28） |
| `x5a_moveit_config/config/X5A.urdf.xacro` / `X5A.srdf` | 运动学模型 |
| `x5a_handeye/x5a_handeye/x5a_fk.py` | 正运动学 `fk_base_tool0(joints)`（`base_link→tool0` 口径） |

### 厂商侧 `~/repos/arx/ARX_X5/ROS2/X5_ws/`
| 路径 | 作用 |
|------|------|
| `install/arx5_arm_msg` | `RobotStatus`(含 `end_pos`)/`RobotCmd` 消息包 |
| `install/arx_x5_controller` | X5Controller 驱动，`open_single_arm.launch.py` 启动 |

注意：`RobotStatus.end_pos` 是 **SDK 运动学基座**口径，≠ URDF `base_link`；统一用
`x5a_fk.fk_base_tool0` 从关节角计算。

### 日志与调试产物
| 路径 | 用途 |
|------|------|
| `~/.ros/log/` | move_group/适配器节点日志——**诊断 execution FAIL(-4) 的第一现场**（搜 `rejecting` / `ARM_ABORT`） |
| `~/arx/arm/logs/ring_task/` | 监控状态机证据帧（stable/trigger）；`watch_latest.jpg` 每 2s 覆盖刷新 = 监控实时视野 |
| `~/arx/arm/logs/circle_detect/` | 圆检测标注图 |

## 5. 使用命令

```bash
# 一键启动（驱动+MoveIt+任务），默认 2 个工件
~/arx/arm/scripts/start_ring_task.sh [cycles]

# 单独跑任务（MoveIt 栈已在时）
/usr/bin/python3 ~/arx/arm/scripts/ring_pick_place.py --cycles 2 \
  --place-good ~/arx/arm/logs/poses/place_good.yaml \
  --place-bad  ~/arx/arm/logs/poses/place_bad.yaml

# 只测检测状态机（不动臂）；或检测预览窗口
/usr/bin/python3 ~/arx/arm/scripts/ring_pick_place.py --monitor-only
/usr/bin/python3 ~/arx/arm/scripts/usb_circle_detect.py --show

# 示教新放置点（手动把臂挪到位后）
/usr/bin/python3 ~/arx/arm/scripts/record_pose.py --name place_good
```

### 调参速查
| 参数 | 默认 | 说明 |
|------|------|------|
| `--stable-sec` | 0.65 | 检测区停稳判定时长 |
| `--countdown-sec` | 6.0 | 传送带运输时间 |
| `--grasp-depth` | 0.04 | 预抓取位向下抓取深度（m） |
| `--zone-frac` | 0.25 | 检测区 = 画面中心 ±比例 |
| `--move-tol-px` | 12 | 圆漂移多少像素内算"停稳" |
| `--vx/--ax` | 0.70/0.25 | 关节段速度/加速度缩放（抓取下降不受影响） |
| `--home-joints` | 全零 | 收尾 home 关节角 |
| `--plan-only` / `--monitor-only` | - | 全流程预演 / 仅检测状态机 |

## 6. 故障对照表

| 现象 | 原因 | 处置 |
|------|------|------|
| `execution=FAIL(-4)`，适配器日志 `has N publishers` | /arm_cmd 上有别的发布者（rqt 插件/残留节点） | 关 rqt；启动脚本已自动清残留 |
| `FAIL(-4)`，`state has not settled after startup` | 启动竞态 | 脚本已等待+任务首段重试，若仍现则等 5s 再跑 |
| `Invalid Trajectory: start point deviates...` | 逻辑状态未回滚（已修复） | 重跑即可 |
| 适配器 `tracking error ... rad` ABORT | 指令发出但臂不响应 | **重启臂电源**（本体侧问题） |
| 任务无声卡住、进程杀不掉 | V4L2 取流阻塞 | 看门狗已处理；复现则重新插拔采集卡 |
| IK/规划 99999 | 目标姿态刁钻 | 快照关节角种子已内置；确认位姿文件未手改坏 |
