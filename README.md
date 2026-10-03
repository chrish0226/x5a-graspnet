# ARX X5A ROS 2 视觉抓取放置系统

稳定抓取路径是 **Python `x5a_pick_place`**，不要用 `./x5a_pick.sh`（那是实验性 MTC）。

## 手机 / QQ 语音抓取 Demo

机械臂主机没有麦克风。推荐用安卓 App 录音，主机用本地 `faster-whisper` 识别，再经 `/x5a/pick_place` 调用现有视觉抓取。不绕过 MoveIt，也不额外发布 `/arm_cmd`。

支持：

- 红 / 白 / 橙
- 「把最近的 / 最远的放入盒子」
- 「把红色和白色放入盒子」（队列连续抓，中间回 Ready，盒子坐标只冻结一次）
- QQ 群 OneBot 机器人（Napcat / Lagrange）

安卓 APK：`android/X5A语音抓取.apk`（自动连接 `192.168.0.50:8000`）

完整说明：[docs/VOICE_INTERACTION.md](docs/VOICE_INTERACTION.md)。

### 终端文字输入层（测试用，不用麦克风）

不想说话/没有手机时，直接用键盘打字，走的是和语音完全相同的那条链路（只替换「语音转文字」这一步）：

```bash
./scripts/text_cli.sh                      # 接入已经在跑的 web_agent（默认 127.0.0.1:8000）
./scripts/text_cli.sh --url http://192.168.0.50:8000

# 或者把输入层塞进 web_agent 进程本身（要先停掉已有的 web_agent）
ros2 run x5a_web_agent web_agent --host 0.0.0.0 --port 8000 --terminal
./run_x5a_voice_demo.sh --terminal
```

新开的终端一定要先 source 环境（或直接用 `scripts/text_cli.sh`，它自己会 source）：
`source /opt/ros/humble/setup.bash && source ~/arx/arm/install/setup.bash`，否则会报 `Package 'x5a_web_agent' not found`。

`/help` 看命令（`/state`、`/cancel`、`/task red`、`/parse <文字>`、`/quit`）。真机模式下打进去的文字会真的动机械臂；只想看解析就用层里的 `/parse`。

## 一键视觉抓取

机械臂、RealSense、CANable2 正确连接后：

```bash
cd ~/arx/arm
./run_x5a_vision_pick.sh --color red
```

```bash
./run_x5a_vision_pick.sh --color white
./run_x5a_vision_pick.sh --color orange
./run_x5a_vision_pick.sh --vision-only
./run_x5a_vision_pick.sh --dry-run --color red
```

脚本会 source ROS/厂商/项目环境，复用健康的既有节点，并检查 CAN、关节反馈、MoveIt、RealSense、Eye-to-Hand TF 和稳定视觉坐标。默认执行一次抓放后回 Home。日志在 `logs/run_YYYYMMDD_HHMMSS/`。

前三步（控制器、MoveIt、视觉）已经在跑时，直接：

```bash
source /opt/ros/humble/setup.bash
source ~/arx/arm/install/setup.bash
ros2 launch x5a_pick_place pick_place.launch.py \
  plan_only:=false vision_enabled:=true target_color:=red
```

## 项目简介

这是一个在真实 ARX X5A 上运行的 ROS 2 Humble 项目，使用 MoveIt 2、Intel RealSense RGB-D 和 Eye-to-Hand 手眼标定，实现视觉定位、抓取、放到黑色海绵目标并回 Home。

换相机位置后已用 2026-08-13 标定重新闭环。当前正式链：

```text
EYE-TO-HAND CALIBRATION: PASS
VISION PICK AND PLACE: PASS
```

## 已实现功能

- X5A `robot_description`、URDF/Xacro 和 mesh
- MoveIt 2 OMPL planning、Planning Scene 和真机 ExecuteTrajectory
- `RobotStatus` → `/joint_states` 状态反馈
- `FollowJointTrajectory` → ARX `RobotCmd` 正式适配器
- 标准 `control_msgs/action/GripperCommand` 夹爪控制
- 视觉坐标冻结后按预抓取姿态 → 接近 → 抓取 → 短抬起 → 放置 → 回 Home
- 抓取使用已验证的近竖直姿态；放置会尝试多组朝向和多种子 IK
- ChArUco Eye-to-Hand 标定与 `base_link -> camera_link` TF 发布
- RealSense aligned RGB-D 红、白、橙三色方块独立定位
- 黑色海绵区域中心作为可移动放置目标
- 工作区包络过滤，避免把不可达点交给规划器

## 系统架构

正式真机执行链：

```text
RealSense
   |
   v
x5a_vision -- selected cube pose + /x5a_vision/box_pose
   |
   v
TF2 / Eye-to-Hand
   |
   v
x5a_pick_place
   |
   v
MoveIt 2 ExecuteTrajectory
   |
   v
FollowJointTrajectory + GripperCommand
   |
   v
x5a_moveit_official_adapter
   |
   v
ARX /arm_cmd (RobotCmd)
   |
   v
official X5Controller -> CAN -> X5A
```

反馈链：

```text
X5A -> CAN -> X5Controller -> /arm_status (RobotStatus)
    -> x5a_moveit_official_adapter -> /joint_states -> MoveIt / RViz
```

`x5a_control_bridge` 是项目早期 bridge/测试实现，保留用于历史和实验参考。稳定版真机运行必须使用 `x5a_moveit_official_adapter`，禁止两个 bridge/adapter 同时运行；`/arm_cmd` publisher count 必须严格等于 1。

## 软件环境

- Ubuntu 22.04.5 LTS (x86_64)
- ROS 2 Humble
- MoveIt 2.5.9
- Intel RealSense D435i，librealsense 2.58.2
- `realsense2_camera` 4.58.2
- OpenCV / `cv_bridge` 3.2.1
- Python 3.10（ROS 2 Humble 系统 Python）

## ROS 2 packages

| Package | 用途 |
|---|---|
| `X5A` | 厂商 X5A description/mesh 与本项目 MoveIt TCP wrapper |
| `x5a_moveit_config` | MoveIt 2 模型、规划参数、控制器映射、真机 bringup |
| `x5a_moveit_official_adapter` | `/arm_status` 状态适配，FJT/Gripper action 到官方 `/arm_cmd` |
| `x5a_pick_place` | **正式**视觉/固定坐标抓放状态机 |
| `x5a_task_server` | 长期运行的 `/x5a/pick_place` Action，复用 `x5a_pick_place` |
| `x5a_web_agent` | 手机网页、语音文字、规则解析、Action Client |
| `x5a_vision` | HSV + RGB-D + TF2 的红/白/橙方块及黑色海绵中心定位 |
| `x5a_remote_grasp` | 远程 SAM 3.1 + GraspNet 候选、TF/RViz/IK/MoveIt 纯规划 |
| `x5a_handeye` | ChArUco 检测、采样、Eye-to-Hand 求解、验证与 TF 发布 |
| `x5a_mtc_pick_place` | 实验性 MTC 整链规划；日常抓取不要用 |
| `x5a_task_interfaces` | MTC 的 `PickPlace` 以及语音链路的 `X5aPickPlace` action |
| `x5a_control_bridge` | 早期实验 bridge；稳定真机链不启动 |

厂商 `arx5_arm_msg`、`arx_x5_controller` 和 SDK 保持为外部依赖，详情见 [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)。

## 获取依赖与编译

先安装 ROS 2 Humble、MoveIt 2、RealSense ROS wrapper 和 OpenCV ROS packages，并按厂商文档准备 ARX 工作区：

```bash
git clone https://github.com/ARXroboticsX/ARX_X5.git ~/repos/arx/ARX_X5
cd ~/repos/arx/ARX_X5/ROS2/X5_ws
source /opt/ros/humble/setup.bash
colcon build
```

克隆本项目并编译：

```bash
git clone https://github.com/wangfengnan521/arm.git ~/arx/arm
cd ~/arx/arm
source /opt/ros/humble/setup.bash
source ~/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## CAN

真机验证使用 CANable2、slcan、`can1`、1 Mbps（`slcand -s8`）。设备 udev alias 为 `/dev/arxcan1` 时：

```bash
cd ~/arx/arm
./scripts/setup_can_x5a.sh
```

脚本不会保存 sudo 密码，并会拒绝为同一接口启动第二个 `slcand`。启动后应看到 `can1` 为 `UP`、`ERROR-ACTIVE`。不要运行多个 `slcand` 或看门狗循环。

## 启动

以下命令分别在独立终端运行。真实运动前确认工作区无人、急停可用、CAN 正常，并确保只有一个 `X5Controller` 和一个正式适配器。

### 1. 官方 X5A 控制器

```bash
source /opt/ros/humble/setup.bash
source ~/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash
ros2 launch arx_x5_controller open_single_arm.launch.py
```

### 2. X5A + MoveIt + RViz

```bash
source /opt/ros/humble/setup.bash
source ~/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash
source ~/arx/arm/install/setup.bash
ros2 launch x5a_moveit_config x5a_real_moveit.launch.py \
  start_driver:=false start_adapter:=true use_rviz:=true
```

也可以由该 launch 启动官方 driver，此时必须保证外部 `X5Controller` 未运行，并使用 `start_driver:=true`。

### 3. RealSense / Eye-to-Hand TF / Vision

```bash
source /opt/ros/humble/setup.bash
source ~/arx/arm/install/setup.bash
ros2 launch x5a_vision vision.launch.py start_camera:=true
```

若相机已经运行，使用 `start_camera:=false`，不要启动第二个 RealSense 节点。

### 4. Pick & Place

先只规划：

```bash
ros2 launch x5a_pick_place pick_place.launch.py \
  plan_only:=true vision_enabled:=true target_color:=red
```

确认 `/arm_cmd` publisher count 为 1、现场安全检查完成后真机执行：

```bash
ros2 launch x5a_pick_place pick_place.launch.py \
  plan_only:=false vision_enabled:=true target_color:=red
```

启动日志应出现 `HANDEYE ... manual_samples_recalib_20260813b.json`。正式链禁止直接发布 `/arm_cmd`、禁止同时启动 `x5a_control_bridge`、禁止绕过 MoveIt 插值关节轨迹。

## Eye-to-Hand calibration

当前实机标定从 `src/x5a_handeye/config/handeye_result.yaml` 读取（换相机位置后于 2026-08-13 重标）：

- Board：ChArUco，`DICT_4X4_50`，5 x 7；square 0.020 m；marker 0.014 m
- Solver：PARK
- 样本：`src/x5a_handeye/data/manual_samples_recalib_20260813b.json`（28 组，用了 22）
- 训练均值 / 最大：5.12 mm / 11.22 mm
- Hold-out 均值 / 最大：4.18 mm / 5.47 mm
- 变换：`T_base_camera_color_optical`
- 标定 verdict：PASS

运行时只额外发布 `base_link -> camera_link`；RealSense 自己发布 `camera_link -> ... -> camera_color_optical_frame`。禁止给光学系再挂第二个 parent。改过 `handeye_result.yaml` 后必须重启 `vision.launch.py`，因为 install 里的 yaml 在非 symlink 情况下不会自动更新。

相机或支架移动后必须重标。重力采样（先停 adapter / pick_place，保证 `/arm_cmd` publisher 为 0）：

```bash
# 终端 1：重力模式，按 g 拖动，h 锁住
source /opt/ros/humble/setup.bash
source ~/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash
source ~/arx/arm/install/setup.bash
ros2 run x5a_handeye gravity_teach
```

```bash
# 终端 2：检测板 + 采样（新文件，不要和旧批次混）
cd ~/arx/arm
source /opt/ros/humble/setup.bash
source install/setup.bash
bash src/x5a_handeye/scripts/standard_handeye_pipeline.sh start_prereq
SAMPLES="$PWD/src/x5a_handeye/data/manual_samples_recalib_YYYYMMDD.json" \
  bash src/x5a_handeye/scripts/standard_handeye_pipeline.sh sample
SAMPLES="$PWD/src/x5a_handeye/data/manual_samples_recalib_YYYYMMDD.json" \
  bash src/x5a_handeye/scripts/standard_handeye_pipeline.sh solve
```

求解须 `verdict: PASS` 后再重启视觉。标定板夹死、每点停稳再按 `s`，并覆盖工作区 X/Y 与多种板朝向。

## Visual Pick & Place

15 帧 median/std 过滤；方块和海绵中心同时冻结，本轮不再跟视觉漂移。流程：

1. `MOVE_READY`：`joint1 = atan2(y, x)`，其余用 `pre_grasp_ready_joints`
2. 预抓取位，接近/接触/短抬起保持同一抓取姿态（默认 RPY `[0, 1.45, 0]`）
3. 放置尝试多组朝向和多种子 IK
4. 松开、后退、回 Home

当前速度缩放（`pick_place.yaml`）：transit `0.75 / 0.38`，接近放下 `0.28 / 0.16`，抬起后退 `0.35 / 0.20`。适配器关节速度上限仍是 `1.0 rad/s`。

## 参数修改

- 物块尺寸、HSV、深度和 workspace：`src/x5a_vision/config/vision.yaml`
- 抓取 offset、物块尺寸、table、place pose、速度：`src/x5a_pick_place/config/pick_place.yaml`
- FollowJointTrajectory/夹爪适配器频率与保护：`src/x5a_moveit_official_adapter/config/adapter.yaml`
- 手眼外参：`src/x5a_handeye/config/handeye_result.yaml`
- ChArUco board：`src/x5a_handeye/config/board.yaml`
- MoveIt joint limits/kinematics/controller：`src/x5a_moveit_config/config/`

当前实机验证的抓取 `z_offset` 为 0.040 m。除非有标定证据，不应通过修改手眼 TF 凑抓取结果。

## 已知限制

- 检测器针对已知 HSV 的小方块，不是通用物体识别。
- `tool0` 是标定和规划使用的 TCP。
- joint limits 来自 X5 软件限位和本机实测，不是完整机械范围认证。`joint4` 主动伺服下限约 `-1.28 rad`。
- 正式桥是项目实现的 FJT / Gripper 到 ARX `RobotCmd` 适配器，不是完整 `ros2_control` HardwareInterface。
- 视觉工作区约 `x∈[0.08,0.41]`、`y∈[0.08,0.48]`、`r≤0.54 m`。更远的近竖直抓取会受臂长和 `joint4` 限制。
- `x5a_mtc_pick_place` 仍是实验包；整链 MTC 在远放置点上不如 Python 状态机稳定。
- 相机 serial 写在 `x5a_vision/launch/vision.launch.py`；换相机要改参数。
- 真机必须有熟悉机械臂、CAN 和急停的人现场监护。
