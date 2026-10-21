# X5A 腕部 RGB-D 语言抓取与放置

面向 **ARX X5A 单臂、原厂平行夹爪、腕部 Orbbec Gemini 345Lg、ROS 2 Humble / MoveIt 2** 的工程工作区。主要流程为 **SAM 文本分割 → 本地深度筛选 → GraspNet 抓取候选 → TRAC-IK / MoveIt 规划 → 独立执行器**。另保留本地颜色方块抓放。

本分支整理自 `wrist_ws`，与原固定相机工程使用不同的手眼模型和命名空间。不要混用两套工作区的外参和构建产物。

> **验证状态（2026-10-17）：** 已有实机抓放记录，但复杂叠放仍会失败。最新方块换面通过部分保存场景的离线规划，尚未实机验收。配置是本机实验配置，标定的严格 3 mm 验收仍为 FAIL。尚无统一场景、足够样本的抓取成功率统计。

## 功能与入口

| 项目 | SAM 腕部模式 | 本地颜色模式 |
| --- | --- | --- |
| 定位 | 云端 SAM 文本分割 + 本地深度 | HSV 红 / 白 / 橙方块及盒沿 |
| 抓姿 | GraspNet 原生候选 + 经检查的几何备用姿态 | 本地方块策略 |
| 求解与规划 | TRAC-IK Speed、4 个 IK worker、OMPL RRTConnect | 默认 KDL、MoveIt |
| 交互 | 分开抓放、整句抓放、多步骤队列 | 持物颜色与代词承接、最多 8 步连续抓放、观察侧、清障及运行内场景记忆 |
| 启动 | `bash scripts/start_sam.sh` | `bash scripts/start.sh` |
| 不执行动作的检查 | `bash scripts/start_sam.sh --plan-only`：推理与规划 | `bash scripts/start.sh --dry-run`：仅交互，不规划 |
| 页面 | HTTP 8000 / HTTPS 8443 | HTTP 8001 / HTTPS 8444，另提供旧端口兼容 |

颜色流程已增加口语同义词、三色顺序收纳、持物/识别查询、叠放四方向搜索，并优化速度、预抓取候选和 5 mm 桌面碰撞余量，详见[颜色优化与指令示例](docs/COLOR_WORKFLOW.md#2026-10-20-速度与容差优化)。

两种模式共用进程锁，不能同时控制机械臂。SAM 没有移植颜色流程中的镜像准备、自动清障和叠放记忆。详见 [SAM 说明](docs/SAM_WRIST.md)、[颜色流程](docs/COLOR_WORKFLOW.md)。

## 系统流程

```mermaid
flowchart LR
    A[文字 / 手机录音] --> B[本地 SenseVoice 转写]
    B --> C[指令分步 + 云端语义解析]
    A --> C
    C --> D[同步 RGB-D + 曝光时刻 TF]
    D --> E[SAM 掩码 + 本地深度筛选]
    E --> F[GraspNet 抓取候选]
    F --> G[几何检查 + TRAC-IK + MoveIt]
    G --> H[完整阶段计划]
    H --> I[独立执行器 / 轨迹适配器]
    I --> J[厂商 SDK / CAN / 机械臂]
```

SAM 决定目标像素，GraspNet 预测候选抓姿，MoveIt 对应用给定的目标进行求解与避障。模型分数高不代表可达；换抓取面、换放置面、候选排序和失败恢复需要应用明确实现。

## 安装

### 1. 环境和外部依赖

已验证主机为 Ubuntu 22.04、x86_64、ROS 2 Humble。先安装 ROS 2，再安装构建与运行依赖，例如：

```bash
sudo apt install git build-essential python3-colcon-common-extensions python3-rosdep \
  python3-pytest python3-numpy python3-scipy python3-opencv python3-yaml \
  python3-fastapi python3-uvicorn python3-multipart python3-requests \
  python3-paho-mqtt python3-pil python3-wsproto python3-pip curl openssl libssl-dev \
  ros-humble-moveit ros-humble-cv-bridge ros-humble-tf2-geometry-msgs \
  ros-humble-realsense2-camera
```

`realsense2_camera` 用于兼容包依赖，腕部实际使用 Orbbec。需另装匹配的 `orbbec_camera`，确认存在 `launch/gemini345_lg.launch.py`。

按 [ARX_X5 官方工程](https://github.com/ARXroboticsX/ARX_X5) 编译 `ROS2/X5_ws`，提供 `arx5_arm_msg`、厂商头文件及 `libarx_x5_src.so`。完整 SDK、相机驱动、模型权重和 CUDA 算子不包含在仓库内。

**腕部驱动兼容层只支持已校验的 x86_64 SDK ABI**。构建和运行均校验 SDK SHA-256：

```text
d07d53d14da17f0a1db2753e3339f9e4c055dad72df613c811b9692c4bede85c
```

不同 SDK 需重新核验兼容层；不能通过删除哈希检查解决兼容问题。实现见 [驱动目录](src/x5a_wrist_driver)。

### 2. 获取腕部版本

```bash
mkdir -p ~/arx
git clone --branch wrist-sam https://github.com/chrish0226/x5a-graspnet.git ~/arx/wrist_ws
cd ~/arx/wrist_ws
```

### 3. 修改本机配置

仓库保留原实机参数。换用户、目录、机械臂或相机时，构建前核对：

| 文件 | 必须核对的内容 |
| --- | --- |
| `scripts/env.bash` | ROS / 厂商 overlay；默认厂商路径 `/home/quella/repos/arx/ARX_X5/ROS2/X5_ws` |
| `src/x5a_wrist_driver/CMakeLists.txt` | `VENDOR_ROOT`、`SDK` 路径及 SDK ABI |
| `scripts/start_sam.sh` | 私有密钥文件路径，默认 `/home/quella/.config/x5a_remote_grasp/key.env` |
| `config/calibration/handeye_model_candidate.yaml` | `robot_calibration.candidate_urdf` 改为本克隆中的 URDF；外参只适用原安装 |
| `config/task_plan.yaml`、`config/task_execute.yaml` | 标定结果路径、观察位和允许关节范围 |
| `config/sam_grasp.yaml` | 模型服务地址、抓帧目录、工作空间、桌面与相机模型、准备位 |
| `config/driver.yaml` | CAN 接口，原机为 `can1`；按厂商方式预先配置 CAN |
| `src/x5a_handeye/launch/eye_in_hand.launch.py` | 相机序列号和驱动入口 |

`/home/quella/arx/wrist_ws` 是原工作区的绝对路径，换目录必须更新。`reachability_table` 指向未提供的离线表，不应视为已标定的可达性数据。原始标定图像未上传；保存的验证报告不能替代新安装上的重新标定。

### 4. 构建和语音模型

```bash
source scripts/env.bash
rosdep install --from-paths src --ignore-src -r -y
bash scripts/setup_trac_ik.bash
bash scripts/build.sh
source scripts/env.bash
```

TRAC-IK 脚本下载固定提交、应用 Humble 补丁，在 `.deps/trac_ik_ws` 内构建，不替换系统插件。需要网络和可用 apt 索引。使用 `/usr/bin/python3`，避免 Conda 环境混入 ROS。

手机语音另需安装 ASR；仅用文字可跳过：

```bash
bash scripts/install_asr.sh
```

它下载 SenseVoiceSmall ONNX INT8，把固定版本 sherpa-onnx 安装到 `vendor/asr/`，不替换 ROS NumPy。

### 5. 云端模型和密钥

SAM、GraspNet、Ollama 在 GPU 服务器运行，部署和接口见 [server/README.md](server/README.md)。使用现有部署时，确认健康状态并配置自己的服务地址。

客户端通过 `SAM_GRASPNET_API_KEY` 读取密钥。可在启动脚本指定的私有 `key.env` 中设置：

```bash
export SAM_GRASPNET_API_KEY='<替换为服务密钥：64 位十六进制>'
```

解析地址由 `X5A_CLOUD_PARSE_URL` 控制；迁移服务器时需同时修改它和 `config/sam_grasp.yaml` 的 `remote.base_url`：

```bash
export X5A_CLOUD_PARSE_URL='http://<模型服务器>:8300/parse'
```

配置里的 IP 是原部署地址，不是随克隆提供的公共服务。密钥、证书私钥和模型权重均未上传。

## 启动和使用

确认原 SAM/颜色启动终端已退出，CAN、相机正常后运行：

```bash
cd ~/arx/wrist_ws
bash scripts/start_sam.sh
```

默认启动只提供服务，收到指令后才运动。手机访问 `http://<机械臂主机IP>:8000`；按住录音使用 `https://<机械臂主机IP>:8443`，首次需接受本机证书并允许麦克风。横幅中的 `192.168.0.50` 是原主机地址，按实际 IP 访问。

独立文字终端：

```bash
cd ~/arx/wrist_ws
bash scripts/text_sam.sh
```

典型指令：

```text
准备
抓红色方块
放到白色方块上
把白色方块放到橙色方块上，然后把红色方块放到白色方块上
把瓶子放进盒子里
回零
```

“抓起后另行指定放置”保留持物状态；队列逐步解析、观察、规划和执行，任一步失败即停止。SAM 目前只配置当前侧观察位。回零、松爪和停止应结合当前持物状态操作。

- `bash scripts/start_sam.sh --plan-only`：推理与规划，准备、回零、抓取、松爪均不执行。
- `bash scripts/start_sam.sh --verbose`：终端展开完整底层日志。
- `bash scripts/stop_sam.sh --check`：只检查当前工作区的相关进程。
- `bash scripts/stop_sam.sh` 或启动终端 `Ctrl+C`：结束本地程序，不停止云端服务。

终端显示任务编号、步骤、阶段、耗时与持物状态；底层日志在 ROS 日志目录（默认 `~/.ros/log/`）。重复的 `MULTI DETECTION` 已降为 DEBUG。网页取消和程序退出不等同硬件急停。持物几何主要在进程内维护，持物时重启后不能假定新进程自动恢复原状态。

## 当前规划与优化

**每一步先规划完整阶段，再执行。** 单抓为 `PREGRASP → GRASP → LIFT → RETREAT`；整句抓放为六段，含 `TRANSPORT / PLACE`；持物纯放置为 `TRANSPORT → PLACE → RETREAT`。多步骤队列逐步规划，不在开局一次计算整条队列。

| 项目 | 当前配置 / 行为 |
| --- | --- |
| IK / 规划器 | TRAC-IK Speed / OMPL RRTConnect；另加载 Pilz、CHOMP，切换仍需验证 |
| 候选规划预算 | `planning.total_budget_s: 8.0`；不包含全部采图、语义解析、网络和执行时间，也不是整个请求的硬超时 |
| 超时后重新观察 | 空手抓取耗尽规划预算时，可尝试一次独立完整的 OBSERVE 路径，靠近约 3–6 cm，再取新图规划；持物纯放置不适用 |
| 前缀复用 | 同一次观测内复用已检查的阶段前缀；新观测重新计算 |
| 接近 / 放置 | 笛卡尔路径完成比例至少 0.9999，并核对 FK 终点；不执行失败计划的残缺片段 |
| 静态执行 | `grasp.pre_execution_visual_recheck: false`，执行前不再调用 SAM；保留指纹、时效、反馈和执行状态检查 |
| 桌面接触 | 仅两指 `link7/8` 有 5 mm 模型接触容差，下方另设边界；本体、腕部和相机仍检查碰撞 |
| 方块对齐 | 支撑中心、夹持偏移和顶面边方向；叠放中心偏移限约 3 mm，释放间隙尝试 10–20 mm |
| 方块换面 | 可信的约 20–55 mm 小立方体枚举最多 24 种面/边等价姿态，各自重新检查完整路径；其他物体保留原朝向约束 |
| 平滑松爪 | 约 0.65 s 渐开，再保持/验证约 0.8 s |
| 精度容忍 | 额外 TCP 误差为提示；小关节最终残差在有响应且稳定后可按 0.06 rad 容差继续；硬范围、跟踪和反馈保护仍有效 |

当前 `gripper.model_path` 为空，使用原厂 URDF/CAD 碰撞网格和原简化接触参数。[共享夹爪模型](models/README.md)是保留但关闭的实验选项。

## 验证和限制

与本分支相关的回归命令，使用独立 DDS 域：

```bash
source scripts/env.bash
X5A_TEST_ROS_DOMAIN_ID=157 /usr/bin/python3 -m pytest -q \
  src/x5a_remote_grasp/test src/x5a_web_agent/test src/x5a_pick_place/test tests \
  src/x5a_vision/test src/x5a_handeye/test --disable-warnings
```

2026-10-21 颜色续接指令修复后的完整回归 **708 项通过**。`conftest.py` 从 `X5A_TEST_ROS_DOMAIN_ID` 设置测试域，仅设置 `ROS_DOMAIN_ID` 会被覆盖。保存场景回放、性能测量及实机验证边界见 [验证记录](docs/VALIDATION.md)。

- 标定留出集最大误差约 4.7 mm，严格 3 mm 验收失败；棱角对齐尚无高精度实机堆叠保证。
- SAM 持物点云尚未作为完整附着碰撞体覆盖所有运动阶段，其他物体的场景建模也不完整。完整轨迹不代表所有物体与环境接触都已验证。
- 静态执行要求物品保持静止；遮挡、透明/反光物体、缺深度和未观测障碍仍可能导致失败。
- 分开抓放时，当前抓姿未必适合后续放置；24 姿态枚举不能解决所有碰撞与臂展问题。
- 持物判断主要依赖夹爪开度/反馈，不是力或视觉闭环证明；薄长工具、空抓和滑落仍需验证。
- 云端可能误解析 box/cube、颜色或关系；先核对解析、掩码和错误阶段，再区分感知、IK、碰撞与执行问题。

## 目录

```text
config/                         本机参数、标定候选、TRAC-IK 补丁
scripts/                        构建、启停、文字输入、只读检查
src/x5a_remote_grasp/           云端接口、几何筛选、规划、独立执行器
src/x5a_parallel_ik/            同模型的独立 IK worker
src/x5a_wrist_driver/           经过 SDK ABI 校验的腕部驱动
src/x5a_moveit_official_adapter/  MoveIt 轨迹适配
src/x5a_web_agent/              手机页面、语音、队列和日志
src/x5a_handeye/                标定、模型检查和腕部 TF
src/x5a_vision/                 RGB-D、颜色检测、顶面/盒沿
src/X5A/                       机器人描述和必要网格
server/                        SAM、语义网关、GraspNet 服务源码
android/                       可选原生手机端源码
models/                        模型说明和未启用的夹爪实验资源
docs/                          使用、验证和第三方说明
tests/                         跨模块回归
```

`.gitignore` 排除构建目录、日志、原始标定数据、模型权重、APK、密钥、私有环境文件和本地报告备份。Android 需自行编译，仓库不提供原设备签名和 APK 下载服务。

## 参考与许可

- [GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)：RGB-D 候选抓姿与碰撞筛选。
- [SAM 3](https://github.com/facebookresearch/sam3)：文本提示分割上游；当前服务适配已部署的 SAM 3.1 multiplex checkpoint，需匹配模型/API 版本。
- [TRAC-IK](https://github.com/traclabs/trac_ik)：IK 插件，固定提交和补丁见安装脚本。
- [MoveIt Planning Scene](https://moveit.picknik.ai/humble/doc/examples/planning_scene/planning_scene_tutorial.html)：碰撞场景和接触策略。
- [公开路线调研](docs/TABLETOP_ROBOT_ROUTES_20261016.md)、[第三方依赖](docs/THIRD_PARTY.md)。

沿用 [BSD-3-Clause](LICENSE)，保留原工程及厂商版权声明。外部 SDK、模型和依赖遵循各自许可。
