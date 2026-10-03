# X5A 手机语音视觉抓取

在已经能真机抓取红 / 白 / 橙方块的项目上，增加「手机说话 → 主机执行一次抓放」的 Demo。

第一版不用大模型。语音在手机浏览器转成文字，主机用关键词解析成：

```json
{"action": "pick_place", "target_color": "red"}
```

然后只调用现有 MoveIt 抓取链。

## 参考过的开源项目

这些项目说明「语音 + 机器人」常见，但都不能直接套到本仓库的安全控制链上：

| 项目 | 可借鉴 | 为什么不直接用 |
|---|---|---|
| [imnuman/robotic-arm-ros2](https://github.com/imnuman/robotic-arm-ros2) | ROS 2 Humble + 视觉 + 语音抓取 | 语音在主机麦克风，控制链不同 |
| [aniskoubaa/rosgpt](https://github.com/aniskoubaa/rosgpt) | 自然语言 → 结构化任务 | 容易让 LLM 直接发 ROS 命令，这里明确禁止 |
| [RobotWebTools/ros2-web-bridge](https://github.com/RobotWebTools/ros2-web-bridge) | 浏览器 WebSocket | 浏览器不应直接碰到 `/arm_cmd` |
| [hucebot/fast_whisper_ROS2](https://github.com/hucebot/fast_whisper_ROS2) | 服务器 ASR | 作为以后的方案 B，第一版用手机 Web Speech API |
| [victorcarvesk/ros-voice-assistant](https://github.com/victorcarvesk/ros-voice-assistant) | ROS 2 语音包分层 | 依赖主机麦和云端 STT |

本仓库采用：手机 ASR → FastAPI → 白名单 JSON → ROS 2 Action → 现有 `x5a_pick_place`。

## 现有代码可直接复用

不要重写这些：

- `src/x5a_pick_place/x5a_pick_place/pick_place_node.py` 里已经验证的抓取流程
- `src/x5a_vision` 红 / 白 / 橙 + 黑色海绵定位
- `src/x5a_moveit_official_adapter` 唯一的 `/arm_cmd` publisher
- `src/x5a_handeye` TF
- `src/x5a_moveit_config` MoveIt
- 现有 `run_x5a_vision_pick.sh` 一键单次抓取

`src/x5a_task_interfaces/action/PickPlace.action` 被实验性 MTC 使用，字段是 `color`，**不要改**。语音链路使用新增的 `X5aPickPlace.action`（字段 `target_color`）。

## 修改 / 新增文件

修改：

- `src/x5a_pick_place/x5a_pick_place/pick_place_node.py`
- `src/x5a_task_interfaces/CMakeLists.txt`
- `README.md`
- `.gitignore`

新增：

- `src/x5a_task_interfaces/action/X5aPickPlace.action`
- `src/x5a_task_server/`
- `src/x5a_web_agent/`
- `run_x5a_voice_demo.sh`
- `docs/VOICE_INTERACTION.md`

后期追加（终端文字测试）：

- `src/x5a_web_agent/x5a_web_agent/terminal.py`（终端输入层，键盘 = 转写文本）
- `src/x5a_web_agent/x5a_web_agent/text_cli.py`（独立进程版，走 agent 的 HTTP 接口）
- 改 `src/x5a_web_agent/x5a_web_agent/app.py`（`--terminal`、hub 观察者、`_request_cancel()`）
- 改 `src/x5a_web_agent/setup.py`（`text_cli` 入口）和 `run_x5a_voice_demo.sh`（`--terminal`）

## 最终调用链

```text
手机浏览器（Web Speech API 或文字 / 颜色按钮）
        ↓  text / WebSocket
FastAPI  x5a_web_agent.app
        ↓
command_parser.RuleBasedCommandParser
        ↓  {"action":"pick_place","target_color":"red"}
ros_bridge.RosBridge   （Action Client only）
        ↓
ROS 2 Action  /x5a/pick_place
        ↓
x5a_task_server   busy lock + feedback.stage
        ↓
PickPlaceNode.execute_pick_place("red")
        ↓  清空上一轮 cube/box 缓存，重新等视觉稳定并冻结
现有 MoveIt 规划 / 执行 / 夹爪
        ↓
x5a_moveit_official_adapter
        ↓
/arm_cmd   （仍然只有这一个 publisher）
        ↓
ARX X5A
```

## 1. 安装依赖

优先用 Ubuntu 22.04 系统包（对应 ROS Humble 的 Python 3.10）：

```bash
sudo apt update
sudo apt install -y python3-pip python3-fastapi python3-uvicorn python3-websockets
```

或者用 pip：

```bash
sudo apt install -y python3-pip
/usr/bin/python3 -m pip install --user -r ~/arx/arm/src/x5a_web_agent/requirements.txt
```

`requirements.txt`：

```
fastapi>=0.63
uvicorn>=0.15
websockets>=9.1
```

## 2. 编译

```bash
cd ~/arx/arm
source /opt/ros/humble/setup.bash
source ~/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash
colcon build --symlink-install \
  --packages-select x5a_task_interfaces x5a_pick_place x5a_task_server x5a_web_agent \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

改了 action 定义后必须先编译 `x5a_task_interfaces`。

## 3. 启动顺序

**不要**同时启动：

- `x5a_pick_place` 的一次性 `pick_place_node`
- `x5a_task_server`
- `x5a_mtc_task_server`

也不要同时启动两个 `/arm_cmd` publisher。

### 终端 1：CAN + 官方控制器

```bash
source /opt/ros/humble/setup.bash
source ~/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash
ros2 launch arx_x5_controller open_single_arm.launch.py
```

### 终端 2：MoveIt + official adapter

```bash
source /opt/ros/humble/setup.bash
source ~/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash
source ~/arx/arm/install/setup.bash
ros2 launch x5a_moveit_config x5a_real_moveit.launch.py \
  start_driver:=false start_adapter:=true use_rviz:=false arm_can_id:=can1
```

确认：

```bash
ros2 topic info /arm_cmd -v
# Publisher count: 1
# Node name: x5a_official_trajectory_adapter
```

### 终端 3：RealSense + 手眼 TF + 视觉

```bash
source /opt/ros/humble/setup.bash
source ~/arx/arm/install/setup.bash
ros2 launch x5a_vision vision.launch.py start_camera:=true
```

### 终端 4：长期运行 Task Server

plan_only 验证：

```bash
source /opt/ros/humble/setup.bash
source ~/arx/arm/install/setup.bash
ros2 launch x5a_task_server task_server.launch.py \
  plan_only:=true vision_enabled:=true
```

真机：

```bash
ros2 launch x5a_task_server task_server.launch.py \
  plan_only:=false vision_enabled:=true
```

### 终端 5：Web 服务

只测网页 / 解析，不发 Action：

```bash
source /opt/ros/humble/setup.bash
source ~/arx/arm/install/setup.bash
ros2 run x5a_web_agent web_agent --host 0.0.0.0 --port 8000 --dry-run
```

联调 Task Server：

```bash
ros2 run x5a_web_agent web_agent --host 0.0.0.0 --port 8000
```

手机语音（Web Speech API 通常要求 HTTPS）：

```bash
ros2 run x5a_web_agent web_agent --host 0.0.0.0 --port 8000 --https
```

前三步已经在跑时，也可以：

```bash
cd ~/arx/arm
./run_x5a_voice_demo.sh --plan-only
./run_x5a_voice_demo.sh --https
./run_x5a_voice_demo.sh --terminal
```

### 终端 5（可选）：终端文字输入层

不想对着手机说话时，用键盘打字测试。这一层只替换掉「语音转文字」那一步，后面
的云端解析、白名单、数字人握手、Action 调用和手机说话完全一样，也没有新增
任何 `/arm_cmd` 出口。

方式 A，加在 agent 进程里（要和 web_agent 一起启动；已有 web_agent 在跑就先停掉它，
否则 8000 端口会冲突）：

```bash
ros2 run x5a_web_agent web_agent --host 0.0.0.0 --port 8000 --terminal
# 或
./run_x5a_voice_demo.sh --terminal
```

方式 B，独立进程接入已经在跑的 agent（**不用重启 agent**，它只走 `/api/command`、
`/api/task`、`/api/cancel`、`/api/state`、`/api/health` 这些本来就有的接口）：

```bash
./scripts/text_cli.sh                                  # 自带环境，默认 127.0.0.1:8000
./scripts/text_cli.sh --url http://192.168.0.50:8000
# 等价的原始写法（要先 source ROS + 本工作空间）
ros2 run x5a_web_agent text_cli --url http://127.0.0.1:8000
```

⚠️ 新开的终端只 source 了 `/opt/ros/humble`（fishros 那段）时，`ros2 run` 会报
`Package 'x5a_web_agent' not found`——补一句
`source ~/arx/arm/install/setup.bash`，或者直接用 `scripts/text_cli.sh`（它 source 了
`scripts/stack_env.sh`，顺手屏蔽 conda）。

两种方式都是同一个交互：

```text
x5a> 抓起红色方块                 # 回车发送，等同手机说这句话
[解析] {"action": "remote_grasp", "prompt": "red cube", ...}
[阶段] 任务已接受
[阶段] 正在寻找目标
...
[阶段] 任务完成
x5a> /state
[状态] 阶段=任务完成 忙=否 手上=空 task_server=就绪
```

命令：

| 命令 | 作用 |
|---|---|
| 直接输入文字 | 走云端解析 + 语音那条链路（真机模式下会真的动） |
| `/help`、`/?` | 帮助 |
| `/state` | 打印 agent 状态（阶段 / 忙闲 / 手上物体 / task_server） |
| `/cancel` | 取消当前任务，等同手机页「停止」 |
| `/task red` | 跳过云端解析，直接发结构化任务（red / white / orange / nearest / farthest） |
| `/parse <文字>` | 只跑本地规则解析器看 JSON，不发任何任务；离线也能用 |
| `/quit`、`/exit`、`/q` | 退出输入层，agent 进程继续跑 |

要点：

- 终端层不是新的控制通道。它调的还是 `handle_text_command()` / `/api/command`，
  所以「手上还拿着东西」「机器人正在执行上一任务」这些拒绝逻辑照样生效。
- 结果行会带上任务标签（`[结果] 抓取「black pen」: ...`）。真机链路是异步的：
  结果可能在你敲下一条指令之后才回来，没有标签就会误以为是新指令的结果。
- ⚠️ `[阶段] 任务完成` 不能当成「你刚发的那条完成了」的证据：
  `x5a_task_server` 每 0.5 秒重发一次它的最后阶段（`task_server_node.py` 的
  `create_timer(0.5, self._republish_status)`），agent 的 `_apply_stage` 会照单
  全收，所以任务进行中显示也会被刷成「任务完成」（手机页同样如此）。以带标签的
  `[结果]` 行为准。
- 真机模式下打进去的每一句都会真的动机械臂。只想验证解析时用 `--dry-run`
  （方式 A）或只敲 `/parse`、`/task`（方式 B 走的是真 agent）。
- 阶段和结果两行是 hub 广播（方式 A）或轮询 `/api/state`（方式 B，默认 1 秒一次，
  可用 `--poll-interval` 调整）打出来的，所以和手机页看到的是同一份状态。
- `/quit` 只关掉输入层；Ctrl-C 才停整个 agent。

## 4. 手机访问

1. 手机和主机连同一个局域网。
2. 主机查看 IP：`hostname -I`
3. 手机浏览器打开：
   - `http://<robot-ip>:8000`
   - 或 `https://<robot-ip>:8000`（自签名证书，手机上选继续访问）
4. 推荐手机 Chrome。HTTP 下如果不能说话，用文字框或红/白/橙按钮。

## 5. 测试顺序

### 测试 1：网页能发文字

Web 用 `--dry-run`。手机输入「抓红色」，页面应显示识别文字。

```bash
curl -s http://127.0.0.1:8000/api/health
```

### 测试 2：解析 JSON

```bash
/usr/bin/python3 ~/arx/arm/src/x5a_web_agent/x5a_web_agent/command_parser.py

curl -s -X POST http://127.0.0.1:8000/api/parse \
  -H 'Content-Type: application/json' \
  -d '{"text":"帮我抓一下红色的"}'
```

期望：

```json
{"action":"pick_place","target_color":"red"}
```

### 测试 3：Web → Action，机械臂不动

Task Server：`plan_only:=true`  
或 Web：`--dry-run`

### 测试 4：手动发 Action

```bash
ros2 action list | grep x5a
ros2 topic echo /x5a/task_stage

ros2 action send_goal /x5a/pick_place x5a_task_interfaces/action/X5aPickPlace \
  "{target_color: red}" --feedback
```

忙碌时再发一次，应被拒绝。

### 测试 5：plan_only 视觉规划

桌面上放红色方块和黑色海绵，确认日志出现 `VISION PICK AND PLACE PLAN: PASS`。

### 测试 6：真机低风险

周围清空，`plan_only:=false`，先按页面「红色」按钮，不要先说话。

### 测试 7：完整语音

手机说：「帮我抓红色的」

期望页面依次显示：

正在寻找目标 → 已定位目标 → 正在移动到抓取位置 → 正在抓取 → 正在放置 → 任务完成

完成后可以说「抓白色」，必须重新定位，不能沿用上一轮红色坐标。

### 测试 8：终端文字（不用手机、不用麦克风）

Web 用 `--dry-run` 时，开一个终端层直接打字：

```bash
./scripts/text_cli.sh                                  # 接入已经跑着的 agent
# 或者单独起一个 dry-run agent + 输入层（真机栈在跑时用别的端口，别抢 8000）
ros2 run x5a_web_agent web_agent --host 127.0.0.1 --port 8099 --no-https --dry-run --terminal
```

期望：敲「抓红色」后出现 `[解析] ...`、`[阶段] 任务已接受`，dry-run 下最后是
`[阶段] 演练模式：未发送运动指令`；敲 `/parse 抓红色` 只打 JSON 不动机械臂。

## 6. 常见错误

| 现象 | 处理 |
|---|---|
| 网页打不开 | 主机防火墙、IP 是否同一网段、服务是否绑 `0.0.0.0` |
| 不能说话 | 改 HTTPS，或用文字/颜色按钮 |
| `Task Server 未就绪` | 先启动 `x5a_task_server`，不要只开 Web |
| `机器人正在执行上一任务` | 等当前任务结束；BUSY 锁是故意的 |
| 第二次仍抓红色 | 确认走的是 task_server 而不是旧的一次性 `pick_place_node` |
| `/arm_cmd` publisher ≠ 1 | 立刻停掉多余 adapter / bridge |
| 同时有 pick_place 和 task_server | 只留 task_server |
| 视觉超时 | 对应颜色方块要在视野内且稳定；默认等待 30 s |
| TF 失败 | 确认 `x5a_handeye_tf` 在发 `base_link → camera_link` |
| MoveIt 规划失败 | 当前任务终止，节点不退出，可再发下一条 |
| 终端输入层敲字没反应 | agent 是不是用 `--terminal` 启动的；如果 agent 的 stdin 是 `/dev/null`、systemd 或 `&` 后台启动，输入层会立刻 EOF 退出 → 改用 `text_cli` 接 |
| `text_cli` 报连不上 | `--url` 要和 agent 的 host/port 对上；agent 只绑 `127.0.0.1` 时不能在别的机器上连 |
| `Package 'x5a_web_agent' not found` | 终端没 source 工作空间：`source ~/arx/arm/install/setup.bash`，或直接用 `scripts/text_cli.sh` |
| 终端说「手上还拿着…」 | agent 的会话镜像认为夹爪里有东西（可从执行器 resync 得到），先「放下」或重启 web_agent 重新对齐 |
| 终端里 `[解析]` 是 `unknown` | 云端网关不通：真机模式下检查 `X5A_CLOUD_PARSE` 与 `X5A_CLOUD_PARSE_URL`，离线测试用 `/parse`（本地规则解析器） |

「停止当前任务」只取消任务层，等当前 MoveIt 轨迹走完后停止。它不是硬件急停。

## 7. 以后接 LLM

`command_parser.py` 里已预留 `LLMCommandParser`。

替换方式：

1. 只让模型输出固定 JSON Schema。
2. 必须经过 `LLMCommandParser.validate_model_json()` / `sanitize_task()`。
3. 模型只能决定 `pick_place + target_color`。
4. 禁止模型输出关节角、轨迹、`/arm_cmd`、shell、`ros2 topic pub`、Python 代码。

运动安全仍然只由现有 ROS 2 / MoveIt 决定。

## 8. 以后做「先红色再白色」

解析器已经能把

「先抓红色，再抓白色」

解析成：

```json
{
  "action": "sequence",
  "tasks": [
    {"action": "pick_place", "target_color": "red"},
    {"action": "pick_place", "target_color": "white"}
  ]
}
```

第一版会提示：一次只执行一个任务。

以后要串行执行时，在 `app.py` 的 `dispatch_parsed()` 里对 `tasks` 逐个调用 `ros_bridge.send_pick_place()`。Task Server 的 BUSY 锁保证不会并发两个 MoveIt 任务。不要在 LLM / Web 层自己规划轨迹。
