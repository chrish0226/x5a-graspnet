# X5A 手机语音抓取 Demo

> 腕部工作区的当前入口与两套模式见[主 README](../../README.md)、[纯颜色流程](../../docs/COLOR_WORKFLOW.md)和[SAM 流程](../../docs/SAM_WRIST.md)。下文保留原固定相机/数字人联动接口说明；当前腕部启动脚本默认关闭 MQTT 与文字转发。

手机当麦克风，Ubuntu 主机跑 Web + ROS 2 Task Server，复用已经验证过的 `x5a_pick_place`。

```text
手机浏览器
  → FastAPI / WebSocket
  → command_parser（规则解析）
  → ros_bridge（只发 ROS 2 Action）
  → /x5a/pick_place
  → x5a_task_server
  → x5a_pick_place.execute_pick_place()
  → MoveIt 2
  → x5a_moveit_official_adapter
  → /arm_cmd
  → ARX X5A
```

Web / ASR / 自然语言层 **不会** 发布 `/arm_cmd`，也不会直接控制关节。

## 安装 Python 依赖

使用 ROS 2 Humble 的系统 Python 3.10：

```bash
sudo apt install -y python3-pip python3-fastapi python3-uvicorn python3-websockets
# 或者
/usr/bin/python3 -m pip install --user -r ~/arx/arm/src/x5a_web_agent/requirements.txt
```

## 编译

```bash
cd ~/arx/arm
source /opt/ros/humble/setup.bash
source ~/repos/arx/ARX_X5/ROS2/X5_ws/install/setup.bash
colcon build --symlink-install \
  --packages-select x5a_task_interfaces x5a_pick_place x5a_task_server x5a_web_agent \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

腕部启动、测试和排错见仓库根目录 [README.md](../../README.md)。

## 把识别原文发到远程服务器

语音识别成功后，主机按照《交互式机械臂联调接口表 V1.3》只发送最终中文原文，不发送音频、解析结果或 partial/interim 文本：

```http
POST http://118.253.179.80:8210/api/v1/interaction/text
Content-Type: application/json; charset=utf-8

{"text":"机械臂向右移动一点"}
```

默认使用 8210 端口。可通过环境变量改目标地址：

```bash
export X5A_TEXT_FORWARD_URL=http://118.253.179.80:8210/api/v1/interaction/text
```

数字人正常受理返回 HTTP 200；参数错误返回 400；已有联动任务返回 409。关掉转发：

```bash
export X5A_TEXT_FORWARD=0
```

联动握手（文字 / 语音指令，不含网页颜色按钮和 `--dry-run`）：

1. 主机 POST 最终中文原文，必须收到 HTTP 200。
2. 数字人先播报「正在执行中」。
3. **只有这段前置播报真正播放完成后**，数字人才向 MQTT 发布开始标志，并把本地 `waiting_finish` 置为 true：

```text
Topic: robot_arm/task_busy
Payload: true
QoS: 1
Retain: false
```

4. 主机收到字符串 `true` 后才发 ROS Action，机械臂开始动。
5. 任务完成、失败或取消后，Task Server 发布结束标志：

```text
Topic: robot_arm/task_busy
Payload: false
QoS: 1
Retain: false
```

连续任务只在最后一个动作完成后发布 `false`；若中途失败则立即发布。默认最多等开始标志 30 秒。可通过
`X5A_MQTT_HOST`、`X5A_MQTT_PORT`、`X5A_MQTT_TOPIC`、
`X5A_MQTT_USERNAME`、`X5A_MQTT_PASSWORD` 修改连接参数，
`X5A_MQTT_START_TIMEOUT` 修改等待秒数，或设置
`X5A_MQTT_ENABLED=0` / `X5A_MQTT_WAIT_START=0` 关闭。

## 接到 QQ 群机器人

网页麦克风经常被 QQ/微信内置浏览器拦住。更稳的方式是：群里发「抓红色」或发一条语音，机器人回群并调用现有抓取链。

兼容 **OneBot v11**（Napcat / Lagrange / llonebot 都可以）。

1. 先让网页服务跑起来（它同时提供 QQ 回调地址）：

```bash
source /opt/ros/humble/setup.bash
source ~/arx/arm/install/setup.bash
export X5A_QQ_API=http://127.0.0.1:3000
export X5A_QQ_GROUP_IDS=你的群号
ros2 run x5a_web_agent web_agent --host 0.0.0.0 --port 8000
```

2. 在 Napcat 里登录机器人 QQ，打开 HTTP 服务（默认 `http://127.0.0.1:3000`），并把 **HTTP 上报** 填成：

```text
http://127.0.0.1:8000/api/onebot
```

如果 Napcat 和网页不在同一台电脑，把 `127.0.0.1` 换成机械臂主机 IP。

3. 进群发：

- 文字：`抓红色` / `帮我拿白色`
- 或直接发一条 **语音**

群里会回：`收到，开始抓取红色方块`，结束后再回完成/失败。

可选环境变量：

| 变量 | 含义 |
|---|---|
| `X5A_QQ_API` | Napcat HTTP 地址，默认 `http://127.0.0.1:3000` |
| `X5A_QQ_GROUP_IDS` | 允许的群号，逗号分隔。不填则所有群都听 |
| `X5A_QQ_TOKEN` | Napcat token，没有就空着 |
| `X5A_QQ_AT_ONLY=1` | 必须 @机器人 才执行 |
| `X5A_QQ_SELF_ID` | 机器人 QQ 号，配合 @ 使用 |
