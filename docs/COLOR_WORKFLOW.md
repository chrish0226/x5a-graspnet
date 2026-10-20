# 本地颜色方块流程

这是独立于 SAM 的腕部颜色流程，使用同一 Gemini、标定候选、驱动与 MoveIt 适配器。启动与构建准备见[主 README](../README.md)。

```bash
bash scripts/start.sh
# 仅测试交互，不启动机械臂/相机，不做路径规划：
bash scripts/start.sh --dry-run
# 已启动时，另一个终端输入：
bash scripts/text.sh
```

网页 HTTP 8001 / HTTPS 8444，同时提供 8000 / 8443 兼容入口。它与 SAM 使用同一进程锁，不能同时运行。语音默认本地 SenseVoice，规则解析在本机完成。

这套流程的图像检测、中文规则解析和路径规划均可在机械臂主机本地运行，不依赖 SAM / GraspNet / Ollama 云端服务。语音需要先下载本地 ASR 模型；纯文字操作可跳过语音模型安装。颜色识别仍使用 RGB-D 中的深度定位，并不是只有 RGB 的二维抓取。

## 保留清单

| 环节 | 当前保留文件 |
| --- | --- |
| 启动与文字输入 | `scripts/start.sh`、`scripts/text.sh` |
| 腕部系统 launch | `src/x5a_pick_place/launch/wrist_stack.launch.py`、`wrist_pick.launch.py` |
| RGB-D 与 HSV 识别 | `src/x5a_vision/x5a_vision/cube_detector.py`、`config/vision.yaml` |
| 本地规则与网页 | `src/x5a_web_agent/x5a_web_agent/wrist_commands.py`、`wrist_app.py`、`wrist_bridge.py` |
| 抓放与场景状态 | `src/x5a_task_server/x5a_task_server/wrist_task_server.py`、`clearing.py`、`scene_memory.py` |
| 配置 | `config/task_plan.yaml`、`task_execute.yaml`、`buffer.yaml`、`calibration/` |
| 回归 | `tests/test_wrist_flow.py`、`test_wrist_launch.py`、`test_clearing.py`、`test_scene_memory.py` 及 `src/x5a_vision/test/` |

它与 SAM 共存于当前工作区；原主机 `backups/` 中还有颜色识别和场景记忆修改前的历史副本。这些本机备份未上传，保留的是上表所列当前源码和配置。

支持红、白、橙方块、盒沿、完整抓放、先抓后放、当前侧/镜像侧准备、遮挡清理和当前进程内叠放记忆。自动暂放使用当前 RGB-D 观察到的可用区域，配置在 `config/buffer.yaml`；固定暂放模式需另行配置。

```text
准备
抓红色方块
放到白色方块上
把红色方块放进盒子里
```

主要配置：

| 文件 | 用途 |
| --- | --- |
| `config/vision.yaml` | 颜色阈值、目标尺寸、同步、顶面与盒沿 |
| `config/task_plan.yaml` / `task_execute.yaml` | 阶段、规划、执行与场景策略 |
| `config/buffer.yaml` | 自动/固定暂放 |
| `src/x5a_task_server/.../wrist_task_server.py` | 任务、遮挡与状态管理 |

当前演示配置关闭场景方块、支撑方块和盒沿的环境碰撞模型（`wrist.scene_objects_collision_enabled: false`、`box.collision.enabled: false`）；桌面和自身碰撞仍参与检查。它降低了局部场景建模覆盖，不能据此宣称完整杂物避障。

叠放记忆只在当前服务运行期间有效，最长一小时；不跨观察坐标系使用，重启清空，不会自动纳入人工堆叠或旧历史动作。单帧异常不直接清空关系，但也不允许使用不匹配记忆。

自动候选可达不等于完整路径通过；运动中执行失败会结束任务。检测与标定同样受原实验模型精度限制。以上功能不可直接推断为 SAM 已支持。
