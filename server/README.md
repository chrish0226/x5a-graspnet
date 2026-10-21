# 云端模型服务

本目录包含当前腕部流程所需的三层服务源码，不包含模型权重、完整上游模型仓库或服务器私有环境文件。

| 文件 / 服务 | 作用 | 默认端口 |
| --- | --- | --- |
| `sam3_server.py` | SAM 3.1 multiplex 文本分割；启动预热，模型常驻；注册网关 | 8300 |
| `llm_gateway.py` | `/parse` 调 Ollama；`/v1/locate`、`/v1/grasp` 代理到 GraspNet | 与 SAM 共用 8300 |
| `grasp_server.py` | RGB-D 处理、目标定位、GraspNet 候选与场景碰撞过滤 | 127.0.0.1:8301 |
| 外部 Ollama | `qwen3:4b` 中文结构化解析 | 127.0.0.1:11434 |

`grasp_server.py` 对应当前回退后的活动后端，没有强制共享夹爪 `model_id`。`models/x5a_stock_gripper.json` 不是当前线上协议必需项。不要把之前的统一模型实验说明套用到本版本。

## 模型和环境准备

1. 按 [SAM 官方仓库](https://github.com/facebookresearch/sam3)及实际 checkpoint 来源建立独立 GPU Python 环境。此适配器要求 `build_sam3_image_model` 和 `Sam3Processor`，使用的权重是 `sam3.1_multiplex_fp32.pt`。任意旧版 SAM 或普通 SAM3 权重不保证接口兼容；checkpoint 与上游版本应配套。权重访问及许可按上游要求办理。
2. 按 [GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)准备单独环境、`checkpoint-rs.tar`、pointnet2 / knn CUDA 扩展及上游依赖。不要把面向新 PyTorch 的 SAM 环境直接当作 GraspNet 的兼容环境。
3. 两个服务环境都需 FastAPI、uvicorn、python-multipart、Pillow、NumPy；GraspNet 额外使用 SciPy / 上游点云依赖。先完成各模型的上游单机推理，再安装 HTTP 适配代码。
4. 安装 Ollama，拉取 `qwen3:4b` 或兼容模型，并确认本机 11434 可访问。解析模型通过 `OLLAMA_MODEL` 可配置。

服务端环境没有完整 lockfile；本仓库验证的是原服务器既有模型环境加上这些适配器，不声称全新环境的一键安装已验收。

## 部署步骤

以下路径是示例，替换为服务器实际目录。SAM 与 GraspNet 使用不同终端、各自已准备好的 Python 环境。

### 1. GraspNet 后端

`grasp_server.py` 通过自身目录寻找上游 `models/`、`dataset/`、`utils/` 和 checkpoint，因此应复制到上游 GraspNet 根目录运行：

```bash
export X5A_REPO_ROOT="$HOME/arx/wrist_ws"
export X5A_GRASPNET_ROOT="$HOME/graspnet-baseline"
cp "$X5A_REPO_ROOT/server/grasp_server.py" "$X5A_GRASPNET_ROOT/grasp_server.py"
cp "$X5A_REPO_ROOT/src/x5a_remote_grasp/x5a_remote_grasp/grasp_quality.py" \
  "$X5A_GRASPNET_ROOT/grasp_quality.py"
# checkpoint-rs.tar 预先放在 X5A_GRASPNET_ROOT 中
cd "$X5A_GRASPNET_ROOT"
# 先激活 GraspNet 的独立环境
export SAM_URL='http://127.0.0.1:8300/segment_text'
CUDA_VISIBLE_DEVICES=0 python -m uvicorn grasp_server:app --host 127.0.0.1 --port 8301
```

### 2. SAM 与网关

```bash
cd ~/arx/wrist_ws/server
# 先激活 SAM 的独立环境
export SAM3_REPO="$HOME/vendor/sam3"
export SAM3_CHECKPOINT="$HOME/models/sam3.1_multiplex_fp32.pt"
# 必要时设置 SAM3_BPE，否则使用 SAM3_REPO 下的标准资源路径
export GRASP_API_KEY='<与客户端相同的 64 位十六进制密钥>'
export OLLAMA_MODEL='qwen3:4b'
CUDA_VISIBLE_DEVICES=1 python -m uvicorn sam3_server:app --host 127.0.0.1 --port 8300
```

单 GPU 机器把两个 `CUDA_VISIBLE_DEVICES` 按本机资源调整。服务启动后执行实际 CUDA 预热，之后每次请求复用已加载模型，并不是每次重新加载权重。

网关可从 `GATEWAY_ENV_FILE` 指定文件读取 `GRASP_API_KEY`，默认同目录的 `gateway.env`。推荐启动时导出密钥，不提交私有 env。`gateway.env.example` 仅为占位模板。

当前 `/v1/locate`、`/v1/grasp` 和 `/health` 验证 `X-API-Key`；兼容 `/grasp` 和 `/parse` 没有完整同等鉴权，`/segment_text` 默认供本机后端调用。保持以上 loopback 绑定，通过 SSH 隧道 / 受控网络代理连接机器人；直接暴露整个 8300 到公网需另加统一鉴权和访问控制。后端调用 `/segment_text` 未传密钥，不应单独设置 `SAM3_API_KEY` 后就假定内部调用仍正常。

SSH 隧道示例（机械臂主机上运行）：

```bash
ssh -N -L 8300:127.0.0.1:8300 <用户>@<模型服务器>
```

此时机器人 `remote.base_url` 设为 `http://127.0.0.1:8300`，`X5A_CLOUD_PARSE_URL` 设为 `http://127.0.0.1:8300/parse`。

原服务器已有的 `~/model_start.sh`、`~/model_status.sh` 和 watchdog 属于部署现场，不包含在仓库中；不要把这些私有脚本当作克隆后必有的入口。

## 检查服务

```bash
# 在服务器检查 GraspNet：
curl -fsS http://127.0.0.1:8301/grasp_health
# 客户端先加载自己的 SAM_GRASPNET_API_KEY，再通过隧道或实际地址检查：
curl -fsS -H "X-API-Key: $SAM_GRASPNET_API_KEY" http://127.0.0.1:8300/health
# 只测试解析，不提交机械臂任务：
curl -fsS -H 'Content-Type: application/json' \
  -d '{"text":"把白色方块放到橙色方块上"}' http://127.0.0.1:8300/parse
```

健康响应需 `status: ready`、`sam.ready: true`、`graspnet.ready: true`。它不检查机器人驱动，也不证明 `/parse` 中 Ollama 可用，解析需单独测。SAM 预热失败会终止启动；GraspNet 的异步预热失败可能只表现为 `model_loaded: false`，需检查后端日志。

## HTTP 契约

| 接口 | 输入 / 输出要点 |
| --- | --- |
| `POST /parse` | JSON `text`，返回 `result`，包括 object/color/sam_prompt/reference_prompt/relation/place_prompt/place_rel/action |
| `POST /segment_text` | multipart `file/prompt/threshold`，返回实例评分、框和 PNG 掩码 |
| `POST /v1/locate` | multipart `rgb/depth/prompt/fx/fy/cx/cy/depth_scale`，返回目标掩码和 `quality_schema: 2` |
| `POST /v1/grasp` | 同上，可携带本地清理的 `mask`，返回候选位置/旋转/开口及碰撞过滤标志 |
| `GET /health` | 网关聚合 SAM 与 GraspNet 就绪状态 |

v1 深度约定为 **原始深度除以 `depth_scale` 得米**，毫米 PNG 对应 1000。旧 `/grasp` 的同名参数是乘数，不能混用。抓姿在相机光学系（x 右、y 下、z 前），不是基座坐标；机器人侧按曝光时刻 TF 转换。

当前缓存只复用完全相同 RGB 像素的编码特征，最多 2 帧、15 s TTL，不缓存不同画面的旧掩码。GPU 串行锁保护模型，推理在线程中运行以避免阻塞网关事件循环。性能测量见[验证记录](../docs/VALIDATION.md)。
