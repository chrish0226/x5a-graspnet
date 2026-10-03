# GraspNet 抓取精度与可解性修复（2026-10-03）

已更新云端 GraspNet、ROS 规划器、执行器和网页后端。真实 RGB-D 回放能得到完整抓取与抓放规划；本次没有执行机械臂或夹爪动作，因此不把规划通过率当成实际抓取成功率。

## 几何与候选选择

- 推理上下文使用轻度膨胀（8px → 2px）；接触测量、目标归属和高度计算始终使用原始 SAM 掩码。碰撞检查使用目标周围完整深度场景，包含邻近物体和桌面。
- 返回位置必须投影到原始目标掩码，并距原始目标点云不超过 6mm。此前被忽略的 `target_distance_thresh` 现在实际生效。
- 开口按抓取中心计算 `2*max(abs(y_low), abs(y_high))`，不再只算目标自身跨度。稀疏或空接触直接拒绝；开口检查放在 NMS 和 top-k 之前。
- NMS 同时判断位置与姿态，保留同位置的不同抓取方向；平行夹爪绕接近轴旋转 180° 视为等价。默认返回候选增加至 20，允许上限 40。
- 同一输入点云使用稳定采样种子，并保护模型初始化，减少重复推理时的随机变化。两组数据在正式 GPU 服务上重复请求得到一致的候选。
- 原生抓取映射明确区分插入深度与工具标定：当前同轴配置下，`tool0 = prediction + (depth_m - tip_offset_m)*approach`。指尖位置因此与 GraspNet 的 `prediction + depth_m*approach` 对齐。
- 优先搜索原生姿态及等价夹爪交换；原生阶段最多占规划预算 45%。替代姿态从原始目标点云独立生成，包括目标主轴方向和多个接触高度，每个姿态重新检查实际开口、目标接触、开爪时的手指/掌部、进入路径及合爪扫掠碰撞。替代姿态分数为 0，不冒用原网络分数。
- 高度先将全部原始目标点变换到 base 系，再计算分位数。取消将两个压缩的相机 Z 端点当作物体厚度的做法。

## 规划与执行

- PREGRASP 与 GRASP 保持同一经过验证的姿态，最后进入目标时不改变夹爪方向。LIFT 可以使用可达的运输姿态；RETREAT 沿此时工具方向退出。
- 接触高度增加有限的向上搜索，以保留足够目标接触、避开低位 IK 边界。优先级保留可达性表顺序，毫米级开口差异不会让反向姿态抢在可达姿态之前。
- GRASP/PLACE 的笛卡尔轨迹要求至少 99.99% 完成，且 MoveIt 返回成功。部分路径不再改走任意关节路径然后合爪。每段轨迹还通过 FK 检查实际规划端点，位置容差 3mm、姿态容差 0.05rad。
- 关节目标容差由 0.05rad 收紧至 0.003rad；规划等待遵守总预算，已经用多种种子证明无 IK 的姿态不再重复消耗一次完整姿态规划超时。保留成功的关节解供后续搜索使用。
- 放置点搜索会尝试所有可达姿态，增加 0.90/0.75rad 的较浅俯角，避免第一个在工作区内但不可解的姿态遮住其他解。
- 执行前重新定位目标、检查目标移动及抓取几何，并核对规划指纹。图像最长 60s、收到的规划最长 30s 有效；消费过的规划不能重复执行。
- 到达 GRASP 后检查新鲜 TF 中的实际工具位姿：误差超过 7mm 或 0.12rad 时拒绝合爪。闭合后与提起后用新鲜夹爪间隙检查空抓/滑落；缺失或过期反馈返回 unknown，不报告成功。
- 夹爪适配器区分 `reached_goal` 和 `stalled`。不再将 PLACE 的通用轨迹错误 -4 自动解释为“触桌成功”并释放物体。

## 验证结果

测试命令：

```bash
source scripts/stack_env.sh
PYTHONPATH="src/x5a_remote_grasp:$PYTHONPATH" /usr/bin/python3 -m pytest src/x5a_remote_grasp/test -q
```

37 项通过，覆盖深度映射、偏心开口、旋转 NMS、完整场景碰撞、改变姿态后的重新检查、倾斜相机下的高度、原生候选优先级、部分笛卡尔轨迹、FK 端点、预算、时间消息、空抓、反馈未知、执行前复核失败和放置错误不释放，以及合爪邻物碰撞、完整指长开口和无效噪声参数。

三个修改的 Python ROS 包通过 colcon 构建；修改文件通过编译及差异空白检查。正式鉴权网关的 SAM + GraspNet 请求返回 `quality_schema=2`、原始目标掩码和碰撞检查标记。

正式服务真实数据回放：

| 采集 | 原始目标点数 | 新推理上下文点数 | 抓取四段搜索 | 抓放六段搜索 |
|---|---:|---:|---:|---:|
| 20261003_124328_379018 | 1921 | 2225 | 通过，1.60s | 通过，1.67s |
| 20261003_130521_424692 | 2022 | 2307 | 通过，4.70s | 通过，3.08s |

以上时间为候选与 MoveIt 搜索耗时，排除云端推理及真实执行。六段测试使用合成放置点 base=(0.22, 0.32, 0.05)m，验证完整规划链路，不代表该处实际存在放置容器。两组最终采用独立几何姿态；第二组的原生候选因实际夹爪场景碰撞被拒绝。开口覆盖手指全长的目标几何；额外的合爪扫掠检查避免夹住开爪间隙内的邻近物体。

复现工具为 `scripts/validate_grasp_precision.py`，可使用 `--plan` 或 `--plan --place-xyz 0.22 0.32 0.05`。它使用独立 ROS 名称，不发布可执行的 DisplayTrajectory，不调用控制器。推理端口默认指向临时 SSH 转发，需要自行连接运行中的 GraspNet 8301 端口。

具体结果见 `precision_plan_replay.json`、`precision_pick_place_replay.json`、各采集目录的 `selected_grasp.json` 与 `grasp_quality.json`，以及 `gateway_precision_check.json`。

## 参考实现

场景碰撞、空抓过滤和 GraspNet 深度几何参考 [GraspNet 官方碰撞检查](https://github.com/graspnet/graspnet-baseline/blob/main/utils/collision_detector.py)。位置与旋转联合去重参考 [GraspNet API 的 NMS](https://github.com/graspnet/graspnetAPI/blob/master/graspnetAPI/grasp.py)。上下文推理与目标接触归属分开处理参考 [Contact-GraspNet 的目标过滤](https://github.com/NVlabs/contact_graspnet/blob/main/contact_graspnet/contact_grasp_estimator.py)。本次保留已有模型权重。

## 运行状态与回滚

云端 `/home/gjj/graspnet-baseline/grasp_server.py` 与 `grasp_quality.py` 已部署；服务在 8301，SAM 网关仍在 8300。部署前原服务 SHA256 校验一致，备份保存于服务器：

`/home/gjj/graspnet-baseline/backups/precision_20261003_143245/grasp_server.py`

共享几何模块 SHA256 为 `b70459ef8ab4b0fc0af928e3b6402a2d5bbf9583077b05efd8c5f3804c02ab59`。

新版服务 SHA256 为 `05c13d1b2568f2205fd450687bf93aaa539ba3c5f9649a50bcbefe5cea77657b`。

本地变更前备份为 `/tmp/x5a_grasp_precision_before_20261003/`，本目录也保存修改前后的 Python 差异。回滚时需要同时恢复云端和本地规划/执行文件并重新构建、重启，避免新规划器连接不返回原始掩码的旧服务。原有其他未提交的修改保留。

本地规划器、执行器、夹爪适配器和网页后端在确认空闲、无持物后重启；旧缓存规划已清除。控制器与相机保持运行，`/arm_cmd` 仍只有一个发布者。本次所有验证仅推理和规划。

## 实物标定与测量边界

当前夹爪参数：tool0 到指尖 15mm、手指长度 100mm、厚度 10mm、高度 20mm、接触垫长度 30mm、总开口 88mm。它们包含已有实测信息和近似尺寸，需要按实际安装夹爪核对；这次没有改变手眼外参。

点云几何检查无法验证遮挡背面的形状、摩擦力或真实夹紧力。间隙只能证明有阻挡/接触，不能单独证明夹住了指定目标。实际抓取精度与成功率仍需通过实机重复抓取测量；软件回放和 FK 检查不替代手眼、深度与指尖标定。
