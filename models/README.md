# 模型资源与当前夹爪配置

## 当前启用状态

`config/sam_grasp.yaml` 中 **`gripper.model_path: ""`**，共享夹爪模型实验已经回退。活动 SAM 流程使用原厂 URDF/STL 碰撞几何，接触参数取 YAML：15 mm 插入参考、100 mm 简化手指长度、10 mm 厚度、20 mm 高度、30 mm pad。不要把这些简化参数当作独立精密实测值。

原生映射保留 `tool0 = t + (depth - 0.015) * approach`。腕部手眼和工具帧没有因保留实验模型而自动改变。

## 保留但默认关闭的实验资源

- `x5a_stock_gripper.json`：原厂 CAD 与用户确认的 80 mm 本体前端面至指尖、约 15 mm 平直接触段形成的候选统一模型。
- `x5a_stock_collision_link6/7/8.stl`：该模型派生的碰撞网格，不是活动默认配置中的原厂网格。
- `config/gripper_measurements.yaml`、`scripts/build_gripper_model.py`：模型测量输入与生成器。

这些测量来自直尺与 CAD 核对，不是独立计量标定。生成器会记录输入/网格哈希，模型 ID 用于检测两端几何不一致。若重新启用，需要同时实现并验证客户端、MoveIt、并行 IK、执行器及云端的相同模型协议；本仓库当前 `server/grasp_server.py` 是回退后的活动后端，不包含共享模型注册接口，不能仅修改 YAML 就认为端到端接入完成。

本机模型重建入口（输出会覆盖本目录实验资源）：

```bash
source scripts/env.bash
/usr/bin/python3 scripts/build_gripper_model.py
```

## 不提交的模型

SenseVoice ONNX INT8、SAM checkpoint、GraspNet checkpoint、压缩包和推理运行库不上传 Git。

- 本地语音：`bash scripts/install_asr.sh`。
- GPU 推理：按[服务端说明](../server/README.md)准备对应权重和上游环境。

外部模型遵循各自访问条件和许可。首次获取需要网络；仅克隆本仓库不能直接加载全部模型。
