# 第三方和厂商依赖

本项目沿用根目录 BSD-3-Clause，保留原工程版权。仓库包含所需 X5A 描述与网格，不复制完整 SDK、模型权重、MoveIt 或相机驱动仓库。

| 依赖 | 来源与本项目用途 |
| --- | --- |
| ARX_X5 | [官方仓库](https://github.com/ARXroboticsX/ARX_X5)，厂商消息、SDK、CAN、机器人描述；旧工程记录基线 c783287，腕部 SDK 还需独立通过 ABI 哈希检查 |
| ROS 2 Humble | [官方文档](https://docs.ros.org/en/humble/)，通信、launch 和构建 |
| MoveIt 2 | [官方仓库](https://github.com/moveit/moveit2)，运动学、碰撞、路径规划；本机原记录 2.5.9 |
| Orbbec SDK ROS 2 | [官方仓库](https://github.com/orbbec/OrbbecSDK_ROS2)，Gemini 腕部相机；需匹配本机 gemini345_lg launch |
| TRAC-IK | [官方仓库](https://github.com/traclabs/trac_ik)，脚本固定提交 ac8ad6ef9f8dddd04524545af79b86c4a7df1c18，补丁在 config/third_party/ |
| GraspNet Baseline | [作者仓库](https://github.com/graspnet/graspnet-baseline)，模型及 CUDA 算子在外部环境安装 |
| SAM | [官方仓库](https://github.com/facebookresearch/sam3)，本项目 HTTP 适配使用当前部署的 SAM 3.1 multiplex API/权重 |
| SenseVoice / sherpa-onnx | [官方仓库](https://github.com/k2-fsa/sherpa-onnx)，本地转写；安装脚本固定 sherpa 1.13.8 |
| Ollama | [官方仓库](https://github.com/ollama/ollama)，本地服务运行 qwen3:4b 解析；模型另遵循发布方许可 |

厂商资源许可副本见 [src/X5A/LICENSE.ARX_X5](../src/X5A/LICENSE.ARX_X5)。外部依赖的源码、二进制、模型及服务适配分别遵循各自许可；本项目许可证不替代其条款。
