# Coord-OPD

连续视觉坐标动作的 On-Policy Distillation。

**在学生实际选区域的历史上，联合蒸馏文字、粗框和连续坐标细化。**

例子：标签只裁到一半。比较仅蒸馏文字、仅蒸馏连续动作、两者联合，能否让实际 crop 更完整且答案更准确。

## 当前状态

本仓库已有研究计划、开发文档及若干研究模块初稿；训练器、数据流水线和 GPU 实验尚未完成。现有脚本的 CPU 合成自检通过，但不代表已完成真实轨迹重放。没有实验分数或已验证收益。所有后续状态与证据记录在 [STATUS.md](docs/STATUS.md)。

| 内容 | 入口 |
|---|---|
| 完整研究计划 | [plan.md](plan.md) |
| 首个开发任务和验收 | [START_HERE.md](docs/START_HERE.md) |
| 固定源码来源与许可状态 | [UPSTREAM.md](UPSTREAM.md) |
| 对照实验与成本统计 | [EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| 实验信息模板 | [experiment.template.json](configs/experiment.template.json) |
| 结果记录 | [results/README.md](results/README.md) |
| 现有代码与接入缺口 | [INTEGRATION_NOTES.md](docs/INTEGRATION_NOTES.md) |
| 开发约定 | [AGENTS.md](AGENTS.md) |

## 第一版范围

单图、最多 2 次裁剪；第一版保留上游实际使用的“粗框文本＋连续残差”。

首个交付：确认真正被 train.py 调用的动作头、粗框/raw action/final box 变换和 checkpoint，重放 32 条裁剪轨迹，检查 2B 适配条件。

工程基础为 [kesenzhao/NV-CoT](https://github.com/kesenzhao/NV-CoT/tree/0d36f9c515fc40864c3c4497f5de73148b77b8ac)，核查提交 `0d36f9c515fc40864c3c4497f5de73148b77b8ac`。当前仓库没有导入上游完整训练代码；下列研究模块为本仓库初稿，尚未接通上游训练循环。

## 模型、数据与资源

- 老师/学生：必须同时具有可评分的 VLM 与训练有效的连续头；同架构强 checkpoint 优先，缺少连续头时先建立 teacher。YAML 草稿已改用上游 README 的真实 checkpoint：teacher `kesenZhaoNTU/NVCoT-Qwen3-VL-8B-Joint-GRPO-Step32`（连续头在 `visual_refiner.safetensors`），学生底座 `Qwen/Qwen3-VL-8B-Instruct`；教师优势仍待实测。
- 数据：核验 DeepEyes-Datasets-47k 的训练部分，按原图去重划分；V* Bench 留作外部评测。没有可信 bbox 标签时不报告伪造的 IoU。
- 资源：少量 A800、2 张 L20；准确卡数、显存和可用时段待实测。先做小样本，正式训练排队；是否租卡由测得的资源需求与实验信号决定。
- OPD teacher 必须能在学生实际前缀上提供所需分布；只返回答案的普通 API 不自动满足要求。

## 开工

```bash
git clone https://github.com/ruyishu/Coord-OPD.git
cd Coord-OPD
```

`git clone` 下载仓库，`cd` 进入目录。私有仓库需先正常登录 GitHub。随后阅读 [START_HERE.md](docs/START_HERE.md)，完成 M0—M1。

当前尚无完整训练启动命令；`configs/experiment.template.json` 是待填写的实验记录，不是可交给训练器的配置。跑通后再加入真实启动命令。

## 已有模块与 CPU 自检

- [research/coord_distill.py](research/coord_distill.py)：固定方差高斯 KL 与文本/连续动作联合 loss；自检含解析解、数值积分对照、梯度方向与被 mask 位置的 NaN 防护检查。
- [research/eval_regions.py](research/eval_regions.py)：框变换与 IoU 原型；已按固定上游移植框级变换（文本框 /1000、`0.15 * tanh(raw)` 残差、0.02 最小框），像素级 crop 路径未移植。
- [配置草稿](configs/research/pilot_crop_opd.yaml)：参数已与上游变换对齐，但尚未核验模型/数据字段，不能直接启动训练。具体缺口见 [接入说明](docs/INTEGRATION_NOTES.md)。

这些脚本使用 PyTorch（张量与梯度计算）。以下命令只运行脚本内置的合成自检，需要已有相应 Python 依赖：

```bash
python research/coord_distill.py
python research/eval_regions.py
```

2026-10-13 首次执行通过；2026-10-14 复跑通过（Python 3.12.10、PyTorch 2.14.0+cpu），新增上游变换移植检查与一般方差的数值对照。测试范围、环境版本和仍需处理的问题见 [INTEGRATION_NOTES.md](docs/INTEGRATION_NOTES.md)。

## 实验原则

同初始化、同任务与同推理预算比较 RL 和 OPD；计入 teacher 准备/评分和全部工具调用成本。老师偏好更密，不保证视觉操作正确；研究假设需要对照实验验证。

论文：[原方法](https://arxiv.org/abs/2602.23959)。相邻方法和新颖性边界见 [plan.md](plan.md)。上游来源和许可详见 [UPSTREAM.md](UPSTREAM.md)。
