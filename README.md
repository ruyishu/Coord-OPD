# Coord-OPD: On-Policy Distillation for Continuous Visual Region Selection

> **状态**：研究与工程推进中（阶段 M0—M1）。本仓库代码基于固定版本的上游 NV-CoT 框架扩展，尚未完成全量训练与基准验证。

## 1. 项目简介与目标

在细粒度多模态理解与文档问答（如 DocVQA, ChartQA, V*Bench）中，模型需要对高分辨率图像中的小目标或局部证据进行主动裁剪和放大（Crop & Zoom）。

传统的 NV-CoT 框架引入连续欧氏空间动作头直接预测边界框残差，并采用强化学习（GRPO 风格）进行策略优化。然而，在奖励稀疏或组内无方差的场景下，仅靠最终答案奖励难以归因裁剪动作的好坏。

**Coord-OPD** 提出：
1. **连续坐标与文本推理联合 OPD**：在学生模型采样的 On-Policy 轨迹上，同时最小化文本 Token 分布的 Reverse KL 和连续动作头输出分布的 Gaussian Reverse KL。
2. **细化框动作直接监督**：在学生选出的粗框上下文下，直接由能力更强的教师模型指导连续残差头，让裁剪区域更准确地覆盖决定性视觉证据。

详细实验设计与技术假设请见 **[完整研究计划 (plan.md)](plan.md)**。

---

## 2. 仓库结构规范

```text
coord-opd/
├── README.md              # 项目简介、阶段状态与使用指南
├── plan.md                # 完整的 10 阶段工程与实验计划
├── UPSTREAM.md            # 上游来源 (NV-CoT)、commit 记录与协议审计
├── .gitignore             # 忽略大文件、权重、裁剪中间图与日志
├── configs/
│   └── research/
│       └── pilot_crop_opd.yaml # A800 + 2x L20 适用的最小 Pilot 训练配置
├── research/
│   ├── coord_distill.py   # 连续高斯动作 Reverse KL 与联合 OPD 损失实现
│   └── eval_regions.py    # 坐标变换、IoU、有效区域与越界截断评估工具
└── results/               # 实验指标记录与对照消融报告
```

---

## 3. 上游对应与关键接入点

本项目基于上游开源项目 **[kesenzhao/NV-CoT](https://github.com/kesenzhao/NV-CoT)**（核查 Commit: `0d36f9c515fc40864c3c4497f5de73148b77b8ac`）构建。详细改动与依赖审计见 **[UPSTREAM.md](UPSTREAM.md)**。

核心接入组件：
- **模型与策略加载**：`policy.py` (`load_policy`, `action_distribution`)
- **Rollout 采样与裁剪**：`rollout.py`
- **训练更新**：解耦原 `train.py` 中的 GRPO 组内优势，新增独立 OPD 训练循环
- **蒸馏损失计算**：`research/coord_distill.py`

---

## 4. 首个开发任务 (M0—M1 阶段目标)

当前开发助手需执行的首个任务：
1. **确认混合动作协议**：上游实际入口为“VLM 预测粗框文本 + 连续头预测高斯残差微调”。第一阶段在此混合动作协议上建立基线，不直接做纯无文本连续策略。
2. **检查模型尺寸与教师可用性**：上游 `policy.py` 硬编码了 `hidden_size == 4096` 检查。若适配 2B 学生，需先解除该假设；确保教师模型拥有已训练有效的连续头。
3. **跑通 32 条裁剪轨迹重放**：验证 `research/eval_regions.py` 能够根据记录的 `coarse_box` 与 `raw_action` 完美还原最终裁切图像。

---

## 5. 快速开始与测试

在本地 Python 环境中验证连续动作蒸馏损失与坐标转换：

```bash
# 运行连续动作与文本联合 OPD 损失自检（测试梯度反传与均值对齐）
python research/coord_distill.py

# 运行坐标有效性与 IoU 评估自检
python research/eval_regions.py
```
