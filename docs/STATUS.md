# 项目状态

最后更新：2026-10-14。

## 已完成

- [x] 创建研究说明、独立 plan.md 与开发入口。
- [x] 对计划引用的上游相关文件做过源码核查，并固定提交。
- [x] 明确 teacher 条件、关键 mask/状态约束、对照实验和现有算力假设。
- [x] 建立实验 manifest 与结果表模板；模板中没有虚构实验结果。
- [x] 本地 Markdown 相对链接、JSON 格式、CSV 表头和文件范围检查通过。

- [x] 合并已有远程研究模块；内置 CPU 合成自检通过，范围见 [接入说明](INTEGRATION_NOTES.md)。
- [x] 2026-10-14：`eval_regions.py` 移植固定上游框级变换（/1000、`0.15*tanh`、0.02 最小框）并补自检；`coord_distill.py` 增加一般方差解析 vs 数值对照；YAML 动作参数与上游对齐（仍无训练入口消费）。

## 尚未完成

- [ ] 正式导入/适配训练代码与依赖环境。
- [ ] M0：完整的模型、数据、GPU 和上游运行核验报告。
- [ ] M1：32 条真实交互轨迹与重放报告。
- [ ] 将已有 loss 模块接入真实 teacher 评分与学生权重更新，处理已记录的协议/边界问题。
- [ ] GPU smoke、checkpoint 恢复与 rollout 权重同步。
- [ ] pilot 对照、正式多种子实验和外部测试。

## 当前重点

先处理 [现有模块接入缺口](INTEGRATION_NOTES.md)，再验收真实轨迹。

- 所读 policy.py 入口检查 hidden_size == 4096；README 的模型列表不能替代小模型实测。
- 当前 action_distribution 是固定标准差 0.15 的 Normal；continuous_policy.py 中的另一种头不等于实际训练入口。
- 两侧方差相同且固定时，Gaussian KL 等价于缩放 MSE；不能将这一特例包装成新的不确定性算法。

下一步：确认真正被 train.py 调用的动作头、粗框/raw action/final box 变换和 checkpoint，重放 32 条裁剪轨迹，检查 2B 适配条件。

## 进展记录

| 日期 | 工作 | 验证层级 | 证据 |
|---|---|---|---|
| 2026-10-13 | 研究启动文档与模板初始化 | 文档/结构；未训练 | plan.md、UPSTREAM.md、docs/START_HERE.md |
| 2026-10-14 | 移植上游框级变换并补自检；数值 KL 对照；YAML 动作参数对齐 | CPU 合成自检（Python 3.12.10 / PyTorch 2.14.0+cpu） | research/eval_regions.py、research/coord_distill.py、docs/INTEGRATION_NOTES.md |

有实际验证后追加记录，并链接真实报告。源码审查不能替代完整 M0，更不能替代 GPU 运行。

2026-10-13：保留远程 `78c9c8f` 的研究代码，完成文档合并和 CPU 自检；尚未执行 GPU 训练。
