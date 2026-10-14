# 现有模块、检查结果与接入缺口

核查日期：2026-10-13。保留远程提交 `78c9c8f` 中的研究代码与配置草稿；本次合并补齐项目文档，没有改写这些 Python 模块或 YAML。

## 已执行的检查

在仓库根目录执行：

```bash
python research/coord_distill.py
python research/eval_regions.py
```

全部退出码为 0。环境：Windows、Python 3.12.4、PyTorch 2.14.0+cpu、Pillow 12.1.1、NumPy 1.26.4、PyYAML 6.0.3。该记录描述本次 CPU 检查环境，不是推荐的 GPU 训练依赖锁定。

这些是脚本内置合成自检：验证有限输入上的 loss/梯度或区域/画布计算。没有加载真实模型、读取训练集、运行 GPU rollout 或测得任何任务收益。M0—M1 的完整验收仍按 [START_HERE.md](START_HERE.md) 执行。

## 首批必须处理的接入问题

1. **坐标变换尚不一致**：当前 `eval_regions.py` 默认执行 `coarse + 0.2 * raw`，再排序、截断。固定上游 `rollout.py` 则先将粗框除以 1000 并 sanitize，执行 `apply_text_box_residual(coarse, 0.15 * tanh(raw), 0.02)`，再经过像素映射和 resize。差异包含非线性、幅度、最小框及越界处理；当前脚本不能用作上游轨迹重放的正确性依据。
2. **复用真实变换**：接入时沿用固定上游的变换函数，记录 `coarse_box`、`raw_action`、`final_box`、图像尺寸和变换版本。比较普通框、倒置框、越界框、极小框和大残差；最终还要比对实际 crop 的像素/哈希。所有 RL/OPD 对照使用同一变换。
3. **评分条件**：连续 KL 比较的是学生实际粗框条件下、变换前的 raw action 分布。不能替换为 teacher 自选粗框，也不能用最终截断框充当高斯均值。teacher 必须有经过训练且协议相容的连续头。
4. **模型和梯度**：固定 `policy.py` 实际加载 Qwen3-VL 并检查 `hidden_size == 4096`；不能据维度相同推断支持另一模型。先验证目标学生加载、head/投影维度以及预期学生参数确实有梯度，再决定小模型适配。
5. **有效框指标**：当前 `is_valid_box` 只检查边长，不能替代有限数值、范围和协议检查。IoU 仅用于有可信区域标注的子集；脚本中的合成 IoU 不是任务成绩。

## 配置草稿核查

`configs/research/pilot_crop_opd.yaml` 随远程代码保留，尚无消费该配置的训练入口。`kesenzhao/NV-CoT-Base`、`kesenzhao/NV-CoT-Final` 尚未核实，不能当作已发布 checkpoint。先核验真实权重及连续头；没有强且有效的 teacher 就先解决教师准备。

配置的 `action_scale: 0.2` 与固定上游不一致，接入时要同步改配置和变换实现，不能只改常数而遗漏 `tanh` 与最小框处理。注释中的 A800 80GB、L20 48GB 是草稿假设；准确显存、卡数和设备映射以 M0 实测为准。

## 下一次开发验收

- 先对齐坐标变换，再做 32 条真实裁剪轨迹及重放。
- 用有效 teacher 的连续头完成一次联合更新，检查文本与动作梯度、无动作 batch 和相同奖励 batch。
- 固定方差下 Gaussian KL 等价于缩放 MSE；不得将该自检结果当作算法新颖性或准确率收益。
