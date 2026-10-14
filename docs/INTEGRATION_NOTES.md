# 现有模块、检查结果与接入缺口

核查日期：2026-10-13。保留远程提交 `78c9c8f` 中的研究代码与配置草稿；2026-10-14 起这些 Python 模块与 YAML 已按下文逐项修改。

## 已执行的检查

在仓库根目录执行：

```bash
python research/coord_distill.py
python research/eval_regions.py
```

全部退出码为 0。环境：Windows、Python 3.12.4、PyTorch 2.14.0+cpu、Pillow 12.1.1、NumPy 1.26.4、PyYAML 6.0.3。该记录描述本次 CPU 检查环境，不是推荐的 GPU 训练依赖锁定。2026-10-14 复跑（Windows、Python 3.12.10、PyTorch 2.14.0+cpu）：新增的上游变换移植检查与一般方差数值对照同样退出码 0。

这些是脚本内置合成自检：验证有限输入上的 loss/梯度或区域/画布计算。没有加载真实模型、读取训练集、运行 GPU rollout 或测得任何任务收益。M0—M1 的完整验收仍按 [START_HERE.md](START_HERE.md) 执行。

## 首批必须处理的接入问题

1. **坐标变换已按固定上游移植（2026-10-14，框级）**：`research/eval_regions.py` 新增 `text_box_to_coarse` / `sanitize_normalized_box` / `apply_nv_cot_action_transform`，忠实移植固定上游路径：`sanitize(bbox/1000, 0.02)` → `sanitize(coarse + 0.15 * tanh(raw), 0.02)`；自检覆盖残差有界、反转框、越界回移、最小框与零残差恒等。原 `apply_action_transform`（`coarse + 0.2 * raw`）保留为原型参照，不能用于上游重放。**仍未移植**：像素映射与 resize、多样本 batch、实际 crop 像素/哈希比对——重放正确性仍需 M1 真实轨迹验收。
2. **复用真实变换**：接入时沿用固定上游的变换函数，记录 `coarse_box`、`raw_action`、`final_box`、图像尺寸和变换版本。比较普通框、倒置框、越界框、极小框和大残差；最终还要比对实际 crop 的像素/哈希。所有 RL/OPD 对照使用同一变换。
3. **评分条件**：连续 KL 比较的是学生实际粗框条件下、变换前的 raw action 分布。不能替换为 teacher 自选粗框，也不能用最终截断框充当高斯均值。teacher 必须有经过训练且协议相容的连续头。
4. **模型和梯度**：固定 `policy.py` 实际加载 Qwen3-VL 并检查 `hidden_size == 4096`；不能据维度相同推断支持另一模型。先验证目标学生加载、head/投影维度以及预期学生参数确实有梯度，再决定小模型适配。
5. **有效框指标**：当前 `is_valid_box` 只检查边长，不能替代有限数值、范围和协议检查。IoU 仅用于有可信区域标注的子集；脚本中的合成 IoU 不是任务成绩。
6. **被 mask 位置的 NaN 防护（2026-10-14）**：文本 KL 在算 KL 前先中和无效位置的 teacher logits 并用 `torch.where` 归约（NaN×0 仍是 NaN）；坐标 KL 同样中和无效 crop 槽位的 teacher 均值。自检验证被 mask 位置放 -inf/NaN teacher 值时 loss 与梯度仍有限。只防 teacher 侧的非有限值。

## 配置草稿核查

`configs/research/pilot_crop_opd.yaml` 随远程代码保留，尚无消费该配置的训练入口。YAML 原先写的 `kesenzhao/NV-CoT-Base`、`kesenzhao/NV-CoT-Final` 不在上游 README 中，`kesenzhao` 账号下也没有任何公开模型，基本可判定是脚手架阶段编的 ID。此前据匿名 401 判断「私有或 gated」是错的：标定实测一个肯定不存在的库匿名访问同样返回 401，401 区分不了私有和不存在。2026-10-14 已改用上游 README 的真实 ID：teacher `kesenZhaoNTU/NVCoT-Qwen3-VL-8B-Joint-GRPO-Step32`（连续头权重在 `visual_refiner.safetensors`）；学生用 README 列出的底座 `Qwen/Qwen3-VL-8B-Instruct`（上游没有发布 RL 前的 NV-CoT checkpoint，学生连续头由 `load_policy` 初始化）。另有 `kesenZhaoNTU/NVCoT-Qwen3-VL-2B-Joint-GRPO-Step32`，但需先做 hidden_size 适配。先核验真实权重及连续头；没有强且有效的 teacher 就先解决教师准备。

配置的 `action_scale: 0.2` 与固定上游不一致的问题已处理（2026-10-14）：YAML 改为 `residual_scale: 0.15` + `residual_transform: tanh` + `min_box_size: 0.02` + `text_box_unit: 1000`。与代码的对应关系是：`residual_scale` 对应变换函数同名参数，`min_box_size` 对应 `min_size` 参数；`residual_transform` 与 `text_box_unit` 是仅作记录的字段，对应代码里写死的常量（`tanh`、`/1000`），不会被传入。仍无训练入口消费该配置。注释中的 A800 80GB、L20 48GB 是草稿假设；准确显存、卡数和设备映射以 M0 实测为准。

## 下一次开发验收

- 先对齐坐标变换，再做 32 条真实裁剪轨迹及重放。
- 用有效 teacher 的连续头完成一次联合更新，检查文本与动作梯度、无动作 batch 和相同奖励 batch。
- 固定方差下 Gaussian KL 等价于缩放 MSE；不得将该自检结果当作算法新颖性或准确率收益。
