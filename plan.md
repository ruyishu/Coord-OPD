# Coord-OPD：连续视觉坐标动作的 On-Policy Distillation 实验计划

更新时间：2026-10-13。项目仓库：[Coord-OPD](https://github.com/ruyishu/Coord-OPD)。

状态：拟议研究方案，尚未实现或验证。算力按少量 A800、2 张 L20 安排，先做测量，不假设具体单卡显存或训练吞吐。

## 1. 要验证什么

**学生自己选择裁剪区域，老师在相同视觉历史下指导它的连续坐标动作，同时指导文字推理。**

例子：要读瓶子标签，学生框只覆盖了一半文字。最终答错无法区分“没看全”和“推理错”；坐标分布蒸馏希望直接帮助学生选出足够完整的区域。

假设：在相同模型、数据与裁剪预算下，“文本 OPD + 连续动作 OPD”比只有最终奖励或只有文本 OPD 更有效，尤其在小目标、框偏移和部分遮挡场景。

不能直接声称“连续动作 KL”是新理论；它是已有蒸馏工具。研究价值要通过连续视觉动作中的有效框多解、粗框与细化分工、越界和尺度行为等实际问题建立。

## 2. 已核实的源码与一个重要范围修正

上游：[kesenzhao/NV-CoT](https://github.com/kesenzhao/NV-CoT)。核查提交：`0d36f9c515fc40864c3c4497f5de73148b77b8ac`。

论文提出连续数值坐标策略；**当前所读执行入口是“VLM 先写粗框，再由连续头产生残差进行微调”**，不能直接把它当成完全没有坐标文本的实现。

| 位置 | 已看到的行为 | 对计划的影响 |
|---|---|---|
| `policy.py:load_policy` | 使用 Qwen3-VL；检查 `hidden_size == 4096`；加载 `JointVisualRefiner` | README 提到 2B，不代表此入口直接兼容 2B；需核查并适配 |
| `policy.py:action_distribution` | 返回 Normal 分布，标准差固定为 0.15 | 第一版是固定方差高斯动作蒸馏 |
| `rollout.py:sample` | 生成粗框，采样 raw action，经有界残差与框修正后实际裁剪 | 保存粗框、raw action、最终框和实际 crop |
| `continuous_policy.py` | 另有可配置 Normal/Laplace 的连续头与动作变换 | 这是另一条实现路径，不能只改这里就宣称主训练入口已改变 |
| `train.py` | 使用组内优势，同时更新文本与连续动作；有零方差组重采样 | OPD 路径要独立于 GRPO 组内优势逻辑 |
| `evaluate.py`、`iou_metrics.py` | 推理评估和框指标入口 | 增加文本、粗框、细化分别贡献的分析 |

第一版明确选择：**保留当前混合动作协议，在该协议上建立 OPD 基线。** 完全取消粗框文本的纯连续策略另开实验配置，不能混用结果。

## 3. MVP、老师与数据

MVP 限定单图、最多 2 次裁剪；先用 32 条样本验证接口，再用 1,000—3,000 条训练样本做 pilot。

教师必须同时具备：

- 可计算 token 分布的 VLM，而不只是能返回答案的 API。
- 与学生含义一致且已经训练有效的连续动作头；随机初始化头不能充当老师。
- 相同粗框定义、坐标单位、原图/当前 crop 引用、raw action 参数化和裁剪变换。

优先用同架构、同词表的强 checkpoint 作老师，学生从较早的可用 checkpoint 出发。公开 checkpoint 的 VLM、连续头权重和配置必须一起核验；缺少合格头时，应把“先训练一个 RL teacher”列为成本与前置任务。

如果资源不足，可先冻结 VLM，只验证连续头 OPD。这只是工程和定位实验，不能当成完整多模态推理蒸馏的最终结果。若采用 2B 学生，先移除尺寸假设并验证视觉特征、模型输出和 checkpoint 对齐，再比较算法。

数据优先使用上游 README 指向的 [DeepEyes-Datasets-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)。[V* Bench](https://huggingface.co/datasets/craigwu/vstar_bench) 用于外部评测，不拿其测试样本调参。

训练数据按原图划分开发集；来自多个来源的数据做去重。框指标只能在具有可信框标注的子集上计算。可另做少量合成 OCR/目标定位诊断集，记录生成规则并与正式基准分开报告；不假设原问答数据都含 bbox 真值。

## 4. 交互与蒸馏流程

1. 学生生成文本和粗框 `b_coarse`，保留原始生成 token。
2. 在当前历史和这个学生粗框下，学生连续头得到 `q_S(z | h, b_coarse)`，采样 `z`。
3. 环境按固定变换 `b_final = f(b_coarse, z)` 裁剪，学生看到实际 crop 后继续回答。
4. 老师在同一历史、同一学生粗框下计算 `q_T(z | h, b_coarse)`；老师不能用自己的粗框代替学生粗框。
5. 文本按学生前缀算 token OPD；坐标按 raw action 分布算连续 OPD。
6. 完成本轮更新后重新采样。MVP 每题 1 条 trajectory 即可；不需要 GRPO 那种组内奖励方差。

最小动作记录：`policy_version`、`parent_turn_id`、`image_id`、`image_size`、`coarse_box`、`raw_action`、`final_box`、`action_transform_version`、`context_token_ids`、`crop_hash`、`old_action_logprob`。额外保存实际工具返回图，确保重放不依赖重新猜测坐标。

当前连续微调范围有界。如果粗框落在瓶盖而标签很远，连续头可能根本移不到标签。因此联合训练必须包含粗框文本；不能把所有区域错误都归因于连续头。

## 5. 损失定义与关键细节

```text
L_text = 学生生成位置上的 mean[KL(p_S(.|h_t) || p_T(.|h_t))]
L_coord = 裁剪决策位置上的 mean[KL(q_S(z|h,b_coarse) || q_T(z|h,b_coarse))]
L_joint = L_text + λ_coord * L_coord
```

两个 loss 分别归一化，防止上千文本 token 淹没少数动作。`λ_coord` 从 0.1、1、10 的小网格中按开发集选择，固定最小搜索预算；这只是建议起点，不是已验证最佳值。无裁剪轨迹的坐标项记为零，并单独记录无动作比例。

对角高斯的逐维 reverse KL 为：

```text
KL(S || T) = sum_d [log(σ_T / σ_S)
                   + (σ_S² + (μ_S - μ_T)²) / (2σ_T²) - 1/2]
```

当前两者都使用固定 `σ=0.15` 时，它等价于缩放后的均值平方误差。必须承认这一点；不要把这种特殊情况包装成“学习了复杂不确定性”。可学习方差、Laplace 和多峰分布留到确有证据需要时再做。

梯度和状态约定：

- 教师 `eval/no_grad`，目标分布参数 detach；教师使用自己计算的视觉与语言特征。
- teacher 和 student 条件中的图片、学生粗框及既往 crop 相同，不能偷看本次动作尚未产生的结果。
- KL 在变换前 raw action 空间计算；不能把裁剪、排序、截断后的像素框直接代入原始 Normal 密度。
- 非可逆框修正会使多种 raw action 对应同一个有效框；raw KL 与“有效区域相同”不是一回事，需单独评价最终 crop。
- 输入视觉 token 不直接计算生成词表 KL；ViT 是否被更新由解冻配置和可微前向路径决定。
- 坐标变换和图像裁剪本身不做反传；OPD 指导的是采样状态下的策略分布。
- 文本 full-vocabulary KL 按位置分块；若使用采样策略梯度版 OPD，要正确处理 log-ratio、detach、采样分布与重要性权重，不把 detached 差值直接当可训练 loss。

工程上将原 `train.py` 的组内奖励筛选、零方差重采样与 OPD 分支彻底分开。OPD 能在组内奖励全部相同的批次产生有效监督，只要老师与学生分布有差异。

## 6. 最小工程改动

| 拟新增或修改位置 | 工作 | 验收标准 |
|---|---|---|
| `policy.py` 的 teacher/student 加载配置 | 加载独立模型和连续头，移除已确认的不兼容尺寸假设 | 权重完整、教师冻结，模型尺寸自检 |
| `rollout.py` | 导出完整动作上下文和最终 crop | 原图及历史 crop 索引可重放 |
| `research/coord_distill.py` | 连续分布 KL 和文本 KL | 解析解、数值估计及梯度方向相符 |
| `research/train_opd.py` | 独立 OPD 训练循环 | 不依赖多 rollout 和非零组优势 |
| `research/eval_regions.py` | 区域覆盖、框合法性与答案评估 | 有标注时算 IoU/覆盖，无标注时不伪造指标 |
| `configs/research/` | RL、text-only、coord-only、joint 配置 | 方法差异可追踪 |

先复用实际入口，不把项目强行迁移到最新 `verl`。上游当前是独立 PyTorch 训练入口，不能直接假设已有通用 OPD worker。

## 7. 对照实验

| 实验 | 目的 |
|---|---|
| 同一初始学生，无继续训练 | 起点 |
| 上游 RL 配方 | 任务奖励对照 |
| Text-only OPD | 文字与粗框文本蒸馏的收益 |
| Coord-only OPD | 连续头监督的独立收益 |
| Joint OPD | 两者联合是否更好 |
| Joint OPD + 原任务奖励 | 在 pilot 成功后验证互补性 |

主要指标：问答准确率、固定 GPU 小时的收益、推理裁剪次数与总时延。辅助指标：有效框比例、越界/退化比例、标注子集上的证据覆盖率、框面积、粗框到细化框的改进。

需要额外做两个诊断：

1. 多个有效框都能读出标签时，OPD 是否只逼迫学生追一个老师偏好的框？比较 raw KL、区域有效性与最终答案的关系。
2. 粗框固定与粗框可学习两种设置，区分连续头收益和语言部分收益。固定方差时，KL 与对应 MSE 只能作为数值一致性检查，不能冒充不同算法的胜出证据。

在同初始化、同 LoRA/解冻范围、同图片分辨率与调用上限下比较。文本/动作 loss 统一计数方式。记录教师准备成本和训练期评分成本，不能把额外训练的教师视为免费。pilot 一个种子，正式至少 3 个种子，按原图或问题做配对置信区间。

## 8. A800 / L20 的使用方案

- 第一步先冻结 VLM，用少量前向特征训练/检查小连续头；2 张 L20 可分工学生推理与教师评分，实际以显存测量为准。
- 完整 joint 阶段优先 A800 训练学生，L20 运行冻结 teacher；按同一 checkpoint 的要求加载连续头。
- 若 8B 全参超预算，统一改用 LoRA 的学生对照；不能只给某一个方法降低分辨率或缩短上下文。
- 2B 适配属于独立 M0 工作，因为当前入口有固定维度检查。适配成功前不承诺“小模型直接跑通”。
- 先限制到 1—2 次裁剪和 microbatch=1，测 20 次更新的峰值显存、tokens/s、teacher 延迟，再确定 pilot 时长。
- 租卡优先用于合格 teacher 建立、joint 全参验证和多种子扩展。先拿到测量数据，再估费用和排期。

## 9. 验收与失败处理

- [ ] M0：锁定实际运行路径；确认混合动作形式、模型尺寸、teacher 连续头和预处理一致。
- [ ] M1：32 条样本完成“粗框→raw action→实际裁剪→回答”，重放 crop 相同。
- [ ] M2：同分布 KL≈0；偏移均值后梯度将学生拉向老师；teacher 无梯度；无动作样本不出现 NaN。
- [ ] M3：边界框、窄框、无效框与不同图像尺寸的 raw/final 坐标测试通过；实际看到的是正确 crop。
- [ ] M4：连续 50 次更新、权重同步和恢复训练通过；奖励无方差的批次仍可做 OPD。
- [ ] M5：完成 1k—3k pilot 与必要消融，检查准确率、区域质量和成本。
- [ ] M6：存在稳定收益或可重复的特有失败，再做正式实验。

如果 teacher 区域选择没有优势，先改善老师；如果坐标 KL 下降但 crop 没改善，先查非可逆变换与范围限制；如果只有文本 OPD 有效，应诚实报告连续头蒸馏缺乏增益。暂不直接叠加 bbox 标签、偏好奖励和更复杂分布，避免失去可解释对照。

## 10. 仓库交付与首个任务

提交计划、数据划分清单、来源版本、训练与评测配置、微型轨迹样例和结果表。大数据、权重、完整 rollout 图像及 API 密钥不进入 Git。新增配置项与脚本先在 README 标为计划中，跑通后再提供真实启动命令。

首个开发任务：

> 阅读 plan.md，完成 M0—M1。确认 NV-CoT 固定提交中真正使用的坐标头与 raw/final 动作变换，检查现有 checkpoint 是否同时包含 VLM 和已训练的连续头。先重放 32 条裁剪轨迹，核对 2B 适配和 teacher 可用性；不要只改 continuous_policy.py 就认定已经改到实际训练路径。

参考：[NV-CoT 论文](https://arxiv.org/abs/2602.23959)、[固定版本代码](https://github.com/kesenzhao/NV-CoT/tree/0d36f9c515fc40864c3c4497f5de73148b77b8ac)、[实际动作分布入口](https://github.com/kesenzhao/NV-CoT/blob/0d36f9c515fc40864c3c4497f5de73148b77b8ac/policy.py)、[实际 rollout](https://github.com/kesenzhao/NV-CoT/blob/0d36f9c515fc40864c3c4497f5de73148b77b8ac/rollout.py)。本计划不保证研究新颖性。
