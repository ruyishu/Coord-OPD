# 源码来源与版本

核查日期：2026-10-13。项目：[Coord-OPD](https://github.com/ruyishu/Coord-OPD)。

| 字段 | 值 |
|---|---|
| 上游 | [kesenzhao/NV-CoT](https://github.com/kesenzhao/NV-CoT) |
| 核查分支 | `main` |
| 固定提交 | `0d36f9c515fc40864c3c4497f5de73148b77b8ac` |
| 论文 | [原文](https://arxiv.org/abs/2602.23959) |
| 当前接入状态 | 已审查相关源码并写入计划；已移植两个框级变换函数到 `research/eval_regions.py`（保留许可证）；未导入上游完整训练代码、未做本项目 GPU 复现 |

## 首先阅读的上游文件

以下路径都属于固定上游提交，当前仓库未包含这些文件。

| 文件 | 用途 |
|---|---|
| [policy.py](https://github.com/kesenzhao/NV-CoT/blob/0d36f9c515fc40864c3c4497f5de73148b77b8ac/policy.py) | load_policy、action_distribution、token_logprobs |
| [rollout.py](https://github.com/kesenzhao/NV-CoT/blob/0d36f9c515fc40864c3c4497f5de73148b77b8ac/rollout.py) | 粗框、raw action、实际 crop 和历史保存 |
| [train.py](https://github.com/kesenzhao/NV-CoT/blob/0d36f9c515fc40864c3c4497f5de73148b77b8ac/train.py) | 组内优势更新；OPD 需分离零方差重采样逻辑 |
| [continuous_policy.py](https://github.com/kesenzhao/NV-CoT/blob/0d36f9c515fc40864c3c4497f5de73148b77b8ac/continuous_policy.py) | 另一套可配置动作头与坐标变换 |
| [visual_refiner.py](https://github.com/kesenzhao/NV-CoT/blob/0d36f9c515fc40864c3c4497f5de73148b77b8ac/visual_refiner.py) | 视觉细化头及 detach 路径 |
| [evaluate.py](https://github.com/kesenzhao/NV-CoT/blob/0d36f9c515fc40864c3c4497f5de73148b77b8ac/evaluate.py) | 评测入口 |

## 已发现的注意点

- 所读 policy.py 入口检查 hidden_size == 4096；README 的模型列表不能替代小模型实测。
- 当前 action_distribution 是固定标准差 0.15 的 Normal；continuous_policy.py 中的另一种头不等于实际训练入口。
- 两侧方差相同且固定时，Gaussian KL 等价于缩放 MSE；不能将这一特例包装成新的不确定性算法。

## 版本与接入规则

先将上游放在独立参考目录并固定提交；接入自己仓库前记录实际采用的文件、修改和依赖版本。不得用另一个版本的整个 `verl` 目录直接覆盖。保留现有计划和来源说明。

修复上游运行问题时，保存修改前后证据；所有算法对照共享同一修复。确认是否换过模型、任务、图像分辨率或推理协议，避免把这些变化记为 OPD 增益。

所读上游根目录有 Apache-2.0 LICENSE；接入代码时保留许可证、版权声明和实际来源。当前仓库未导入上游训练代码，也不替上游重新指定许可证。数据和权重分别核查条款。2026-10-14 移植了 `sanitize_normalized_boxes` 与 `apply_text_box_residual`：文件头注明来源与修改，上游许可证原文复制到 `third_party/NV-CoT/LICENSE`。

## 后续接入记录

| 日期 | 实际来源与提交 | 接入/修改内容 | 验证证据 |
|---|---|---|---|
| 2026-10-13 | 上述固定提交 | 源码核查和计划初始化；未导入训练代码 | [计划](plan.md) |
| 2026-10-14 | 上述固定提交的 `continuous_policy.py` | 将 `sanitize_normalized_boxes`、`apply_text_box_residual` 移植为 `research/eval_regions.py` 的纯 Python 单框版；许可证复制到 `third_party/NV-CoT/LICENSE` | 与上游函数在 2 万组随机输入（含反转、越界框）上差分对照，最大误差 3.88e-07（float32/float64 精度差）；对照脚本为临时脚本，未入库 |

后续更新此表时只填写实际完成的动作，不能将计划描述当作运行证据。

## 本仓库现有研究模块

固定 rollout 的 raw 残差经 `0.15 * tanh(raw)` 和最小框 0.02 的修正后才产生最终框。2026-10-14 已在 `research/eval_regions.py` 完成框级对齐；像素映射与 resize 尚未移植，见 [接入说明](docs/INTEGRATION_NOTES.md)。teacher 连续头必须在学生粗框上评分。

已合并本仓库远程提交 `78c9c8f` 的研究模块和配置草稿；本次补全文档时保留这些代码。来源记录与实际检查见 [接入说明](docs/INTEGRATION_NOTES.md)。
