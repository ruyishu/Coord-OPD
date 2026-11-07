# Geo-QKL：视觉文档智能体的精确可行集蒸馏

**[English README](README.md)**

无框标注、无答案标注，用冻结的 `Qwen3-VL-8B-Instruct` 在线蒸馏 `Qwen3-VL-2B-Instruct`。**Geo-QKL** 用精确四坐标几何 Oracle 释放教师在等价裁剪框之间的任意数字偏好，让 **2B 学生在 DocVQA 上反超 8B 教师**。

![Geo-QKL 核心流程](assets/hero_geo_qkl.png)

---

## 1. Motivation（核心痛点）

- **裁剪框是“取证动作”，不是目标检测框**：只框住数值的紧框，和连带左侧字段名/表头一起框住的宽框，都能提供完整答题证据。
- **逐数字蒸馏会误伤好框**：标准 OPD 把 `[x1, y1, x2, y2]` 当普通文本逐位对齐，只要学生的宽框数字与教师第一偏好不同，就会受到与“移偏丢证据”同等严厉的惩罚。
- **直接证据**：固定阅读器在 125 道题上测试，把教师框 $b_T$ 外扩 $1.5\times$ 会使 GT IoU 腰斩（`26.3% → 12.6%`），但答题准确率反而升至 **`87.9` ANLS**（紧框为 `85.9`）；而等面积平移偏离证据则暴跌至 `70.4`。

---

## 2. Method（三步做法）

**核心思想：只蒸馏“哪些框可接受”，不蒸馏“教师偏爱哪组坐标数字”。**

**第一步：定义教师锚定的可接受框集 $\mathcal{A}(b_T)$**  
每个裁剪回合开始时，教师在 `1.0 MP` 视图下从学生框前历史贪心解码参考框 $b_T$。满足“高覆盖+面积受控”或“高重合”的框均视为可接受：

$$
b \in \mathcal{A}(b_T) \iff \left(\frac{|b \cap b_T|}{|b_T|} \ge 0.85 \;\wedge\; \frac{|b|}{|b_T|} \le 3.5\right) \;\lor\; \mathrm{IoU}(b, b_T) \ge 0.5
$$

**第二步：精确四坐标前缀可行性 Oracle（`geo_qkl/geo_feas.py`）**  
学生逐数字生成 `[x1, y1, x2, y2]`。在任意数字前缀 $h_t$ 处，利用整数网格上 $y$ 轴最优响应的闭式区间解与 $x_2 \le t_{x_2}$ 非支配集约简，在 CPU 上**零误差**求出仍能补全进 $\mathcal{A}(b_T)$ 的可行下一数字集合 $\mathcal{F}(h_t)$：

$$
v \in \mathcal{F}(h_t) \iff \exists\, u \;\text{ s.t. }\; \mathrm{parse}(h_t\,v\,u) \in \mathcal{A}(b_T)
$$

**第三步：商空间逆向 KL 损失（`geo_qkl/tk_core.py`）**  
记可行集总概率为 $M_S = \sum_{v \in \mathcal{F}(h_t)} p_S(v\mid h_t)$ 与 $M_T = \sum_{v \in \mathcal{F}(h_t)} p_T(v\mid h_t)$。将可行数字合并为单一事件后计算逆向 KL：

$$
\mathcal{L}_{\mathrm{Geo}}(h_t) = M_S \log \frac{M_S}{M_T} + \sum_{v \notin \mathcal{F}(h_t)} p_S(v\mid h_t) \log \frac{p_S(v\mid h_t)}{p_T(v\mid h_t)}
$$

由 KL 链式法则：

$$
\mathrm{KL}(p_S \| p_T) = \mathcal{L}_{\mathrm{Geo}}(h_t) + M_S\,\mathrm{KL}\big(p_S(\cdot\mid\mathcal{F}) \,\big\|\, p_T(\cdot\mid\mathcal{F})\big)
$$

**Geo-QKL** 精确剔除了集合内部的偏好排序项 $M_S\,\mathrm{KL}(p_S(\cdot\mid\mathcal{F}) \| p_T(\cdot\mid\mathcal{F}))$，同时完整保留可行总概率对齐与不可行方向的逐项纠偏。非坐标文本 token 采用视图对齐的 **Matched-view OPD**（教师同样用 `0.25 MP` 概览图打分）。

---

## 3. Results（最终结果）

![实验结果总览](assets/results_overview.png)

| 方法 | 步数 | DocVQA ANLS ↑ | DocVQA EM ↑ | DocVQA Crops ↓ | InfographicVQA ANLS ↑ | InfographicVQA EM ↑ | InfographicVQA Crops ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Student before distillation (2B) | -- | 68.31 | 56.60 | 1.80 | 34.47 | 28.28 | 2.43 |
| Teacher (8B, frozen) | -- | 81.99 | 73.00 | 1.19 | 65.75 | 60.37 | 1.27 |
| OPD (1 MP teacher on all tokens) | 64 | 76.89 | 68.40 | 1.43 | 43.60 | 36.91 | 1.70 |
| Matched-view OPD | 64 | 76.53 | 69.00 | 1.52 | 44.13 | 37.86 | 2.25 |
| **Geo-QKL (ours)** | 64 | **78.20** | **69.60** | **1.13** | **49.03** | **42.36** | **1.11** |
| OPD (1 MP teacher on all tokens) | 256 | 79.55 | 70.40 | **1.00** | 48.97 | 42.55 | 2.00 |
| Matched-view OPD | 256 | 79.26 | 70.80 | 1.30 | 50.82 | 44.32 | 1.87 |
| **Geo-QKL (ours)** | 256 | **82.90** | **75.40** | 1.19 | **55.31** | **49.08** | **1.64** |

- **DocVQA 反超 8B 教师**：256 步达到 **`82.90` ANLS / `75.40` EM**，领先 Matched-view OPD **`+3.64` ANLS / `+4.60` EM**，反超 8B 教师 **`+0.91` ANLS / `+2.40` EM**。
- **InfographicVQA 大涨且裁剪更少**：64 步 **`+4.90` ANLS**（裁剪次数减半：`1.11` vs. `2.25`），256 步 **`+4.49` ANLS**（比普通 OPD 高 **`+6.34` ANLS**）；4 个随机种子共 16 组配对对比中 ANLS **15/16 为正**、EM **16/16 全正**。
- **消融与零样本迁移全部第一**：显著优于稀疏候选集（开发集 `+3.52` / 测试集 `+1.26` ANLS）、统一降权 $\times 0.25$（`+2.79` ANLS）和整框事后门控（`+1.81` ANLS）；300 状态证据漏框率降至最低（**`12.2%`**）；在 `V*`、`HR-Bench 4K/8K`、`TreeBench` 四榜零样本迁移取得最高平均准确率 **`50.4%`**。

---

## 4. 快速运行

```bash
# 1. CPU 精确几何 Oracle 自检（191,520 次暴力穷举核对 0 误差）
python -m geo_qkl.selftest

# 2. 启动 Geo-QKL 训练与评测（单卡 48GB GPU）
python -m geo_qkl.train WQKG256_s17 256 17 eval

# 3. 从 results/*.csv 重新绘制结果总览图
python scripts/draw_results.py
```
