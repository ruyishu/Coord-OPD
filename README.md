# Crops as Evidence: Feasible-Set Distillation for Visual Document Agents (Geo-QKL)

**[中文版说明 (Motivation · Method · Results)](README_zh.md)**

**Geo-QKL** 是一个面向高分辨率视觉文档智能体（Visual Document Agents）的无标注在线策略蒸馏（On-Policy Distillation, OPD）方法。它通过**精确四坐标整数几何 Oracle** 将教师模型在可接受裁剪框集合内部的任意坐标偏好释放掉，仅蒸馏“哪些裁剪框能保留答题证据”，在不增加任何参数、不使用任何人工框或答案标签的情况下，让 **2B 学生模型在 DocVQA 上反超 8B 教师模型**。

![Geo-QKL Core Method](assets/hero_geo_qkl.png)

---

## 1. Motivation（为什么要做 Geo-QKL？）

高分辨率文档和信息图缩放到全局概览图（`0.25 MP`）后，小字号正文与表格数字会模糊丢失。视觉智能体通过发出裁剪工具调用 `<tool_call>{"name":"crop","arguments":{"bbox":[x1,y1,x2,y2]}}</tool_call>`（`0..1000` 整数网格），切取原图局部高清区域（`1.0 MP`）来读取细节并作答。

当我们用 8B 教师蒸馏 2B 学生（无人工框或答案标签）时，标准 **Token 级 OPD** 把裁剪框坐标当成普通文字，强迫学生在每个数字 token 上照抄教师的概率分布。但**裁剪框是“获取证据的动作”，不是目标检测框**：
- 只框住数值本身的**紧框**，和把左侧字段名/表头一起框进来的**宽框**，都能让模型看清并答对问题；
- 固定阅读器的对照实验证明：把教师框 $b_T$ 外扩 $1.5\times$，它与标注框的 IoU 腰斩了一半（`26.3% → 12.6%`），但问答准确率不降反升（`85.9 → 87.9` ANLS）；而把同样大小的框平移离开证据区，准确率暴跌到 `70.4`。
- 标准 OPD 把“外扩保留上下文的好框”和“移偏丢掉证据的坏框”一视同仁地惩罚，浪费了学生的容量。

---

## 2. Method（Geo-QKL 怎么做？）

我们的核心思路：**只蒸馏“教师认为哪些裁剪框可接受”，不蒸馏“教师偏爱哪一组具体坐标数字”**。

1. **教师锚定的可接受框集合 $\mathcal{A}(b_T)$**：
   在每个裁剪回合开始时，8B 教师从学生的框前历史出发，用 `1.0 MP` 视图贪心解码参考框 $b_T$。只要候选框 $b$ 覆盖了 $b_T$ 的绝大部分且面积不过度膨胀，或者与 $b_T$ 重合度高，就属于可接受集合：
   $$\mathcal{A}(b_T) = \left\{ b \;\middle|\; \left(\frac{|b \cap b_T|}{|b_T|} \ge 0.85 \;\wedge\; \frac{|b|}{|b_T|} \le 3.5\right) \;\vee\; \mathrm{IoU}(b, b_T) \ge 0.5 \right\}$$

2. **精确四坐标前缀可行性 Oracle（`geo_qkl/geo_feas.py`）**：
   学生逐个数字写出 `[x1, y1, x2, y2]`。在任意数字前缀 $h_t$ 处，四个坐标通过交集和面积耦合。我们利用 $y$ 轴最优响应的闭式解与 $x_2 \le t_{x_2}$ 的非支配集约简，在 CPU 上**零误差地精确求出可行下一数字集合**：
   $$\mathcal{F}(h_t) = \left\{ v \;\middle|\; \exists \text{ 合法后缀 } u, \text{ 使得 } \mathrm{parse}(h_t\,v\,u) \in \mathcal{A}(b_T) \right\}$$

3. **Geo-QKL 商空间逆向 KL 损失（`geo_qkl/tk_core.py`）**：
   记学生和教师分给可行集 $\mathcal{F}(h_t)$ 的总概率为 $M_S = \sum_{v \in \mathcal{F}(h_t)} p_S(v\mid h_t)$ 和 $M_T = \sum_{v \in \mathcal{F}(h_t)} p_T(v\mid h_t)$。将可行数字合并为单一等价事件后计算逆向 KL：
   $$\mathcal{L}_{\text{Geo}}(h_t) = M_S \log \frac{M_S}{M_T} + \sum_{v \notin \mathcal{F}(h_t)} p_S(v\mid h_t) \log \frac{p_S(v\mid h_t)}{p_T(v\mid h_t)}$$
   由 KL 链式法则，$\mathrm{KL}(p_S \| p_T) = \mathcal{L}_{\text{Geo}}(h_t) + M_S\,\mathrm{KL}(p_S(\cdot\mid\mathcal{F}) \| p_T(\cdot\mid\mathcal{F}))$。**Geo-QKL 恰好只删掉了集合内部的教师偏好排序项**，同时保留了可行总概率匹配（$M_S \approx M_T$）和对每一个不可行数字的逐项纠偏。非坐标文本 token 则采用视图对齐的 **Matched-view OPD**（教师用与学生相同的 `0.25 MP` 视图打分）。

---

## 3. Results（最终实验结果）

![Geo-QKL Quantitative Results](assets/results_overview.png)

### 3.1 主实验结果（Qwen3-VL-8B → Qwen3-VL-2B，无任何标注）

| 方法 | 步数 | DocVQA (Held-out) ANLS ↑ | DocVQA EM ↑ | DocVQA Crops ↓ | InfographicVQA ANLS ↑ | InfographicVQA EM ↑ | InfographicVQA Crops ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Student before distillation (2B) | -- | 68.31 | 56.60 | 1.80 | 34.47 | 28.28 | 2.43 |
| Teacher (8B, frozen) | -- | 81.99 | 73.00 | 1.19 | 65.75 | 60.37 | 1.27 |
| OPD (1 MP teacher on all tokens) | 64 | 76.89 | 68.40 | 1.43 | 43.60 | 36.91 | 1.70 |
| Matched-view OPD | 64 | 76.53 | 69.00 | 1.52 | 44.13 | 37.86 | 2.25 |
| **Geo-QKL (ours)** | 64 | **78.20** | **69.60** | **1.13** | **49.03** | **42.36** | **1.11** |
| OPD (1 MP teacher on all tokens) | 256 | 79.55 | 70.40 | **1.00** | 48.97 | 42.55 | 2.00 |
| Matched-view OPD | 256 | 79.26 | 70.80 | 1.30 | 50.82 | 44.32 | 1.87 |
| **Geo-QKL (ours)** | 256 | **82.90** | **75.40** | 1.19 | **55.31** | **49.08** | **1.64** |
| $\Delta$ vs. Matched-view OPD | 64 | **+1.67** | **+0.60** | -26% crops | **+4.90** | **+4.50** | -51% crops |
| $\Delta$ vs. Matched-view OPD | 256 | **+3.64** | **+4.60** | -8% crops | **+4.49** | **+4.76** | -12% crops |

### 3.2 关键消融与零样本迁移

- **精确几何 Oracle 不可或缺**：用 $\le 13$ 个候选框的稀疏集合近似 $\mathcal{F}(h_t)$，在 Held-out DocVQA 上掉分 `-1.26` ANLS / `-2.60` EM，在 404 题开发集上掉分 `-3.52` ANLS（`84.09` vs. **`87.61`**）。原因在于稀疏近似在 `84%` 的位置退化为空集或单数字集，仅放松了 `13%` 的坐标 KL 质量；而精确 Oracle 在 `79.4%` 的位置找到多可行数字，放松了 `77.3%` 的集合内 KL 质量。
- **放松偏好 $\neq$ 统一降权**：把坐标损失权重统一乘以 `0.25`，在 256 步 Held-out DocVQA 上落后 Geo-QKL **`2.79` ANLS / `3.80` EM**（`80.11` vs. **`82.90`**）。
- **前缀级放松优于整框事后门控**：等整框写完再门控（Box-level Gate）在开发集上落后 Geo-QKL **`1.81` ANLS**（`85.80` vs. **`87.61`**）。
- **自然图像高分辨率零样本迁移（256 步）**：在 `V*`（**`55.0%`**）、`HR-Bench 4K`（`61.1%`）、`HR-Bench 8K`（`52.5%`）、`TreeBench`（**`33.1%`**）四榜取得最高宏平均准确率 **`50.4%`**。

---

## 4. 仓库结构与快速运行

```text
Coord-OPD/
├── assets/
│   ├── hero_geo_qkl.png        # 核心方法示意图
│   └── results_overview.png    # 论文定量结果与探针图表
├── geo_qkl/
│   ├── geo_feas.py             # 精确四坐标整数几何 Oracle 与数字语法状态机
│   ├── geo_feas_area.py        # 参数化面积上限 P/Q 的精确 Oracle
│   ├── tk_core.py              # Geo-QKL 商空间损失与教师参考框解码
│   ├── v1_core.py              # Qwen3-VL 模型加载、LoRA 与裁剪工具执行
│   ├── pipeline.py             # 在线采样、Matched-view 打分与训练评测主循环
│   ├── train.py                # 训练与评测 CLI 入口
│   ├── evaluate.py             # 官方 ANLS / EM / 循环选项去偏评测脚本
│   └── selftest.py             # 精确几何 Oracle 暴力穷举与语法自检
├── results/
│   ├── qa_selected.csv         # DocVQA 与 InfographicVQA 主实验数据
│   └── box_all_seeds.csv       # 300 固定状态证据漏框率探针数据
└── scripts/
    └── draw_results.py         # 生成 assets/results_overview.png 的绘图脚本
```

```bash
# 1. 运行 CPU 精确几何 Oracle 自检（191,520 次暴力穷举核对 + 语法验证）
python -m geo_qkl.selftest

# 2. 启动 Geo-QKL 训练与评测（单卡 48GB GPU）
python -m geo_qkl.train WQKG256_s17 256 17 eval

# 3. 重新生成定量结果图表
python scripts/draw_results.py
```
