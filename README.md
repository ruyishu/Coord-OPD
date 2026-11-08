# Geo-QKL：零标注、2B 反超 8B 教师与 1.0 MP 高清直读的视觉文档智能体精确可行集蒸馏

![Geo-QKL 核心架构图](assets/hero_geo_qkl.png)

**一句话简介**：**Geo-QKL** 面向高分辨率视觉文档智能体（`Qwen3-VL-8B-Instruct` 蒸馏 `Qwen3-VL-2B-Instruct`），**既不需要任何人工标注框或标准答案标签，也不增加任何可训练参数**，仅靠**读写视图对齐（Matched-view OPD）+ 四坐标精确整数几何 Oracle + 商空间逆向 KL（Quotient Reverse KL）**，在**仅消耗不到 `1.0 MP` 全图直读一半总像素（`~0.48 MP`）且显著减少裁剪工具调用**的前提下，让 **2B 学生在 DocVQA 上以 1/4 参数量同时反超 8B 教师（`82.90% vs. 81.99%`）和 4× 分辨率 `1.0 MP` 高清单视图（`82.90%/75.40% vs. 82.10%/70.40%`）**，并在双基准上全面超越 `ViCrop`、`GKD (ICLR'24)`、`NTL (ICML'25)`、`RAFT (ICLR'25)` 与 `TRT-adapted (2026)`。

---

## 一、Motivation（为什么要做 Geo-QKL？）

高分辨率文档（DocVQA）和信息图（InfographicVQA）缩放到全局概览预算（`0.25 MP`）后，细小文字和密集表格数字会模糊丢失。视觉智能体通过“看 `0.25 MP` 低清概览 $\to$ 发出裁剪工具调用 `[x1, y1, x2, y2]`（`0..1000` 整数网格）$\to$ 切取 `1.0 MP` 高清局部图拼回上下文 $\to$ 最多裁剪 3 次后作答”来读取证据。但在无标注蒸馏裁剪智能体时，现有路线各自卡在一个致命痛点上：

1. **标准逐词在线蒸馏（Token-level OPD / GKD）——把“取证动作”当成了“死板的文本抄写”**
   - 直接把 `[x1, y1, x2, y2]` 的每个坐标数字当成普通文本 token，强迫学生复刻教师在每一位数字上的完整概率分布（Reverse KL 或 Forward KL `GKD`）。
   - **痛点**：**裁剪框是“获取证据的动作（Evidence-Acquisition Action）”，不是单目标检测框**。只框住 `"501"` 的紧框，与连带左侧字段名 `"Serial No. 501"` 一起框进来的宽框，都能提供完整答题证据。固定阅读器实验直接证明：把教师框 $b_T$ 外扩 $1.5\times$，它与人工证据框的 IoU 直接腰斩（`26.3% -> 12.6%`），但答题准确率反而从 `85.9%` 升至 **`87.9%`**；而把等面积框平移偏离证据区则会暴跌至 `70.4%`。
2. **一维数字感知损失（`NTL` ICML'25、`RAFT` ICLR'25）与可行集标量重加权（`TRT-adapted` 2026）——忽略跨数位二维耦合或未合并可行分支**
   - **`NTL`（1D Wasserstein 数字距离）与 `RAFT`（数字高斯核平滑）**：仅在单个十进制数位 $0..9$ 上建模一维数值距离或平滑，忽略了 $(x_1, y_1, x_2, y_2)$ 四个坐标 12 个数位之间的二维包含与前缀耦合约束，甚至不如标准 `OPD`（DocVQA `76.90% / 76.27%` vs. `79.55%`）。
   - **`TRT-adapted`（可行集验证加权 KL）**：虽用前缀可行集 $\mathcal{F}_t$ 校验教师 top-1 数位并做置信度重加权（拿到外部方法最高分 `78.60%`），但内部仍对 $\mathcal{F}_t$ 中的其他合法替代数字施加逐 token KL 惩罚，落后 `Geo-QKL` **`+4.30 ANLS / +6.40 EM`**。
3. **内部坐标放松探索（统一降权、事后整框门控、稀疏候选框近似）**
   - **统一坐标降权（$\times 0.25$）**、**整框事后门控（Box-level Gate）**、**稀疏候选集（$\le 13$ 框）** 无法在逐位生成数字的前缀阶段完整释放所有几何等价分支。

**👉 我们的破局点（Geo-QKL）**：
在**零额外参数、零人工标注**的前提下——**在学生逐位写坐标数字的每一步前缀 $h_t$，用精确几何算法把所有还能补全成好框的下一数字集合 $\mathcal{F}(h_t)$ 100% 算准，把集合内部的老师个人偏好彻底合并释放，同时对一切跳出好框范围的错误数字保持 100% 严厉纠偏！**

---

## 二、Method（核心方法：两步走框架）

整套方法将 8B 教师全程冻结，仅训练 2B 学生语言层的轻量 LoRA（$r=16, \alpha=32$），由前后衔接的两部分组成：

### 第一步：读写视图对齐与 CPU 精确四坐标几何 Oracle（`geo_qkl/geo_feas.py`）

1. **读写视图对齐（`Matched-view OPD`）**
   - 我们在非坐标文本 token 上让教师同样使用 **`0.25 MP` 概览图**（对齐阅读证据），仅在坐标 token 上使用 **`1.0 MP` 概览图**（提供精准定位引导）。
2. **教师锚定的几何可接受框集 $\mathcal{A}(b_T)$**
   - 在每个裁剪回合起点，教师从学生的框前前缀出发，用 `1.0 MP` 视图贪心解码参考框 $b_T$。
   - 凡满足**“高覆盖且面积受控”**（覆盖率 $|b \cap b_T| / |b_T| \ge 0.85$ 且面积比 $|b| / |b_T| \le 3.5$）或**“高交并比”**（$\mathrm{IoU}(b, b_T) \ge 0.5$）的框 $b$，均属于可接受集 $\mathcal{A}(b_T)$。在 $0..1000$ 整数网格上等价于无浮点误差的整数不等式：$(100\,i \ge 85\,a_T \wedge 2a \le 7a_T) \vee 3i \ge a + a_T$。
3. **精确四坐标前缀可行性 Oracle（`GeoOracle` & `GeoGrammar`）**
   - 利用**固定水平区间时 $y$ 轴最优响应的闭式整数区间解**，以及**右边界 $x_2 > t_{x_2}$ 被 $x_2 \le t_{x_2}$ 严格支配（non-dominated）**的几何性质，在 CPU 上向量化求出任意数字前缀 $h_t$ 下仍能补全进 $\mathcal{A}(b_T)$ 的精确可行下一数字集合 $\mathcal{F}(h_t)$（191,520 次子网格暴力穷举核对 **0 误差**，在真实前缀的 **79.4%** 位置识别出多可行数字）。

### 第二步：商空间逆向 KL 蒸馏 `Geo-QKL`（`geo_qkl/tk_core.py` & `pipeline.py`）
- 记学生和教师分给可行集 $\mathcal{F}(h_t)$ 的总概率为 $M_S = \sum_{v \in \mathcal{F}(h_t)} p_S(v \mid h_t)$ 与 $M_T = \sum_{v \in \mathcal{F}(h_t)} p_T(v \mid h_t)$。`Geo-QKL` 将所有可行下一数字合并为单一等价事件后计算商空间逆向 KL：
  $\mathcal{L}_{\mathrm{Geo}}(h_t) = M_S \log(M_S / M_T) + \sum_{v \notin \mathcal{F}(h_t)} p_S(v \mid h_t) \log(p_S(v \mid h_t) / p_T(v \mid h_t))$。
- **链式法则精确手术**：由 KL 散度链式法则，$\mathrm{KL}(p_S \| p_T) = \mathcal{L}_{\mathrm{Geo}}(h_t) + M_S\,\mathrm{KL}(p_S(\cdot \mid \mathcal{F}) \| p_T(\cdot \mid \mathcal{F}))$。`Geo-QKL` **不多不少恰好删掉了集合内部的教师偏好排序项**（占坐标总 KL 质量的 **77.3%**），同时 100% 保留可行总概率对齐（$M_S \approx M_T$）与所有不可行数字的逐项纠偏。

---

## 三、Experiment & Results（主表对比、外部基线与消融实验）

### 1. 主实验对比（Table 1：256 Updates，含 5 个外部方法与高分辨率参照）

| 方法类别 | 方法名称 | 来源 / 机制 | **DocVQA (500p) ANLS / EM ↑** | **DocVQA Crops ↓** | **InfographicVQA (1,577q) ANLS / EM ↑** | **InfoVQA Crops ↓** |
|---|---|---|:---:|:---:|:---:|:---:|
| *Reference (上界与分辨率参照)* | `Teacher (Qwen3-VL-8B)` | 0.25 MP 概览 + 裁剪（4× 参数量） | 81.99 / 73.00 | 1.19 | 65.75 / 60.37 | 1.27 |
| | `High-Res Single-View (2B, 1.0 MP)` | 1.00 MP 全图直读（~2.1× 总像素，不裁剪） | 82.10 / 70.40 | — | 56.31 / 46.23 | — |
| | `Student w/ 1.0 MP Overview (2B)` | 1.00 MP 概览 + 裁剪（未蒸馏智能体） | 72.50 / 61.80 | 0.67 | 46.18 / 38.87 | 1.46 |
| | `Student before distillation (2B)` | 0.25 MP 概览 + 裁剪（蒸馏前起点 S0） | 68.31 / 56.60 | 1.80 | 34.47 / 28.28 | 2.43 |
| *Training-Free 选框* | **`ViCrop`** | Zhang et al., ICLR 2025（注意力显著性选框） | 64.61 / 50.00 | **1.00** | 34.00 / 25.62 | **1.00** |
| *On-Policy Distillation (256 upd.)* | **`RAFT`** | Lukasik et al., ICLR 2025（数字高斯核平滑） | 76.27 / 66.40 | 1.39 | <u>49.81</u> / 42.23 | 2.12 |
| | **`NTL`** | Zausinger et al., ICML 2025（1D Wasserstein 数字距离） | 76.90 / 67.60 | 1.31 | 48.02 / 40.27 | 1.96 |
| | **`GKD`** | Agarwal et al., ICLR 2024（词表级正向 KL） | 77.49 / 66.20 | 1.27 | 48.14 / 40.65 | 1.78 |
| | **`TRT-adapted`** | Huang et al., 2026（前缀可行集校验加权 KL） | 78.60 / 69.00 | 1.29 | 49.54 / <u>42.61</u> | 2.04 |
| | **`OPD (Reverse KL)`** | 标准词表级逆向 KL | <u>79.55</u> / <u>70.40</u> | **1.00** | 48.97 / 42.55 | 2.00 |
| | 🌟 **`Geo-QKL (Ours)`** | **精确前缀可行域商空间逆向 KL** | **82.90 / 75.40** | <u>1.19</u> | **55.31 / 49.08** | <u>1.64</u> |

---

### 2. 低预算对比（64 Updates，仅 1/4 训练步数）与内部消融（Table 2）

| 方法 (64 Updates) | DocVQA (500p) ANLS / EM ↑ | DocVQA Crops ↓ | InfographicVQA (1,577q) ANLS / EM ↑ | InfoVQA Crops ↓ |
|---|:---:|:---:|:---:|:---:|
| `NTL` (ICML 2025) | 73.77 / 64.60 | 1.27 | 44.79 / 36.72 | 1.18 |
| `TRT-adapted` (2026) | 76.16 / 67.40 | 1.08 | 48.19 / 39.57 | 1.16 |
| `GKD` (ICLR 2024) | 76.51 / 65.60 | 1.41 | 47.57 / 40.14 | 2.00 |
| `OPD (Reverse KL)` | 76.89 / 68.40 | 1.43 | 43.60 / 36.91 | 1.70 |
| `Matched-view OPD` | 76.53 / 69.00 | 1.52 | 44.13 / 37.86 | 2.25 |
| 🌟 **`Geo-QKL (Ours, 64 upd.)`** | **78.20 / 69.60** | **1.13** | **49.03 / 42.36** | **1.11** |

![核心消融与机制验证](assets/fig_ablations.png)

---

## 四、仓库结构与快速复现

```text
├── geo_qkl/
│   ├── geo_feas.py           # 精确四坐标整数几何 Oracle 与数字语法状态机 (GeoOracle, GeoGrammar)
│   ├── geo_feas_area.py      # 参数化有理数面积上限 P/Q 的精确几何 Oracle (α ∈ {2.0, 3.5, 5.0})
│   ├── tk_core.py            # 商空间逆向 KL 损失 (L_Geo)、稀疏集近似 (QK1)、整框门控 (WAT) 与教师框解码
│   ├── gkd_core.py           # GKD (ICLR 2024) 正向 KL 坐标蒸馏实现
│   ├── ntl_core.py           # NTL (ICML 2025) 1D 1-Wasserstein 数字距离坐标蒸馏实现
│   ├── raft_core.py          # RAFT (ICLR 2025) 坐标数字高斯核平滑蒸馏实现
│   ├── trt_core.py           # TRT-adapted (2026) 前缀可行集校验与置信度重加权蒸馏实现
│   ├── run_eval_vicrop.py    # ViCrop (ICLR 2025) 免训练注意力显著性裁剪评测脚本
│   ├── run_eval_hires.py     # 1.0 MP 高清单视图参照评测脚本
│   ├── v1_core.py            # Qwen3-VL 师生模型加载、LoRA 注入与高分辨率裁剪工具执行
│   ├── pipeline.py           # 在线策略采样 (On-Policy Rollout)、Matched-view 打分与训练评测主循环
│   ├── train.py              # Geo-QKL 训练与评测命令行入口
│   ├── evaluate.py           # 官方 ANLS / EM / 循环选项去偏重打分与页面聚类统计
│   └── selftest.py           # CPU 精确几何 Oracle 暴力穷举 (191,520 次核对 0 误差) 与语法验证
├── scripts/
│   └── plot_results.py       # 一键生成 assets/ 下的实验结果与消融图表
├── results/
│   ├── qa_selected.csv       # DocVQA 与 InfographicVQA 主实验及外部基线评测结果数据
│   └── box_all_seeds.csv     # 300 个固定 DocVQA 状态的证据漏框率探针数据
└── assets/                   # 核心方法架构图、主实验对比图与消融机制图
```

### 核心复现命令

```bash
# 1. CPU 精确几何 Oracle 自检 (191,520 次子网格暴力穷举 0 误差 + 语法验证)
python -m geo_qkl.selftest

# 2. 启动 Geo-QKL 在线策略蒸馏训练与评测 (64 步 / 256 步，单卡 48GB GPU)
python -m geo_qkl.train WQKG_lr1e4_s2026 64 2026 eval
python -m geo_qkl.train WQKG256_lr1e4_s17 256 17 eval

# 3. 运行全量评测重打分与多基准汇总
python -m geo_qkl.evaluate
```
