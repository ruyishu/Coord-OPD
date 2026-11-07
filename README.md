# Geo-QKL: Feasible-Set Distillation for Visual Document Agents

**[中文版 README](README_zh.md)**

**Geo-QKL** distills a frozen 8B vision-language teacher (`Qwen3-VL-8B-Instruct`) into a 2B visual document agent (`Qwen3-VL-2B-Instruct`) **without box or answer labels**. By merging prefix-feasible coordinate digits under an exact four-coordinate geometry oracle, it removes the teacher's arbitrary preferences among equally valid crops and lets the **2B student surpass the 8B teacher on DocVQA**.

![Geo-QKL Overview](assets/hero_geo_qkl.png)

---

## Motivation

- **A crop is an evidence-acquisition action, not a detection box.** A tight box around a value and a wider box that retains its field label expose the exact same answer evidence.
- **Token-level OPD over-penalizes valid crops.** Standard on-policy distillation (OPD) matches every coordinate digit $[x_1,y_1,x_2,y_2]$ as plain text, penalizing context-preserving crops whenever their digits differ from the teacher's top choice.
- **Direct evidence:** With a fixed reader on 125 DocVQA questions, enlarging the teacher's box $b_T$ by $1.5\times$ halves ground-truth IoU (`26.3% → 12.6%`) yet raises QA accuracy (`85.9 → 87.9` ANLS), whereas shifting the same box off the evidence drops accuracy to `70.4`.

---

## Method

We distill **which crops the teacher accepts**, not **which exact digits the teacher prefers**:

1. **Teacher-Anchored Acceptable Set $\mathcal{A}(b_T)$** — At each crop turn, the teacher greedily decodes a reference box $b_T$ from the student's pre-box prefix under a `1.0 MP` view. A candidate box $b$ is acceptable if it covers $b_T$ within an area cap ($\alpha=3.5$) or overlaps it strongly:

$$
b \in \mathcal{A}(b_T) \iff \left(\frac{|b \cap b_T|}{|b_T|} \ge 0.85 \;\wedge\; \frac{|b|}{|b_T|} \le 3.5\right) \;\lor\; \mathrm{IoU}(b, b_T) \ge 0.5
$$

2. **Exact Prefix-Feasible Oracle (`geo_qkl/geo_feas.py`)** — As the student writes `[x1, y1, x2, y2]` digit by digit, an exact integer oracle on the $0..1000$ grid (closed-form $y$-interval best response + non-dominated $x_2 \le t_{x_2}$ reduction) identifies all next digits $v \in \mathcal{F}(h_t)$ that still admit a valid completion in $\mathcal{A}(b_T)$.

3. **Geo-QKL Quotient Loss (`geo_qkl/tk_core.py`)** — Let $M_S = \sum_{v \in \mathcal{F}(h_t)} p_S(v\mid h_t)$ and $M_T = \sum_{v \in \mathcal{F}(h_t)} p_T(v\mid h_t)$. Merging feasible digits before computing reverse KL yields:

$$
\mathcal{L}_{\mathrm{Geo}}(h_t) = M_S \log \frac{M_S}{M_T} + \sum_{v \notin \mathcal{F}(h_t)} p_S(v\mid h_t) \log \frac{p_S(v\mid h_t)}{p_T(v\mid h_t)}
$$

By the KL chain rule, $\mathrm{KL}(p_S \| p_T) = \mathcal{L}_{\mathrm{Geo}}(h_t) + M_S\,\mathrm{KL}(p_S(\cdot\mid\mathcal{F}) \| p_T(\cdot\mid\mathcal{F}))$. **Geo-QKL** drops only the within-set preference term while keeping total feasible mass matching and per-token infeasible penalties intact.

---

## Results

![Quantitative Results](assets/results_overview.png)

| Method | Updates | DocVQA ANLS ↑ | DocVQA EM ↑ | DocVQA Crops ↓ | InfographicVQA ANLS ↑ | InfographicVQA EM ↑ | InfographicVQA Crops ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Student before distillation (2B) | -- | 68.31 | 56.60 | 1.80 | 34.47 | 28.28 | 2.43 |
| Teacher (8B, frozen) | -- | 81.99 | 73.00 | 1.19 | 65.75 | 60.37 | 1.27 |
| OPD (1 MP teacher on all tokens) | 64 | 76.89 | 68.40 | 1.43 | 43.60 | 36.91 | 1.70 |
| Matched-view OPD | 64 | 76.53 | 69.00 | 1.52 | 44.13 | 37.86 | 2.25 |
| **Geo-QKL (ours)** | 64 | **78.20** | **69.60** | **1.13** | **49.03** | **42.36** | **1.11** |
| OPD (1 MP teacher on all tokens) | 256 | 79.55 | 70.40 | **1.00** | 48.97 | 42.55 | 2.00 |
| Matched-view OPD | 256 | 79.26 | 70.80 | 1.30 | 50.82 | 44.32 | 1.87 |
| **Geo-QKL (ours)** | 256 | **82.90** | **75.40** | 1.19 | **55.31** | **49.08** | **1.64** |

- **2B beats 8B Teacher on DocVQA:** Reaches **`82.90` ANLS / `75.40` EM** at 256 updates (`+3.64` ANLS / `+4.60` EM over Matched-view OPD, `+0.91` ANLS / `+2.40` EM above the 8B teacher).
- **Large gains with fewer crops on InfographicVQA:** Gains **`+4.90` ANLS** (`51%` fewer crops) at 64 updates and **`+4.49` ANLS** (`+6.34` over OPD) at 256 updates; wins **15/16** ANLS and **16/16** EM paired comparisons across 4 seeds.
- **Exact oracle & zero-shot transfer:** Beats the sparse candidate set (`+3.52` Dev / `+1.26` Held-out ANLS), uniform down-weighting (`+2.79` ANLS), and box-level gating (`+1.81` Dev ANLS), while achieving the highest average accuracy (**`50.4%`**) across `V*`, `HR-Bench 4K/8K`, and `TreeBench`.

---

## Quickstart

```bash
# 1. CPU exact oracle verification (191,520 exhaustive checks, 0 mismatches)
python -m geo_qkl.selftest

# 2. Train & evaluate Geo-QKL (single 48GB GPU)
python -m geo_qkl.train WQKG256_s17 256 17 eval

# 3. Redraw quantitative results chart from results/*.csv
python scripts/draw_results.py
```
