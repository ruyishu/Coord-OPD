"""绘制 Geo-QKL 仓库核心实验结果图表（对齐 SpinVL / CodeV 浅色清晰学术风格）：
1. assets/fig_main_results.png：主实验对比（DocVQA 与 InfographicVQA 在 64/256 步的 ANLS & EM 对比 + 平均每题裁剪开销缩减）
2. assets/fig_ablations.png：核心消融与机制验证（404 题开发集坐标目标全消融、固定阅读器 1.5x 扩框 vs 移框干预、300 状态证据漏框率 Miss Rate、精确 Oracle vs 稀疏集可行覆盖率）
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#2C3E50"
plt.rcParams["axes.linewidth"] = 0.9

C_INIT = "#CBD5E1"
C_OPD = "#93C5FD"
C_MV = "#FDBA74"
C_OURS = "#34D399"
C_OURS_DARK = "#059669"
C_TEA = "#7C3AED"
C_BAD = "#FCA5A5"


def plot_main_results():
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(14.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1.15, 1.1]}, dpi=240
    )
    fig.patch.set_facecolor("white")

    # Panel (a): Held-out DocVQA (500 pages)
    groups = ["64 updates", "256 updates"]
    x = np.arange(len(groups))
    w = 0.24
    doc_opd = [76.89, 79.55]
    doc_mv = [76.53, 79.26]
    doc_ours = [78.20, 82.90]

    b1 = ax1.bar(x - w, doc_opd, width=w * 0.92, label="OPD (1 MP all tokens)", color="#CFE3F5", edgecolor="#4E7FA6", linewidth=1.0)
    b2 = ax1.bar(x, doc_mv, width=w * 0.92, label="Matched-view OPD", color="#F8DFC5", edgecolor="#C2824E", linewidth=1.0)
    b3 = ax1.bar(x + w, doc_ours, width=w * 0.92, label="Geo-QKL (Ours)", color="#BCE8D0", edgecolor="#1D6843", linewidth=1.3)

    ax1.axhline(81.99, color=C_TEA, linestyle="--", linewidth=1.4, label="8B Teacher (81.99%)")
    for bars, is_ours in [(b1, False), (b2, False), (b3, True)]:
        for b in bars:
            ax1.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.28, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=8.3 if is_ours else 7.8,
                fontweight="bold" if is_ours else "normal",
                color="#1D6843" if is_ours else "#334155",
            )
    ax1.text(0 + w, 79.35, "+1.67 ANLS", ha="center", fontsize=8.2, fontweight="bold", color="#1D6843",
             bbox=dict(boxstyle="round,pad=0.22", fc="#E7F6EE", ec="#8FD1AE", lw=0.8))
    ax1.text(1 + w, 84.05, "+3.64 ANLS\n(Beats 8B Teacher)", ha="center", fontsize=8.0, fontweight="bold", color="#1D6843",
             bbox=dict(boxstyle="round,pad=0.22", fc="#E7F6EE", ec="#8FD1AE", lw=0.8))
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=9.8, fontweight="bold")
    ax1.set_ylabel("ANLS (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(72.0, 86.2)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("(a) Held-out DocVQA (500 pages)", fontsize=10.8, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="#F8FAFC")

    # Panel (b): InfographicVQA (1,577 questions)
    info_opd = [43.60, 48.97]
    info_mv = [44.13, 50.82]
    info_ours = [49.03, 55.31]

    ib1 = ax2.bar(x - w, info_opd, width=w * 0.92, label="OPD", color="#CFE3F5", edgecolor="#4E7FA6", linewidth=1.0)
    ib2 = ax2.bar(x, info_mv, width=w * 0.92, label="Matched-view OPD", color="#F8DFC5", edgecolor="#C2824E", linewidth=1.0)
    ib3 = ax2.bar(x + w, info_ours, width=w * 0.92, label="Geo-QKL (Ours)", color="#BCE8D0", edgecolor="#1D6843", linewidth=1.3)

    for bars, is_ours in [(ib1, False), (ib2, False), (ib3, True)]:
        for b in bars:
            ax2.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=8.3 if is_ours else 7.8,
                fontweight="bold" if is_ours else "normal",
                color="#1D6843" if is_ours else "#334155",
            )
    ax2.text(0 + w, 50.9, "+4.90 ANLS", ha="center", fontsize=8.2, fontweight="bold", color="#1D6843",
             bbox=dict(boxstyle="round,pad=0.22", fc="#E7F6EE", ec="#8FD1AE", lw=0.8))
    ax2.text(1 + w, 57.1, "+4.49 vs. MV\n+6.34 vs. OPD", ha="center", fontsize=8.0, fontweight="bold", color="#1D6843",
             bbox=dict(boxstyle="round,pad=0.22", fc="#E7F6EE", ec="#8FD1AE", lw=0.8))
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("ANLS (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(38.0, 60.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("(b) InfographicVQA (1,577 questions)", fontsize=10.8, fontweight="bold", pad=10)

    # Panel (c): Average Crops per Question (lower is more efficient)
    settings = ["DocVQA\n(64 upd.)", "DocVQA\n(256 upd.)", "Infographic\n(64 upd.)", "Infographic\n(256 upd.)"]
    y = np.arange(len(settings))[::-1]
    h = 0.24
    c_opd = [1.43, 1.00, 1.70, 2.00]
    c_mv = [1.52, 1.30, 2.25, 1.87]
    c_ours = [1.13, 1.19, 1.11, 1.64]
    pcts = ["-26%", "-8%", "-51%", "-12%"]

    ax3.barh(y + h, c_opd, height=h * 0.9, color="#CFE3F5", edgecolor="#4E7FA6", linewidth=0.9, label="OPD")
    ax3.barh(y, c_mv, height=h * 0.9, color="#F8DFC5", edgecolor="#C2824E", linewidth=0.9, label="Matched-view OPD")
    ax3.barh(y - h, c_ours, height=h * 0.9, color="#BCE8D0", edgecolor="#1D6843", linewidth=1.25, label="Geo-QKL (Ours)")

    for idx in range(len(settings)):
        ax3.text(c_mv[idx] + 0.04, y[idx], f"{c_mv[idx]:.2f}", va="center", fontsize=7.6, color="#9A5F2E")
        ax3.text(c_ours[idx] + 0.04, y[idx] - h, f"{c_ours[idx]:.2f} ({pcts[idx]} crops)", va="center",
                 fontsize=8.0, fontweight="bold", color="#1D6843")
    ax3.set_yticks(y)
    ax3.set_yticklabels(settings, fontsize=9.0, fontweight="bold")
    ax3.set_xlabel("Average Crops per Question (↓ lower is better)", fontsize=9.5, fontweight="bold")
    ax3.set_xlim(0, 2.65)
    ax3.grid(axis="x", linestyle="--", alpha=0.35)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_title("(c) Tool-Call Efficiency (Crops / Question ↓)", fontsize=10.8, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_main_results.png", dpi=240)
    plt.close(fig)


def plot_ablations():
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(14.2, 4.3), gridspec_kw={"width_ratios": [1.25, 1.05, 1.05]}, dpi=240
    )
    fig.patch.set_facecolor("white")

    # Panel (a): Coordinate Objective Ablations on 404-question Dev Split (64 updates)
    names = ["Sparse Set\n(<=13 boxes)", "Down-wt\n(x0.25)", "Token KL\n(Matched)", "OPD\n(1 MP all)", "Box-Level\nGate", "Geo-QKL\n(Ours)"]
    vals = [84.09, 84.98, 85.19, 85.40, 85.80, 87.61]
    cols = ["#E2E8F0", "#FDE68A", "#F8DFC5", "#CFE3F5", "#DDD6FE", "#BCE8D0"]
    edges = ["#64748B", "#B45309", "#C2824E", "#4E7FA6", "#6D28D9", "#1D6843"]
    x1 = np.arange(len(names))
    bars = ax1.bar(x1, vals, width=0.58, color=cols, edgecolor=edges, linewidth=1.15)
    bars[-1].set_linewidth(1.5)
    for idx, b in enumerate(bars):
        ax1.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{b.get_height():.2f}",
            ha="center", va="bottom", fontsize=8.3 if idx == 5 else 7.8,
            fontweight="bold" if idx == 5 else "normal",
            color="#1D6843" if idx == 5 else "#334155",
        )
    ax1.set_xticks(x1)
    ax1.set_xticklabels(names, fontsize=8.3, fontweight="bold")
    ax1.set_ylabel("Dev ANLS (%)", fontsize=9.8, fontweight="bold")
    ax1.set_ylim(82.5, 88.6)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("(a) Coordinate Objective Ablation (DocVQA Dev)", fontsize=10.5, fontweight="bold", pad=10)

    # Panel (b): Fixed-Reader Crop Intervention (125 DocVQA Questions)
    crops = ["Shifted\n(IoU 0.2%)", "Student\n(IoU 14.6%)", "Tight bT\n(IoU 26.3%)", "1.5x bT\n(IoU 12.6%)", "GT Box\n(IoU 96.9%)"]
    c_anls = [70.39, 77.07, 85.87, 87.89, 88.56]
    c_cols = ["#F7D3D0", "#CFE3F5", "#F8DFC5", "#BCE8D0", "#E2E8F0"]
    c_edges = ["#C96A63", "#4E7FA6", "#C2824E", "#1D6843", "#64748B"]
    x2 = np.arange(len(crops))
    b_int = ax2.bar(x2, c_anls, width=0.56, color=c_cols, edgecolor=c_edges, linewidth=1.15)
    b_int[3].set_linewidth(1.5)
    for idx, b in enumerate(b_int):
        ax2.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}",
            ha="center", va="bottom", fontsize=8.3 if idx == 3 else 7.8,
            fontweight="bold" if idx == 3 else "normal",
            color="#1D6843" if idx == 3 else "#334155",
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(crops, fontsize=8.2, fontweight="bold")
    ax2.set_ylabel("Downstream Reader ANLS (%)", fontsize=9.8, fontweight="bold")
    ax2.set_ylim(64.0, 92.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("(b) Fixed-Reader Crop Intervention (125 Qs)", fontsize=10.5, fontweight="bold", pad=10)

    # Panel (c): Exact Oracle vs Sparse Set Coverage & 300-State Evidence Miss Rate
    cats = ["Multi-Token\nFeasible (%)", "KL Mass\nRelaxed (%)", "Evidence Hit\nRate (256u, %)"]
    sparse_v = [16.0, 13.0, 100.0 - 13.2]
    exact_v = [79.4, 77.3, 100.0 - 12.2]
    x3 = np.arange(len(cats))
    w3 = 0.32
    sb1 = ax3.bar(x3 - w3 / 2, sparse_v, width=w3, label="Sparse Set / Baseline", color="#F8DFC5", edgecolor="#C2824E", linewidth=1.0)
    sb2 = ax3.bar(x3 + w3 / 2, exact_v, width=w3, label="Geo-QKL (Exact Oracle)", color="#BCE8D0", edgecolor="#1D6843", linewidth=1.3)
    for b in sb1:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=7.8, color="#9A5F2E")
    for b in sb2:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#1D6843")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(cats, fontsize=8.5, fontweight="bold")
    ax3.set_ylabel("Percentage (%)", fontsize=9.8, fontweight="bold")
    ax3.set_ylim(0, 103)
    ax3.grid(axis="y", linestyle="--", alpha=0.35)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_title("(c) Oracle Coverage & Evidence Preservation", fontsize=10.5, fontweight="bold", pad=10)
    ax3.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="#F8FAFC")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_ablations.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    plot_main_results()
    plot_ablations()
    print("Generated assets/fig_main_results.png and assets/fig_ablations.png successfully.")
