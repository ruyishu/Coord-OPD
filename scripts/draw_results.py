"""Generate assets/results_overview.png from results/qa_selected.csv and results/box_all_seeds.csv."""
from pathlib import Path
import csv
import math
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

REPO = Path(__file__).resolve().parent.parent
HERE = REPO / "results"
ASSETS = REPO / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

for name, file in [("A", "arial.ttf"), ("B", "arialbd.ttf"), ("I", "ariali.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / file)))

def r_half_up(val, decimals="0.1"):
    return float(Decimal(str(round(float(val), 8))).quantize(Decimal(decimals), rounding=ROUND_HALF_UP))

# Load recomputed raw CSVs
def load_csv(name):
    rows = list(csv.DictReader((HERE / name).open(encoding="utf-8")))
    for r in rows:
        for k in ("steps", "seed", "n"):
            if k in r:
                r[k] = int(r[k])
        for k in ("anls", "normalized_em", "crops", "miss", "cover80", "mean_iou"):
            if k in r:
                r[k] = float(r[k])
    return rows

QA_ROWS = load_csv("qa_selected.csv")
BOX_ROWS = load_csv("box_all_seeds.csv")

def get_qa(bench, method, steps):
    return next(r for r in QA_ROWS if (r["benchmark"], r["method"], r["steps"]) == (bench, method, steps))

def get_miss(method, steps):
    vals = [r["miss"] for r in BOX_ROWS if r["method"] == method and r["steps"] == steps]
    assert len(vals) == 4
    return float(np.mean(vals)), sorted(vals)

# Canvas dimensions (matches fig3_framework width=1440 scaled to 504pt two-column width)
W, H = 1440, 565
SCALE = 504.0 / W

C = {
    "ink": "#22303A",
    "muted": "#5E717C",
    "light_muted": "#82949E",
    "grid": "#E6EDF1",
    "axis": "#AEBEC7",
    "card_bg": "#FFFFFF",
    "subcard_bg": "#F8FAFC",
    "subcard_edge": "#DCE5EC",
    # Method 1: OPD (soft sky blue)
    "opd_fill": "#CFE3F5",
    "opd_edge": "#5484B0",
    "opd_text": "#35648E",
    # Method 2: Matched-view OPD (soft warm apricot)
    "mv_fill": "#F8DFC5",
    "mv_edge": "#C2824E",
    "mv_text": "#9A5F2E",
    # Method 3: Geo-QKL (ours, soft mint emerald)
    "ours_fill": "#BCE8D0",
    "ours_em_fill": "#D8F2E4",
    "ours_edge": "#2E855A",
    "ours_text": "#1D6843",
    "ours_badge_bg": "#E7F6EE",
    "ours_badge_edge": "#8FD1AE",
    # Teacher & accents (soft lavender / coral)
    "tea_line": "#7B66A8",
    "tea_bg": "#F1EEFA",
    "tea_edge": "#B9ADD8",
    "coral_fill": "#F7D3D0",
    "coral_edge": "#C96A63",
    "coral_text": "#9E4740",
    # Region header tints (matching fig3_framework)
    "hdr_blue_bg": "#E3F2FA",
    "hdr_blue_edge": "#5D8FBD",
    "hdr_peach_bg": "#FFF1E4",
    "hdr_peach_edge": "#D98B52",
    "hdr_mint_bg": "#E7F6EC",
    "hdr_mint_edge": "#4E9A6F",
}

OUT_PDF = ASSETS / "results_overview.pdf"
c = canvas.Canvas(str(OUT_PDF), pagesize=(W * SCALE, H * SCALE), pageCompression=1)
c.setTitle("Figure 4: Quantitative performance, crop efficiency, and evidence preservation")
c.setAuthor("Anonymous")
c.scale(SCALE, SCALE)

def col(v):
    return white if v == "white" else HexColor(C.get(v, v))

def rr(x, y, w, h, r=8, fill="white", stroke=None, lw=1.1, dash=None):
    c.saveState()
    c.setLineWidth(lw)
    if fill is not None:
        c.setFillColor(col(fill))
    if stroke is not None:
        c.setStrokeColor(col(stroke))
    if dash:
        c.setDash(*dash)
    c.roundRect(x, H - y - h, w, h, r, fill=int(fill is not None), stroke=int(stroke is not None))
    c.restoreState()

def top_round_bar(x, y, w, h, r=4.5, fill="opd_fill", stroke="opd_edge", lw=1.15):
    """Draw a vertical bar with rounded top corners and flat bottom on the axis."""
    c.saveState()
    c.setLineWidth(lw)
    c.setFillColor(col(fill))
    c.setStrokeColor(col(stroke))
    r = min(r, w / 2.0, h / 2.0)
    by = H - y - h
    ty = H - y
    p = c.beginPath()
    p.moveTo(x, by)
    p.lineTo(x, ty - r)
    p.arcTo(x, ty - 2 * r, x + 2 * r, ty, 180, -90)
    p.lineTo(x + w - r, ty)
    p.arcTo(x + w - 2 * r, ty - 2 * r, x + w, ty, 90, -90)
    p.lineTo(x + w, by)
    p.close()
    c.drawPath(p, fill=1, stroke=1)
    c.restoreState()

def right_round_bar(x, y, w, h, r=4.0, fill="opd_fill", stroke="opd_edge", lw=1.1):
    """Draw a horizontal bar with rounded right corners and flat left on the axis."""
    c.saveState()
    c.setLineWidth(lw)
    c.setFillColor(col(fill))
    c.setStrokeColor(col(stroke))
    r = min(r, w / 2.0, h / 2.0)
    by = H - y - h
    ty = H - y
    p = c.beginPath()
    p.moveTo(x, by)
    p.lineTo(x + w - r, by)
    p.arcTo(x + w - 2 * r, by, x + w, by + 2 * r, -90, 90)
    p.lineTo(x + w, ty - r)
    p.arcTo(x + w - 2 * r, ty - 2 * r, x + w, ty, 0, 90)
    p.lineTo(x, ty)
    p.close()
    c.drawPath(p, fill=1, stroke=1)
    c.restoreState()

def tx(x, y, s, size=14, bold=False, color="ink", align="left", italic=False):
    c.saveState()
    c.setFillColor(col(color))
    font = "I" if italic else ("B" if bold else "A")
    c.setFont(font, size)
    {"left": c.drawString, "center": c.drawCentredString, "right": c.drawRightString}[align](
        x, H - y - 0.80 * size, s
    )
    c.restoreState()

def ln(points, color="axis", lw=1.1, dash=None, arrow=False):
    c.saveState()
    c.setStrokeColor(col(color))
    c.setFillColor(col(color))
    c.setLineWidth(lw)
    c.setLineCap(1)
    c.setLineJoin(1)
    if dash:
        c.setDash(*dash)
    p = c.beginPath()
    p.moveTo(points[0][0], H - points[0][1])
    for x, y in points[1:]:
        p.lineTo(x, H - y)
    c.drawPath(p, fill=0, stroke=1)
    if arrow:
        x, y = points[-1]
        px, py = points[-2]
        a = math.atan2(y - py, x - px)
        pa = c.beginPath()
        pa.moveTo(x, H - y)
        for sg in (1, -1):
            pa.lineTo(
                x - 7.0 * math.cos(a) + sg * 3.3 * math.sin(a),
                H - (y - 7.0 * math.sin(a) - sg * 3.3 * math.cos(a)),
            )
        pa.close()
        c.drawPath(pa, fill=1, stroke=0)
    c.restoreState()

def circle(x, y, r=3.0, fill="white", stroke="ink", lw=1.0):
    c.saveState()
    c.setLineWidth(lw)
    c.setFillColor(col(fill))
    c.setStrokeColor(col(stroke))
    c.circle(x, H - y, r, fill=1, stroke=1)
    c.restoreState()

def region_card(x, y, w, h, title, subtitle, bg_key, edge_key):
    rr(x, y, w, h, r=12, fill="card_bg", stroke=edge_key, lw=1.4, dash=(6, 4))
    tw = pdfmetrics.stringWidth(title, "B", 16.5) + 26
    rr(x + 12, y - 1, tw, 30, r=7, fill=bg_key, stroke=edge_key, lw=0.95)
    tx(x + 25, y + 6.5, title, size=16.5, bold=True, color="ink")
    if subtitle:
        tx(x + tw + 22, y + 8.5, subtitle, size=12.5, italic=True, color="muted")

# ==============================================================================
# 0. TOP SHARED LEGEND BAR
# ==============================================================================
rr(170, 8, 1100, 32, r=10, fill="subcard_bg", stroke="subcard_edge", lw=1.0)

leg_items = [
    (198, "opd_fill", "opd_edge", "OPD (1 MP teacher on all tokens)", False),
    (468, "mv_fill", "mv_edge", "Matched-view OPD (baseline)", False),
    (718, "ours_fill", "ours_edge", "Geo-QKL (ours, exact feasible set)", True),
]
for lx, f_k, e_k, label, is_b in leg_items:
    rr(lx, 15.5, 22, 16, r=4, fill=f_k, stroke=e_k, lw=1.25)
    tx(lx + 30, 16.5, label, size=13.5, bold=is_b, color="ours_text" if is_b else "ink")

ln([(1005, 23.5), (1039, 23.5)], color="tea_line", lw=2.0, dash=(4, 3))
tx(1047, 16.5, "8B Teacher reference", size=13.2, bold=False, color="tea_line")

# ==============================================================================
# 1. PANEL A: QUESTION ANSWERING GAINS (DocVQA & InfographicVQA)
# ==============================================================================
PA_X, PA_Y, PA_W, PA_H = 14, 50, 566, 500
region_card(
    PA_X, PA_Y, PA_W, PA_H,
    "A  Question Answering Accuracy",
    "ANLS (%) ↑ & EM (%)",
    "hdr_blue_bg", "hdr_blue_edge",
)

def draw_qa_subchart(sx, sy, sw, sh, bench, title, sub, ymin, ymax, yticks, tea_val=None):
    rr(sx, sy, sw, sh, r=9, fill="subcard_bg", stroke="subcard_edge", lw=0.9)
    tx(sx + 12, sy + 10, title, size=14.5, bold=True, color="ink")
    tx(sx + sw - 10, sy + 11.5, sub, size=11.5, color="muted", align="right")

    plot_x0 = sx + 38
    plot_x1 = sx + sw - 12
    plot_y0 = sy + 68
    plot_y1 = sy + sh - 54
    plot_h = plot_y1 - plot_y0

    def y_of(v):
        return plot_y1 - (v - ymin) / (ymax - ymin) * plot_h

    # Gridlines & Y-ticks
    for yv in yticks:
        yy = y_of(yv)
        ln([(plot_x0, yy), (plot_x1, yy)], color="grid", lw=0.9)
        tx(plot_x0 - 6, yy - 5.5, f"{yv}", size=11.5, color="muted", align="right")

    # Teacher reference line (if within range)
    if tea_val is not None:
        ty = y_of(tea_val)
        ln([(plot_x0, ty), (plot_x1, ty)], color="tea_line", lw=1.6, dash=(5, 3))
        rr(plot_x0 + 4, ty - 19, 128, 16, r=4, fill="tea_bg", stroke="tea_edge", lw=0.7)
        tx(plot_x0 + 68, ty - 16.5, f"8B Teacher: {tea_val:.2f}%", size=10.5, bold=True, color="tea_line", align="center")

    # Bars for 64 and 256 updates
    groups = [("64 updates", 64, 0.27), ("256 updates", 256, 0.75)]
    m_specs = [
        ("WWH", "opd_fill", "opd_edge", "opd_text"),
        ("WHS", "mv_fill", "mv_edge", "mv_text"),
        ("WQKG", "ours_fill", "ours_edge", "ours_text"),
    ]
    bw = 24.0
    gap = 4.5

    for g_label, steps, frac in groups:
        gx = plot_x0 + frac * (plot_x1 - plot_x0)
        xs = [gx - 1.5 * bw - gap, gx - 0.5 * bw, gx + 0.5 * bw + gap]
        vals = []
        for idx, (m_code, f_col, e_col, t_col) in enumerate(m_specs):
            row = get_qa(bench, m_code, steps)
            v_anls = r_half_up(row["anls"], "0.01")
            v_em = r_half_up(row["normalized_em"], "0.01")
            vals.append((xs[idx], v_anls, v_em, row["crops"]))
            by = y_of(v_anls)
            bh = plot_y1 - by
            top_round_bar(
                xs[idx], by, bw, bh, r=4.0,
                fill=f_col, stroke=e_col, lw=1.4 if m_code == "WQKG" else 1.05
            )
            # Value label above bar
            lbl_y = by - (15.5 if m_code == "WQKG" else 14.0)
            tx(
                xs[idx] + bw / 2, lbl_y, f"{v_anls:.2f}",
                size=11.0 if m_code == "WQKG" else 10.0,
                bold=(m_code == "WQKG"),
                color=t_col, align="center",
            )
            # Subtle EM label inside bottom of bar
            tx(
                xs[idx] + bw / 2, plot_y1 - 15, f"{v_em:.1f}",
                size=9.2, bold=(m_code == "WQKG"),
                color=t_col, align="center",
            )

        # Gain callout pill above the 3 bars (using rounded table numbers for exact consistency)
        mv_anls = vals[1][1]
        ours_anls = vals[2][1]
        gain = ours_anls - mv_anls
        top_bar_y = min(y_of(v[1]) for v in vals)
        pill_y = top_bar_y - 38
        pill_w = 80
        rr(gx - pill_w / 2 + 2, pill_y, pill_w, 17, r=5, fill="ours_badge_bg", stroke="ours_badge_edge", lw=0.85)
        tx(gx + 2, pill_y + 2.8, f"+{gain:.2f} ANLS", size=10.8, bold=True, color="ours_text", align="center")

        # X-axis group label
        tx(gx, plot_y1 + 10, g_label, size=12.2, bold=True, color="ink", align="center")

    # Axes lines
    ln([(plot_x0, plot_y0), (plot_x0, plot_y1)], color="axis", lw=1.15)
    ln([(plot_x0, plot_y1), (plot_x1, plot_y1)], color="axis", lw=1.15)
    tx(sx + 12, plot_y1 + 31, "Numbers inside bars: Exact Match (EM %)", size=10.5, italic=True, color="muted")

draw_qa_subchart(
    PA_X + 12, PA_Y + 40, 265, 446,
    "docvqa_confirm", "(a1) Held-out DocVQA", "500 pages",
    72, 85.2, [72, 75, 78, 81, 84], tea_val=81.99,
)
draw_qa_subchart(
    PA_X + 289, PA_Y + 40, 265, 446,
    "infographicvqa_val_sub", "(a2) InfographicVQA", "1,577 questions",
    38, 58.5, [40, 44, 48, 52, 56], tea_val=None,
)

# ==============================================================================
# 2. PANEL B: TOOL-CALL EFFICIENCY & NET GAIN CHARTS
# ==============================================================================
PB_X, PB_Y, PB_W, PB_H = 592, 50, 412, 500
region_card(
    PB_X, PB_Y, PB_W, PB_H,
    "B  Crop Efficiency & Gains",
    "Fewer crops ↓, higher QA ↑",
    "hdr_peach_bg", "hdr_peach_edge",
)

# Top subcard (b1): Horizontal Grouped Bar Chart of Crops per Question
rr(PB_X + 12, PB_Y + 40, PB_W - 24, 252, r=9, fill="subcard_bg", stroke="subcard_edge", lw=0.9)
tx(PB_X + 24, PB_Y + 49, "(b1) Average Crops per Question (↓ lower is better)", size=13.5, bold=True, color="ink")

bx0 = PB_X + 118
bx1 = PB_X + PB_W - 86
by0 = PB_Y + 74
by1 = PB_Y + 264
xmax_crop = 2.50

def x_of_crop(c_val):
    return bx0 + (c_val / xmax_crop) * (bx1 - bx0)

for xv in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
    xx = x_of_crop(xv)
    ln([(xx, by0), (xx, by1)], color="grid", lw=0.85)
    tx(xx, by1 + 5, f"{xv:.1f}", size=10.5, color="muted", align="center")

crop_groups = [
    ("DocVQA\n(64 upd.)", "docvqa_confirm", 64),
    ("DocVQA\n(256 upd.)", "docvqa_confirm", 256),
    ("Infographic\n(64 upd.)", "infographicvqa_val_sub", 64),
    ("Infographic\n(256 upd.)", "infographicvqa_val_sub", 256),
]

row_h = (by1 - by0) / 4.0
bar_h = 10.2
for g_idx, (g_lbl, bench, steps) in enumerate(crop_groups):
    cy = by0 + (g_idx + 0.5) * row_h
    parts = g_lbl.split("\n")
    tx(bx0 - 8, cy - 10, parts[0], size=11.2, bold=True, color="ink", align="right")
    tx(bx0 - 8, cy + 2, parts[1], size=10.0, color="muted", align="right")

    m_list = [
        ("WWH", "opd_fill", "opd_edge", "opd_text", -bar_h - 1.3),
        ("WHS", "mv_fill", "mv_edge", "mv_text", 0.0),
        ("WQKG", "ours_fill", "ours_edge", "ours_text", bar_h + 1.3),
    ]
    c_vals = {}
    for m_code, f_col, e_col, t_col, dy in m_list:
        cv = r_half_up(get_qa(bench, m_code, steps)["crops"], "0.01")
        c_vals[m_code] = cv
        bw_c = x_of_crop(cv) - bx0
        right_round_bar(
            bx0, cy + dy - bar_h / 2, bw_c, bar_h, r=3.0,
            fill=f_col, stroke=e_col, lw=1.2 if m_code == "WQKG" else 0.9
        )
        tx(
            bx0 + bw_c + 4, cy + dy - 4.8, f"{cv:.2f}",
            size=10.0 if m_code == "WQKG" else 9.4,
            bold=(m_code == "WQKG"), color=t_col, align="left"
        )

    # Reduction pill vs Matched-view OPD
    pct = 100.0 * (c_vals["WQKG"] - c_vals["WHS"]) / c_vals["WHS"]
    rr(bx1 + 10, cy - 9.5, 62, 19, r=5, fill="ours_badge_bg", stroke="ours_badge_edge", lw=0.85)
    tx(bx1 + 41, cy - 5.8, f"{pct:.0f}% crops", size=10.2, bold=True, color="ours_text", align="center")

ln([(bx0, by0), (bx0, by1)], color="axis", lw=1.15)
ln([(bx0, by1), (bx1, by1)], color="axis", lw=1.15)

# Bottom subcard (b2): Grouped Horizontal Gain Chart over Matched-View OPD (Δ ANLS & Δ EM)
rr(PB_X + 12, PB_Y + 302, PB_W - 24, 184, r=9, fill="subcard_bg", stroke="subcard_edge", lw=0.9)
tx(PB_X + 24, PB_Y + 311, "(b2) Gain over Matched-View OPD (Δ ANLS & Δ EM ↑)", size=13.2, bold=True, color="ink")

# Small inline legend for Δ ANLS vs Δ EM
rr(PB_X + 250, PB_Y + 330, 14, 9, r=2, fill="ours_fill", stroke="ours_edge", lw=1.0)
tx(PB_X + 268, PB_Y + 328.5, "Δ ANLS", size=10.0, bold=True, color="ours_text")
rr(PB_X + 320, PB_Y + 330, 14, 9, r=2, fill="ours_em_fill", stroke="ours_edge", lw=0.85)
tx(PB_X + 338, PB_Y + 328.5, "Δ EM", size=10.0, bold=False, color="ours_text")

gx0 = PB_X + 118
gx1 = PB_X + PB_W - 52
gy0 = PB_Y + 344
gy1 = PB_Y + 462
g_xmax = 5.5

def x_of_gain(gv):
    return gx0 + (gv / g_xmax) * (gx1 - gx0)

for xv in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
    xx = x_of_gain(xv)
    ln([(xx, gy0), (xx, gy1)], color="grid", lw=0.85)
    tx(xx, gy1 + 5, f"+{xv:.0f}" if xv > 0 else "0", size=10.0, color="muted", align="center")

gain_bars = [
    ("DocVQA (64)", 1.67, 0.60),
    ("DocVQA (256)", 3.64, 4.60),
    ("Infographic (64)", 4.90, 4.50),
    ("Infographic (256)", 4.49, 4.76),
]
grow_h = (gy1 - gy0) / 4.0
gbar_h = 8.8
for g_idx, (g_lbl, da, de) in enumerate(gain_bars):
    cy = gy0 + (g_idx + 0.5) * grow_h
    tx(gx0 - 8, cy - 5, g_lbl, size=10.6, bold=True, color="ink", align="right")
    # Δ ANLS bar
    wa = x_of_gain(da) - gx0
    right_round_bar(gx0, cy - gbar_h - 0.8, wa, gbar_h, r=2.8, fill="ours_fill", stroke="ours_edge", lw=1.15)
    tx(gx0 + wa + 4, cy - gbar_h - 1.5, f"+{da:.2f}", size=9.8, bold=True, color="ours_text")
    # Δ EM bar
    we = x_of_gain(de) - gx0
    right_round_bar(gx0, cy + 0.8, we, gbar_h, r=2.8, fill="ours_em_fill", stroke="ours_edge", lw=0.9)
    tx(gx0 + we + 4, cy + 0.2, f"+{de:.2f}", size=9.4, bold=False, color="ours_text")

ln([(gx0, gy0), (gx0, gy1)], color="axis", lw=1.15)
ln([(gx0, gy1), (gx1, gy1)], color="axis", lw=1.15)

# ==============================================================================
# 3. PANEL C: EVIDENCE PRESERVATION & MECHANISM (300 FIXED STATES + PROBES)
# ==============================================================================
PC_X, PC_Y, PC_W, PC_H = 1016, 50, 410, 500
region_card(
    PC_X, PC_Y, PC_W, PC_H,
    "C  Evidence Preservation",
    "300 fixed DocVQA states",
    "hdr_mint_bg", "hdr_mint_edge",
)

# Top subcard (c1): Evidence Miss Rate (%) ↓ on 300 Fixed States
rr(PC_X + 12, PC_Y + 40, PC_W - 24, 252, r=9, fill="subcard_bg", stroke="subcard_edge", lw=0.9)
tx(PC_X + 24, PC_Y + 49, "(c1) Evidence Miss Rate (%) ↓  (4-seed mean & seeds)", size=13.5, bold=True, color="ink")

mx0 = PC_X + 48
mx1 = PC_X + PC_W - 24
my0 = PC_Y + 76
my1 = PC_Y + 260
m_ymax = 26.5

def y_of_miss(mv):
    return my1 - (mv / m_ymax) * (my1 - my0)

for yv in (0, 5, 10, 15, 20):
    yy = y_of_miss(yv)
    ln([(mx0, yy), (mx1, yy)], color="grid", lw=0.85)
    tx(mx0 - 6, yy - 5.5, f"{yv}", size=11.0, color="muted", align="right")

miss_groups = [("64 updates", 64, 0.28), ("256 updates", 256, 0.75)]
bw_m = 34.0
gap_m = 8.0
for g_lbl, steps, frac in miss_groups:
    gx = mx0 + frac * (mx1 - mx0)
    xs = [gx - 1.5 * bw_m - gap_m, gx - 0.5 * bw_m, gx + 0.5 * bw_m + gap_m]
    m_specs = [
        ("WWH", "opd_fill", "opd_edge", "opd_text"),
        ("WHS", "mv_fill", "mv_edge", "mv_text"),
        ("WQKG", "ours_fill", "ours_edge", "ours_text"),
    ]
    means_rounded = []
    top_label_ys = []
    for idx, (m_code, f_col, e_col, t_col) in enumerate(m_specs):
        mean_v, seed_vals = get_miss(m_code, steps)
        mean_disp = r_half_up(mean_v, "0.1")
        means_rounded.append(mean_disp)
        by = y_of_miss(mean_v)
        bh = my1 - by
        top_round_bar(
            xs[idx], by, bw_m, bh, r=4.0,
            fill=f_col, stroke=e_col, lw=1.4 if m_code == "WQKG" else 1.05
        )
        # Plot individual seed dots cleanly
        offsets_x = [-8.5, -2.8, 2.8, 8.5]
        for s_idx, sv in enumerate(seed_vals):
            sy_dot = y_of_miss(sv)
            circle(xs[idx] + bw_m / 2 + offsets_x[s_idx], sy_dot, r=2.5, fill="white", stroke=e_col, lw=0.95)
        # Place mean label cleanly ABOVE the highest seed dot so it never collides!
        top_y = y_of_miss(max(mean_v, max(seed_vals)))
        lbl_y = top_y - 14.5
        top_label_ys.append(lbl_y)
        tx(
            xs[idx] + bw_m / 2, lbl_y, f"{mean_disp:.1f}%",
            size=10.8 if m_code == "WQKG" else 10.0,
            bold=(m_code == "WQKG"), color=t_col, align="center"
        )

    # Delta callout pill placed well above all 3 bar labels in the group
    diff = means_rounded[2] - means_rounded[1]
    pill_y = min(top_label_ys[1], top_label_ys[2]) - 20.5
    rr(xs[2] + bw_m / 2 - 36, pill_y, 72, 16.5, r=5, fill="ours_badge_bg", stroke="ours_badge_edge", lw=0.85)
    tx(xs[2] + bw_m / 2, pill_y + 2.5, f"{diff:.1f}% miss", size=10.2, bold=True, color="ours_text", align="center")
    tx(gx, my1 + 9, g_lbl, size=12.0, bold=True, color="ink", align="center")

ln([(mx0, my0), (mx0, my1)], color="axis", lw=1.15)
ln([(mx0, my1), (mx1, my1)], color="axis", lw=1.15)

# Bottom-left subcard (c2): Area Cap Sensitivity (Miss Rate ↓ & Median Area)
rr(PC_X + 12, PC_Y + 302, 188, 184, r=9, fill="subcard_bg", stroke="subcard_edge", lw=0.9)
tx(PC_X + 22, PC_Y + 311, "(c2) Area Cap α (64 upd.)", size=12.8, bold=True, color="ink")
tx(PC_X + 22, PC_Y + 328, "Wider cap → fewer misses", size=10.5, italic=True, color="muted")

cap_data = [
    ("α = 2.0", 14.1, 4.0, "opd_fill", "opd_edge", "opd_text"),
    ("α = 3.5", 13.8, 4.5, "ours_fill", "ours_edge", "ours_text"),
    ("α = 5.0", 12.5, 5.5, "ours_fill", "ours_edge", "ours_text"),
]
cx0 = PC_X + 34
cx1 = PC_X + 188
cy0 = PC_Y + 356
cy1 = PC_Y + 452
for idx, (c_lbl, miss_v, area_v, f_col, e_col, t_col) in enumerate(cap_data):
    bx = cx0 + idx * 50
    bh = (miss_v / 16.5) * (cy1 - cy0)
    by = cy1 - bh
    top_round_bar(bx, by, 34, bh, r=3.5, fill=f_col, stroke=e_col, lw=1.25 if idx >= 1 else 1.0)
    tx(bx + 17, by - 14, f"{miss_v:.1f}%", size=10.5, bold=(idx >= 1), color=t_col, align="center")
    tx(bx + 17, cy1 - 14, f"{area_v:.1f}×", size=9.2, bold=True, color=t_col, align="center")
    tx(bx + 17, cy1 + 7, c_lbl, size=10.5, bold=(idx == 1), color="ink", align="center")
ln([(cx0 - 8, cy1), (cx1, cy1)], color="axis", lw=1.1)
tx(PC_X + 106, PC_Y + 470, "Inside bar: median box area vs. GT", size=9.4, italic=True, color="muted", align="center")

# Bottom-right subcard (c3): Fixed-Reader Crop Intervention (125 DocVQA Qs)
rr(PC_X + 208, PC_Y + 302, 190, 184, r=9, fill="subcard_bg", stroke="subcard_edge", lw=0.9)
tx(PC_X + 218, PC_Y + 311, "(c3) Fixed-Reader Probe", size=12.8, bold=True, color="ink")
tx(PC_X + 218, PC_Y + 328, "1.5× crop keeps ANLS high", size=10.5, italic=True, color="muted")

probe_rows = [
    ("Shifted", 70.4, "IoU 0.2%", "coral_fill", "coral_edge", "coral_text"),
    ("Student", 77.1, "IoU 14.6%", "opd_fill", "opd_edge", "opd_text"),
    ("Tight bT", 85.9, "IoU 26.3%", "mv_fill", "mv_edge", "mv_text"),
    ("1.5× bT", 87.9, "IoU 12.6%", "ours_fill", "ours_edge", "ours_text"),
]
px0 = PC_X + 262
px1 = PC_X + 362
py0 = PC_Y + 350
for idx, (p_lbl, p_anls, p_iou, f_col, e_col, t_col) in enumerate(probe_rows):
    ry = py0 + idx * 28.5
    tx(px0 - 6, ry + 2, p_lbl, size=10.2, bold=(idx == 3), color="ink", align="right")
    pw = ((p_anls - 48.0) / 42.0) * (px1 - px0)
    right_round_bar(px0, ry, pw, 15, r=3.5, fill=f_col, stroke=e_col, lw=1.25 if idx == 3 else 0.95)
    tx(px0 + 4.5, ry + 2.3, p_iou, size=8.5, bold=(idx == 3), color=t_col)
    tx(px0 + pw + 4, ry + 2, f"{p_anls:.1f}", size=10.0, bold=(idx == 3), color=t_col)
ln([(px0, py0 - 4), (px0, py0 + 112)], color="axis", lw=1.1)
tx(PC_X + 303, PC_Y + 470, "Downstream ANLS (%) on 125 Qs", size=9.4, italic=True, color="muted", align="center")

c.save()

import pymupdf
doc = pymupdf.open(str(OUT_PDF))
page = doc[0]
pix = page.get_pixmap(dpi=400)
pix.save(str(ASSETS / "results_overview.png"))
doc.close()
OUT_PDF.unlink(missing_ok=True)
print("Generated assets/results_overview.png successfully.")
