#!/usr/bin/env python3
"""Generate the four manuscript figures as editable SVG.

All numbers are taken from the A/B stage development-set results:
  runs/stage_A_summary.json
  runs/stage_B_summary.json
  runs/B_beats_last2_TF_FE_t07_f*/validation_subject_predictions.jsonl

Outputs: fig1_workflow.svg, fig2_fe_gain.svg, fig3_frontier.svg, fig4_confusion.svg
Preview PNGs are rendered with cairosvg when available.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_DARK = "#104281"
BLUE_WASH = "#f4f9ff"
ORANGE = "#eb6834"
ORANGE_WASH = "#fff4ef"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#0ca30c"
GREEN_DARK = "#006300"
GREEN_WASH = "#eaf7ea"
GRAY_LINE = "#b5b3aa"

FONT = "Noto Sans CJK SC, Noto Sans SC, Source Han Sans SC, Microsoft YaHei, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"

# sequential blue ramp used by the spectrogram tile and the confusion matrix
SEQ = [
    (0.00, "#f4f9ff"),
    (0.12, "#cde2fb"),
    (0.35, "#86b6ef"),
    (0.60, "#3987e5"),
    (0.82, "#1c5cab"),
    (1.00, "#0d366b"),
]


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_hex(r: float, g: float, b: float) -> str:
    return f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def lerp_hex(a: str, b: str, t: float) -> str:
    ar, ag, ab = hex_rgb(a)
    br, bg, bb = hex_rgb(b)
    return rgb_hex(ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t)


def seq_color(t: float) -> str:
    t = max(0.0, min(1.0, t))
    for i in range(len(SEQ) - 1):
        t0, c0 = SEQ[i]
        t1, c1 = SEQ[i + 1]
        if t0 <= t <= t1:
            return lerp_hex(c0, c1, (t - t0) / (t1 - t0))
    return SEQ[-1][1]


def relative_luminance(h: str) -> float:
    r, g, b = hex_rgb(h)
    vals = []
    for c in (r, g, b):
        c = c / 255
        vals.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]


# ---------------------------------------------------------------- primitives
def text(x, y, s, size=12, anchor="start", fill=INK, weight="400", style="normal",
         opacity=1.0, rotate=None, family=FONT, spacing=None) -> str:
    attrs = [
        f'x="{x:.1f}"', f'y="{y:.1f}"', f'font-family="{family}"',
        f'font-size="{size}"', f'text-anchor="{anchor}"', f'fill="{fill}"',
        f'font-weight="{weight}"',
    ]
    if style != "normal":
        attrs.append(f'font-style="{style}"')
    if opacity != 1.0:
        attrs.append(f'opacity="{opacity}"')
    if rotate is not None:
        attrs.append(f'transform="rotate({rotate} {x:.1f} {y:.1f})"')
    if spacing is not None:
        attrs.append(f'letter-spacing="{spacing}"')
    return f'<text {" ".join(attrs)}>{esc(s)}</text>'


def rect(x, y, w, h, fill=SURFACE, stroke=None, rx=0, sw=1.0, opacity=1.0) -> str:
    attrs = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'width="{w:.1f}"', f'height="{h:.1f}"',
             f'fill="{fill}"', f'rx="{rx}"']
    if stroke is not None:
        attrs += [f'stroke="{stroke}"', f'stroke-width="{sw}"']
    if opacity != 1.0:
        attrs.append(f'opacity="{opacity}"')
    return f'<rect {" ".join(attrs)}/>'


def line(x1, y1, x2, y2, stroke=AXIS, sw=1.0, opacity=1.0, cap="butt") -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}" opacity="{opacity}"/>')


def circle(cx, cy, r, fill, stroke=None, sw=0.0, opacity=1.0) -> str:
    attrs = [f'cx="{cx:.1f}"', f'cy="{cy:.1f}"', f'r="{r}"', f'fill="{fill}"']
    if stroke is not None:
        attrs += [f'stroke="{stroke}"', f'stroke-width="{sw}"']
    if opacity != 1.0:
        attrs.append(f'opacity="{opacity}"')
    return f'<circle {" ".join(attrs)}/>'


def arrowhead_points(x1, y1, x2, y2, head=8.0) -> str:
    ang = math.atan2(y2 - y1, x2 - x1)
    p2 = (x2 - head * math.cos(ang - 0.42), y2 - head * math.sin(ang - 0.42))
    p3 = (x2 - head * math.cos(ang + 0.42), y2 - head * math.sin(ang + 0.42))
    return f"{x2:.1f},{y2:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}"


def polyline(points, color=INK2, sw=2.0, arrow=True, head=8.0, dash=None) -> str:
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    out = (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
           f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>')
    if arrow:
        (x1, y1), (x2, y2) = points[-2], points[-1]
        out += f'<polygon points="{arrowhead_points(x1, y1, x2, y2, head)}" fill="{color}"/>'
    return out


def svg_doc(width, height, body, title, desc) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
     viewBox="0 0 {width} {height}" role="img" aria-labelledby="fig-title fig-desc">
  <title id="fig-title">{esc(title)}</title>
  <desc id="fig-desc">{esc(desc)}</desc>
  <style>
    text {{ font-family: {FONT}; }}
    .mono {{ font-family: {MONO}; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" fill="{SURFACE}"/>
  <g shape-rendering="geometricPrecision">
{body}
  </g>
</svg>
'''


# ---------------------------------------------------------------- figure 1
def fig1() -> str:
    W, H = 1080, 560
    b = []

    b.append(text(40, 42, "同一功率谱的两种组织与门控融合", 20, weight="600"))
    b.append(text(40, 66, "TF 保留时频纹理；FE 在功率域对时间平均，得到 128 维频带能量轮廓。",
                  11.5, fill=INK2))

    # legend
    for x, c, lab in [(820, BLUE, "TF 分支"), (918, ORANGE, "FE 分支")]:
        b.append(line(x, 40, x + 16, 40, c, 2.5))
        b.append(text(x + 22, 44, lab, 10, fill=INK2))

    # input box
    b.append(rect(40, 110, 160, 300, fill="#f9f9f7", stroke=GRID, rx=10))
    b.append(text(120, 138, "六元音输入", 14, anchor="middle", weight="600"))
    vowels = [("a", 62, 160), ("e", 100, 160), ("i", 138, 160),
              ("o", 62, 196), ("u", 100, 196), ("ü", 138, 196)]
    for v, x, y in vowels:
        b.append(rect(x, y, 32, 26, fill=SURFACE, stroke=AXIS, rx=13))
        b.append(text(x + 16, y + 17.5, v, 11, anchor="middle", fill=INK2))

    # stylized waveform
    pts = []
    for i in range(41):
        x = 55 + i * (130 / 40)
        y = 252 + 9 * math.sin(i * 0.55) * math.exp(-((i - 20) ** 2) / 500) + 3 * math.sin(i * 1.3)
        pts.append((x, y))
    b.append(polyline(pts, BLUE, 1.5, arrow=False))
    b.append(text(120, 276, "16 kHz · 单声道", 9.5, anchor="middle", fill=MUTED))
    b.append(text(120, 294, "同一有效帧与掩码", 9.5, anchor="middle", fill=MUTED))
    b.append(text(120, 340, "同一音频 · 同一裁剪", 9.5, anchor="middle", fill=MUTED))
    b.append(text(120, 357, "同一功率 Mel 谱", 9.5, anchor="middle", fill=MUTED))

    # input -> spectrogram
    b.append(polyline([(200, 260), (245, 260)], INK2, 2.0, arrow=True))

    # spectrogram box
    b.append(rect(250, 150, 170, 220, fill=SURFACE, stroke=AXIS, rx=8))
    b.append(text(335, 178, "功率 Mel 谱", 14, anchor="middle", weight="600"))
    b.append(text(335, 196, "P[t,f]", 10.5, anchor="middle", fill=MUTED, family=MONO))
    # heatmap
    hx, hy, hw, hh = 268, 210, 134, 120
    cols, rows = 26, 14
    cw, ch = hw / cols, hh / rows
    for r in range(rows):
        for c in range(cols):
            val = (0.16 + 0.62 * math.exp(-((r - 11) ** 2) / 18)
                   + 0.13 * math.sin(c * 0.62 + r * 0.45)
                   + 0.07 * math.cos(c * 0.27 - r * 0.8))
            val = max(0.0, min(1.0, val))
            b.append(rect(hx + c * cw, hy + r * ch, cw + 0.2, ch + 0.2,
                          fill=seq_color(val), rx=0))
    b.append(rect(hx, hy, hw, hh, fill="none", stroke=AXIS, rx=1, sw=0.8))
    b.append(text(262, 275, "128 频带", 9, anchor="middle", fill=MUTED, rotate=-90))
    b.append(text(335, 346, "时间 →", 9, anchor="middle", fill=MUTED))
    b.append(text(335, 362, "同一有效帧 · 同一裁剪", 9, anchor="middle", fill=MUTED))

    # branch arrows
    b.append(polyline([(420, 230), (455, 230), (455, 155), (485, 155)], BLUE, 2.2, arrow=True))
    b.append(polyline([(420, 330), (455, 330), (455, 405), (485, 405)], ORANGE, 2.2, arrow=True))

    # TF view
    b.append(rect(485, 115, 175, 82, fill=BLUE_WASH, stroke=BLUE, rx=8))
    b.append(text(572, 140, "TF 视图", 14, anchor="middle", weight="600"))
    b.append(text(572, 162, "X_TF[t,f] = L[t,f]", 10, anchor="middle", family=MONO))
    b.append(text(572, 182, "保留时间—频率纹理", 9.5, anchor="middle", fill=MUTED))

    # BEATs
    b.append(rect(690, 115, 155, 82, fill=SURFACE, stroke=AXIS, rx=8))
    b.append(text(767, 140, "BEATs 骨干", 13, anchor="middle", weight="600"))
    b.append(text(767, 162, "冻结 / 末两层微调", 9.5, anchor="middle", fill=INK2))
    b.append(text(767, 182, "90.4 M 总参数", 9.5, anchor="middle", fill=MUTED))

    # FE view
    b.append(rect(485, 365, 175, 82, fill=ORANGE_WASH, stroke=ORANGE, rx=8))
    b.append(text(572, 390, "FE 视图", 14, anchor="middle", weight="600"))
    b.append(text(572, 412, "x_FE[f] = ln(mean_t P[t,f])", 10, anchor="middle", family=MONO))
    b.append(text(572, 432, "128 维频带能量轮廓", 9.5, anchor="middle", fill=MUTED))

    # MLP
    b.append(rect(690, 365, 155, 82, fill=SURFACE, stroke=AXIS, rx=8))
    b.append(text(767, 390, "MLP", 13, anchor="middle", weight="600"))
    b.append(text(767, 412, "128 → 128 → 128", 9.5, anchor="middle", fill=INK2, family=MONO))
    b.append(text(767, 432, "GELU · LayerNorm", 9.5, anchor="middle", fill=MUTED))

    # encoders -> fusion
    b.append(polyline([(660, 156), (690, 156)], BLUE, 2.0, arrow=True))
    b.append(polyline([(845, 156), (858, 156), (858, 260), (870, 260)], BLUE, 2.0, arrow=True))
    b.append(text(852, 238, "z_TF ∈ R^128", 8.5, anchor="end", fill=MUTED, family=MONO))
    b.append(polyline([(660, 406), (690, 406)], ORANGE, 2.0, arrow=True))
    b.append(polyline([(845, 406), (858, 406), (858, 320), (870, 320)], ORANGE, 2.0, arrow=True))
    b.append(text(852, 350, "z_FE ∈ R^128", 8.5, anchor="end", fill=MUTED, family=MONO))

    # fusion
    b.append(rect(870, 235, 125, 110, fill="#f9f9f7", stroke=INK2, rx=10))
    b.append(text(932.5, 260, "门控融合", 13, anchor="middle", weight="600"))
    b.append(text(932.5, 282, "α = softmax(s)", 9.5, anchor="middle", family=MONO))
    b.append(text(932.5, 302, "z = Σ α_v z_v", 9.5, anchor="middle", family=MONO))
    b.append(text(932.5, 326, "五类 logit", 9.5, anchor="middle", fill=MUTED))

    # output
    b.append(polyline([(995, 290), (1015, 290)], INK2, 2.0, arrow=True))
    b.append(rect(1015, 220, 55, 140, fill=SURFACE, stroke=AXIS, rx=8))
    b.append(text(1042.5, 242, "输出", 9, anchor="middle", fill=MUTED))
    for i, lab in enumerate(["NV", "SD", "UVFP", "BSVFL", "MVFT"]):
        b.append(text(1042.5, 262 + i * 22, lab, 10, anchor="middle", fill=INK2))

    # notes
    b.append(text(40, 470, "同一音频、同一有效帧、同一功率 Mel 谱；FE 不引入新的声学前端或预训练权重。",
                  9.5, fill=MUTED))
    b.append(text(40, 490, "两条分支均投影到 128 维；门控融合按样本产生视图权重，再输出五类 logit。",
                  9.5, fill=MUTED))
    b.append(text(40, 510, "图中频谱为示意，非特定样本。", 9.5, fill=MUTED))

    return svg_doc(W, H, "\n".join(b),
                   "同一功率谱的两种组织与门控融合",
                   "TF 与 FE 来自同一功率 Mel 谱；TF 分支经 BEATs 骨干，FE 分支经轻量 MLP；两支路在门控融合处加权求和并输出五类 logit。")


# ---------------------------------------------------------------- figure 2
def fig2() -> str:
    W, H = 1000, 410
    b = []
    b.append(text(50, 40, "A 阶段 FE 增益的折级一致性", 20, weight="600"))
    b.append(text(50, 64, "Δ macro-F1 = TF+FE − TF；3 个骨干 × 3 折，9/9 为正",
                  11.5, fill=INK2))

    # 9/9 badge
    b.append(rect(846, 30, 110, 26, fill=GREEN_WASH, stroke=GREEN, rx=13, sw=0.8))
    b.append(text(862, 48, "✓", 12, fill=GREEN_DARK, weight="600"))
    b.append(text(878, 48, "9/9 为正", 10.5, fill=GREEN_DARK, weight="600"))

    panels = [
        ("BEATs 末两层", 95, 0.08, [0, 0.02, 0.04, 0.06, 0.08],
         [0.0535, 0.0389, 0.0608], 0.0511),
        ("BEATs 冻结", 385, 0.08, [0, 0.02, 0.04, 0.06, 0.08],
         [0.0521, 0.0685, 0.0541], 0.0583),
        ("ResNet18 参照", 675, 0.50, [0, 0.1, 0.2, 0.3, 0.4],
         [0.4114, 0.3405, 0.3579], 0.3699),
    ]
    y0, h, w = 125, 170, 210

    for title, x0, ymax, ticks, vals, mean in panels:
        b.append(text(x0 + w / 2, 110, title, 13, anchor="middle", weight="600"))
        # grid + ticks
        for tv in ticks:
            y = y0 + h - (tv / ymax) * h
            b.append(line(x0, y, x0 + w, y, GRID, 1.0))
            lab = f"{tv:g}" if tv in (0, 0.1, 0.2, 0.3, 0.4) and ymax == 0.50 else f"{tv:.2f}"
            b.append(text(x0 - 10, y + 3.5, lab, 9, anchor="end", fill=MUTED))
        b.append(line(x0, y0 + h, x0 + w, y0 + h, AXIS, 1.0))
        b.append(text(x0 + w, y0 - 12, f"均值 +{mean:.4f}", 9.5, anchor="end", fill=INK2))

        xs = [x0 + 35, x0 + 105, x0 + 175]
        for i, (x, v) in enumerate(zip(xs, vals)):
            y = y0 + h - (v / ymax) * h
            b.append(circle(x, y, 5, BLUE, SURFACE, 2.0))
            b.append(text(x, y - 12, f"+{v:.4f}", 9, anchor="middle", fill=INK2))
            b.append(text(x, y0 + h + 20, f"折 {i}", 9.5, anchor="middle", fill=MUTED))

    b.append(text(55, 220, "Δ macro-F1", 11, anchor="middle", fill=INK2, rotate=-90))
    b.append(text(490, 355, "交叉验证折", 10.5, anchor="middle", fill=INK2))
    b.append(text(50, 398, "注：ResNet18 参照的 Δ 量级大于 BEATs 骨干，故使用独立纵轴；每个点为一个折的 TF+FE − TF 差值。",
                  8.5, fill=MUTED))

    return svg_doc(W, H, "\n".join(b),
                   "A 阶段 FE 增益的折级一致性",
                   "三个骨干各三个折的 TF+FE 相对 TF 的 macro-F1 增益；九个点全部位于零线以上。")


# ---------------------------------------------------------------- figure 3
def fig3() -> str:
    W, H = 760, 560
    b = []
    b.append(text(60, 40, "B 阶段优胜者的精度—参数前沿", 20, weight="600"))
    b.append(text(60, 64, "横轴为可训练参数（对数刻度）；纵轴为三折均值 macro-F1",
                  11.5, fill=INK2))

    # legend
    lx = 500
    b.append(circle(lx, 36, 4, ORANGE))
    b.append(text(lx + 10, 40, "效率端点", 9.5, fill=INK2))
    b.append(circle(lx + 78, 36, 4, BLUE))
    b.append(text(lx + 88, 40, "峰值", 9.5, fill=INK2))
    b.append(circle(lx + 136, 36, 4, MUTED))
    b.append(text(lx + 146, 40, "其他配置", 9.5, fill=INK2))

    x0, y0, w, h = 110, 105, 560, 330
    xmin, xmax = 80_000, 20_000_000
    ymin, ymax = 0.15, 0.85

    def X(v):
        return x0 + (math.log10(v) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * w

    def Y(v):
        return y0 + h - (v - ymin) / (ymax - ymin) * h

    # grid
    for tv in [0.2, 0.4, 0.6, 0.8]:
        y = Y(tv)
        b.append(line(x0, y, x0 + w, y, GRID, 1.0))
        b.append(text(x0 - 10, y + 3.5, f"{tv:.1f}", 9, anchor="end", fill=MUTED))
    for tv in [100_000, 1_000_000, 10_000_000]:
        x = X(tv)
        b.append(line(x, y0, x, y0 + h, GRID, 1.0))
        lab = {100_000: "100K", 1_000_000: "1M", 10_000_000: "10M"}[tv]
        b.append(text(x, y0 + h + 18, lab, 9, anchor="middle", fill=MUTED))

    # axes
    b.append(line(x0, y0 + h, x0 + w, y0 + h, AXIS, 1.0))
    b.append(line(x0, y0, x0, y0 + h, AXIS, 1.0))
    b.append(text(65, 270, "macro-F1（三折均值）", 10.5, anchor="middle", fill=INK2, rotate=-90))
    b.append(text(390, 490, "可训练参数（对数刻度）", 10.5, anchor="middle", fill=INK2))

    # peak reference
    yp = Y(0.8092)
    b.append(line(x0, yp, x0 + w, yp, GRID, 1.0))

    # points
    pts = {
        "frozen_tf": (99_077, 0.7246),
        "frozen_tffe": (136_774, 0.7984),
        "last2_tf": (14_281_261, 0.7953),
        "last2_tffe": (14_318_958, 0.8092),
        "resnet": (11_242_821, 0.1913),
    }
    # frontier
    b.append(polyline([(X(pts["frozen_tf"][0]), Y(pts["frozen_tf"][1])),
                       (X(pts["frozen_tffe"][0]), Y(pts["frozen_tffe"][1])),
                       (X(pts["last2_tffe"][0]), Y(pts["last2_tffe"][1]))],
                      GRAY_LINE, 1.5, arrow=False))
    b.append(text(400, 112, "部署前沿", 9, anchor="middle", fill=MUTED))

    # halo + points
    b.append(circle(X(pts["frozen_tffe"][0]), Y(pts["frozen_tffe"][1]), 14, ORANGE, opacity=0.10))
    b.append(circle(X(pts["frozen_tffe"][0]), Y(pts["frozen_tffe"][1]), 7, ORANGE, SURFACE, 2.0))
    b.append(circle(X(pts["last2_tffe"][0]), Y(pts["last2_tffe"][1]), 7, BLUE, SURFACE, 2.0))
    b.append(circle(X(pts["frozen_tf"][0]), Y(pts["frozen_tf"][1]), 5, MUTED, SURFACE, 2.0))
    b.append(circle(X(pts["last2_tf"][0]), Y(pts["last2_tf"][1]), 5, MUTED, SURFACE, 2.0))
    b.append(circle(X(pts["resnet"][0]), Y(pts["resnet"][1]), 5, MUTED, SURFACE, 2.0))

    # labels with leader lines
    fx, fy = X(pts["frozen_tf"][0]), Y(pts["frozen_tf"][1])
    b.append(text(fx + 14, fy + 21, "冻结 TF 0.7246", 9.5, fill=INK2))

    fxx, fyy = X(pts["frozen_tffe"][0]), Y(pts["frozen_tffe"][1])
    b.append(text(fxx + 20, fyy - 12, "冻结 TF+FE 0.7984", 9.5, fill=INK2))
    b.append(line(fxx + 8, fyy - 2, fxx + 20, fyy - 8, AXIS, 1.0))
    # callout
    b.append(rect(fxx + 20, fyy + 4, 206, 36, fill=SURFACE, stroke=GRID, rx=6))
    b.append(text(fxx + 30, fyy + 19, "效率端点：136,774 参数", 8.5, fill=INK2))
    b.append(text(fxx + 30, fyy + 33, "峰值的 98.7% · 仅 0.96% 可训练参数", 8.5, fill=MUTED))
    b.append(line(fxx + 8, fyy + 10, fxx + 20, fyy + 14, AXIS, 1.0))

    lx2, ly2 = X(pts["last2_tffe"][0]), Y(pts["last2_tffe"][1])
    b.append(text(lx2 - 14, ly2 - 14, "末两层 TF+FE 0.8092", 9.5, anchor="end", fill=INK2))
    b.append(line(lx2 - 6, ly2 - 4, lx2 - 14, ly2 - 10, AXIS, 1.0))
    lx3, ly3 = X(pts["last2_tf"][0]), Y(pts["last2_tf"][1])
    b.append(text(lx3 - 14, ly3 + 22, "末两层 TF 0.7953", 9.5, anchor="end", fill=INK2))
    b.append(line(lx3 - 6, ly3 + 5, lx3 - 14, ly3 + 16, AXIS, 1.0))

    rx, ry = X(pts["resnet"][0]), Y(pts["resnet"][1])
    b.append(text(rx + 16, ry + 2, "ResNet18 TF", 9.5, fill=INK2))
    b.append(text(rx + 16, ry + 14, "0.1913（参照）", 9.5, fill=MUTED))

    b.append(text(110, 525, "ResNet18 为监督参照，不参与 BEATs 骨干排序；门控融合比拼接少 28,735 个可训练参数。",
                  8.5, fill=MUTED))

    return svg_doc(W, H, "\n".join(b),
                   "B 阶段优胜者的精度—参数前沿",
                   "五条管线优胜者的三折均值 macro-F1 与可训练参数；冻结 BEATs + TF+FE 以 136,774 个可训练参数达到峰值的 98.7%。")


# ---------------------------------------------------------------- figure 4
def fig4() -> str:
    W, H = 720, 640
    b = []
    b.append(text(60, 40, "入选模型的混淆矩阵", 20, weight="600"))
    b.append(text(60, 64, "757 个开发目录；行为真实类别，列为预测类别",
                  11.5, fill=INK2))

    labels = ["NV", "SD", "UVFP", "BSVFL", "MVFT"]
    matrix = [
        [82, 1, 1, 6, 4],
        [0, 138, 1, 3, 2],
        [0, 0, 110, 10, 5],
        [24, 7, 7, 183, 27],
        [19, 2, 5, 23, 97],
    ]
    row_n = [sum(r) for r in matrix]
    diag = [matrix[i][i] for i in range(5)]
    mx = max(diag)

    x0, y0, cell, gap = 150, 120, 78, 2
    pitch = cell + gap
    mxw = 5 * cell + 4 * gap

    # frame
    b.append(rect(x0 - 1, y0 - 1, mxw + 2, mxw + 2, fill="none", stroke=GRID, rx=2))

    # top labels
    for j, lab in enumerate(labels):
        b.append(text(x0 + j * pitch + cell / 2, 108, lab, 11, anchor="middle",
                      weight="600", fill=INK2))
    # left labels
    for i, lab in enumerate(labels):
        b.append(text(x0 - 12, y0 + i * pitch + cell / 2 + 4, lab, 11, anchor="end",
                      weight="600", fill=INK2))

    # axis titles
    b.append(text(x0 + mxw / 2, 80, "预测类别", 10.5, anchor="middle", fill=INK2))
    b.append(text(95, y0 + mxw / 2, "真实类别", 10.5, anchor="middle", fill=INK2, rotate=-90))

    # cells
    for i in range(5):
        for j in range(5):
            v = matrix[i][j]
            t = v / mx
            fill = seq_color(t)
            cx = x0 + j * pitch + gap / 2
            cy = y0 + i * pitch + gap / 2
            b.append(rect(cx, cy, cell, cell, fill=fill, rx=3))
            fg = "#ffffff" if relative_luminance(fill) < 0.42 else INK
            if v == 0:
                b.append(text(cx + cell / 2, cy + cell / 2 + 4, "0", 12,
                              anchor="middle", fill=MUTED))
            else:
                b.append(text(cx + cell / 2, cy + cell / 2 + (1 if i == j else 5),
                               str(v), 16, anchor="middle", fill=fg, weight="600"))
                if i == j:
                    pct = v / row_n[i] * 100
                    b.append(text(cx + cell / 2, cy + cell / 2 + 17, f"{pct:.1f}%",
                                  8.5, anchor="middle", fill=fg, opacity=0.85))

    # row totals
    b.append(text(x0 + mxw + 10, 108, "n", 10, anchor="middle", fill=MUTED))
    for i, n in enumerate(row_n):
        b.append(text(x0 + mxw + 10, y0 + i * pitch + cell / 2 + 4, str(n), 9.5,
                      anchor="middle", fill=MUTED))

    # colorbar
    cbx, cby, cbw, cbh = 590, 120, 14, mxw
    stops = "\n".join(
        f'<stop offset="{t * 100:.0f}%" stop-color="{c}"/>' for t, c in reversed(SEQ)
    )
    b.append(f'''<defs>
    <linearGradient id="blueSeq" x1="0" y1="0" x2="0" y2="1">
      {stops}
    </linearGradient>
  </defs>''')
    b.append(rect(cbx, cby, cbw, cbh, fill="url(#blueSeq)", stroke=AXIS, rx=2, sw=0.8))
    b.append(text(cbx + cbw / 2, 108, "计数", 9.5, anchor="middle", fill=MUTED))
    for v in [0, 50, 100, 150, 183]:
        y = cby + cbh - (v / mx) * cbh
        b.append(line(cbx + cbw, y, cbx + cbw + 4, y, AXIS, 0.8))
        b.append(text(cbx + cbw + 8, y + 3, str(v), 8.5, fill=MUTED))

    # notes
    b.append(text(60, 570, "对角线为正确分类；MVFT 与 BSVFL 的双向混淆共 50 例。",
                  9.5, fill=MUTED))
    b.append(text(60, 590, "颜色越深计数越大；对角线第二行为该类召回率，n 为行和。",
                  9.5, fill=MUTED))

    return svg_doc(W, H, "\n".join(b),
                   "入选模型的混淆矩阵",
                   "757 个开发目录上五类真实标签与预测标签的混淆矩阵；对角线标注计数与该类召回率。")


# ---------------------------------------------------------------- figure 5
def fig5() -> str:
    data = json.loads((OUT / "roc_data.json").read_text(encoding="utf-8"))
    W, H = 820, 620
    b = []
    b.append(text(60, 40, "入选模型的五类 one-vs-rest ROC", 20, weight="600"))
    b.append(text(60, 64, "757 个开发目录的折外预测；AUROC 与 Average Precision 见图例",
                  11.5, fill=INK2))

    x0, y0, w, h = 100, 115, 430, 430

    # grid and ticks
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = x0 + t * w
        y = y0 + (1 - t) * h
        b.append(line(x0, y, x0 + w, y, GRID, 1.0))
        b.append(line(x, y0, x, y0 + h, GRID, 1.0))
        b.append(text(x, y0 + h + 18, f"{t:.1f}", 9, anchor="middle", fill=MUTED))
        b.append(text(x0 - 10, y + 3.5, f"{t:.1f}", 9, anchor="end", fill=MUTED))

    # axes
    b.append(line(x0, y0 + h, x0 + w, y0 + h, AXIS, 1.0))
    b.append(line(x0, y0, x0, y0 + h, AXIS, 1.0))
    b.append(text(x0 + w / 2, 585, "假阳性率 (FPR)", 10.5, anchor="middle", fill=INK2))
    b.append(text(55, y0 + h / 2, "真阳性率 (TPR)", 10.5, anchor="middle", fill=INK2, rotate=-90))

    # chance line
    b.append(line(x0, y0 + h, x0 + w, y0, AXIS, 1.0))
    b.append(text(330, 332, "随机基线", 8.5, fill=MUTED, rotate=-45))

    styles = {
        "NV": (BLUE, None),
        "SD": (ORANGE, "8 4"),
        "UVFP": (AQUA, "8 3 2 3"),
        "BSVFL": (YELLOW, "2 3"),
        "MVFT": (MAGENTA, "12 4"),
    }
    # draw lower-AUC curves first so the strongest curves sit on top
    for label in ["BSVFL", "MVFT", "NV", "UVFP", "SD"]:
        curve = data["curves"][label]
        color, dash = styles[label]
        pts = [(x0 + fx * w, y0 + (1 - fy) * h)
               for fx, fy in zip(curve["fpr"], curve["tpr"])]
        b.append(polyline(pts, color, 2.2, arrow=False, dash=dash))

    # legend as a small table (also serves as the contrast relief)
    def fmt3(x):
        # round half up for display, matching the manuscript tables
        return f"{x + 1e-9:.3f}"

    lx = 580
    b.append(text(lx + 34, 125, "类别", 9.5, fill=MUTED))
    b.append(text(lx + 78, 125, "AUROC", 9.5, fill=MUTED))
    b.append(text(lx + 138, 125, "AP", 9.5, fill=MUTED))
    for i, label in enumerate(data["labels"]):
        y = 150 + i * 30
        color, dash = styles[label]
        b.append(polyline([(lx, y - 3), (lx + 24, y - 3)], color, 2.2, arrow=False, dash=dash))
        c = data["curves"][label]
        b.append(text(lx + 34, y, label, 9.5, fill=INK2, weight="600"))
        b.append(text(lx + 78, y, fmt3(c["auc"]), 9.5, fill=INK2))
        b.append(text(lx + 138, y, fmt3(c["ap"]), 9.5, fill=INK2))
    b.append(line(lx, 292, lx + 200, 292, GRID, 1.0))
    b.append(text(lx, 314, f"Macro AUROC {fmt3(data['macro_auc'])}", 11, fill=INK, weight="600"))
    b.append(text(lx, 330, "五类等权平均", 8.5, fill=MUTED))
    b.append(text(lx, 348, "线型 + 颜色双重编码", 8.5, fill=MUTED))
    b.append(text(lx, 366, f"n = {data['n']}", 8.5, fill=MUTED))

    b.append(text(100, 605, "AUROC 与 AP 由 757 个开发目录的折外预测计算；AP 为 one-vs-rest average precision。",
                  8.5, fill=MUTED))

    return svg_doc(W, H, "\n".join(b),
                   "入选模型的五类 one-vs-rest ROC",
                   "757 个开发目录上五类 one-vs-rest ROC 曲线；图例给出每类 AUROC、AP 与宏平均 AUROC。")


def main() -> None:
    figs = {
        "fig1_workflow": fig1(),
        "fig2_fe_gain": fig2(),
        "fig3_frontier": fig3(),
        "fig4_confusion": fig4(),
        "fig5_roc": fig5(),
    }
    for name, svg in figs.items():
        svg_path = OUT / f"{name}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        print(f"wrote {svg_path}")
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                             write_to=str(OUT / f"{name}_preview.png"), scale=2)
            print(f"  preview {name}_preview.png")
        except Exception as exc:  # pragma: no cover
            print(f"  preview skipped: {exc}")


if __name__ == "__main__":
    main()
