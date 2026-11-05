#!/usr/bin/env python3
"""Generate mechanism figures 6 and 7 from mechanism_data.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from make_figures import (
    AXIS,
    BLUE,
    GRID,
    INK,
    INK2,
    MUTED,
    ORANGE,
    ORANGE_WASH,
    OUT,
    SURFACE,
    circle,
    line,
    polyline,
    rect,
    svg_doc,
    text,
)

LABELS = ["NV", "SD", "UVFP", "BSVFL", "MVFT"]
NEUTRAL = "#f0efec"
RED = "#e34948"
DATA = json.loads((OUT / "mechanism_data.json").read_text(encoding="utf-8"))


def box_stats(values):
    v = sorted(values)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo = min(x for x in v if x >= q1 - 1.5 * iqr)
    hi = max(x for x in v if x <= q3 + 1.5 * iqr)
    return float(q1), float(med), float(q3), float(lo), float(hi)


def bar_path(x, base, w, h, fill, r=4.0):
    if h <= 0:
        return ""
    top = base - h
    r = min(r, h, w / 2)
    return (
        f'<path d="M {x:.1f} {base:.1f} L {x:.1f} {top + r:.1f} '
        f'Q {x:.1f} {top:.1f} {x + r:.1f} {top:.1f} '
        f'L {x + w - r:.1f} {top:.1f} Q {x + w:.1f} {top:.1f} {x + w:.1f} {top + r:.1f} '
        f'L {x + w:.1f} {base:.1f} Z" fill="{fill}"/>'
    )


def diverging(t, vmin, vmax):
    t = max(-1.0, min(1.0, (t - (vmin + vmax) / 2) / ((vmax - vmin) / 2)))
    if t < 0:
        return _lerp_hex(NEUTRAL, BLUE, -t)
    return _lerp_hex(NEUTRAL, RED, t)


def _lerp_hex(a, b, f):
    a = a.lstrip("#")
    b = b.lstrip("#")
    ar, ag, ab = int(a[:2], 16), int(a[2:4], 16), int(a[4:], 16)
    br, bg, bb = int(b[:2], 16), int(b[2:4], 16), int(b[4:], 16)
    return f"#{round(ar + (br - ar) * f):02x}{round(ag + (bg - ag) * f):02x}{round(ab + (bb - ab) * f):02x}"


def fig6():
    W, H = 1240, 460
    b = []
    b.append(text(50, 40, "门控路由与视图依赖", 20, weight="600"))
    b.append(text(50, 64, "开发集 757 个目录的折外验证预测；A：FE 门控权重；B：视图替换后的 F1 下降；C：类均值路由与依赖",
                  11.5, fill=INK2))

    # ---------------- panel A
    x0, y0, w, h = 75, 110, 300, 270
    b.append(text(x0, 95, "A  门控路由", 12, weight="600"))
    yt = [0, 0.25, 0.5, 0.75, 1.0]
    for t in yt:
        y = y0 + h - t * h
        b.append(line(x0, y, x0 + w, y, GRID, 1.0))
        b.append(text(x0 - 8, y + 3.5, f"{t:g}", 9, anchor="end", fill=MUTED))
    b.append(text(38, y0 + h / 2, "α_FE", 10.5, anchor="middle", fill=INK2, rotate=-90))
    b.append(line(x0, y0 + h, x0 + w, y0 + h, AXIS, 1.0))
    # equal-weight reference behind the boxes
    y_half = y0 + h - 0.5 * h
    b.append(line(x0, y_half, x0 + w, y_half, AXIS, 1.0))
    b.append(text(x0 + w, y_half - 6, "等权 0.5", 8.5, anchor="end", fill=MUTED))

    xs = [x0 + 35 + i * 55 for i in range(5)]
    for i, label in enumerate(LABELS):
        vals = np.array(DATA["alpha_fe_by_class"][label])
        q1, med, q3, lo, hi = box_stats(vals)
        mean = float(vals.mean())
        cx = xs[i]
        # individual points, deterministic jitter
        for k, v in enumerate(vals):
            jitter = (((k * 37) % 101) / 100 - 0.5) * 16
            b.append(circle(cx + jitter, y0 + h - v * h, 1.5, ORANGE, opacity=0.15))
        # whiskers
        b.append(line(cx, y0 + h - q3 * h, cx, y0 + h - hi * h, ORANGE, 1.0))
        b.append(line(cx - 5, y0 + h - hi * h, cx + 5, y0 + h - hi * h, ORANGE, 1.0))
        b.append(line(cx, y0 + h - q1 * h, cx, y0 + h - lo * h, ORANGE, 1.0))
        b.append(line(cx - 5, y0 + h - lo * h, cx + 5, y0 + h - lo * h, ORANGE, 1.0))
        # box
        b.append(rect(cx - 13, y0 + h - q3 * h, 26, (q3 - q1) * h,
                      fill=ORANGE_WASH, stroke=ORANGE, rx=3, sw=1.1))
        b.append(line(cx - 13, y0 + h - med * h, cx + 13, y0 + h - med * h, ORANGE, 2.0))
        # mean diamond
        my = y0 + h - mean * h
        b.append(f'<polygon points="{cx:.1f},{my-4:.1f} {cx+4:.1f},{my:.1f} {cx:.1f},{my+4:.1f} {cx-4:.1f},{my:.1f}" '
                 f'fill="{SURFACE}" stroke="{ORANGE}" stroke-width="1.2"/>')
        b.append(text(cx, y0 + h + 18, label, 9, anchor="middle", fill=INK2, weight="600"))

    # ---------------- panel B
    x0b, y0b, wb, hb = 440, 110, 390, 270
    b.append(text(x0b, 95, "B  视图依赖", 12, weight="600"))
    ymax = 0.9
    for t in [0, 0.3, 0.6, 0.9]:
        y = y0b + hb - t / ymax * hb
        b.append(line(x0b, y, x0b + wb, y, GRID, 1.0))
        b.append(text(x0b - 8, y + 3.5, f"{t:g}", 9, anchor="end", fill=MUTED))
    b.append(text(405, y0b + hb / 2, "F1 下降", 10.5, anchor="middle", fill=INK2, rotate=-90))
    b.append(line(x0b, y0b + hb, x0b + wb, y0b + hb, AXIS, 1.0))
    # legend
    b.append(rect(x0b + wb - 170, 99, 9, 9, fill=ORANGE, rx=1))
    b.append(text(x0b + wb - 156, 107, "FE 替换", 8.5, fill=INK2))
    b.append(rect(x0b + wb - 92, 99, 9, 9, fill=BLUE, rx=1))
    b.append(text(x0b + wb - 78, 107, "TF 替换", 8.5, fill=INK2))

    centers = [x0b + 45 + i * 72 for i in range(5)]
    for i, label in enumerate(LABELS):
        fe_v = DATA["ablation_drop"]["fe"][i]
        tf_v = DATA["ablation_drop"]["tf"][i]
        cx = centers[i]
        feh = fe_v / ymax * hb
        tfh = tf_v / ymax * hb
        b.append(bar_path(cx - 21, y0b + hb, 20, feh, ORANGE))
        b.append(bar_path(cx + 1, y0b + hb, 20, tfh, BLUE))
        b.append(text(cx - 11, y0b + hb - feh - 4, f"{fe_v:.3f}", 8, anchor="middle", fill=INK2))
        b.append(text(cx + 11, y0b + hb - tfh - 4, f"{tf_v:.3f}", 8, anchor="middle", fill=INK2))
        b.append(text(cx, y0b + hb + 18, label, 9, anchor="middle", fill=INK2, weight="600"))

    # ---------------- panel C
    x0c, y0c, wc, hc = 870, 110, 300, 270
    b.append(text(x0c, 95, "C  路由与依赖", 12, weight="600"))
    xmin, xmax = 0.22, 0.56
    ymin, ymaxc = 0.0, 0.65

    def X(v):
        return x0c + (v - xmin) / (xmax - xmin) * wc

    def Y(v):
        return y0c + hc - (v - ymin) / (ymaxc - ymin) * hc

    for t in [0.25, 0.35, 0.45, 0.55]:
        x = X(t)
        b.append(line(x, y0c, x, y0c + hc, GRID, 1.0))
        b.append(text(x, y0c + hc + 18, f"{t:.2f}", 9, anchor="middle", fill=MUTED))
    for t in [0, 0.2, 0.4, 0.6]:
        y = Y(t)
        b.append(line(x0c, y, x0c + wc, y, GRID, 1.0))
        b.append(text(x0c - 8, y + 3.5, f"{t:g}", 9, anchor="end", fill=MUTED))
    b.append(line(x0c, y0c + hc, x0c + wc, y0c + hc, AXIS, 1.0))
    b.append(line(x0c, y0c, x0c, y0c + hc, AXIS, 1.0))
    b.append(text(x0c + wc / 2, 420, "类均值 α_FE", 9.5, anchor="middle", fill=INK2))
    b.append(text(835, y0c + hc / 2, "FE 替换 F1 下降", 9.5, anchor="middle", fill=INK2, rotate=-90))

    alphas = np.array([DATA["alpha_fe_mean_by_class"][l] for l in LABELS])
    drops = np.array(DATA["ablation_drop"]["fe"])
    # least-squares trend line (descriptive)
    coef = np.polyfit(alphas, drops, 1)
    xx = np.linspace(alphas.min(), alphas.max(), 2)
    pts = [(X(x), Y(np.polyval(coef, x))) for x in xx]
    b.append(polyline(pts, "#b5b3aa", 1.0, arrow=False))

    offsets = {
        "NV": (-8, -8, "end"),
        "SD": (-8, 12, "end"),
        "UVFP": (-8, -10, "end"),
        "BSVFL": (8, 4, "start"),
        "MVFT": (8, 4, "start"),
    }
    for i, label in enumerate(LABELS):
        x, y = X(alphas[i]), Y(drops[i])
        b.append(circle(x, y, 5, ORANGE, SURFACE, 2.0))
        dx, dy, anc = offsets[label]
        b.append(text(x + dx, y + dy, label, 9, anchor=anc, fill=INK2, weight="600"))

    b.append(text(x0c + 10, y0c + 18, "Spearman ρ = 1.00（5 类）", 9, fill=INK2))
    b.append(text(x0c + 10, y0c + 32, "Pearson r = 0.75（描述性）", 8.5, fill=MUTED))
    b.append(text(x0c + 10, y0c + 46, "A 阶段 FE 增益：r = 0.826", 8.5, fill=MUTED))

    b.append(text(50, 430, "FE 替换：FE 置为训练均值；TF 替换：TF 置为训练均值谱。值越大，该视图对该类越重要。",
                  8.5, fill=MUTED))
    return svg_doc(W, H, "\n".join(b),
                   "门控路由与视图依赖",
                   "A：各类别的 FE 门控权重分布；B：FE 或 TF 被训练参考替换后各类 F1 的下降；C：类均值门控权重与 FE 消融下降的关系。")


def fig7():
    W, H = 1000, 440
    b = []
    b.append(text(50, 40, "两视图表征的互补性", 20, weight="600"))
    b.append(text(50, 64, "A：TF 与 FE 的 128 维表征在开发目录上的余弦相似度；B：各类 FE 频带能量轮廓",
                  11.5, fill=INK2))

    # ---------------- panel A histogram
    x0, y0, w, h = 80, 110, 400, 270
    b.append(text(x0, 95, "A  表征相似度", 12, weight="600"))
    vals = np.array(DATA["cosine"]["values"])
    bins = np.arange(-0.5, 0.6001, 0.05)
    counts, _ = np.histogram(vals, bins=bins)
    ymax = 80
    for t in [0, 20, 40, 60, 80]:
        y = y0 + h - t / ymax * h
        b.append(line(x0, y, x0 + w, y, GRID, 1.0))
        b.append(text(x0 - 8, y + 3.5, str(t), 9, anchor="end", fill=MUTED))
    for t in [-0.5, -0.25, 0, 0.25, 0.5]:
        x = x0 + (t + 0.5) / 1.1 * w
        b.append(text(x, y0 + h + 18, f"{t:g}", 9, anchor="middle", fill=MUTED))
    b.append(line(x0, y0 + h, x0 + w, y0 + h, AXIS, 1.0))
    b.append(line(x0, y0, x0, y0 + h, AXIS, 1.0))
    b.append(text(42, y0 + h / 2, "目录数", 10.5, anchor="middle", fill=INK2, rotate=-90))
    b.append(text(x0 + w / 2, 420, "中心化余弦相似度", 9.5, anchor="middle", fill=INK2))

    bw = w * 0.05 / 1.1
    for i, c in enumerate(counts):
        if c == 0:
            continue
        left = x0 + (bins[i] + 0.5) / 1.1 * w
        bh = c / ymax * h
        b.append(rect(left + 1, y0 + h - bh, bw - 2, bh, fill=BLUE, rx=1.5))
    # zero and mean references
    xz = x0 + 0.5 / 1.1 * w
    b.append(line(xz, y0, xz, y0 + h, AXIS, 1.0))
    xm = x0 + (DATA["cosine"]["mean"] + 0.5) / 1.1 * w
    b.append(line(xm, y0, xm, y0 + h, ORANGE, 1.2))
    b.append(text(xm + 4, y0 + 14, f"均值 {DATA['cosine']['mean']:.3f}", 8.5, fill=INK2))
    b.append(text(x0 + 8, y0 + 18, f"线性 CKA {DATA['cka']['overall']:.3f}", 9, fill=INK2))
    b.append(text(x0 + 8, y0 + 32, f"置换基线 {DATA['cka']['perm_mean']:.3f}（95% {DATA['cka']['perm_p95']:.3f}）",
                  8.5, fill=MUTED))
    b.append(text(x0 + 8, y0 + 46, f"三折 CKA {min(DATA['cka']['per_fold']):.3f}–{max(DATA['cka']['per_fold']):.3f}",
                  8.5, fill=MUTED))

    # ---------------- panel B FE heatmap
    x0b, y0b, cw, chh, gap = 560, 110, 10, 42, 1
    b.append(text(x0b, 95, "B  FE 频带轮廓", 12, weight="600"))
    z = np.array(DATA["fe_profile"]["z"])
    vmin, vmax = -0.65, 0.65
    for i in range(5):
        for j in range(32):
            b.append(rect(x0b + j * cw + gap / 2, y0b + i * chh + gap / 2,
                          cw - gap, chh - gap, fill=diverging(z[i, j], vmin, vmax), rx=1))
        b.append(text(x0b - 10, y0b + i * chh + chh / 2 + 3, LABELS[i], 9,
                      anchor="end", fill=INK2, weight="600"))

    centers = DATA["fe_profile"]["band_centers_hz"]
    for hz, lab in [(0, "0"), (1000, "1k"), (2000, "2k"), (4000, "4k"), (6000, "6k"), (8000, "8k")]:
        idx = int(np.argmin(np.abs(np.array(centers) - hz)))
        x = x0b + idx * cw + cw / 2
        b.append(text(x, y0b + 5 * chh + 18, lab, 8.5, anchor="middle", fill=MUTED))
    b.append(text(x0b + 160, 420, "频率 (Hz)", 9.5, anchor="middle", fill=INK2))

    # colorbar
    cbx, cby, cbw, cbh = 900, 110, 12, 210
    stops = "\n".join([
        '<stop offset="0%" stop-color="#e34948"/>',
        '<stop offset="50%" stop-color="#f0efec"/>',
        '<stop offset="100%" stop-color="#2a78d6"/>',
    ])
    b.append(f'''<defs><linearGradient id="divSeq" x1="0" y1="0" x2="0" y2="1">{stops}</linearGradient></defs>''')
    b.append(rect(cbx, cby, cbw, cbh, fill="url(#divSeq)", stroke=AXIS, rx=2, sw=0.8))
    b.append(text(cbx + cbw / 2, 104, "z", 9.5, anchor="middle", fill=MUTED))
    for v, lab in [(0.6, "+0.6"), (0.0, "0"), (-0.6, "−0.6")]:
        y = cby + cbh * (1 - (v - (-0.65)) / 1.3)
        b.append(line(cbx + cbw, y, cbx + cbw + 4, y, AXIS, 0.8))
        b.append(text(cbx + cbw + 8, y + 3, lab, 8.5, fill=MUTED))

    b.append(text(50, 430, "每类为开发目录的平均 FE 轮廓，按频带跨目录 z 标准化；32 个 4 频带组。",
                  8.5, fill=MUTED))
    return svg_doc(W, H, "\n".join(b),
                   "两视图表征的互补性",
                   "A：TF 与 FE 的 128 维表征在开发目录上的中心化余弦相似度分布；B：各类 FE 频带能量轮廓的 z 标准化热图。")


def main():
    for name, svg in {"fig6_gate_ablation": fig6(), "fig7_complementarity": fig7()}.items():
        path = OUT / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        print(f"wrote {path}")
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                             write_to=str(OUT / f"{name}_preview.png"), scale=2)
            print(f"  preview {name}_preview.png")
        except Exception as exc:
            print(f"  preview skipped: {exc}")


if __name__ == "__main__":
    main()
