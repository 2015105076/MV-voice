#!/usr/bin/env python3
"""Generate mechanism-evidence figures 8 (reference sensitivity) and 9
(FE band-group perturbation) from their JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from make_figures import (
    AXIS,
    BLUE,
    BLUE_WASH,
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
    rect,
    svg_doc,
    text,
)

LABELS = ["NV", "SD", "UVFP", "BSVFL", "MVFT"]


def box_stats(values):
    v = sorted(values)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo = min(x for x in v if x >= q1 - 1.5 * iqr)
    hi = max(x for x in v if x <= q3 + 1.5 * iqr)
    return float(q1), float(med), float(q3), float(lo), float(hi)


def fig8():
    d = json.loads((OUT / "reference_sensitivity.json").read_text(encoding="utf-8"))
    W, H = 1040, 430
    b = []
    b.append(text(50, 40, "输入级参考敏感性", 20, weight="600"))
    b.append(text(50, 64, "20 个训练受试者参考 × 3 折；参考保留元音身份；替换一个视图，另一个视图与模型不变",
                  11.5, fill=INK2))

    # ---------------- panel A
    x0, y0, w, h = 80, 110, 440, 250
    b.append(text(x0, 95, "A  macro-F1 的参考分布", 12, weight="600"))
    xmin, xmax = 0.0, 0.9
    for t in [0, 0.3, 0.6, 0.9]:
        x = x0 + (t - xmin) / (xmax - xmin) * w
        b.append(line(x, y0, x, y0 + h, GRID, 1.0))
        b.append(text(x, y0 + h + 18, f"{t:g}", 9, anchor="middle", fill=MUTED))
    b.append(line(x0, y0 + h, x0 + w, y0 + h, AXIS, 1.0))
    b.append(text(x0 + w / 2, 400, "macro-F1", 9.5, anchor="middle", fill=INK2))

    def X(v):
        return x0 + (v - xmin) / (xmax - xmin) * w

    rows = [
        ("FE 替换", d["fe"], ORANGE, ORANGE_WASH, y0 + 0.30 * h, d["mean_reference"]["fe_macro_f1"]),
        ("TF 替换", d["tf"], BLUE, BLUE_WASH, y0 + 0.70 * h, d["mean_reference"]["tf_macro_f1"]),
    ]
    for label, block, color, wash, cy, mean_ref in rows:
        vals = np.array(block["values"])
        q1, med, q3, lo, hi = box_stats(vals)
        for k, v in enumerate(vals):
            jitter = (((k * 41) % 101) / 100 - 0.5) * 18
            b.append(circle(X(v), cy + jitter, 1.8, color, opacity=0.20))
        b.append(line(X(lo), cy, X(q1), cy, color, 1.0))
        b.append(line(X(lo), cy - 5, X(lo), cy + 5, color, 1.0))
        b.append(line(X(q3), cy, X(hi), cy, color, 1.0))
        b.append(line(X(hi), cy - 5, X(hi), cy + 5, color, 1.0))
        b.append(rect(X(q1), cy - 12, X(q3) - X(q1), 24, fill=wash, stroke=color, rx=3, sw=1.0))
        b.append(line(X(med), cy - 12, X(med), cy + 12, color, 2.0))
        b.append(f'<polygon points="{X(vals.mean()):.1f},{cy-4:.1f} {X(vals.mean())+4:.1f},{cy:.1f} '
                 f'{X(vals.mean()):.1f},{cy+4:.1f} {X(vals.mean())-4:.1f},{cy:.1f}" '
                 f'fill="{SURFACE}" stroke="{color}" stroke-width="1.2"/>')
        b.append(text(70, cy + 3.5, label, 10, anchor="end", fill=INK2, weight="600"))
        # training-mean reference marker
        mx = X(mean_ref)
        b.append(circle(mx, cy, 3.5, SURFACE, color, 1.4))
        b.append(text(mx + 6, cy - 18, f"均值参考 {mean_ref:.3f}", 8, fill=MUTED))

    # clean reference
    xc = X(d["clean"]["macro_f1"])
    b.append(line(xc, y0, xc, y0 + h, AXIS, 1.2))
    b.append(text(xc - 4, y0 + 14, f"干净 {d['clean']['macro_f1']:.3f}", 8.5, anchor="end", fill=INK2))
    b.append(text(x0 + 8, y0 + 18, "FE > TF：59/60 次", 8.5, fill=INK2))
    b.append(text(x0 + 8, y0 + 32, "平均差 +0.360", 8.5, fill=MUTED))

    # ---------------- panel B
    x0b, y0b, wb, hb = 620, 110, 370, 250
    b.append(text(x0b, 95, "B  逐类 F1：干净 vs 替换", 12, weight="600"))
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y0b + hb - t * hb
        b.append(line(x0b, y, x0b + wb, y, GRID, 1.0))
        b.append(text(x0b - 8, y + 3.5, f"{t:g}", 9, anchor="end", fill=MUTED))
    b.append(line(x0b, y0b + hb, x0b + wb, y0b + hb, AXIS, 1.0))
    b.append(text(578, y0b + hb / 2, "F1", 10.5, anchor="middle", fill=INK2, rotate=-90))

    centers = [x0b + 45 + i * 72 for i in range(5)]
    clean = np.array(d["clean"]["per_class_f1"])
    fe_mean = np.array(d["fe"]["per_class_mean"])
    fe_sd = np.array(d["fe"]["per_class_sd"])
    tf_mean = np.array(d["tf"]["per_class_mean"])
    tf_sd = np.array(d["tf"]["per_class_sd"])

    def Y(v):
        return y0b + hb - v * hb

    for i, label in enumerate(LABELS):
        cx = centers[i]
        b.append(text(cx, y0b + hb + 18, label, 9, anchor="middle", fill=INK2, weight="600"))
        # clean diamond
        b.append(f'<polygon points="{cx-14:.1f},{Y(clean[i])-4:.1f} {cx-10:.1f},{Y(clean[i]):.1f} '
                 f'{cx-14:.1f},{Y(clean[i])+4:.1f} {cx-18:.1f},{Y(clean[i]):.1f}" '
                 f'fill="{SURFACE}" stroke="{MUTED}" stroke-width="1.1"/>')
        # FE error bar
        fe_lo, fe_hi = max(0.0, fe_mean[i] - fe_sd[i]), min(1.0, fe_mean[i] + fe_sd[i])
        b.append(line(cx, Y(fe_lo), cx, Y(fe_hi), ORANGE, 1.0, 0.7))
        b.append(line(cx - 4, Y(fe_hi), cx + 4, Y(fe_hi), ORANGE, 1.0, 0.7))
        b.append(line(cx - 4, Y(fe_lo), cx + 4, Y(fe_lo), ORANGE, 1.0, 0.7))
        b.append(circle(cx, Y(fe_mean[i]), 4, ORANGE, SURFACE, 1.8))
        # TF error bar
        bx = cx + 14
        tf_lo, tf_hi = max(0.0, tf_mean[i] - tf_sd[i]), min(1.0, tf_mean[i] + tf_sd[i])
        b.append(line(bx, Y(tf_lo), bx, Y(tf_hi), BLUE, 1.0, 0.7))
        b.append(line(bx - 4, Y(tf_hi), bx + 4, Y(tf_hi), BLUE, 1.0, 0.7))
        b.append(line(bx - 4, Y(tf_lo), bx + 4, Y(tf_lo), BLUE, 1.0, 0.7))
        b.append(rect(bx - 3.5, Y(tf_mean[i]) - 3.5, 7, 7, fill=BLUE, stroke=SURFACE, rx=1, sw=1.2))

    # legend
    lx = x0b + wb - 190
    b.append(f'<polygon points="{lx},{102} {lx+4},{106} {lx},{110} {lx-4},{106}" fill="{SURFACE}" stroke="{MUTED}" stroke-width="1.0"/>')
    b.append(text(lx + 8, 109, "干净", 8.5, fill=INK2))
    b.append(circle(lx + 62, 106, 3.5, ORANGE))
    b.append(text(lx + 70, 109, "FE 替换", 8.5, fill=INK2))
    b.append(rect(lx + 126, 102.5, 6, 6, fill=BLUE, rx=1))
    b.append(text(lx + 136, 109, "TF 替换", 8.5, fill=INK2))

    b.append(text(50, 415, "误差线为 60 次参考抽取的 ±1 SD；干净为折外点估计。", 8.5, fill=MUTED))
    return svg_doc(W, H, "\n".join(b),
                   "输入级参考敏感性",
                   "20 个训练受试者参考 × 3 折的视图替换分布；A：macro-F1 参考分布；B：逐类 F1 的均值与 ±1 SD。")



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


def fig9():
    d = json.loads((OUT / "band_perturbation.json").read_text(encoding="utf-8"))
    W, H = 1040, 430
    b = []
    b.append(text(50, 40, "FE 频带分组扰动", 20, weight="600"))
    b.append(text(50, 64, "16 个连续 8 频带组；替换为训练均值；A：逐组真实类概率下降；B：最高 1/2/4 组与 100 个随机组",
                  11.5, fill=INK2))

    # ---------------- panel A
    x0, y0, w, h = 80, 110, 460, 250
    b.append(text(x0, 95, "A  逐组扰动", 12, weight="600"))
    groups = d["groups"]
    vals = np.array([g["true_prob_drop"] for g in groups])
    ymax = max(vals.max() * 1.18, 0.01)
    for t in [0, ymax / 2, ymax]:
        y = y0 + h - t / ymax * h
        b.append(line(x0, y, x0 + w, y, GRID, 1.0))
        b.append(text(x0 - 8, y + 3.5, f"{t:.3f}", 9, anchor="end", fill=MUTED))
    b.append(line(x0, y0 + h, x0 + w, y0 + h, AXIS, 1.0))
    b.append(text(40, y0 + h / 2, "真实类概率下降", 10.5, anchor="middle", fill=INK2, rotate=-90))

    top = set(d["topk"]["4"]["groups"])
    step = w / 16
    for i, g in enumerate(groups):
        v = g["true_prob_drop"]
        bh = v / ymax * h
        color = ORANGE if i in top else BLUE
        bx = x0 + i * step + step * 0.18
        bw = step * 0.64
        b.append(bar_path(bx, y0 + h, bw, bh, color))
        if i in top:
            b.append(text(bx + bw / 2, y0 + h - bh - 4, f"{v:.3f}", 8, anchor="middle", fill=INK2))
        if i % 2 == 0:
            b.append(text(x0 + i * step + step / 2, y0 + h + 16, str(i), 8.5,
                          anchor="middle", fill=MUTED))
    # frequency annotations at selected groups
    for i in [0, 3, 6, 9, 12, 15]:
        hz = groups[i]["center_hz"]
        lab = f"{hz/1000:.1f}k" if hz >= 1000 else f"{hz:.0f}"
        b.append(text(x0 + i * step + step / 2, y0 + h + 30, lab, 8.5,
                      anchor="middle", fill=MUTED))
    b.append(text(x0 + w / 2, 400, "Mel 频带组（低频 → 高频；上排为组号，下排为近似中心频率）",
                  9, anchor="middle", fill=INK2))

    # ---------------- panel B
    x0b, y0b, wb, hb = 620, 110, 370, 250
    b.append(text(x0b, 95, "B  最高组 vs 随机组", 12, weight="600"))
    all_vals = []
    for k in ["1", "2", "4"]:
        all_vals.append(d["topk"][k]["observed_drop"])
        all_vals.extend(d["topk"][k]["random_values"])
    xmax = max(all_vals) * 1.12
    for t in [0, xmax / 2, xmax]:
        x = x0b + t / xmax * wb
        b.append(line(x, y0b, x, y0b + hb, GRID, 1.0))
        b.append(text(x, y0b + hb + 18, f"{t:.3f}", 9, anchor="middle", fill=MUTED))
    b.append(line(x0b, y0b + hb, x0b + wb, y0b + hb, AXIS, 1.0))
    b.append(text(x0b + wb / 2, 400, "真实类概率下降", 9.5, anchor="middle", fill=INK2))

    rows = [("1", "top-1", y0b + 50), ("2", "top-2", y0b + 120), ("4", "top-4", y0b + 190)]
    for k, lab, cy in rows:
        block = d["topk"][k]
        obs = block["observed_drop"]
        rnd = np.array(block["random_values"])
        for j, v in enumerate(rnd):
            jitter = (((j * 47) % 101) / 100 - 0.5) * 18
            b.append(circle(x0b + v / xmax * wb, cy + jitter, 1.6, MUTED, opacity=0.28))
        mx = x0b + rnd.mean() / xmax * wb
        b.append(line(mx, cy - 12, mx, cy + 12, MUTED, 1.0))
        px = x0b + np.percentile(rnd, 95) / xmax * wb
        b.append(line(px, cy - 12, px, cy + 12, AXIS, 1.0))
        ox = x0b + obs / xmax * wb
        b.append(circle(ox, cy, 5, ORANGE, SURFACE, 2.0))
        b.append(text(x0b - 8, cy + 3.5, lab, 9.5, anchor="end", fill=INK2, weight="600"))
        b.append(text(x0b + wb, cy - 12, f"p = {block['p_empirical']:.3f}", 8.5,
                      anchor="end", fill=MUTED))

    b.append(text(50, 415, "扰动为 FE 频带组置为训练均值；随机组为同大小、随机抽取的 100 组；灰线为随机均值，浅灰线为 95% 分位。",
                  8.5, fill=MUTED))
    return svg_doc(W, H, "\n".join(b),
                   "FE 频带分组扰动",
                   "16 个 FE 频带组逐组替换为训练均值后的真实类概率下降；最高 1/2/4 组与 100 个随机组的比较。")


def fig10():
    fe = json.loads((OUT / "band_perturbation.json").read_text(encoding="utf-8"))
    sh = json.loads((OUT / "shared_spectrum_perturbation.json").read_text(encoding="utf-8"))
    W, H = 1040, 430
    b = []
    b.append(text(50, 40, "共享 log-Mel 谱扰动", 20, weight="600"))
    b.append(text(50, 64, "同一组 Mel 频带替换为训练均值，TF 与 FE 同时重算；A：FE-only 与共享谱的逐组下降；B：共享 top-k 与随机组",
                  11.5, fill=INK2))

    # ---------------- panel A: dumbbell FE-only vs shared
    x0, y0, w, h = 80, 110, 470, 250
    b.append(text(x0, 95, "A  逐组扰动：FE-only vs 共享谱", 12, weight="600"))
    fe_vals = np.array([g["true_prob_drop"] for g in fe["groups"]])
    sh_vals = np.array([g["true_prob_drop"] for g in sh["groups"]])
    ymax = max(fe_vals.max(), sh_vals.max()) * 1.18
    for t in [0, ymax / 2, ymax]:
        y = y0 + h - t / ymax * h
        b.append(line(x0, y, x0 + w, y, GRID, 1.0))
        b.append(text(x0 - 8, y + 3.5, f"{t:.3f}", 9, anchor="end", fill=MUTED))
    b.append(line(x0, y0 + h, x0 + w, y0 + h, AXIS, 1.0))
    b.append(text(40, y0 + h / 2, "真实类概率下降", 10.5, anchor="middle", fill=INK2, rotate=-90))

    step = w / 16
    for i in range(16):
        cx = x0 + i * step + step / 2
        y_fe = y0 + h - fe_vals[i] / ymax * h
        y_sh = y0 + h - sh_vals[i] / ymax * h
        b.append(line(cx - 4, y_fe, cx + 4, y_sh, "#b5b3aa", 1.2))
        b.append(circle(cx - 4, y_fe, 3.5, ORANGE, SURFACE, 1.4))
        b.append(circle(cx + 4, y_sh, 3.5, BLUE, SURFACE, 1.4))
        if i % 2 == 0:
            b.append(text(cx, y0 + h + 16, str(i), 8.5, anchor="middle", fill=MUTED))
    for i in [0, 3, 6, 9, 12, 15]:
        hz = sh["groups"][i]["center_hz"]
        lab = f"{hz/1000:.1f}k" if hz >= 1000 else f"{hz:.0f}"
        b.append(text(x0 + i * step + step / 2, y0 + h + 30, lab, 8.5,
                      anchor="middle", fill=MUTED))
    # legend
    lx = x0 + 230
    b.append(circle(lx, 100, 4, ORANGE))
    b.append(text(lx + 8, 103, "FE-only", 8.5, fill=INK2))
    b.append(circle(lx + 78, 100, 4, BLUE))
    b.append(text(lx + 86, 103, "共享 TF+FE", 8.5, fill=INK2))
    b.append(text(x0 + w / 2, 400, "Mel 频带组（低频 → 高频；上排为组号，下排为近似中心频率）",
                  9, anchor="middle", fill=INK2))

    # ---------------- panel B: shared top-k vs random
    x0b, y0b, wb, hb = 620, 110, 370, 250
    b.append(text(x0b, 95, "B  共享 top-k vs 随机组", 12, weight="600"))
    all_vals = []
    for k in ["1", "2", "4"]:
        all_vals.append(sh["topk"][k]["observed_drop"])
        all_vals.extend(sh["topk"][k]["random_values"])
    xmax = max(all_vals) * 1.12
    for t in [0, xmax / 2, xmax]:
        x = x0b + t / xmax * wb
        b.append(line(x, y0b, x, y0b + hb, GRID, 1.0))
        b.append(text(x, y0b + hb + 18, f"{t:.3f}", 9, anchor="middle", fill=MUTED))
    b.append(line(x0b, y0b + hb, x0b + wb, y0b + hb, AXIS, 1.0))
    b.append(text(x0b + wb / 2, 400, "真实类概率下降", 9.5, anchor="middle", fill=INK2))

    rows = [("1", "top-1", y0b + 50), ("2", "top-2", y0b + 120), ("4", "top-4", y0b + 190)]
    for k, lab, cy in rows:
        block = sh["topk"][k]
        obs = block["observed_drop"]
        rnd = np.array(block["random_values"])
        for j, v in enumerate(rnd):
            jitter = (((j * 47) % 101) / 100 - 0.5) * 18
            b.append(circle(x0b + v / xmax * wb, cy + jitter, 1.6, MUTED, opacity=0.28))
        mx = x0b + rnd.mean() / xmax * wb
        b.append(line(mx, cy - 12, mx, cy + 12, MUTED, 1.0))
        px = x0b + np.percentile(rnd, 95) / xmax * wb
        b.append(line(px, cy - 12, px, cy + 12, AXIS, 1.0))
        ox = x0b + obs / xmax * wb
        b.append(circle(ox, cy, 5, BLUE, SURFACE, 2.0))
        b.append(text(x0b - 8, cy + 3.5, lab, 9.5, anchor="end", fill=INK2, weight="600"))
        b.append(text(x0b + wb, cy - 12, f"p = {block['p_empirical']:.3f}", 8.5,
                      anchor="end", fill=MUTED))

    b.append(text(50, 415, "共享扰动同时重算 TF 与 FE；top-k 组沿用 FE-only 的最高 4 组；随机组为同大小、随机抽取的 100 组。FE-only 与共享谱的 FE 参考量不同（FE 均值 vs TF 均值），组间差值只作方向性比较。",
                  8.5, fill=MUTED))
    return svg_doc(W, H, "\n".join(b),
                   "共享 log-Mel 谱扰动",
                   "同一组 Mel 频带替换为训练均值后同时重算 TF 与 FE；FE-only 与共享谱的逐组下降及共享 top-k 与随机组的比较。")

def main():
    figs = {"fig8_reference_sensitivity": fig8(), "fig9_band_perturbation": fig9()}
    if (OUT / "shared_spectrum_perturbation.json").exists():
        figs["fig10_shared_spectrum"] = fig10()
    for name, svg in figs.items():
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
