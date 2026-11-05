#!/usr/bin/env python3
"""Compute one-vs-rest ROC curves for the selected B-stage model.

Reads the three fold validation prediction files of
B_beats_last2_TF_FE_t07 and writes roc_data.json for make_figures.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "experiments" / "runs"
RUN_IDS = [f"B_beats_last2_TF_FE_t07_f{i}_s3407" for i in range(3)]
LABELS = ["NV", "SD", "UVFP", "BSVFL", "MVFT"]


def main() -> None:
    rows = []
    for rid in RUN_IDS:
        path = RUNS / rid / "validation_subject_predictions.jsonl"
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh)

    y = np.array([r["y_true"] for r in rows])
    probs = np.array([r["probabilities"] for r in rows])

    curves = {}
    for i, label in enumerate(LABELS):
        y_bin = (y == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        curves[label] = {
            "fpr": [round(float(x), 6) for x in fpr],
            "tpr": [round(float(x), 6) for x in tpr],
            "auc": round(float(roc_auc_score(y_bin, probs[:, i])), 4),
            "ap": round(float(average_precision_score(y_bin, probs[:, i])), 4),
        }

    macro_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    out = {
        "n": int(len(rows)),
        "labels": LABELS,
        "macro_auc": round(macro_auc, 4),
        "curves": curves,
    }
    (ROOT / "roc_data.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote roc_data.json: n={len(rows)}, macro_auc={macro_auc:.4f}")
    for label in LABELS:
        print(f"  {label}: AUC {curves[label]['auc']:.4f}  AP {curves[label]['ap']:.4f}")


if __name__ == "__main__":
    main()
