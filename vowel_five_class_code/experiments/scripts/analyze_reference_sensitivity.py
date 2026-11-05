#!/usr/bin/env python3
"""Input-level reference sensitivity for view replacement.

For each fold, draw 20 training subjects as references (vowel identity is
preserved) and replace either FE or TF of every validation item with the
same-vowel reference input. The other view and the trained model stay fixed.
The output is the distribution of macro-F1 and per-class F1 over the 20
references, per fold.

This is the sensitivity check for the single training-mean reference used in
analyze_mechanism.py; it does not retrain and does not use the test split.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_data import read_jsonl  # noqa: E402
from models import MultiViewModel  # noqa: E402

FIG = ROOT.parent / "figures"
LABELS = ["NV", "SD", "UVFP", "BSVFL", "MVFT"]
RUN = "B_beats_last2_TF_FE_t07"
FOLDS = [0, 1, 2]
N_REF = 20


def load_item(record, stats):
    with np.load(ROOT / record["feature_path"], allow_pickle=False) as cached:
        tf = torch.from_numpy(cached["TF"].copy()).cuda()
        fe = torch.from_numpy(cached["FE"].copy()).cuda()
    fe = (fe - stats["FE_mean"].cuda()) / stats["FE_std"].cuda()
    return tf, fe


def embed_tf(model, tf):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.tf_projection(model.tf_embedding(tf)).float()


def embed_fe(model, fe):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.fe(fe[None]).float()


def head_batch(model, z_tf, z_fe):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        stack = torch.stack([z_tf, z_fe], dim=1)
        weights = model.score(stack).softmax(1)
        logits = model.head((stack * weights).sum(1))
    return logits.float(), weights[:, :, 0].float()


def subject_probs(probs, subjects):
    grouped = defaultdict(list)
    for p, s in zip(probs, subjects):
        grouped[s].append(p)
    rows = []
    for s in sorted(grouped):
        rows.append((s, np.mean(grouped[s], axis=0)))
    return rows


def metrics(rows, label_by_subject):
    y = np.array([label_by_subject[s] for s, _ in rows])
    p = np.array([p for _, p in rows])
    pred = p.argmax(1)
    per_class = f1_score(y, pred, labels=list(range(5)), average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class_f1": [float(x) for x in per_class],
        "mean_true_prob": float(np.mean([p[y[i]] for i, (_, p) in enumerate(rows)])),
    }


def main() -> None:
    torch.set_num_threads(12)
    records = read_jsonl(ROOT / "features" / "development_manifest.jsonl")
    fold_map = {
        f["subject_id"]: f["validation_fold"]
        for f in read_jsonl(ROOT / "splits" / "development_folds.jsonl")
        if f["fold_count"] == 3
    }
    label_by_subject = {}
    for r in records:
        label_by_subject[r["subject_id"]] = LABELS.index(r["label"])

    results = {"fe": [], "tf": [], "fe_per_fold": [], "tf_per_fold": []}
    clean_metrics = []

    for fold in FOLDS:
        print(f"fold {fold}: loading", flush=True)
        val = [r for r in records if fold_map[r["subject_id"]] == fold]
        train = [r for r in records if fold_map[r["subject_id"]] != fold]
        train_by_subject = defaultdict(list)
        for r in train:
            train_by_subject[r["subject_id"]].append(r)
        train_subjects = sorted(train_by_subject)
        rng = np.random.default_rng(3407 + fold)
        ref_subjects = rng.choice(train_subjects, size=N_REF, replace=False).tolist()

        payload = torch.load(
            ROOT / "runs" / f"{RUN}_f{fold}_s3407" / "best.pt",
            map_location="cpu", weights_only=False,
        )
        stats = payload["statistics"]
        model = MultiViewModel("beats_last2", "TF+FE", dropout=0.2, fusion="gated").cuda()
        model.load_state_dict(payload["model"])
        model.eval()

        # validation embeddings (clean)
        z_tf_val, z_fe_val, val_subjects, val_vowels = [], [], [], []
        for r in val:
            tf, fe = load_item(r, stats)
            z_tf_val.append(embed_tf(model, tf)[0])
            z_fe_val.append(embed_fe(model, fe)[0])
            val_subjects.append(r["subject_id"])
            val_vowels.append(r["vowel"])
        z_tf_val = torch.stack(z_tf_val)
        z_fe_val = torch.stack(z_fe_val)
        val_subjects = np.array(val_subjects)
        val_vowels = np.array(val_vowels)

        # reference embeddings for the 20 training subjects (six vowels each)
        ref_tf = {}
        ref_fe = {}
        for s in ref_subjects:
            ref_tf[s], ref_fe[s] = {}, {}
            for r in train_by_subject[s]:
                tf, fe = load_item(r, stats)
                ref_tf[s][r["vowel"]] = embed_tf(model, tf)[0]
                ref_fe[s][r["vowel"]] = embed_fe(model, fe)[0]

        # clean metrics
        logits, _ = head_batch(model, z_tf_val, z_fe_val)
        clean = metrics(subject_probs(logits.softmax(-1).cpu().numpy(), val_subjects),
                        label_by_subject)
        clean_metrics.append(clean)

        fe_fold, tf_fold = [], []
        for ri, s in enumerate(ref_subjects):
            z_tf_ref = torch.stack([ref_tf[s][v] for v in val_vowels])
            z_fe_ref = torch.stack([ref_fe[s][v] for v in val_vowels])

            logits_fe, _ = head_batch(model, z_tf_val, z_fe_ref)
            fe_rows = subject_probs(logits_fe.softmax(-1).cpu().numpy(), val_subjects)
            fe_fold.append(metrics(fe_rows, label_by_subject))

            logits_tf, _ = head_batch(model, z_tf_ref, z_fe_val)
            tf_rows = subject_probs(logits_tf.softmax(-1).cpu().numpy(), val_subjects)
            tf_fold.append(metrics(tf_rows, label_by_subject))
            if (ri + 1) % 5 == 0:
                print(f"  fold {fold}: reference {ri + 1}/{N_REF}", flush=True)

        results["fe_per_fold"].append(fe_fold)
        results["tf_per_fold"].append(tf_fold)
        results["fe"].extend(fe_fold)
        results["tf"].extend(tf_fold)
        del model
        torch.cuda.empty_cache()

    def summarize(rows):
        macro = np.array([r["macro_f1"] for r in rows])
        per_class = np.array([r["per_class_f1"] for r in rows])
        return {
            "values": macro.tolist(),
            "mean": float(macro.mean()),
            "sd": float(macro.std(ddof=0)),
            "min": float(macro.min()),
            "max": float(macro.max()),
            "per_class_mean": per_class.mean(0).tolist(),
            "per_class_sd": per_class.std(0, ddof=0).tolist(),
        }

    out = {
        "n": len({r["subject_id"] for r in records}),
        "n_reference": N_REF,
        "clean": {
            "macro_f1": float(np.mean([c["macro_f1"] for c in clean_metrics])),
            "per_class_f1": np.mean([c["per_class_f1"] for c in clean_metrics], axis=0).tolist(),
        },
        "mean_reference": {"fe_macro_f1": 0.5918474348212911, "tf_macro_f1": 0.2916648433673867},
        "fe": summarize(results["fe"]),
        "tf": summarize(results["tf"]),
        "fe_per_fold": [[r["macro_f1"] for r in fold] for fold in results["fe_per_fold"]],
        "tf_per_fold": [[r["macro_f1"] for r in fold] for fold in results["tf_per_fold"]],
        "labels": LABELS,
    }
    (FIG / "reference_sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("wrote reference_sensitivity.json")
    print(json.dumps({
        "clean_macro_f1": out["clean"]["macro_f1"],
        "fe_mean": out["fe"]["mean"], "fe_range": [out["fe"]["min"], out["fe"]["max"]],
        "tf_mean": out["tf"]["mean"], "tf_range": [out["tf"]["min"], out["tf"]["max"]],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
