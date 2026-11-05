#!/usr/bin/env python3
"""Exploratory mechanism analysis for the selected B-stage model.

For each development validation subject, this script reconstructs the selected
model (BEATs last-two + TF/FE, gated fusion) from its best checkpoint and
collects:

  * gate weights (alpha_FE) for the clean input;
  * per-subject z_TF and z_FE embeddings;
  * clean, FE-ablated and TF-ablated predictions;
  * class-mean FE profiles.

Ablation replaces one view with its training reference:
  FE -> zero in normalized space (the training mean FE profile);
  TF -> the training mean log-Mel spectrum repeated over the item's frames.

The output is descriptive evidence for the discussion section, not a new model
selection criterion.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_data import read_jsonl  # noqa: E402
from models import MultiViewModel  # noqa: E402

FIG = ROOT.parent / "figures"
LABELS = ["NV", "SD", "UVFP", "BSVFL", "MVFT"]
RUN = "B_beats_last2_TF_FE_t07"
FOLDS = [0, 1, 2]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    denom = np.linalg.norm(xtx, "fro") * np.linalg.norm(yty, "fro")
    return float(np.sum(xty ** 2) / denom)


def mel_centers(n_mels=128, fmin=20.0, fmax=8000.0):
    mel = lambda f: 1127.0 * np.log1p(f / 700.0)
    hz = lambda m: 700.0 * np.expm1(m / 1127.0)
    edges = np.linspace(mel(fmin), mel(fmax), n_mels + 2)[1:-1]
    return hz(edges)


def load_item(record, stats):
    with np.load(ROOT / record["feature_path"], allow_pickle=False) as cached:
        tf = torch.from_numpy(cached["TF"].copy()).cuda()
        fe = torch.from_numpy(cached["FE"].copy()).cuda()
        raw_fe = cached["FE"].copy()
    fe = (fe - stats["FE_mean"].cuda()) / stats["FE_std"].cuda()
    return tf, fe, raw_fe


def embed_tf(model, tf, stats, mode="clean"):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if mode == "mean":
            tf_in = stats["TF_mean"].cuda()[None, :].expand(tf.shape[0], -1)
        else:
            tf_in = tf
        return model.tf_projection(model.tf_embedding(tf_in)).float()


def embed_fe(model, fe, mode="clean"):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fe_in = torch.zeros_like(fe) if mode == "mean" else fe
        return model.fe(fe_in[None]).float()


def head_from_embeddings(model, z_tf, z_fe):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        stack = torch.stack([z_tf, z_fe], dim=1)
        weights = model.score(stack).softmax(1)
        logits = model.head((stack * weights).sum(1))
    return logits.float(), weights[0, :, 0].float()


def main() -> None:
    torch.set_num_threads(12)
    records = read_jsonl(ROOT / "features" / "development_manifest.jsonl")
    fold_map = {
        f["subject_id"]: f["validation_fold"]
        for f in read_jsonl(ROOT / "splits" / "development_folds.jsonl")
        if f["fold_count"] == 3
    }
    assert all(r["split"] == "development" for r in records)

    subject_rows = []
    fe_profiles = {}
    cka_per_fold = []
    clean_diff = 0.0

    for fold in FOLDS:
        print(f"fold {fold}: loading validation items", flush=True)
        val = [r for r in records if fold_map[r["subject_id"]] == fold]
        payload = torch.load(
            ROOT / "runs" / f"{RUN}_f{fold}_s3407" / "best.pt",
            map_location="cpu", weights_only=False,
        )
        stats = payload["statistics"]
        model = MultiViewModel("beats_last2", "TF+FE", dropout=0.2, fusion="gated").cuda()
        model.load_state_dict(payload["model"])
        model.eval()

        # saved clean subject predictions, for an exactness check
        saved = {}
        with (ROOT / "runs" / f"{RUN}_f{fold}_s3407" /
              "validation_subject_predictions.jsonl").open() as fh:
            for line in fh:
                row = json.loads(line)
                saved[row["subject_id"]] = np.array(row["probabilities"])

        by_subject = defaultdict(list)
        for record in val:
            by_subject[record["subject_id"]].append(record)

        for si, (subject, items) in enumerate(sorted(by_subject.items())):
            clean_probs, fe_probs, tf_probs = [], [], []
            alpha_clean, alpha_fe_abl, alpha_tf_abl = [], [], []
            z_tf_list, z_fe_list = [], []
            raw_fe_list = []
            for record in items:
                tf, fe, raw_fe = load_item(record, stats)
                raw_fe_list.append(raw_fe)

                z_tf_clean = embed_tf(model, tf, stats)
                z_fe_clean = embed_fe(model, fe)
                z_tf_mean = embed_tf(model, tf, stats, mode="mean")
                z_fe_mean = embed_fe(model, fe, mode="mean")

                logits, w = head_from_embeddings(model, z_tf_clean, z_fe_clean)
                clean_probs.append(logits.softmax(-1)[0].cpu().numpy())
                alpha_clean.append(float(w[1]))
                z_tf_list.append(z_tf_clean[0].cpu().numpy())
                z_fe_list.append(z_fe_clean[0].cpu().numpy())

                logits_f, w_f = head_from_embeddings(model, z_tf_clean, z_fe_mean)
                fe_probs.append(logits_f.softmax(-1)[0].cpu().numpy())
                alpha_fe_abl.append(float(w_f[1]))

                logits_t, w_t = head_from_embeddings(model, z_tf_mean, z_fe_clean)
                tf_probs.append(logits_t.softmax(-1)[0].cpu().numpy())
                alpha_tf_abl.append(float(w_t[1]))

            clean = np.mean(clean_probs, axis=0)
            fe_abl = np.mean(fe_probs, axis=0)
            tf_abl = np.mean(tf_probs, axis=0)
            if subject in saved:
                clean_diff = max(clean_diff, float(np.abs(clean - saved[subject]).max()))

            subject_rows.append({
                "subject_id": subject,
                "leakage_group_id": items[0]["leakage_group_id"],
                "fold": fold,
                "y_true": LABELS.index(items[0]["label"]),
                "prob_clean": clean.tolist(),
                "prob_fe_ablation": fe_abl.tolist(),
                "prob_tf_ablation": tf_abl.tolist(),
                "alpha_fe": float(np.mean(alpha_clean)),
                "alpha_fe_under_fe_ablation": float(np.mean(alpha_fe_abl)),
                "alpha_fe_under_tf_ablation": float(np.mean(alpha_tf_abl)),
                "z_tf": np.mean(z_tf_list, axis=0).tolist(),
                "z_fe": np.mean(z_fe_list, axis=0).tolist(),
            })
            fe_profiles[subject] = np.mean(raw_fe_list, axis=0)
            if (si + 1) % 50 == 0:
                print(f"  fold {fold}: {si + 1}/{len(by_subject)} subjects", flush=True)

        # fold-level CKA on subject-mean embeddings
        ztf = np.array([r["z_tf"] for r in subject_rows if r["fold"] == fold])
        zfe = np.array([r["z_fe"] for r in subject_rows if r["fold"] == fold])
        cka_per_fold.append(linear_cka(ztf, zfe))
        print(f"fold {fold} CKA {cka_per_fold[-1]:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    y = np.array([r["y_true"] for r in subject_rows])
    clean_pred = np.array([np.argmax(r["prob_clean"]) for r in subject_rows])
    fe_pred = np.array([np.argmax(r["prob_fe_ablation"]) for r in subject_rows])
    tf_pred = np.array([np.argmax(r["prob_tf_ablation"]) for r in subject_rows])

    def metric_block(pred):
        per_class = f1_score(y, pred, labels=list(range(5)), average=None, zero_division=0)
        return {
            "macro_f1": float(f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "per_class_f1": [float(x) for x in per_class],
        }

    clean_metrics = metric_block(clean_pred)
    fe_metrics = metric_block(fe_pred)
    tf_metrics = metric_block(tf_pred)

    ztf = np.array([r["z_tf"] for r in subject_rows])
    zfe = np.array([r["z_fe"] for r in subject_rows])
    cka_overall = linear_cka(ztf, zfe)

    # permutation null: shuffle the subject order of FE embeddings
    rng = np.random.default_rng(3407)
    null = []
    for _ in range(200):
        idx = rng.permutation(len(zfe))
        null.append(linear_cka(ztf, zfe[idx]))
    null = np.array(null)

    # centered cosine similarity per subject
    ztf_c = ztf - ztf.mean(0, keepdims=True)
    zfe_c = zfe - zfe.mean(0, keepdims=True)
    cos = np.sum(ztf_c * zfe_c, axis=1) / (
        np.linalg.norm(ztf_c, axis=1) * np.linalg.norm(zfe_c, axis=1)
    )

    # class-wise alpha_FE
    alpha_by_class = {
        label: [r["alpha_fe"] for r in subject_rows if r["y_true"] == i]
        for i, label in enumerate(LABELS)
    }
    alpha_mean_by_class = {k: float(np.mean(v)) for k, v in alpha_by_class.items()}
    alpha_median_by_class = {k: float(np.median(v)) for k, v in alpha_by_class.items()}

    correct = clean_pred == y
    correct_alpha = {
        "correct_mean": float(np.mean([r["alpha_fe"] for r, c in zip(subject_rows, correct) if c])),
        "incorrect_mean": float(np.mean([r["alpha_fe"] for r, c in zip(subject_rows, correct) if not c])),
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
    }

    # class-mean FE profile, z-scored per band across subjects
    subjects_sorted = sorted(fe_profiles)
    raw = np.stack([fe_profiles[s] for s in subjects_sorted])
    z = (raw - raw.mean(0, keepdims=True)) / raw.std(0, keepdims=True).clip(min=1e-6)
    labels_sorted = np.array([r["y_true"] for r in subject_rows])
    # subject_rows and subjects_sorted have the same sorted order? subject_rows is
    # appended in sorted(subject) order inside each fold, and folds are processed
    # in order; for a per-fold concatenation this is consistent enough for a
    # descriptive class mean. Rebuild explicitly by subject id.
    row_by_subject = {r["subject_id"]: r for r in subject_rows}
    profile_by_class = {}
    for i, label in enumerate(LABELS):
        rows = [row_by_subject[s]["subject_id"] for s in subjects_sorted
                if row_by_subject[s]["y_true"] == i]
        idx = [subjects_sorted.index(s) for s in rows]
        profile_by_class[label] = z[idx].mean(0)
    # bin 128 bands into 32 groups of 4
    profile_32 = np.stack([
        profile_by_class[label].reshape(32, 4).mean(1) for label in LABELS
    ])
    centers = mel_centers(128).reshape(32, 4).mean(1)

    # A-stage per-class F1 gain on the same backbone, OOF predictions
    def load_subject(run_ids):
        out = []
        for rid in run_ids:
            with (ROOT / "runs" / rid / "validation_subject_predictions.jsonl").open() as fh:
                out.extend(json.loads(line) for line in fh)
        return out

    a_tf = load_subject([f"A_beats_last2_TF_f{i}_s3407" for i in range(3)])
    a_fe = load_subject([f"A_beats_last2_TF_FE_f{i}_s3407" for i in range(3)])
    y_a = np.array([r["y_true"] for r in a_tf])
    p_tf = np.array([r["probabilities"] for r in a_tf]).argmax(1)
    p_fe = np.array([r["probabilities"] for r in a_fe]).argmax(1)
    f1_tf = f1_score(y_a, p_tf, labels=list(range(5)), average=None, zero_division=0)
    f1_fe = f1_score(y_a, p_fe, labels=list(range(5)), average=None, zero_division=0)
    gain = [float(a - b) for a, b in zip(f1_fe, f1_tf)]
    from scipy.stats import pearsonr, spearmanr
    alpha_vec = np.array([alpha_mean_by_class[l] for l in LABELS])
    pear = float(pearsonr(alpha_vec, gain)[0])
    spear = float(spearmanr(alpha_vec, gain)[0])

    out = {
        "n": len(subject_rows),
        "labels": LABELS,
        "class_counts": {l: int((y == i).sum()) for i, l in enumerate(LABELS)},
        "clean_prediction_max_abs_diff": clean_diff,
        "clean": clean_metrics,
        "fe_ablation": fe_metrics,
        "tf_ablation": tf_metrics,
        "ablation_drop": {
            "fe": [float(a - b) for a, b in zip(clean_metrics["per_class_f1"], fe_metrics["per_class_f1"])],
            "tf": [float(a - b) for a, b in zip(clean_metrics["per_class_f1"], tf_metrics["per_class_f1"])],
        },
        "alpha_fe_by_class": alpha_by_class,
        "alpha_fe_mean_by_class": alpha_mean_by_class,
        "alpha_fe_median_by_class": alpha_median_by_class,
        "alpha_fe_under_ablation": {
            "clean": float(np.mean([r["alpha_fe"] for r in subject_rows])),
            "fe_ablation": float(np.mean([r["alpha_fe_under_fe_ablation"] for r in subject_rows])),
            "tf_ablation": float(np.mean([r["alpha_fe_under_tf_ablation"] for r in subject_rows])),
        },
        "correct_alpha": correct_alpha,
        "cka": {
            "overall": cka_overall,
            "per_fold": cka_per_fold,
            "perm_mean": float(null.mean()),
            "perm_p95": float(np.percentile(null, 95)),
        },
        "cosine": {
            "values": [float(v) for v in cos],
            "mean": float(cos.mean()),
            "median": float(np.median(cos)),
        },
        "fe_profile": {
            "z": profile_32.tolist(),
            "band_centers_hz": [float(x) for x in centers],
            "vmin": float(profile_32.min()),
            "vmax": float(profile_32.max()),
        },
        "a_stage_gain": {
            "per_class_f1_gain": gain,
            "pearson_with_alpha": pear,
            "spearman_with_alpha": spear,
        },
    }
    (FIG / "mechanism_data.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("wrote mechanism_data.json")
    print(json.dumps({
        "n": out["n"],
        "clean_diff": out["clean_prediction_max_abs_diff"],
        "clean_macro_f1": out["clean"]["macro_f1"],
        "fe_ablation_macro_f1": out["fe_ablation"]["macro_f1"],
        "tf_ablation_macro_f1": out["tf_ablation"]["macro_f1"],
        "alpha_fe_mean": out["alpha_fe_under_ablation"]["clean"],
        "cka": out["cka"],
        "cosine_mean": out["cosine"]["mean"],
        "gain_pearson": out["a_stage_gain"]["pearson_with_alpha"],
        "gain_spearman": out["a_stage_gain"]["spearman_with_alpha"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
