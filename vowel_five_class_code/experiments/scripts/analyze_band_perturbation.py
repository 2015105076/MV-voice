#!/usr/bin/env python3
"""FE frequency-band group perturbation.

The 128 normalized FE bands are grouped into 16 contiguous groups of 8 bands.
Each group is replaced by its training mean (zero in normalized space), the FE
embedding and the gated head are recomputed, and the true-class probability and
macro-F1 are recorded. The top 1/2/4 groups are then compared with 100 random
sets of the same size, following the E6 region-perturbation design.

This is a view-specific FE perturbation; the full shared-log-Mel perturbation
that recomputes TF and FE is reserved for the locked E6 phase.
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
N_GROUPS = 16
BANDS_PER_GROUP = 8
N_RANDOM = 100


def mel_centers(n_mels=128, fmin=20.0, fmax=8000.0):
    mel = lambda f: 1127.0 * np.log1p(f / 700.0)
    hz = lambda m: 700.0 * np.expm1(m / 1127.0)
    edges = np.linspace(mel(fmin), mel(fmax), n_mels + 2)[1:-1]
    return hz(edges)


def load_item(record, stats):
    with np.load(ROOT / record["feature_path"], allow_pickle=False) as cached:
        tf = torch.from_numpy(cached["TF"].copy()).cuda()
        fe = torch.from_numpy(cached["FE"].copy()).cuda()
    fe = (fe - stats["FE_mean"].cuda()) / stats["FE_std"].cuda()
    return tf, fe


def embed_tf(model, tf):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.tf_projection(model.tf_embedding(tf)).float()


def head_batch(model, z_tf, z_fe):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        stack = torch.stack([z_tf, z_fe], dim=1)
        weights = model.score(stack).softmax(1)
        logits = model.head((stack * weights).sum(1))
    return logits.float()


def subject_probs(probs, subjects, labels):
    grouped = defaultdict(list)
    for p, s, y in zip(probs, subjects, labels):
        grouped[s].append((p, y))
    rows = []
    for s in sorted(grouped):
        items = grouped[s]
        rows.append((s, np.mean([p for p, _ in items], axis=0), items[0][1]))
    return rows


def metric_block(rows):
    y = np.array([y for _, _, y in rows])
    p = np.array([p for _, p, _ in rows])
    pred = p.argmax(1)
    per_class = f1_score(y, pred, labels=list(range(5)), average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class_f1": [float(x) for x in per_class],
        "mean_true_prob": float(np.mean([p[y[i]] for i, (_, p, _) in enumerate(rows)])),
    }


def main() -> None:
    torch.set_num_threads(12)
    records = read_jsonl(ROOT / "features" / "development_manifest.jsonl")
    fold_map = {
        f["subject_id"]: f["validation_fold"]
        for f in read_jsonl(ROOT / "splits" / "development_folds.jsonl")
        if f["fold_count"] == 3
    }
    label_by_subject = {r["subject_id"]: LABELS.index(r["label"]) for r in records}

    # cache per-fold validation embeddings; phase 2 reuses them
    folds = {}
    for fold in FOLDS:
        print(f"fold {fold}: phase 1", flush=True)
        val = [r for r in records if fold_map[r["subject_id"]] == fold]
        payload = torch.load(
            ROOT / "runs" / f"{RUN}_f{fold}_s3407" / "best.pt",
            map_location="cpu", weights_only=False,
        )
        stats = payload["statistics"]
        model = MultiViewModel("beats_last2", "TF+FE", dropout=0.2, fusion="gated").cuda()
        model.load_state_dict(payload["model"])
        model.eval()

        z_tf, fe_val, subjects, vowels, y = [], [], [], [], []
        for r in val:
            tf, fe = load_item(r, stats)
            z_tf.append(embed_tf(model, tf)[0])
            fe_val.append(fe)
            subjects.append(r["subject_id"])
            vowels.append(r["vowel"])
            y.append(label_by_subject[r["subject_id"]])
        z_tf = torch.stack(z_tf)
        fe_val = torch.stack(fe_val)

        fold_data = {
            "z_tf": z_tf,
            "fe": fe_val,
            "subjects": subjects,
            "y": y,
            "model": model,
        }

        def run(mask_groups):
            fe_pert = fe_val.clone()
            if mask_groups:
                idx = np.concatenate([np.arange(g * BANDS_PER_GROUP, (g + 1) * BANDS_PER_GROUP)
                                      for g in mask_groups])
                fe_pert[:, idx] = 0.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z_fe = model.fe(fe_pert)
            logits = head_batch(model, z_tf, z_fe)
            probs = logits.softmax(-1).cpu().numpy()
            return subject_probs(probs, subjects, y)

        fold_data["clean_rows"] = run([])
        fold_data["group_rows"] = {g: run([g]) for g in range(N_GROUPS)}
        folds[fold] = fold_data
        del model
        torch.cuda.empty_cache()

    # pooled clean and group metrics
    def pooled(rows_by_fold):
        rows = []
        for fold in FOLDS:
            rows.extend(rows_by_fold[fold])
        return rows

    clean_rows = pooled({f: folds[f]["clean_rows"] for f in FOLDS})
    clean = metric_block(clean_rows)
    groups = []
    centers = mel_centers(128).reshape(N_GROUPS, BANDS_PER_GROUP).mean(1)
    for g in range(N_GROUPS):
        rows = pooled({f: folds[f]["group_rows"][g] for f in FOLDS})
        m = metric_block(rows)
        groups.append({
            "group": g,
            "bands": [g * BANDS_PER_GROUP, (g + 1) * BANDS_PER_GROUP - 1],
            "center_hz": float(centers[g]),
            "true_prob_drop": float(clean["mean_true_prob"] - m["mean_true_prob"]),
            "macro_f1_drop": float(clean["macro_f1"] - m["macro_f1"]),
            "per_class_f1_drop": [float(a - b) for a, b in
                                  zip(clean["per_class_f1"], m["per_class_f1"])],
        })
    ranking = sorted(range(N_GROUPS), key=lambda g: groups[g]["true_prob_drop"], reverse=True)
    print("group ranking", ranking[:4], flush=True)

    # phase 2: top-k and random sets
    rng = np.random.default_rng(20261102)
    random_sets = {}
    for k in [1, 2, 4]:
        # sample with replacement: for k=1 there are only 16 distinct groups,
        # so 100 distinct singleton sets do not exist. Duplicate draws are a
        # valid randomization null.
        random_sets[k] = [
            tuple(sorted(rng.choice(N_GROUPS, size=k, replace=False).tolist()))
            for _ in range(N_RANDOM)
        ]

    def evaluate_mask(mask_groups):
        rows = []
        for fold in FOLDS:
            fd = folds[fold]
            model = fd["model"]
            z_tf = fd["z_tf"]
            fe_val = fd["fe"]
            fe_pert = fe_val.clone()
            if mask_groups:
                idx = np.concatenate([np.arange(g * BANDS_PER_GROUP, (g + 1) * BANDS_PER_GROUP)
                                      for g in mask_groups])
                fe_pert[:, idx] = 0.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z_fe = model.fe(fe_pert)
            logits = head_batch(model, z_tf, z_fe)
            probs = logits.softmax(-1).cpu().numpy()
            rows.extend(subject_probs(probs, fd["subjects"], fd["y"]))
        return metric_block(rows)

    topk = {}
    for k in [1, 2, 4]:
        top = tuple(ranking[:k])
        observed = evaluate_mask(top)
        random_drops = []
        for j, s in enumerate(random_sets[k]):
            m = evaluate_mask(s)
            random_drops.append(clean["mean_true_prob"] - m["mean_true_prob"])
            if (j + 1) % 20 == 0:
                print(f"  k={k}: random {j + 1}/{N_RANDOM}", flush=True)
        observed_drop = clean["mean_true_prob"] - observed["mean_true_prob"]
        random_drops = np.array(random_drops)
        topk[str(k)] = {
            "groups": list(top),
            "observed_drop": float(observed_drop),
            "observed_macro_f1": float(observed["macro_f1"]),
            "random_mean": float(random_drops.mean()),
            "random_p95": float(np.percentile(random_drops, 95)),
            "p_empirical": float((1 + np.sum(random_drops >= observed_drop)) / (1 + len(random_drops))),
            "random_values": random_drops.tolist(),
        }
        print(f"k={k} top={top} observed={observed_drop:.4f} random_mean={random_drops.mean():.4f} "
              f"p95={np.percentile(random_drops, 95):.4f} p={topk[str(k)]['p_empirical']:.3f}", flush=True)

    out = {
        "n": len(clean_rows),
        "labels": LABELS,
        "clean": clean,
        "groups": groups,
        "ranking": ranking,
        "topk": topk,
        "bands_per_group": BANDS_PER_GROUP,
        "n_random": N_RANDOM,
    }
    (FIG / "band_perturbation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("wrote band_perturbation.json")


if __name__ == "__main__":
    main()
