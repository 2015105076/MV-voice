#!/usr/bin/env python3
"""Shared log-Mel spectrum perturbation (E6 core).

The 128 Mel bands are grouped into 16 contiguous groups of 8 bands. A group is
replaced by its training mean in the shared log-Mel spectrum; both views are
recomputed (TF through the batched BEATs encoder, FE from the modified power
spectrum), and the true-class probability and macro-F1 are recorded. The
top-1/2/4 groups identified by the FE-only analysis are compared with 100
random groups of the same size.

The FE update is exact and cheap: after replacing log-power values in a band
with a constant (the training mean log-Mel), the time-averaged log-power FE in
that band equals the same constant. Thus only the TF branch needs a new forward
pass; FE is the clean FE with the masked bands overwritten.

This script uses a batched BEATs path validated against the saved predictions
(fold 0 clean macro-F1 0.8288 vs saved 0.8282; max probability difference
0.0032). It does not retrain and does not read the test split.
"""

from __future__ import annotations

import json
import sys
import time
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
BATCH = 256


def mel_centers(n_mels=128, fmin=20.0, fmax=8000.0):
    mel = lambda f: 1127.0 * np.log1p(f / 700.0)
    hz = lambda m: 700.0 * np.expm1(m / 1127.0)
    edges = np.linspace(mel(fmin), mel(fmax), n_mels + 2)[1:-1]
    return hz(edges)


def load_model(fold):
    payload = torch.load(
        ROOT / "runs" / f"{RUN}_f{fold}_s3407" / "best.pt",
        map_location="cpu", weights_only=False,
    )
    model = MultiViewModel("beats_last2", "TF+FE", dropout=0.2, fusion="gated").cuda()
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload["statistics"]


def beats_batch(model, tfs):
    """Batched official BEATs forward with padding masks.

    tfs: list of CPU tensors [T,128]. Returns pooled [B,768] float32.
    The truncation to a multiple of 16 matches the single-item convolution.
    """
    device = next(model.parameters()).device
    B = len(tfs)
    tmax = max((tf.shape[0] // 16) * 16 for tf in tfs)
    L = torch.zeros(B, tmax, 128, device=device)
    mask = torch.ones(B, tmax, dtype=torch.bool, device=device)
    for i, tf in enumerate(tfs):
        t = (tf.shape[0] // 16) * 16
        L[i, :t] = tf[:t].to(device)
        mask[i, :t] = False
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        normalized = (L - 15.41663) / (2 * 6.55582)
        features = model.backbone.patch_embedding(normalized.unsqueeze(1))
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        features = model.backbone.layer_norm(features)
        padding = model.backbone.forward_padding_mask(features, mask)
        if model.backbone.post_extract_proj is not None:
            features = model.backbone.post_extract_proj(features)
        x = model.backbone.dropout_input(features)
        x, _ = model.backbone.encoder(x, padding_mask=padding)
        valid = (~padding).float()
        pooled = (x * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True)
    return pooled.float()


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
    fe_only = json.loads((FIG / "band_perturbation.json").read_text(encoding="utf-8"))
    top4 = fe_only["topk"]["4"]["groups"]
    print("pre-specified FE-only top groups", top4, flush=True)

    rng = np.random.default_rng(20261102)
    random_sets = {
        k: [tuple(sorted(rng.choice(N_GROUPS, size=k, replace=False).tolist()))
            for _ in range(N_RANDOM)]
        for k in [1, 2, 4]
    }

    conditions = {"clean": []}
    for g in range(N_GROUPS):
        conditions[f"group_{g}"] = []
    for k in [1, 2, 4]:
        conditions[f"top_{k}"] = []
        for j in range(N_RANDOM):
            conditions[f"random_{k}_{j}"] = []

    for fold in FOLDS:
        print(f"fold {fold}: loading", flush=True)
        records = read_jsonl(ROOT / "features" / "development_manifest.jsonl")
        fold_map = {
            f["subject_id"]: f["validation_fold"]
            for f in read_jsonl(ROOT / "splits" / "development_folds.jsonl")
            if f["fold_count"] == 3
        }
        label_by_subject = {r["subject_id"]: LABELS.index(r["label"]) for r in records}
        val = [r for r in records if fold_map[r["subject_id"]] == fold]
        model, stats = load_model(fold)
        tf_mean = stats["TF_mean"].cpu()
        fe_mean = stats["FE_mean"].cpu()
        fe_std = stats["FE_std"].cpu()

        tfs, fe_norm, subjects, y = [], [], [], []
        for r in val:
            with np.load(ROOT / r["feature_path"], allow_pickle=False) as cached:
                tfs.append(torch.from_numpy(cached["TF"].copy()))
                fe_raw = torch.from_numpy(cached["FE"].copy())
            fe_norm.append((fe_raw - fe_mean) / fe_std)
            subjects.append(r["subject_id"])
            y.append(label_by_subject[r["subject_id"]])
        fe_norm = torch.stack(fe_norm)

        def run_condition(mask_groups):
            if mask_groups:
                idx = np.concatenate([np.arange(g * BANDS_PER_GROUP, (g + 1) * BANDS_PER_GROUP)
                                      for g in mask_groups])
                fe_patch = (tf_mean[idx] - fe_mean[idx]) / fe_std[idx]
            else:
                idx = None
                fe_patch = None
            probs = []
            for start in range(0, len(tfs), BATCH):
                tf_batch = []
                for tf in tfs[start:start + BATCH]:
                    if idx is not None:
                        t = tf.clone()
                        t[:, idx] = tf_mean[idx]
                        tf_batch.append(t)
                    else:
                        tf_batch.append(tf)
                fe_batch = fe_norm[start:start + BATCH].clone()
                if fe_patch is not None:
                    fe_batch[:, idx] = fe_patch
                z_tf = model.tf_projection(beats_batch(model, tf_batch)).float()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    z_fe = model.fe(fe_batch.cuda()).float()
                logits = head_batch(model, z_tf, z_fe)
                probs.append(logits.softmax(-1).cpu().numpy())
            probs = np.concatenate(probs, axis=0)
            return subject_probs(probs, subjects, y)

        conditions["clean"].extend(run_condition([]))
        for g in range(N_GROUPS):
            conditions[f"group_{g}"].extend(run_condition([g]))
            print(f"  fold {fold}: group {g + 1}/{N_GROUPS}", flush=True)
        for k in [1, 2, 4]:
            conditions[f"top_{k}"].extend(run_condition(list(top4[:k])))
            for j, s in enumerate(random_sets[k]):
                conditions[f"random_{k}_{j}"].extend(run_condition(list(s)))
                if (j + 1) % 25 == 0:
                    print(f"  fold {fold} k={k}: random {j + 1}/{N_RANDOM}", flush=True)
        del model
        torch.cuda.empty_cache()
        print(f"fold {fold}: done", flush=True)

    clean_rows = conditions["clean"]
    clean = metric_block(clean_rows)
    centers = mel_centers(128).reshape(N_GROUPS, BANDS_PER_GROUP).mean(1)
    groups = []
    for g in range(N_GROUPS):
        m = metric_block(conditions[f"group_{g}"])
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
    print("shared-spectrum ranking", ranking[:4], flush=True)

    topk = {}
    for k in [1, 2, 4]:
        observed = metric_block(conditions[f"top_{k}"])
        random_drops = []
        for j in range(N_RANDOM):
            m = metric_block(conditions[f"random_{k}_{j}"])
            random_drops.append(clean["mean_true_prob"] - m["mean_true_prob"])
        observed_drop = clean["mean_true_prob"] - observed["mean_true_prob"]
        random_drops = np.array(random_drops)
        topk[str(k)] = {
            "groups": list(top4[:k]),
            "observed_drop": float(observed_drop),
            "observed_macro_f1": float(observed["macro_f1"]),
            "random_mean": float(random_drops.mean()),
            "random_p95": float(np.percentile(random_drops, 95)),
            "p_empirical": float((1 + np.sum(random_drops >= observed_drop)) / (1 + len(random_drops))),
            "random_values": random_drops.tolist(),
        }
        print(f"k={k} top={top4[:k]} observed={observed_drop:.4f} "
              f"random_mean={random_drops.mean():.4f} p95={np.percentile(random_drops, 95):.4f} "
              f"p={topk[str(k)]['p_empirical']:.3f}", flush=True)

    out = {
        "n": len(clean_rows),
        "labels": LABELS,
        "clean": clean,
        "groups": groups,
        "ranking_shared": ranking,
        "fe_only_top4": top4,
        "topk": topk,
        "bands_per_group": BANDS_PER_GROUP,
        "n_random": N_RANDOM,
        "batched_validation": "fold 0 clean macro-F1 0.8288 vs saved 0.8282; max prob diff 0.0032",
    }
    (FIG / "shared_spectrum_perturbation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("wrote shared_spectrum_perturbation.json")


if __name__ == "__main__":
    main()
