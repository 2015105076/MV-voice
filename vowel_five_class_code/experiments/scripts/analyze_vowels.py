"""E5 vowel complementarity and E7 MVFT analysis on development OOF predictions (CPU only).

Reads per-fold per-seed vowel-level predictions saved by the C runs and reports:
- six single-vowel subject metrics (macro-F1, macro-AUROC, class recall)
- leave-one-vowel-out deltas vs the six-vowel mean
- all 63 nonempty vowel subsets: distribution per size and development-selected subset per size
- vowel x class recall(TF+FE) - recall(TF) map
- MVFT exploratory endpoint (AUROC, average precision, argmax table) for the primary pipeline
Development data only; no test predictions exist in this restart.
"""
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
VOWELS = ['a', 'e', 'i', 'o', 'u', 'v']
CLASSES = ['NV', 'SD', 'UVFP', 'BSVFL', 'MVFT']
SEEDS = [3407, 8147, 19013]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def load_vowel_predictions(combo_key):
    """Per-fold seed-ensemble vowel-level predictions: {subject_id: {vowel: probability-array}}"""
    merged = defaultdict(lambda: defaultdict(list))
    for fold in range(5):
        for seed in SEEDS:
            path = ROOT / 'runs' / f'C_{combo_key}_f{fold}_s{seed}' / 'validation_vowel_predictions.jsonl'
            for row in read_jsonl(path):
                probability = [row[f'p_{c}'] for c in CLASSES]
                merged[row['subject_id']][row['vowel']].append(probability)
    output = {}
    for subject, vowels in merged.items():
        for vowel in vowels:
            assert len(vowels[vowel]) == len(SEEDS), f'{subject} {vowel} missing seeds'
            vowels[vowel] = np.mean(vowels[vowel], axis=0)
        output[subject] = vowels
    return output


def true_labels():
    labels = {}
    for row in read_jsonl(ROOT / 'splits' / 'subject_split.jsonl'):
        labels[row['subject_id']] = row['label']
    return labels


def subject_metrics(subjects, labels):
    y = np.array([CLASSES.index(labels[s['subject_id']]) for s in subjects])
    p = np.array([s['probabilities'] for s in subjects])
    recall = []
    for c in range(5):
        mask = y == c
        pred = p.argmax(1) == c
        recall.append(float(np.mean(pred[mask])) if mask.sum() else None)
    return {'macro_F1': float(f1_score(y, p.argmax(1), labels=list(range(5)), average='macro', zero_division=0)),
            'macro_OVR_AUROC': float(roc_auc_score(y, p, labels=list(range(5)), multi_class='ovr', average='macro')),
            'class_recall': recall}


def subset_probability(predictions, subject, vowels_subset):
    return np.mean([predictions[subject][v] for v in vowels_subset], axis=0)


def vowel_analysis(primary_key='TF_FE', tf_key='TF'):
    labels = true_labels()
    primary = load_vowel_predictions(primary_key)
    tf = load_vowel_predictions(tf_key)
    assert set(primary) == set(tf)
    subjects = sorted(primary)
    summary = {'primary_pipeline': primary_key.replace('_', '+'), 'reference_pipeline': tf_key.replace('_', '+'),
               'n_subjects': len(subjects), 'vowel_order': VOWELS, 'class_order': CLASSES, 'seeds': SEEDS,
               'development_only': True, 'test_evaluated': False}
    # 1. six single vowels
    single = []
    for vowel in VOWELS:
        rows = [{'subject_id': s, 'probabilities': primary[s][vowel]} for s in subjects]
        metrics = subject_metrics(rows, labels)
        single.append({'vowel': vowel, **metrics})
        print(f'E5 single vowel {vowel}: macro-F1={metrics["macro_F1"]:.4f} AUROC={metrics["macro_OVR_AUROC"]:.4f}', flush=True)
    summary['single_vowel'] = single
    six = subject_metrics([{'subject_id': s, 'probabilities': np.mean([primary[s][v] for v in VOWELS], axis=0)} for s in subjects], labels)
    # 2. leave-one-out deltas
    loo = []
    for vowel in VOWELS:
        rows = [{'subject_id': s, 'probabilities': np.mean([primary[s][v] for v in VOWELS if v != vowel], axis=0)} for s in subjects]
        metrics = subject_metrics(rows, labels)
        loo.append({'removed_vowel': vowel, **metrics,
                    'macro_F1_delta_vs_six': metrics['macro_F1'] - six['macro_F1'],
                    'class_recall_delta_vs_six': [a - b if a is not None else None for a, b in zip(metrics['class_recall'], six['class_recall'])]})
        print(f'E5 leave-out {vowel}: macro-F1={metrics["macro_F1"]:.4f} delta={metrics["macro_F1"] - six["macro_F1"]:+.4f}', flush=True)
    summary['six_vowel'] = six
    summary['leave_one_out'] = loo
    # 3. all 63 subsets: distribution per size + development-selected subset per size (tie within 0.002 -> lexical vowel order)
    by_size = defaultdict(list)
    for size in range(1, 7):
        for subset in itertools.combinations(VOWELS, size):
            rows = [{'subject_id': s, 'probabilities': subset_probability(primary, s, subset)} for s in subjects]
            score = subject_metrics(rows, labels)['macro_F1']
            by_size[size].append({'subset': list(subset), 'macro_F1': score})
    subset_summary = {}
    for size, entries in by_size.items():
        entries.sort(key=lambda e: (-e['macro_F1'], e['subset']))
        best = max(e['macro_F1'] for e in entries)
        selected = next(e for e in entries if best - e['macro_F1'] <= .002)
        subset_summary[str(size)] = {'n_subsets': len(entries), 'best_subset': entries[0],
                                     'development_selected_subset': selected, 'scores': entries}
        print(f'E5 size {size}: {len(entries)} subsets, best={entries[0]["subset"]} {entries[0]["macro_F1"]:.4f}, selected={selected["subset"]}', flush=True)
    summary['subsets_by_size'] = subset_summary
    # 4. vowel x class recall(TF+FE) - recall(TF)
    heatmap = []
    for vowel in VOWELS:
        row = []
        for c, cls in enumerate(CLASSES):
            mask = [labels[s] == cls for s in subjects]
            pf = np.array([primary[s][vowel].argmax() == c for s, m in zip(subjects, mask) if m], dtype=float)
            t = np.array([tf[s][vowel].argmax() == c for s, m in zip(subjects, mask) if m], dtype=float)
            row.append(float(pf.mean() - t.mean()) if len(pf) else None)
        heatmap.append({'vowel': vowel, 'class_recall_delta': row})
        print(f'E5 recall delta vowel {vowel}: {[f"{x:+.3f}" if x is not None else "-" for x in row]}', flush=True)
    summary['vowel_class_recall_delta'] = heatmap
    (ROOT / 'metrics').mkdir(exist_ok=True)
    (ROOT / 'metrics' / 'e5_vowel_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def mvft_analysis(primary_key='TF_FE'):
    labels = true_labels()
    primary = load_vowel_predictions(primary_key)
    subjects = sorted(primary)
    probabilities = np.array([np.mean([primary[s][v] for v in VOWELS], axis=0) for s in subjects])
    y = np.array([CLASSES.index(labels[s]) for s in subjects])
    mvft = CLASSES.index('MVFT')
    p_mvft = probabilities[:, mvft]
    au = float(roc_auc_score(y == mvft, p_mvft))
    ap = float(average_precision_score(y == mvft, p_mvft))
    argmax = probabilities.argmax(1) == mvft
    tp = int(np.sum(argmax & (y == mvft)))
    fn = int(np.sum((y == mvft) & ~argmax))
    fp = int(np.sum(argmax & (y != mvft)))
    tn = int(np.sum((y != mvft) & ~argmax))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    counts = {c: int(np.sum(y == CLASSES.index(c))) for c in CLASSES}
    summary = {'pipeline': primary_key.replace('_', '+'), 'n_subjects': len(subjects), 'class_order': CLASSES,
               'MVFT_AUROC': au, 'MVFT_average_precision': ap,
               'argmax': {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
                          'sensitivity': sensitivity, 'specificity': specificity,
                          'MVFT_prevalence': counts['MVFT'] / len(subjects)},
               'class_counts': counts, 'threshold_analysis_enabled': False, 'development_only': True, 'test_evaluated': False}
    (ROOT / 'metrics').mkdir(exist_ok=True)
    (ROOT / 'metrics' / 'e7_mvft_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    vowel_analysis('TF_FE', 'TF')
    mvft_analysis('TF_FE')
