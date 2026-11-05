"""C baselines: three-fold development tuning then five-fold three-seed subject OOF evaluation.

LTAS_GBM, MFCC_XGBoost and FE_GBM run here; logMel_ResNet18 reuses the identical
ResNet18 TF deep pipeline runs dispatched by run_stage_c. Development data only.
"""
import argparse
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from audit_waveforms import digest_file, write_json, write_jsonl
from freeze_data import ROOT, read_jsonl
from train import aggregate, metrics


def subject_weight(labels_subject, class_order):
    counts = Counter(labels_subject.values())
    n = len(labels_subject)
    return {s: n / (5 * counts[l]) for s, l in labels_subject.items()}


def load_matrices(feature_key):
    records = read_jsonl(ROOT / 'features' / 'development_manifest.jsonl')
    x, rows = [], []
    for record in records:
        path = ROOT / record['feature_path']
        if digest_file(path) != record['feature_sha256']:
            raise RuntimeError('feature cache fingerprint mismatch')
        with np.load(path, allow_pickle=False) as cached:
            x.append(np.asarray(cached[feature_key], dtype=np.float64))
        rows.append(record)
    return np.stack(x), rows


def fold_map(count):
    return {r['subject_id']: r['validation_fold'] for r in read_jsonl(ROOT / 'splits' / 'development_folds.jsonl') if r['fold_count'] == count}


def fit_evaluate(classifier_factory, x, rows, train_indices, val_indices, sample_weight):
    labels = ['NV', 'SD', 'UVFP', 'BSVFL', 'MVFT']
    y = np.array([labels.index(r['label']) for r in rows])
    model = classifier_factory()
    model.fit(x[train_indices], y[train_indices], sample_weight=sample_weight[train_indices])
    probabilities = model.predict_proba(x[val_indices])
    records = [rows[i] for i in val_indices]
    subjects = aggregate(records, probabilities)
    return subjects, metrics(subjects)


def grid_configs(spec):
    keys = [k for k in spec['grid'] if k != 'min_samples_leaf']
    values = [spec['grid'][k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_baseline(baseline_id, spec):
    out = ROOT / 'runs' / f'baselines_{baseline_id}.json'
    if out.exists():
        return json.loads(out.read_text(encoding='utf-8'))
    protocol = json.loads((ROOT / 'protocol.execution.v1.2.json').read_text(encoding='utf-8'))
    lock = json.loads((ROOT / 'locks' / 'data_lock.json').read_text(encoding='utf-8'))
    if digest_file(ROOT / 'protocol.execution.v1.2.json') != lock['protocol_sha256']:
        raise RuntimeError('protocol changed after data lock')
    for rel, fingerprint in lock['files_sha256'].items():
        if digest_file(ROOT / rel.replace('\\', '/')) != fingerprint:
            raise RuntimeError(f'locked artifact changed: {rel}')
    feature_key = 'LTAS' if baseline_id == 'LTAS_GBM' else ('MFCC' if baseline_id == 'MFCC_XGBoost' else 'FE')
    x, rows = load_matrices(feature_key)
    class_order = protocol['data']['class_order']
    seeds = protocol['training']['confirmation_seeds']
    subject_labels = {r['subject_id']: r['label'] for r in rows}
    configs = grid_configs(spec)

    def factory(cfg, seed):
        if baseline_id == 'MFCC_XGBoost':
            from xgboost import XGBClassifier
            return XGBClassifier(objective='multi:softprob', num_class=5, tree_method='hist', random_state=seed,
                                 n_estimators=cfg['n_estimators'], learning_rate=cfg['learning_rate'],
                                 max_depth=cfg['max_depth'], subsample=cfg['subsample'],
                                 colsample_bytree=cfg['colsample_bytree'], reg_lambda=cfg['reg_lambda'],
                                 min_child_weight=cfg['min_child_weight'])
        return GradientBoostingClassifier(loss='log_loss', random_state=seed, n_estimators=cfg['n_estimators'],
                                          learning_rate=cfg['learning_rate'], max_depth=cfg['max_depth'],
                                          min_samples_leaf=spec['grid']['min_samples_leaf'][0], subsample=cfg['subsample'])

    started = time.perf_counter()
    folds3 = fold_map(3)
    tuning = []
    for config in configs:
        scores = []
        for fold in range(3):
            train_idx = [i for i, r in enumerate(rows) if folds3[r['subject_id']] != fold]
            val_idx = [i for i, r in enumerate(rows) if folds3[r['subject_id']] == fold]
            if {rows[i]['leakage_group_id'] for i in train_idx} & {rows[i]['leakage_group_id'] for i in val_idx}:
                raise RuntimeError('group leakage in baseline tuning fold')
            train_labels = {rows[i]['subject_id']: rows[i]['label'] for i in train_idx}
            weights = subject_weight(train_labels, class_order)
            sample_weight = np.zeros(len(rows))
            for i in train_idx:
                sample_weight[i] = weights[rows[i]['subject_id']]
            _, result = fit_evaluate(lambda: factory(config, 3407), x, rows, train_idx, val_idx, sample_weight)
            scores.append(result['macro_F1'])
        entry = {**config, 'mean_fold_macro_F1': float(np.mean(scores)), 'folds': scores}
        tuning.append(entry)
        print(f'{baseline_id} tuning {tuning.index(entry)+1}/{len(configs)}: mean={entry["mean_fold_macro_F1"]:.4f}', flush=True)
    selected = max(tuning, key=lambda e: e['mean_fold_macro_F1'])
    folds5 = fold_map(5)
    fold_results, oof = [], []
    for fold in range(5):
        train_idx = [i for i, r in enumerate(rows) if folds5[r['subject_id']] != fold]
        val_idx = [i for i, r in enumerate(rows) if folds5[r['subject_id']] == fold]
        if {rows[i]['leakage_group_id'] for i in train_idx} & {rows[i]['leakage_group_id'] for i in val_idx}:
            raise RuntimeError('group leakage in baseline evaluation fold')
        train_labels = {rows[i]['subject_id']: rows[i]['label'] for i in train_idx}
        weights = subject_weight(train_labels, class_order)
        sample_weight = np.zeros(len(rows))
        for i in train_idx:
            sample_weight[i] = weights[rows[i]['subject_id']]
        seed_subjects = []
        for seed in seeds:
            subjects, _ = fit_evaluate(lambda: factory(selected, seed), x, rows, train_idx, val_idx, sample_weight)
            seed_subjects.append(subjects)
        per_seed = [{s['subject_id']: s['probabilities'] for s in subjects} for subjects in seed_subjects]
        probabilities = {subject_id: np.mean([mapping[subject_id] for mapping in per_seed], axis=0)
                         for subject_id in per_seed[0]}
        ensemble = [{**seed_subjects[0][j], 'probabilities': probabilities[seed_subjects[0][j]['subject_id']].tolist()}
                    for j in range(len(seed_subjects[0]))]
        result = metrics(ensemble)
        fold_results.append(result)
        for entry in ensemble:
            entry['baseline'] = baseline_id
            entry['fold'] = fold
            oof.append(entry)
        print(f'{baseline_id} eval fold {fold}: ensemble macro-F1={result["macro_F1"]:.4f}', flush=True)
    (ROOT / 'predictions').mkdir(exist_ok=True)
    write_jsonl(ROOT / 'predictions' / f'baseline_{baseline_id}_oof.jsonl', oof)
    summary = {'baseline': baseline_id, 'feature': feature_key, 'protocol_sha256': digest_file(ROOT / 'protocol.execution.v1.2.json'),
               'data_lock_sha256': digest_file(ROOT / 'locks' / 'data_lock.json'),
               'feature_manifest_sha256': digest_file(ROOT / 'features' / 'development_manifest.jsonl'),
               'source_sha256': {n: digest_file(ROOT / 'scripts' / n) for n in ['train_baselines.py', 'train.py', 'features.py']},
               'seeds': seeds, 'tuning_folds': 3, 'evaluation_folds': 5, 'grid_trials': len(configs),
               'tuning': tuning, 'selected_configuration': selected,
               'mean_fold_macro_F1': float(np.mean([r['macro_F1'] for r in fold_results])), 'fold_results': fold_results,
               'wall_time_seconds': time.perf_counter() - started, 'test_evaluated': False}
    write_json(out, summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, choices=['LTAS_GBM', 'MFCC_XGBoost', 'FE_GBM'])
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'protocol.execution.v1.2.json').read_text(encoding='utf-8'))
    spec = next(b for b in protocol['baselines'] if b['id'] == args.baseline)
    summary = run_baseline(args.baseline, spec)
    print(json.dumps({k: summary[k] for k in ['baseline', 'selected_configuration', 'mean_fold_macro_F1', 'wall_time_seconds']}, ensure_ascii=False), flush=True)
