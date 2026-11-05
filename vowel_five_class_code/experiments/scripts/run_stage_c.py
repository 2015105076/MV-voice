"""Run/resume prespecified C: five-fold three-seed seven-view comparison, second-backbone check,
ResNet18 TF baseline and tuned literature baselines. Stops before D and test access."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

from audit_waveforms import digest_file, write_json, write_jsonl
from freeze_data import ROOT, read_jsonl
from search_plan import rank_near_ties
from train import metrics


def state_write(state):
    temporary = ROOT / 'runs' / 'stage_C_state.tmp'
    write_json(temporary, state)
    temporary.replace(ROOT / 'runs' / 'stage_C_state.json')


def freeze_c_plan():
    path = ROOT / 'locks' / 'stage_C_plan.json'
    protocol_path = ROOT / 'protocol.execution.v1.2.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    b = json.loads((ROOT / 'runs' / 'stage_B_summary.json').read_text(encoding='utf-8'))
    frozen_tf = next(w for w in b['pipeline_winners'] if w['backbone'] == 'beats_frozen' and w['views'] == 'TF')
    frozen_tf_fe = next(w for w in b['pipeline_winners'] if w['backbone'] == 'beats_frozen' and w['views'] == 'TF+FE')
    resnet = b['resnet18_TF_baseline_trial']
    source_names = ['train_c.py', 'train.py', 'models.py', 'features.py', 'search_plan.py', 'run_stage_c.py', 'train_baselines.py']
    plan = {'stage': 'C',
            'protocol_sha256': digest_file(protocol_path),
            'data_lock_sha256': digest_file(ROOT / 'locks' / 'data_lock.json'),
            'stage_B_summary_sha256': digest_file(ROOT / 'runs' / 'stage_B_summary.json'),
            'stage_A_summary_sha256': digest_file(ROOT / 'runs' / 'stage_A_summary.json'),
            'feature_manifest_sha256': digest_file(ROOT / 'features' / 'development_manifest.jsonl'),
            'selected_backbone': b['selected_backbone'],
            'selected_fusion': b['selected_fusion'],
            'common_hyperparameters': b['common_training_hyperparameters'],
            'folds': protocol['search']['stage_C']['folds'],
            'seeds': protocol['training']['confirmation_seeds'],
            'maximum_epochs': protocol['training']['maximum_epochs'],
            'early_stopping_patience': protocol['training']['early_stopping_patience'],
            'view_combinations': protocol['models']['view_combinations'],
            'view_selection_rule': protocol['search']['stage_C']['view_selection'],
            'near_tie_rule': protocol['search']['stage_B']['near_tie_rule'],
            'second_backbone_check': [
                {'backbone': 'beats_frozen', 'views': 'TF', 'parameters': frozen_tf['parameters'], 'folds': 5, 'seeds': [3407],
                 'role': 'exploratory transfer-of-gain analysis; not part of primary selection'},
                {'backbone': 'beats_frozen', 'views': 'TF+FE', 'parameters': frozen_tf_fe['parameters'], 'folds': 5, 'seeds': [3407],
                 'role': 'exploratory transfer-of-gain analysis; not part of primary selection'}],
            'resnet18_TF_baseline': {'backbone': 'resnet18_imagenet', 'views': 'TF', 'parameters': resnet['parameters'],
                                     'folds': 5, 'seeds': protocol['training']['confirmation_seeds'],
                                     'role': 'logMel_ResNet18 baseline; identical pipeline reused, no separate control model'},
            'baselines': [{'id': 'LTAS_GBM'}, {'id': 'MFCC_XGBoost'}, {'id': 'FE_GBM'}],
            'reuse_note': 'no pre-existing five-fold seed-3407 runs in this restart; all C runs are trained fresh',
            'source_revision_note': '2026-11-03: fixed np.mean generator bug in the aggregation paths of run_stage_c.py and train_baselines.py; no run-level training or metric computation is affected',
            'test_used': False,
            'source_sha256': {n: digest_file(ROOT / 'scripts' / n) for n in source_names}}
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != plan:
            raise RuntimeError('frozen C plan changed; do not regenerate after observing results')
        return existing
    write_json(path, plan)
    return plan


def deep_task(plan, backbone, views, parameters, fold, seed, run_id):
    return {'run_id': run_id, 'backbone': backbone, 'views': views, 'fold': fold, 'seed': seed,
            'head_lr': parameters['head_lr'], 'backbone_multiplier': parameters['backbone_multiplier'],
            'weight_decay': parameters['weight_decay'], 'dropout': parameters['dropout'],
            'fusion': parameters['fusion'], 'fold_count': plan['folds'], 'epochs': plan['maximum_epochs']}


def queue(plan):
    # Light non-TF combinations first so the five-fold plumbing is validated early.
    combos = ['TE', 'FE', 'TE+FE', 'TF', 'TF+TE', 'TF+FE', 'TF+TE+FE']
    tasks = []
    for combo in combos:
        single = '+' not in combo
        parameters = {**plan['common_hyperparameters'], 'fusion': 'concat' if single else plan['selected_fusion']}
        for fold in range(plan['folds']):
            for seed in plan['seeds']:
                run_id = f'C_{combo.replace("+", "_")}_f{fold}_s{seed}'
                tasks.append(deep_task(plan, plan['selected_backbone'], combo, parameters, fold, seed, run_id))
    for check in plan['second_backbone_check']:
        for fold in range(plan['folds']):
            for seed in check['seeds']:
                run_id = f'C_{check["backbone"]}_{check["views"].replace("+", "_")}_f{fold}_s{seed}'
                tasks.append(deep_task(plan, check['backbone'], check['views'], check['parameters'], fold, seed, run_id))
    base = plan['resnet18_TF_baseline']
    for fold in range(plan['folds']):
        for seed in base['seeds']:
            run_id = f'C_{base["backbone"]}_TF_f{fold}_s{seed}'
            tasks.append(deep_task(plan, base['backbone'], 'TF', base['parameters'], fold, seed, run_id))
    return tasks


def execute(task):
    command = [sys.executable, '-u', str(ROOT / 'scripts' / 'train_c.py'), '--backbone', task['backbone'],
               '--views', task['views'], '--fold', str(task['fold']), '--fold-count', str(task['fold_count']),
               '--seed', str(task['seed']), '--epochs', str(task['epochs']), '--dropout', str(task['dropout']),
               '--fusion', task['fusion'], '--head-lr', str(task['head_lr']),
               '--backbone-multiplier', str(task['backbone_multiplier']), '--weight-decay', str(task['weight_decay']),
               '--run-id', task['run_id']]
    subprocess.run(command, cwd=ROOT, check=True)


def ensemble_subjects(combo_key, fold, seeds):
    merged = defaultdict(list)
    for seed in seeds:
        run_id = f'C_{combo_key}_f{fold}_s{seed}'
        path = ROOT / 'runs' / run_id / 'validation_subject_predictions.jsonl'
        for row in read_jsonl(path):
            merged[row['subject_id']].append(row)
    output = []
    for subject, rows in merged.items():
        probability = np.mean([r['probabilities'] for r in rows], axis=0)
        output.append({'subject_id': subject, 'leakage_group_id': rows[0]['leakage_group_id'],
                       'y_true': rows[0]['y_true'], 'probabilities': probability.tolist()})
    return output


def single_seed_subjects(combo_key, fold, seed):
    run_id = f'C_{combo_key}_f{fold}_s{seed}'
    return read_jsonl(ROOT / 'runs' / run_id / 'validation_subject_predictions.jsonl')


def aggregate_views(plan):
    results = []
    (ROOT / 'predictions').mkdir(exist_ok=True)
    for combo in plan['view_combinations']:
        key = combo.replace('+', '_')
        fold_scores, oof = [], []
        for fold in range(plan['folds']):
            subjects = ensemble_subjects(key, fold, plan['seeds'])
            score = metrics(subjects)
            fold_scores.append(score)
            for entry in subjects:
                oof.append({**entry, 'pipeline': combo, 'fold': fold})
        write_jsonl(ROOT / 'predictions' / f'c_subject_oof_{key}.jsonl', oof)
        seed_run = ROOT / 'runs' / f'C_{key}_f0_s{plan["seeds"][0]}' / 'result.json'
        run_result = json.loads(seed_run.read_text(encoding='utf-8'))
        results.append({'pipeline': combo, 'views': combo, 'fusion': 'concat' if '+' not in combo else plan['selected_fusion'],
                        'mean_fold_macro_F1': float(np.mean([s['macro_F1'] for s in fold_scores])),
                        'fold_results': fold_scores, 'total_parameters': run_result['total_parameters'],
                        'trainable_parameters': run_result['trainable_parameters']})
        print(f'C view {combo}: mean five-fold seed-ensemble macro-F1={results[-1]["mean_fold_macro_F1"]:.4f}', flush=True)
    ranked = rank_near_ties(results)
    return results, ranked


def aggregate_second_backbone(plan):
    output = []
    for check in plan['second_backbone_check']:
        key = f'{check["backbone"]}_{check["views"].replace("+", "_")}'
        fold_scores = []
        for fold in range(plan['folds']):
            subjects = single_seed_subjects(key, fold, check['seeds'][0])
            fold_scores.append(metrics(subjects))
        output.append({'backbone': check['backbone'], 'views': check['views'], 'seed': check['seeds'][0],
                       'mean_fold_macro_F1': float(np.mean([s['macro_F1'] for s in fold_scores])),
                       'fold_results': fold_scores, 'role': check['role']})
        print(f'C second backbone {key}: mean five-fold macro-F1={output[-1]["mean_fold_macro_F1"]:.4f}', flush=True)
    return output


def aggregate_resnet18(plan):
    base = plan['resnet18_TF_baseline']
    key = 'resnet18_imagenet_TF'
    fold_scores, oof = [], []
    for fold in range(plan['folds']):
        subjects = ensemble_subjects(key, fold, base['seeds'])
        score = metrics(subjects)
        fold_scores.append(score)
        for entry in subjects:
            oof.append({**entry, 'pipeline': 'logMel_ResNet18', 'fold': fold})
    write_jsonl(ROOT / 'predictions' / 'baseline_logMel_ResNet18_oof.jsonl', oof)
    print(f'C ResNet18 TF baseline: mean five-fold seed-ensemble macro-F1={float(np.mean([s["macro_F1"] for s in fold_scores])):.4f}', flush=True)
    return {'pipeline': 'logMel_ResNet18', 'mean_fold_macro_F1': float(np.mean([s['macro_F1'] for s in fold_scores])),
            'fold_results': fold_scores}


def aggregate_baselines(plan):
    rows = []
    for baseline in plan['baselines']:
        summary = json.loads((ROOT / 'runs' / f'baselines_{baseline["id"]}.json').read_text(encoding='utf-8'))
        rows.append({'pipeline': baseline['id'], 'mean_fold_macro_F1': summary['mean_fold_macro_F1'],
                     'fold_results': summary['fold_results'],
                     'selected_configuration': summary['selected_configuration']})
        print(f'C baseline {baseline["id"]}: mean five-fold three-seed macro-F1={summary["mean_fold_macro_F1"]:.4f}', flush=True)
    return rows


def main(plan_only=False):
    plan = freeze_c_plan()
    for name, fingerprint in plan['source_sha256'].items():
        if digest_file(ROOT / 'scripts' / name) != fingerprint:
            raise RuntimeError(f'C code changed since plan freeze: {name}')
    if plan_only:
        print(f'Frozen C plan: {len(plan["view_combinations"])} views x {plan["folds"]} folds x {len(plan["seeds"])} seeds '
              f'+ second-backbone + ResNet18 baseline + {len(plan["baselines"])} tuned baselines.', flush=True)
        return
    a = json.loads((ROOT / 'runs' / 'stage_A_state.json').read_text(encoding='utf-8'))
    b = json.loads((ROOT / 'runs' / 'stage_B_summary.json').read_text(encoding='utf-8'))
    if a['status'] != 'complete' or a['completed_runs'] != 18 or b['selected_backbone'] != plan['selected_backbone']:
        raise RuntimeError('A screening and B selection must finish before C')
    verification = json.loads((ROOT / 'locks' / 'pipeline_verification.json').read_text(encoding='utf-8'))
    for name in ['train.py', 'models.py', 'features.py']:
        if digest_file(ROOT / 'scripts' / name) != verification['source_sha256'][name]:
            raise RuntimeError(f'pipeline source changed since M1 verification: {name}')
    tasks = queue(plan)
    mutex = ROOT / 'runs' / 'stage_C.running'
    try:
        descriptor = os.open(mutex, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        previous = int(mutex.read_text().strip())
        try:
            os.kill(previous, 0)
        except OSError:
            mutex.unlink()
            descriptor = os.open(mutex, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f'C queue already active: process {previous}')
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(str(os.getpid()))
    state = {'stage': 'C', 'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
             'plan_sha256': digest_file(ROOT / 'locks' / 'stage_C_plan.json'),
             'deep_tasks': len(tasks), 'completed_tasks': 0, 'baselines': ['LTAS_GBM', 'MFCC_XGBoost', 'FE_GBM'],
             'test_evaluated': False}
    try:
        for task in tasks:
            if (ROOT / 'runs' / 'STOP').exists():
                state.update(status='stopped_between_runs')
                state_write(state)
                return
            run = ROOT / 'runs' / task['run_id']
            run.mkdir(exist_ok=True)
            write_json(run / 'task.json', task)
            state.update(active_run=task['run_id'])
            state_write(state)
            if not (run / 'result.json').exists():
                print(f'C: {task["run_id"]}', flush=True)
                execute(task)
            result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
            if result['test_evaluated']:
                raise RuntimeError('invalid worker result')
            state['completed_tasks'] += 1
            state_write(state)
        for baseline in state['baselines']:
            if (ROOT / 'runs' / 'STOP').exists():
                state.update(status='stopped_between_runs')
                state_write(state)
                return
            state.update(active_run=f'baselines_{baseline}')
            state_write(state)
            print(f'C baseline: {baseline}', flush=True)
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / 'train_baselines.py'), '--baseline', baseline],
                           cwd=ROOT, check=True)
        views, ranked = aggregate_views(plan)
        second = aggregate_second_backbone(plan)
        resnet = aggregate_resnet18(plan)
        baselines = aggregate_baselines(plan)
        candidates = [{'trial_id': f'resnet18_TF', 'pipeline': 'logMel_ResNet18',
                       'mean_fold_macro_F1': resnet['mean_fold_macro_F1']}] + baselines
        strongest_baseline = rank_near_ties([{'trial_id': c['pipeline'], 'mean_fold_macro_F1': c['mean_fold_macro_F1'],
                                              'total_parameters': 0, 'fusion': 'concat'} for c in candidates])[0]
        view_winner = ranked[0]
        tf = next(r for r in views if r['pipeline'] == 'TF')
        tf_fe = next(r for r in views if r['pipeline'] == 'TF+FE')
        summary = {'stage': 'C', 'plan_sha256': digest_file(ROOT / 'locks' / 'stage_C_plan.json'),
                   'seven_view_results': views, 'seven_view_ranking': [r['pipeline'] for r in ranked],
                   'selected_view_combination': view_winner['pipeline'],
                   'selected_view_combination_macro_F1': view_winner['mean_fold_macro_F1'],
                   'second_backbone_check': second, 'logMel_ResNet18_baseline': resnet,
                   'baselines': baselines,
                   'strongest_baseline': strongest_baseline['trial_id'],
                   'strongest_baseline_macro_F1': strongest_baseline['mean_fold_macro_F1'],
                   'C1_development': {'positive': 'TF+FE', 'reference': 'TF',
                                      'delta_macro_F1': tf_fe['mean_fold_macro_F1'] - tf['mean_fold_macro_F1'],
                                      'positive_macro_F1': tf_fe['mean_fold_macro_F1'], 'reference_macro_F1': tf['mean_fold_macro_F1']},
                   'C2_development': {'positive': view_winner['pipeline'], 'reference': strongest_baseline['trial_id'],
                                      'delta_macro_F1': view_winner['mean_fold_macro_F1'] - strongest_baseline['mean_fold_macro_F1']},
                   'selection_only': True, 'test_evaluated': False,
                   'next_stage': 'D locked full-development three-seed fits and model/comparison locks; not dispatched by C-only queue'}
        write_json(ROOT / 'runs' / 'stage_C_summary.json', summary)
        state.update(status='complete', active_run=None, finished_unix=time.time())
        state_write(state)
        print('Stage C complete; seven-view and baseline development results saved. No test evaluation.', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        state_write(state)
        raise
    finally:
        mutex.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    main(parser.parse_args().plan_only)
