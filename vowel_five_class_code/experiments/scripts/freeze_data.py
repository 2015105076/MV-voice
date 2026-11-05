"""Freeze directory-confirmed cohort and duplication-safe development/test splits."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from audit_waveforms import digest_file, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    if (ROOT/'locks'/'data_lock.json').exists():
        raise RuntimeError('data already locked; do not silently regenerate splits')
    original = ROOT/'protocol.json'
    protocol = json.loads(original.read_text(encoding='utf-8'))
    protocol['schema_version'] = '1.1'
    protocol['protocol_id'] = 'vowel_five_class_v1_1'
    protocol['status'] = 'execution_started_before_any_model_or_test_results'
    protocol['amendment'] = {
        'date':'2026-10-29', 'stage':'E0 before split and before any training or test results',
        'authority':'User confirms current directory data and diagnoses are established; no further clinical or demographic review requested; v represents ü.',
        'identity_basis':'Each original_label:folder is a user-confirmed recording bundle. Real-world independent identities are not inferred from audio.',
        'duplicate_resolution':'Retain all complete-six folders; connected components of cross-folder exact decoded duplicates OR screening correlation >=0.995 are indivisible leakage groups. Grouping is conservative leakage protection, not an assertion of same patient.',
        'stereo_resolution':'All 252 stereo recordings have sample-for-sample identical channels; average these identical channels. No heterogeneous stereo selection is needed.',
        'split_change':'Stratify unique leakage groups by class, then assign every member folder together. Use original split/CV seeds. Approximately 80/20 groups; folder ratio may differ slightly.',
        'inference_change':'Primary point metrics remain folder/recording-bundle level with equal six-vowel probabilities. Report bundle and leakage-group counts. Any later uncertainty estimation must resample entire leakage groups stratified by class, never claim the 957 folders are 957 verified independent people.',
        'near_duplicate_limitation':'Bounded signal screening cannot establish unique clinical identity or exhaustively detect every edited duplicate.'}
    protocol['data']['identity']['require_cross_folder_identity_review_before_split'] = False
    protocol['data']['identity']['clinical_labels'] = 'User-confirmed folder mapping; diagnostic source accepted by user instruction.'
    protocol['data']['identity']['independent_group'] = 'conservative duplicate connected component; not verified clinical identity'
    protocol['split']['unit'] = 'duplicate_leakage_group'
    protocol['split']['write_split_now'] = True
    protocol['statistics']['bootstrap_unit'] = 'leakage group stratified by true class; preserve all member bundles and their six vowels; compute bundle-level metrics'
    protocol['statistics']['paired_randomization'] = '10000 leakage-group-wise swaps of all member bundles and all class probabilities; two-sided absolute macro-F1 difference; add-one p-value'
    protocol['statistics']['independent_n'] = 'test leakage groups; also report recording-bundle count; clinical independence is not verified'
    protocol['locks']['data_lock_required_before_split'] = ['user-confirmed directory identity and class mapping','quality audit and missing-vowel exclusions','exact and near-duplicate leakage grouping','input content hashes','identical stereo confirmation']
    rows = read_jsonl(ROOT/'e0_audit'/'waveform_audit.jsonl')
    subjects = read_jsonl(ROOT/'inventory'/'subject_inventory.jsonl')
    parent = {s['candidate_group_id']:s['candidate_group_id'] for s in subjects}
    lookup = {r['relative_path']:r for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a,b):
        a,b = find(a),find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)

    for group in read_jsonl(ROOT/'e0_audit'/'exact_duplicate_groups.jsonl'):
        for path in group['relative_paths'][1:]:
            union(lookup[group['relative_paths'][0]]['candidate_group_id'],lookup[path]['candidate_group_id'])
    for pair in read_jsonl(ROOT/'e0_audit'/'near_duplicate_candidates.jsonl'):
        union(lookup[pair['left']]['candidate_group_id'],lookup[pair['right']]['candidate_group_id'])
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row['candidate_group_id']].append(row)
        if not row['decode_ok'] or not row.get('finite'):
            raise RuntimeError('decode/finite problem requires explicit resolution')
        if row['channels'] > 1 and not row.get('first_two_channels_equal'):
            raise RuntimeError('non-identical stereo requires resolution')
        if any(f!='channel_semantics_unverified' for f in row['review_flags']):
            raise RuntimeError('quality flag requires resolution')
    included, excluded = [], []
    for s in subjects:
        item = {**s, 'subject_id':s['candidate_group_id'], 'leakage_group_id':find(s['candidate_group_id']),
                'identity_basis':'directory confirmed by user; clinical independence not asserted'}
        if s['candidate_for_complete_six_analysis']:
            included.append(item)
        else:
            excluded.append({**item,'exclusion_reason':'incomplete six-vowel set'})
    groups = defaultdict(list)
    for s in included:
        groups[s['leakage_group_id']].append(s)
    group_ids = sorted(groups)
    for g in group_ids:
        if len({s['label'] for s in groups[g]}) != 1:
            raise RuntimeError('cross-class duplicate component requires explicit resolution')
    labels = [groups[g][0]['label'] for g in group_ids]
    dev, test = train_test_split(group_ids,test_size=.2,random_state=protocol['split']['split_seed'],stratify=labels)
    dev, test = sorted(dev),sorted(test)
    assignments = {g:'development' for g in dev} | {g:'test' for g in test}
    split_rows = [{**s,'split':assignments[s['leakage_group_id']]} for s in included]
    folds = []
    dev_labels = [groups[g][0]['label'] for g in dev]
    for count in [3,5]:
        cv = StratifiedKFold(count,shuffle=True,random_state=protocol['split']['cv_seed'])
        for fold, (_,val) in enumerate(cv.split(np.zeros(len(dev)),dev_labels)):
            for p in val:
                for s in groups[dev[p]]:
                    folds.append({'subject_id':s['subject_id'],'leakage_group_id':dev[p], 'fold_count':count,'validation_fold':fold})
    # Check every group and every exact/near pair, not merely matching directory IDs.
    assert not (set(dev)&set(test))
    assert len(folds)==len([s for s in split_rows if s['split']=='development'])*2
    for count in [3,5]:
        check = defaultdict(set)
        for f in folds:
            if f['fold_count']==count:
                check[f['leakage_group_id']].add(f['validation_fold'])
        assert all(len(v)==1 for v in check.values())
    for directory in ['locks','splits']:
        (ROOT/directory).mkdir(exist_ok=True)
    write_json(ROOT/'protocol.execution.v1.1.json',protocol)
    write_jsonl(ROOT/'inventory'/'included_subjects.jsonl',included)
    write_jsonl(ROOT/'inventory'/'excluded_subjects.jsonl',excluded)
    write_jsonl(ROOT/'splits'/'subject_split.jsonl',split_rows)
    write_jsonl(ROOT/'splits'/'development_folds.jsonl',folds)
    write_jsonl(ROOT/'inventory'/'leakage_groups.jsonl',[{'leakage_group_id':g,'label':groups[g][0]['label'],'member_subject_ids':[s['subject_id'] for s in groups[g]]} for g in group_ids])
    lock = {'protocol_sha256':digest_file(ROOT/'protocol.execution.v1.1.json'),'original_protocol_sha256':digest_file(original),
            'identity_and_label_basis':protocol['amendment']['authority'],
            'included_recording_bundles':len(included),'included_leakage_groups':len(group_ids),'excluded_incomplete_bundles':len(excluded),
            'clinical_identity_independently_verified':False,'channel_rule':'average identical channels; all stereo channels verified equal by exact sample comparison',
            'raw_content_hash_manifest':'e0_audit/waveform_audit.jsonl','duplicate_resolution':protocol['amendment']['duplicate_resolution'],
            'by_split':{split:{'bundles':sum(s['split']==split for s in split_rows), 'leakage_groups':len({s['leakage_group_id'] for s in split_rows if s['split']==split}),
                               'class_bundles':dict(Counter(s['label'] for s in split_rows if s['split']==split))} for split in ['development','test']},
            'files_sha256':{str(p.relative_to(ROOT)):digest_file(p) for p in [ROOT/'inventory'/'included_subjects.jsonl', ROOT/'inventory'/'excluded_subjects.jsonl',ROOT/'inventory'/'leakage_groups.jsonl',ROOT/'splits'/'subject_split.jsonl',ROOT/'splits'/'development_folds.jsonl',ROOT/'e0_audit'/'waveform_audit.jsonl',ROOT/'e0_audit'/'exact_duplicate_groups.jsonl',ROOT/'e0_audit'/'near_duplicate_candidates.jsonl']},
            'split_checks_passed':True,'test_predictions_generated':False}
    write_json(ROOT/'locks'/'data_lock.json',lock)
    print(json.dumps(lock,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
