"""E0 read-only waveform audit. Review findings never automatically exclude subjects.

Near-duplicate screening is candidate retrieval plus waveform correlation, not an
identity test. No mono/stereo decision, data lock, split, or model is produced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy import signal

from audit_data import quantiles


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def write_csv(path, fields, rows):
    # Review decisions are user-owned; never replace a completed or edited table.
    if path.exists():
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def zero_bounds(x, sr, seconds=.05):
    active = np.flatnonzero(np.any(x != 0, axis=1))
    if not len(active):
        return len(x), 0, 0, 0
    first, last = int(active[0]), int(active[-1]) + 1
    leading, trailing = first, len(x) - last
    start = first if leading >= math.ceil(sr * seconds) else 0
    end = last if trailing >= math.ceil(sr * seconds) else len(x)
    return leading, trailing, start, end


def waveform_hash(x, sr):
    # Native decoded PCM, exact float64, channel order and sample rate preserved.
    canonical = np.ascontiguousarray(x, dtype='<f8').copy()
    canonical[canonical == 0] = 0  # canonicalize signed zero
    h = hashlib.sha256(json.dumps([int(sr), list(x.shape)]).encode('ascii'))
    h.update(canonical.tobytes())
    return h.hexdigest()


def aligned_similarity(a, b, sr=4000):
    """Maximum normalized correlation, +/-250 ms, at least 90% overlap."""
    if min(len(a), len(b)) < .9 * max(len(a), len(b)):
        return None
    a = a.astype(np.float64) - np.mean(a)
    b = b.astype(np.float64) - np.mean(b)
    corr = signal.correlate(a, b, mode='full', method='fft')
    lag = signal.correlation_lags(len(a), len(b))
    starts_a = np.maximum(lag, 0)
    starts_b = np.maximum(-lag, 0)
    overlap = np.minimum(len(a) - starts_a, len(b) - starts_b)
    eligible = (np.abs(lag) <= sr // 4) & (overlap >= .9 * max(len(a), len(b)))
    positions = np.flatnonzero(eligible)
    if not len(positions):
        return None
    sa, sb, n = starts_a[positions], starts_b[positions], overlap[positions]
    aa = np.r_[0., np.cumsum(a * a)]
    bb = np.r_[0., np.cumsum(b * b)]
    denom = np.sqrt((aa[sa+n] - aa[sa]) * (bb[sb+n] - bb[sb]))
    values = np.divide(np.abs(corr[positions]), denom, out=np.zeros_like(denom), where=denom > 1e-20)
    best = int(np.argmax(values))
    return {'absolute_correlation': float(min(values[best], 1.)),
            'lag_seconds': float(lag[positions[best]] / sr),
            'overlap_seconds': float(n[best] / sr),
            'overlap_fraction': float(n[best] / max(len(a), len(b)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        parser.error('output must be outside raw data')
    if (out / 'waveform_summary.json').exists():
        parser.error('completed output exists; choose a fresh output directory')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    quality = protocol['data']['quality']
    records = [json.loads(line) for line in (args.inventory / 'audio_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    subjects = [json.loads(line) for line in (args.inventory / 'subject_inventory.jsonl').read_text(encoding='utf-8').splitlines()]
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.wav'}
    if actual != {r['relative_path'] for r in records}:
        parser.error('source inventory changed; rerun header audit first')
    out.mkdir(parents=True, exist_ok=True)
    rows, waves, features = [], {}, {}
    raw_groups, decoded_groups = defaultdict(list), defaultdict(list)
    for index, source in enumerate(records):
        path = root / source['relative_path']
        row = dict(source)
        row['audit_id'] = f'Q{index+1:05d}'
        row['review_flags'] = []
        row['decode_ok'] = False
        before = path.stat()
        row['file_sha256'] = digest_file(path)
        raw_groups[row['file_sha256']].append(index)
        try:
            x, sr = sf.read(path, dtype='float64', always_2d=True)
            row.update(decode_ok=True, decoded_sample_rate_hz=sr, decoded_frames=len(x), decoded_channels=x.shape[1], finite=bool(np.isfinite(x).all()))
            if not row['finite']:
                row['review_flags'].append('nonfinite_samples')
            else:
                row['decoded_sha256'] = waveform_hash(x, sr)
                decoded_groups[row['decoded_sha256']].append(index)
                leading, trailing, start, end = zero_bounds(x, sr)
                valid = x[start:end]
                row.update(leading_zero_seconds=leading/sr, trailing_zero_seconds=trailing/sr,
                           suggested_trim_start_frame=start, suggested_trim_end_frame=end,
                           post_zero_trim_seconds=len(valid)/sr)
                if len(valid)/sr < quality['minimum_valid_audio_seconds']:
                    row['review_flags'].append('less_than_minimum_after_digital_zero_trim')
                row['channel_metrics'] = []
                for channel in range(x.shape[1]):
                    channel_x = x[:, channel]
                    _, _, cs, ce = zero_bounds(x[:, channel:channel+1], sr)
                    metrics = {'channel_index_0based': channel,
                               'rms': float(np.sqrt(np.mean(channel_x**2))),
                               'peak_abs': float(np.max(np.abs(channel_x))),
                               'digital_zero_fraction': float(np.mean(channel_x == 0)),
                               'near_full_scale_fraction': float(np.mean(np.abs(channel_x) >= quality['near_full_scale_threshold'])),
                               'post_zero_trim_seconds': (ce-cs)/sr}
                    row['channel_metrics'].append(metrics)
                if any(c['near_full_scale_fraction'] > quality['near_full_scale_fraction_review_threshold'] for c in row['channel_metrics']):
                    row['review_flags'].append('near_full_scale_review')
                if x.shape[1] > 1:
                    row['review_flags'].append('channel_semantics_unverified')
                    a, b = x[:, 0], x[:, 1]
                    row['first_two_channels_equal'] = bool(np.array_equal(a, b))
                    denom = float(np.linalg.norm(a)*np.linalg.norm(b))
                    row['first_two_channel_cosine'] = float(np.clip(np.dot(a, b)/denom, -1, 1)) if denom else None
                    if denom and row['first_two_channel_cosine'] < -.95:
                        row['review_flags'].append('possible_antiphase')
                if (sr != source['sample_rate_hz'] or len(x) != source['frames'] or x.shape[1] != source['channels']):
                    row['review_flags'].append('decoded_header_mismatch')
                # Highest-RMS channel is ONLY a retrieval aid; no primary audio is made.
                channel = int(np.argmax([c['rms'] for c in row['channel_metrics']]))
                row['duplicate_screen_channel_0based'] = channel
                if len(valid) >= sr * .5 and row['channel_metrics'][channel]['rms'] > 0:
                    gcd = math.gcd(sr, 4000)
                    w = signal.resample_poly(valid[:, channel], 4000//gcd, sr//gcd).astype(np.float32)
                    w -= w.mean()
                    w /= max(float(np.linalg.norm(w)), 1e-20)
                    waves[index] = w
                    _, psd = signal.welch(w, fs=4000, nperseg=512, noverlap=256)
                    f = np.log(np.maximum(psd[2:130], 1e-20))
                    f -= f.mean()
                    f /= max(float(np.linalg.norm(f)), 1e-20)
                    features[index] = f.astype(np.float32)
        except (RuntimeError, ValueError, OSError) as exc:
            row['decode_ok'] = False
            row['decode_error'] = str(exc)
            row['review_flags'].append('decode_failure')
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError(f'source changed during read: {path}')
        rows.append(row)
        if (index+1) % 500 == 0:
            print(f'Decoded and hashed {index+1}/{len(records)}', flush=True)
    write_jsonl(out / 'waveform_audit.jsonl', rows)
    exact = []
    for kind, groups in [('file_sha256', raw_groups), ('decoded_sha256', decoded_groups)]:
        for fingerprint, indices in groups.items():
            if len(indices) > 1:
                exact.append({'kind': kind, 'sha256': fingerprint,
                              'relative_paths': [rows[i]['relative_path'] for i in indices],
                              'candidate_groups': sorted({rows[i]['candidate_group_id'] for i in indices})})
    write_jsonl(out / 'exact_duplicate_groups.jsonl', exact)
    # Retrieve top 10 spectra globally and top 10 within vowel, then verify waveform.
    candidates = set()
    keys = sorted(features)
    matrix = np.stack([features[i] for i in keys]) if keys else np.empty((0, 128))
    for offset in range(0, len(keys), 128):
        scores = matrix[offset:offset+128] @ matrix.T
        for local, index in enumerate(keys[offset:offset+128]):
            eligible = [p for p, other in enumerate(keys)
                        if rows[other]['candidate_group_id'] != rows[index]['candidate_group_id']
                        and min(len(waves[index]),len(waves[other])) >= .9*max(len(waves[index]),len(waves[other]))]
            same = [p for p in eligible if rows[keys[p]]['vowel_file_token'] == rows[index]['vowel_file_token']]
            for pool in (eligible, same):
                if pool:
                    best = sorted(pool, key=lambda p: (-float(scores[local,p]), keys[p]))[:10]
                    candidates.update(tuple(sorted((index,keys[p]))) for p in best)
    print(f'Near-duplicate waveform verification: {len(candidates)} candidate pairs', flush=True)
    near = []
    for n, (a,b) in enumerate(sorted(candidates)):
        result = aligned_similarity(waves[a], waves[b])
        if result and result['absolute_correlation'] >= .995:
            near.append({'left': rows[a]['relative_path'], 'right': rows[b]['relative_path'],
                         'left_audit_id': rows[a]['audit_id'], 'right_audit_id': rows[b]['audit_id'],
                         'same_original_label': rows[a]['original_label'] == rows[b]['original_label'],
                         'review_status': 'pending; correlation is not identity proof', **result})
        if (n+1) % 10000 == 0:
            print(f'Verified {n+1}/{len(candidates)} candidate pairs', flush=True)
    write_jsonl(out / 'near_duplicate_candidates.jsonl', near)
    write_csv(out / 'identity_label_review.csv',
              ['candidate_group_id','original_label','proposed_class','verified_subject_id','index_visit_id','diagnostic_source_reference','verified_class','identity_review_status','diagnosis_review_status','reviewer','notes'],
              [dict(candidate_group_id=s['candidate_group_id'], original_label=s['original_label'], proposed_class=s['label'],
                    verified_subject_id='', index_visit_id='', diagnostic_source_reference='', verified_class='',
                    identity_review_status='pending', diagnosis_review_status='pending', reviewer='', notes='') for s in subjects])
    write_csv(out / 'channel_review.csv', ['audit_id','relative_path','channel_metrics_json','selected_rule','source_reference','review_status','reviewer','notes'],
              [dict(audit_id=r['audit_id'],relative_path=r['relative_path'],channel_metrics_json=json.dumps(r.get('channel_metrics',[])),
                    selected_rule='',source_reference='',review_status='pending',reviewer='',notes='') for r in rows if r.get('decoded_channels',0)>1])
    write_csv(out / 'blinded_quality_review.csv', ['audit_id','flags','post_zero_trim_seconds','decision','reason','reviewer'],
              [dict(audit_id=r['audit_id'],flags=';'.join(f for f in r['review_flags'] if f!='channel_semantics_unverified'),
                    post_zero_trim_seconds=r.get('post_zero_trim_seconds',''),decision='pending',reason='',reviewer='')
               for r in rows if any(f!='channel_semantics_unverified' for f in r['review_flags'])])
    summary = {'created_utc':datetime.now(timezone.utc).isoformat(), 'stage':'E0 automated audit; human verification pending',
               'protocol_sha256':digest_file(args.protocol), 'source_manifest_sha256':digest_file(args.inventory/'audio_manifest.jsonl'),
               'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'soundfile':sf.__version__},
               'wav_files':len(rows),'decode_successes':sum(r['decode_ok'] for r in rows),
               'nonfinite_files':sum(r.get('finite') is False for r in rows),
               'flags':dict(Counter(f for r in rows for f in r['review_flags'])),
               'post_zero_trim_seconds':quantiles([r['post_zero_trim_seconds'] for r in rows if 'post_zero_trim_seconds' in r]),
               'raw_duplicate_groups':sum(g['kind']=='file_sha256' for g in exact),
               'decoded_duplicate_groups':sum(g['kind']=='decoded_sha256' for g in exact),
               'near_duplicate_pairs':len(near),'near_duplicate_candidate_pairs_checked':len(candidates),
               'near_duplicate_method':{'retrieval':'top 10 centered log-Welch PSD cosine globally and within vowel, cross-folder, duration ratio >=0.90',
                                        'waveform':'4000 Hz band-limited resampling, highest-RMS channel for screening only, zero-mean absolute normalized cross-correlation >=0.995, lag <=250 ms, overlap >=90% of longer waveform',
                                        'limitation':'bounded candidate retrieval is not exhaustive; gain, time shift and polarity handled; identity, speech validity and clinical labels require source review'},
               'blocking_requirements':['verified cross-folder subject identity and prespecified index visit','diagnostic source and five-class mapping confirmation','stereo channel semantics','quality and duplicate adjudication','v token display confirmation'],
               'data_lock_created':False,'split_assigned':False,'training_started':False,
               'output_sha256':{name:digest_file(out/name) for name in ['waveform_audit.jsonl','exact_duplicate_groups.jsonl','near_duplicate_candidates.jsonl']}}
    write_json(out / 'waveform_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
