"""Shared native-scale features. Only development audio is cached by this command."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import fft, signal
from torchaudio.compliance import kaldi

from audit_waveforms import digest_file, write_json, write_jsonl
from freeze_data import ROOT, read_jsonl


def settings(protocol):
    p = protocol['preprocessing']['shared_fbank']
    return dict(blackman_coeff=.42, channel=-1, dither=p['dither'], energy_floor=1.,
                frame_length=p['frame_length_ms'],frame_shift=p['frame_shift_ms'],high_freq=p['high_freq_kaldi'],
                htk_compat=False,low_freq=p['low_freq_hz'],min_duration=0.,num_mel_bins=p['num_mel_bins'],
                preemphasis_coefficient=p['preemphasis_coefficient'],raw_energy=True,remove_dc_offset=p['remove_dc_offset'],
                round_to_power_of_two=p['round_to_power_of_two'],sample_frequency=p['sample_frequency'],
                snip_edges=p['snip_edges'],subtract_mean=False,use_energy=p['use_energy'],use_log_fbank=p['use_log_fbank'],
                use_power=p['use_power'],vtln_high=-500.,vtln_low=100.,vtln_warp=1.,window_type=p['window_type'])


def feature_version(protocol):
    config = {'preprocessing':protocol['preprocessing'],'channel':'mean of exactly identical channels',
              'resampling':'scipy.signal.resample_poly, window=(kaiser,5.0), padtype=constant',
              'code_sha256':digest_file(Path(__file__)), 'torch':torch.__version__,'torchaudio':torchaudio.__version__,
              'fbank_arguments':settings(protocol),'fbank_power_floor':float(torch.finfo(torch.float32).eps)}
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),config


def analyzed_waveform(record,protocol):
    path = Path(protocol['data']['root'])/record['relative_path']
    if digest_file(path) != record['file_sha256']:
        raise RuntimeError(f'raw content changed after lock: {path}')
    x,sr = sf.read(path,dtype='float64',always_2d=True)
    if x.shape[1]>1 and not np.all(x==x[:,0:1]):
        raise RuntimeError('channel mismatch against data lock')
    x = x[record['suggested_trim_start_frame']:record['suggested_trim_end_frame']].mean(axis=1)
    target = protocol['preprocessing']['sample_rate_hz']
    divisor = math.gcd(sr,target)
    x = signal.resample_poly(x,target//divisor,sr//divisor,window=('kaiser',5.),padtype='constant')
    maximum = int(target*protocol['preprocessing']['max_analysis_seconds'])
    start = max(0,(len(x)-maximum)//2)
    return torch.from_numpy(x[start:start+maximum].astype(np.float32))


def views(L):
    double = L.double()
    FE = torch.logsumexp(double,dim=0)-math.log(len(L))
    TE = torch.logsumexp(double,dim=1)-math.log(L.shape[1])
    return FE.float(),TE.float()


def extract(record,protocol):
    wave = analyzed_waveform(record,protocol)
    L = kaldi.fbank(wave[None]*protocol['preprocessing']['shared_fbank']['amplitude_multiplier'],**settings(protocol))
    FE,TE = views(L)
    mfcc = torch.from_numpy(fft.dct(L.numpy().astype(np.float64),type=2,axis=1,norm='ortho')[:,:20].astype(np.float32)).T
    delta = torchaudio.functional.compute_deltas(mfcc,win_length=5,mode='replicate')
    delta2 = torchaudio.functional.compute_deltas(delta,win_length=5,mode='replicate')
    MFCC = torch.cat([t for arr in [mfcc,delta,delta2] for t in [arr.mean(1),arr.std(1,unbiased=False)]]).numpy()
    hz,psd = signal.welch(wave.numpy(),fs=16000,window='hann',nperseg=400,noverlap=240,nfft=512,detrend='constant',scaling='density',average='mean')
    valid = (hz>=20)&(hz<=8000)
    LTAS = np.log10(np.maximum(psd[valid],np.finfo(np.float32).tiny))
    return {'TF':L.numpy(),'FE':FE.numpy(),'TE':TE.numpy(),'MFCC':MFCC,'LTAS':LTAS.astype(np.float32),
            'waveform_samples':np.array(len(wave)),'valid_frames':np.array(len(L))}


def main():
    torch.set_num_threads(12)
    protocol = json.loads((ROOT/'protocol.execution.v1.2.json').read_text(encoding='utf-8'))
    lock = json.loads((ROOT/'locks'/'data_lock.json').read_text(encoding='utf-8'))
    for relative,fingerprint in lock['files_sha256'].items():
        if digest_file(ROOT/relative.replace('\\','/'))!=fingerprint:
            raise RuntimeError(f'data lock mismatch: {relative}')
    dev = {s['subject_id']:s for s in read_jsonl(ROOT/'splits'/'subject_split.jsonl') if s['split']=='development'}
    records = [r for r in read_jsonl(ROOT/'e0_audit'/'waveform_audit.jsonl') if r['candidate_group_id'] in dev]
    version,config = feature_version(protocol)
    cache = ROOT/'features'/version
    cache.mkdir(parents=True,exist_ok=True)
    manifest = []
    for i,record in enumerate(records):
        key = hashlib.sha256((record['file_sha256']+version).encode()).hexdigest()
        path = cache/(key+'.npz')
        if not path.exists():
            data = extract(record,protocol)
            if not all(np.isfinite(v).all() for v in data.values()):
                raise RuntimeError('nonfinite features')
            np.savez(path,**data)
        with np.load(path,allow_pickle=False) as data:
            frames = int(data['valid_frames'])
        manifest.append({'subject_id':record['candidate_group_id'],'leakage_group_id':dev[record['candidate_group_id']]['leakage_group_id'],
                         'label':record['label'],'vowel':record['vowel_file_token'],'relative_path':record['relative_path'],
                         'file_sha256':record['file_sha256'],'feature_path':str(path.relative_to(ROOT)),'feature_sha256':digest_file(path),
                         'feature_version':version,'valid_frames':frames,'split':'development'})
        if (i+1)%500==0:
            print(f'Cached development features {i+1}/{len(records)}',flush=True)
    write_jsonl(ROOT/'features'/'development_manifest.jsonl',manifest)
    mel,center = kaldi.get_mel_banks(128,512,16000,20,0,100,-500,1)
    # get_mel_banks returns center frequencies in Hz in the pinned torchaudio implementation.
    write_json(ROOT/'features'/'feature_metadata.json',{'feature_version':version,'config':config,
               'mel_center_frequencies_hz':center.tolist(),'LTAS_frequencies_hz':np.fft.rfftfreq(512,1/16000)[1:].tolist(),
               'development_recordings':len(manifest),'test_feature_extraction_performed':False,
               'feature_manifest_sha256':digest_file(ROOT/'features'/'development_manifest.jsonl'),
               'normalization':'raw features only; fitted fold statistics are separate and training-only'})
    print(f'Development feature extraction complete: {len(manifest)} recordings',flush=True)


if __name__=='__main__':
    main()
