"""Required M1 development-only numerical, padding and GPU backward checks."""
import gc
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_waveforms import digest_file,write_json
from features import analyzed_waveform,settings,views
from freeze_data import ROOT,read_jsonl
from models import MaskedGroupNorm,MultiViewModel,TemporalEncoder,beats_from_logmel,load_beats
from train import aggregate,fit_statistics,load_data,normalize


def main():
    torch.set_num_threads(12)
    torch.manual_seed(3407)
    protocol=json.loads((ROOT/'protocol.execution.v1.2.json').read_text(encoding='utf-8'))
    records=read_jsonl(ROOT/'features'/'development_manifest.jsonl')
    selected=sorted(records,key=lambda r:r['valid_frames'])
    examples=[selected[0],selected[len(selected)//3],selected[2*len(selected)//3],selected[-1]]
    data=load_data(examples)
    stats=fit_statistics(data,list(range(len(data))))
    checks={}
    L=torch.tensor([[1.,2.,3.],[2.,4.,6.],[3.,2.,1.]])
    fe,te=views(L)
    assert torch.allclose(fe,L.double().exp().mean(0).log().float())
    assert torch.allclose(te,L.double().exp().mean(1).log().float())
    checks['linear_power_before_log']=True
    # Explicit valid-frame slicing makes the view statistics invariant to padding.
    for padding in [1,31,211]:
        padded=F.pad(L,(0,0,0,padding),value=999.)
        padded_fe,padded_te=views(padded[:len(L)])
        assert torch.equal(padded_fe,fe) and torch.equal(padded_te,te)
    checks['view_padding_invariance']=True
    normalizer=MaskedGroupNorm(32)
    x=torch.randn(2,32,57)
    assert torch.allclose(normalizer(x,torch.ones(2,57,dtype=torch.bool)),F.group_norm(x,8,normalizer.weight,normalizer.bias),atol=1e-6)
    checks['masked_groupnorm_matches_unpadded_groupnorm']=True
    te_model=TemporalEncoder(.3).eval()
    sequence=torch.randn(1,57)
    expected=te_model(sequence,torch.ones_like(sequence,dtype=torch.bool))
    for padding in [1,19,101]:
        padded=F.pad(sequence,(0,padding),value=999.)
        mask=torch.arange(padded.shape[-1])[None]<57
        actual=te_model(padded,mask)
        assert torch.allclose(expected,actual,atol=2e-6,rtol=1e-5)
    checks['TE_encoder_padding_invariance']=True
    probabilities=np.array([[.6,.1,.1,.1,.1],[.1,.6,.1,.1,.1],[.1,.1,.6,.1,.1],
                            [.1,.1,.1,.6,.1],[.1,.1,.1,.1,.6],[.2,.2,.2,.2,.2]])
    aggregation_records=[{'subject_id':'synthetic','leakage_group_id':'synthetic','label':'NV','vowel':v} for v in ['a','e','i','o','u','v']]
    actual=np.array(aggregate(aggregation_records,probabilities)[0]['probabilities'])
    assert np.allclose(actual,np.array([.2]*5))
    seed_ensemble=np.mean([probabilities,probabilities[:,::-1],probabilities],axis=0).mean(axis=0)
    assert np.allclose(seed_ensemble,.2)
    assert int(actual.argmax())==0
    checks['six_vowel_and_three_seed_probability_aggregation']=True
    # Check official preprocessing AND the entire shared-spectrum bridge in float32.
    waveform_records={r['relative_path']:r for r in read_jsonl(ROOT/'e0_audit'/'waveform_audit.jsonl')}
    from torchaudio.compliance import kaldi
    beats=load_beats().cuda().eval()
    source=analyzed_waveform(waveform_records[examples[1]['relative_path']],protocol).cuda()[None]
    with torch.no_grad():
        shared=kaldi.fbank(source*32768,**settings(protocol))
        official=beats.preprocess(source)
        normalized=(shared[None]-15.41663)/(2*6.55582)
        torch.testing.assert_close(normalized,official,atol=1e-6,rtol=1e-6)
        official_tokens,official_mask=beats.extract_features(source,padding_mask=torch.zeros_like(source,dtype=torch.bool))
        tokens,mask=beats_from_logmel(beats,shared[None])
        torch.testing.assert_close(tokens,official_tokens,atol=1e-5,rtol=1e-5)
        assert torch.equal(mask,official_mask)
    checks['official_BEATs_preprocessing_max_abs_difference']=float((normalized-official).abs().max())
    checks['official_BEATs_bridge_max_abs_difference']=float((tokens-official_tokens).abs().max())
    del beats,source,tokens,official_tokens
    gc.collect();torch.cuda.empty_cache()
    runs=[]
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    combinations=protocol['models']['view_combinations']
    for backbone,combos in [('resnet18_imagenet',combinations),('beats_frozen',['TF','TF+FE']),('beats_last2',['TF','TF+FE'])]:
        for combo in combos:
            started=time.perf_counter()
            model=MultiViewModel(backbone,combo).cuda().train()
            torch.cuda.reset_peak_memory_stats()
            inputs=[normalize(item,stats,backbone) for item in data]
            with torch.autocast('cuda',dtype=dtype):
                logits=model(inputs)
                loss=F.cross_entropy(logits.float(),torch.tensor([0,1,2,3],device='cuda'))
            loss.backward()
            assert logits.shape==(4,5) and torch.isfinite(logits).all()
            gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().sum()>0 for g in gradients)
            if backbone=='beats_frozen' and 'TF' in combo:
                assert not model.backbone.training and all(p.grad is None for p in model.backbone.parameters())
            if backbone=='beats_last2':
                assert not model.backbone.encoder.layers[0].training
                assert model.backbone.encoder.layers[-1].training
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.encoder.layers[-1].parameters())
            torch.cuda.synchronize()
            entry={'backbone':backbone,'views':combo,'loss':float(loss.detach()),'seconds':time.perf_counter()-started,
                   'peak_memory_MB':torch.cuda.max_memory_allocated()/1024**2,
                   'total_parameters':sum(p.numel() for p in model.parameters()),
                   'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
            runs.append(entry)
            print(json.dumps(entry),flush=True)
            del model,inputs,logits,loss,gradients
            gc.collect();torch.cuda.empty_cache()
    checks['all_branch_and_backbone_GPU_forward_backward']=True
    write_json(ROOT/'locks'/'pipeline_verification.json',{'checks':checks,'GPU':torch.cuda.get_device_name(),
               'precision':str(dtype),'development_only':True,'test_evaluated':False,'smoke_runs':runs,
               'source_sha256':{name:digest_file(ROOT/'scripts'/name) for name in ['features.py','models.py','train.py','verify_pipeline.py']}})
    print('All M1 checks passed',flush=True)


if __name__=='__main__':
    main()
