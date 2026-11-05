"""Verify frozen plans, gated gradients, and exact interrupted-training continuation."""
import json
import subprocess
import sys

import torch

from audit_waveforms import digest_file,write_json
from freeze_data import ROOT
from models import MultiViewModel
from search_plan import freeze_plan,make_plan,rank_near_ties


def main():
    torch.set_num_threads(12)
    previous=json.loads((ROOT/'locks'/'pipeline_verification.json').read_text(encoding='utf-8'))
    for name,fingerprint in previous['source_sha256'].items():
        if digest_file(ROOT/'scripts'/name)!=fingerprint:
            raise RuntimeError('original verified feature/model code changed')
    plan=freeze_plan()
    assert plan==make_plan()
    assert len(plan['pipelines'])==5
    for pipeline in plan['pipelines']:
        assert len(pipeline['trials'])==8
        if pipeline['views']=='TF+FE':
            assert sum(t['parameters']['fusion']=='gated' for t in pipeline['trials'])==4
        assert len({json.dumps(t['parameters'],sort_keys=True) for t in pipeline['trials']})==8
    test=[dict(trial_id='c',mean_fold_macro_F1=.801,total_parameters=100,fusion='gated'),
          dict(trial_id='b',mean_fold_macro_F1=.800,total_parameters=100,fusion='concat'),
          dict(trial_id='a',mean_fold_macro_F1=.7995,total_parameters=99,fusion='gated')]
    assert [c['trial_id'] for c in rank_near_ties(test)]==['a','b','c']
    for fusion in ['concat','gated']:
        model=MultiViewModel('beats_frozen','TF+FE',.2,fusion).cuda().train()
        inputs=[{'FE':torch.randn(128,device='cuda'),'frozen_tf_embedding':torch.randn(768,device='cuda')} for _ in range(4)]
        logits=model(inputs)
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,2,3],device='cuda')).backward()
        assert torch.isfinite(logits).all()
        assert model.fe[0].weight.grad is not None
        if fusion=='gated':
            assert model.score[0].weight.grad is not None and model.score[0].weight.grad.abs().sum()>0
        del model,inputs,logits
        torch.cuda.empty_cache()
    trial=next(t for p in plan['pipelines'] if p['backbone']=='beats_frozen' and p['views']=='TF+FE'
               for t in p['trials'] if t['parameters']['fusion']=='gated')
    tasks=[]
    for label in ['continuous','resumed']:
        run_id=f'B_verification_{label}'
        task={**trial,'run_id':run_id,'fold':0,'plan_sha256':digest_file(ROOT/'locks'/'stage_B_plan.json'),'verification_only':True}
        directory=ROOT/'runs'/run_id
        directory.mkdir(exist_ok=True)
        write_json(directory/'task.json',task)
        tasks.append(directory/'task.json')
    for task,budget in [(tasks[0],2),(tasks[1],1),(tasks[1],2)]:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/'train_search.py'),'--task',str(task),'--budget',str(budget)],cwd=ROOT,check=True)
    first=torch.load(tasks[0].parent/'last.pt',map_location='cpu',weights_only=False)
    second=torch.load(tasks[1].parent/'last.pt',map_location='cpu',weights_only=False)
    assert first['epoch']==second['epoch']==1
    for name,value in first['model'].items():
        assert torch.equal(value,second['model'][name]),f'resume weight mismatch: {name}'
    assert first['best_score']==second['best_score']
    assert torch.equal(first['torch_state'],second['torch_state'])
    assert all(torch.equal(a,b) for a,b in zip(first['cuda_states'],second['cuda_states']))
    assert first['optimizer']['param_groups']==second['optimizer']['param_groups']
    for key,value in first['optimizer']['state'].items():
        for name,state in value.items():
            other=second['optimizer']['state'][key][name]
            assert torch.equal(state,other) if isinstance(state,torch.Tensor) else state==other
    checks={'deterministic_40_trial_plan':True,'five_pipelines_eight_unique_trials_each':True,
            'fusion_balanced_4_plus_4':True,'near_tie_ranking':True,'concat_and_gated_GPU_gradients':True,
            'uninterrupted_vs_resumed_two_epochs_exact_model_optimizer_and_RNG_match':True,
            'verification_used_development_fold_only':True,'verification_runs_excluded_from_search':True,'test_evaluated':False}
    names=['train_search.py','train.py','models.py','features.py','search_plan.py','verify_search.py','run_stage_b.py']
    write_json(ROOT/'locks'/'stage_B_verification.json',{'checks':checks,'plan_sha256':digest_file(ROOT/'locks'/'stage_B_plan.json'),
               'source_sha256':{n:digest_file(ROOT/'scripts'/n) for n in names},'GPU':torch.cuda.get_device_name()})
    print('B search verification passed, including exact checkpoint continuation',flush=True)


if __name__=='__main__':
    main()
