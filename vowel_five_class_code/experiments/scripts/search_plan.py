"""Deterministic, bounded B search plan. No model/test predictions are read."""
import itertools
import json
import random

from audit_waveforms import digest_file,write_json
from freeze_data import ROOT


def rank_near_ties(candidates,tolerance=.002):
    pending=list(candidates)
    ranked=[]
    while pending:
        high=max(c['mean_fold_macro_F1'] for c in pending)
        ties=[c for c in pending if high-c['mean_fold_macro_F1']<=tolerance]
        chosen=min(ties,key=lambda c:(c['total_parameters'],0 if c['fusion']=='concat' else 1,c['trial_id']))
        ranked.append(chosen)
        pending.remove(chosen)
    return ranked


def make_plan():
    protocol_path=ROOT/'protocol.execution.v1.2.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    a_path=ROOT/'runs'/'stage_A_summary.json'
    screening=json.loads(a_path.read_text(encoding='utf-8'))
    spec=protocol['search']['stage_B']
    retained=screening['retained_backbones']
    assert len(retained)==2 and len(set(retained))==2
    pipes=[(b,v) for b in retained for v in ['TF','TF+FE']]
    if ('resnet18_imagenet','TF') not in pipes:
        pipes.append(('resnet18_imagenet','TF'))
    # Immutable frozen embeddings make these pipelines quickest; order affects time only.
    pipes.sort(key=lambda p:({'beats_frozen':0,'resnet18_imagenet':1,'beats_last2':2}[p[0]],p[1]))
    rng=random.Random(spec['trial_seed'])
    plans=[]
    for backbone,views in pipes:
        multipliers=[1.] if backbone=='beats_frozen' else spec['backbone_lr_multiplier_candidates']
        combinations=[dict(head_lr=h,backbone_multiplier=b,weight_decay=w,dropout=d)
                      for h,b,w,d in itertools.product(spec['head_lr_candidates'],multipliers,spec['weight_decay_candidates'],spec['dropout_candidates'])]
        if views=='TF+FE':
            choices=[]
            for fusion in ['concat','gated']:
                selected=rng.sample(combinations,4)
                choices.extend({**c,'fusion':fusion} for c in selected)
            rng.shuffle(choices)
        else:
            choices=[{**c,'fusion':'concat'} for c in rng.sample(combinations,8)]
        assert len({json.dumps(c,sort_keys=True) for c in choices})==8
        pipe_id=f'{backbone}_{views.replace("+","_")}'
        trials=[{'trial_id':f'B_{pipe_id}_t{i:02d}','backbone':backbone,'views':views,'parameters':parameters}
                for i,parameters in enumerate(choices)]
        plans.append({'pipeline_id':pipe_id,'backbone':backbone,'views':views,'trials':trials})
    return {'stage':'B','protocol_sha256':digest_file(protocol_path),'data_lock_sha256':digest_file(ROOT/'locks'/'data_lock.json'),
            'screening_summary_sha256':digest_file(a_path),'seed':spec['seed'],'trial_seed':spec['trial_seed'],
            'folds':spec['folds'],'resource_schedule_epochs':spec['resource_schedule_epochs'],'survivors':spec['survivors'],
            'scheduler_horizon_epochs':60,'early_stopping_patience':10,
            'budget_interpretation':'Continue identical model/optimizer/RNG state across rungs with one fixed 60-epoch LR schedule. Each rung sets an epoch cap. Prespecified patience-10 early stopping is retained; a stopped trial is not restarted. Report actual epochs and early-stop reason. Final selection only compares the two trials assigned the common terminal 60-epoch cap in each pipeline.',
            'frozen_multiplier_rule':'Frozen backbone has no optimizer parameters; canonical multiplier 1 avoids repeated effective configurations.',
            'tie_rule':spec['near_tie_rule'],
            'main_backbone_candidates':retained,
            'fusion_if_TF_wins':'Take the fusion of the terminal-rung TF+FE winner on the selected backbone. Shared training hyperparameters come from the selected main trial. Record both sources before C.',
            'test_used':False,'pipelines':plans}


def freeze_plan():
    plan=make_plan()
    path=ROOT/'locks'/'stage_B_plan.json'
    if path.exists():
        existing=json.loads(path.read_text(encoding='utf-8'))
        if existing!=plan:
            raise RuntimeError('frozen B plan changed; do not regenerate after observing results')
        return existing
    write_json(path,plan)
    return plan


if __name__=='__main__':
    plan=freeze_plan()
    print(json.dumps({'pipelines':len(plan['pipelines']),'initial_trials':sum(len(p['trials']) for p in plan['pipelines']),
                      'initial_fold_runs':sum(len(p['trials'])*plan['folds'] for p in plan['pipelines']),
                      'rung_limits':plan['resource_schedule_epochs'],'seed':plan['trial_seed']},indent=2))
