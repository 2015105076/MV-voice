"""B worker: persist optimizer/RNG state across successive resource caps."""
import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

from audit_waveforms import digest_file,write_json,write_jsonl
from freeze_data import ROOT,read_jsonl
from models import MultiViewModel
from train import aggregate,fit_statistics,load_data,metrics,normalize,precision_context


def save_atomic(payload,path):
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def run_task(task_path,budget):
    task=json.loads(Path(task_path).read_text(encoding='utf-8'))
    plan_path=ROOT/'locks'/'stage_B_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if task['plan_sha256']!=digest_file(plan_path):
        raise RuntimeError('task does not match frozen B plan')
    matches=[t for p in plan['pipelines'] for t in p['trials'] if t['trial_id']==task['trial_id']]
    if len(matches)!=1 or any(task[key]!=matches[0][key] for key in ['backbone','views','parameters']):
        raise RuntimeError('task parameters outside frozen search plan')
    if not 1<=budget<=plan['scheduler_horizon_epochs']:
        raise RuntimeError('invalid epoch budget')
    if not task.get('verification_only') and budget not in plan['resource_schedule_epochs']:
        raise RuntimeError('unplanned production rung')
    if not 0<=task['fold']<plan['folds']:
        raise RuntimeError('invalid development fold')
    protocol_path=ROOT/'protocol.execution.v1.2.json'
    lock_path=ROOT/'locks'/'data_lock.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    lock=json.loads(lock_path.read_text(encoding='utf-8'))
    if digest_file(protocol_path)!=plan['protocol_sha256'] or digest_file(lock_path)!=plan['data_lock_sha256']:
        raise RuntimeError('protocol or data lock changed')
    for path,fingerprint in lock['files_sha256'].items():
        if digest_file(ROOT/path.replace('\\','/'))!=fingerprint:
            raise RuntimeError(f'locked artifact changed: {path}')
    torch.set_num_threads(12)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    seed=plan['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    records=read_jsonl(ROOT/'features'/'development_manifest.jsonl')
    splits={s['subject_id']:s for s in read_jsonl(ROOT/'splits'/'subject_split.jsonl')}
    if not all(r['split']=='development' and splits[r['subject_id']]['split']=='development' for r in records):
        raise RuntimeError('search may only consume development records')
    foldmap={r['subject_id']:r['validation_fold'] for r in read_jsonl(ROOT/'splits'/'development_folds.jsonl') if r['fold_count']==plan['folds']}
    training=[i for i,r in enumerate(records) if foldmap[r['subject_id']]!=task['fold']]
    validation=[i for i,r in enumerate(records) if foldmap[r['subject_id']]==task['fold']]
    if {records[i]['leakage_group_id'] for i in training}&{records[i]['leakage_group_id'] for i in validation}:
        raise RuntimeError('group leakage in search fold')
    run=ROOT/'runs'/task['run_id']
    run.mkdir(parents=True,exist_ok=True)
    names=['train_search.py','train.py','models.py','features.py','search_plan.py']
    config={'task':task,'plan_sha256':digest_file(plan_path),'protocol_sha256':digest_file(protocol_path),
            'data_lock_sha256':digest_file(lock_path),'split_sha256':digest_file(ROOT/'splits'/'subject_split.jsonl'),
            'folds_sha256':digest_file(ROOT/'splits'/'development_folds.jsonl'),
            'feature_manifest_sha256':digest_file(ROOT/'features'/'development_manifest.jsonl'),
            'environment_lock_sha256':digest_file(ROOT/'locks'/'environment.json'),
            'source_sha256':{n:digest_file(ROOT/'scripts'/n) for n in names},
            'maximum_epochs':plan['scheduler_horizon_epochs'],'seed':seed,'effective_vowel_batch_size':96,'initial_microbatch':24,
            'patience':plan['early_stopping_patience'],'scheduler_horizon_fixed_across_rungs':True,
            'train_bundles':len({records[i]['subject_id'] for i in training}),
            'validation_bundles':len({records[i]['subject_id'] for i in validation}),
            'checkpoint_source_sha256':digest_file(ROOT/'pretrained'/('resnet18-f37072fd.pth' if task['backbone']=='resnet18_imagenet' else 'BEATs_iter3.pt'))}
    if (run/'config.json').exists():
        if json.loads((run/'config.json').read_text(encoding='utf-8'))!=config:
            raise RuntimeError('resume task, code, or data fingerprint mismatch')
    else:
        write_json(run/'config.json',config)
        (run/'source').mkdir(exist_ok=True)
        for name in names:
            (run/'source'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    result_path=run/f'rung_{budget}.json'
    if result_path.exists():
        return json.loads(result_path.read_text(encoding='utf-8'))
    params=task['parameters']
    data=load_data(records)
    stats=fit_statistics(data,training)
    if not (run/'normalization.pt').exists():
        torch.save(stats,run/'normalization.pt')
    model=MultiViewModel(task['backbone'],task['views'],params['dropout'],params['fusion']).cuda()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    if task['backbone']=='beats_frozen' and 'TF' in task['views']:
        cache=ROOT/'features'/f'frozen_beats_{records[0]["feature_version"]}_{config["checkpoint_source_sha256"][:12]}.pt'
        payload=torch.load(cache,map_location='cpu',weights_only=True)
        assert payload['paths']==[r['relative_path'] for r in records]
        for item,embedding in zip(data,payload['embeddings']):
            item['frozen_tf_embedding']=embedding
        del payload
    named=list(model.named_parameters())
    groups=[{'params':[p for name,p in named if p.requires_grad and not name.startswith('backbone.')],'lr':params['head_lr']}]
    backbone_params=[p for name,p in named if p.requires_grad and name.startswith('backbone.')]
    if backbone_params:
        rate=(.0001 if task['backbone']=='resnet18_imagenet' else .00001)*params['backbone_multiplier']
        groups.append({'params':backbone_params,'lr':rate})
    base_lrs=[g['lr'] for g in groups]
    optimizer=torch.optim.AdamW(groups,weight_decay=params['weight_decay'])
    labels=protocol['data']['class_order']
    subject_labels={records[i]['subject_id']:records[i]['label'] for i in training}
    counts={c:sum(v==c for v in subject_labels.values()) for c in labels}
    weights=torch.tensor([len(subject_labels)/(5*counts[c]) for c in labels],device='cuda')
    start_epoch,best_epoch,best_score,stale,microbatch,elapsed_before=0,-1,-1.,0,24,0.
    last=run/'last.pt'
    if last.exists():
        payload=torch.load(last,map_location='cpu',weights_only=False) # locally created trusted checkpoint including RNG state
        model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler'])
        start_epoch=payload['epoch']+1
        best_epoch,best_score,stale,microbatch,elapsed_before=payload['best_epoch'],payload['best_score'],payload['stale'],payload['microbatch'],payload['elapsed_seconds']
        random.setstate(payload['random_state']);np.random.set_state(payload['numpy_state'])
        torch.set_rng_state(payload['torch_state']);torch.cuda.set_rng_state_all(payload['cuda_states'])
        del payload
    started=time.perf_counter()
    write_json(run/'state.json',{'status':'running','run_id':task['run_id'],'resource_cap':budget,'epochs_completed':start_epoch})
    epochs_completed=start_epoch
    val_records=[records[i] for i in validation]
    for epoch in range(start_epoch,budget):
        if stale>=plan['early_stopping_patience']:
            break
        epoch_start=time.perf_counter()
        factor=(epoch+1)/5 if epoch<5 else (1+math.cos(math.pi*(epoch-5)/(plan['scheduler_horizon_epochs']-5)))/2
        for group,rate in zip(optimizer.param_groups,base_lrs):
            group['lr']=rate*factor
        model.train()
        order=np.random.default_rng(seed+epoch).permutation(training).tolist()
        loss_sum=0.
        for start in range(0,len(order),96):
            indices=order[start:start+48]
            # An OOM retry must not double-update BatchNorm buffers or consume dropout RNG.
            retry_buffers={name:b.detach().clone() for name,b in model.named_buffers()}
            retry_cpu=torch.get_rng_state();retry_cuda=torch.cuda.get_rng_state_all()
            while True:
                optimizer.zero_grad(set_to_none=True)
                batch_loss=0.
                try:
                    for offset in range(0,len(indices),microbatch):
                        selected=indices[offset:offset+microbatch]
                        inputs=[normalize(data[i],stats,task['backbone']) for i in selected]
                        targets=torch.tensor([labels.index(records[i]['label']) for i in selected],device='cuda')
                        with precision_context(dtype):
                            logits=model(inputs)
                            loss=torch.nn.functional.cross_entropy(logits.float(),targets,weight=weights,reduction='sum')/len(indices)
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite training loss')
                        scaler.scale(loss).backward()
                        batch_loss+=float(loss.detach())*len(indices)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    scaler.step(optimizer);scaler.update()
                    loss_sum+=batch_loss
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if microbatch==1:
                        raise
                    microbatch//=2
                    with torch.no_grad():
                        for name,buffer in model.named_buffers():
                            buffer.copy_(retry_buffers[name])
                    torch.set_rng_state(retry_cpu);torch.cuda.set_rng_state_all(retry_cuda)
                    inputs=None;logits=None;loss=None
                    gc.collect();torch.cuda.empty_cache()
                    print(f'OOM fallback microbatch={microbatch}, effective batch=96',flush=True)
        model.eval();probabilities=[]
        with torch.no_grad(),precision_context(dtype):
            for start in range(0,len(validation),4):
                selected=validation[start:start+4]
                logits=model([normalize(data[i],stats,task['backbone']) for i in selected])
                probabilities.extend(logits.float().softmax(-1).cpu().tolist())
        subjects=aggregate(val_records,probabilities)
        result=metrics(subjects)
        if result['macro_F1']>best_score:
            best_score,best_epoch,stale=result['macro_F1'],epoch,0
            save_atomic({'model':model.state_dict(),'statistics':stats,'epoch':epoch,'config':config},run/'best.pt')
            write_jsonl(run/'validation_subject_predictions.jsonl',subjects)
            write_jsonl(run/'validation_vowel_predictions.jsonl',[
                {'subject_id':r['subject_id'],'leakage_group_id':r['leakage_group_id'],'split':'development','fold':task['fold'],
                 'pipeline_id':task['trial_id'],'seed':seed,'vowel':r['vowel'],'y_true':labels.index(r['label']),
                 **{'p_'+c:p[j] for j,c in enumerate(labels)},'valid_frames':r['valid_frames'],'feature_version':r['feature_version']}
                for r,p in zip(val_records,probabilities)])
        else:
            stale+=1
        epochs_completed=epoch+1
        elapsed=elapsed_before+time.perf_counter()-started
        log={'epoch':epochs_completed,'resource_cap':budget,'training_loss':loss_sum/len(training),'validation':result,
             'best_epoch':best_epoch+1,'best_macro_F1':best_score,'epoch_seconds':time.perf_counter()-epoch_start,
             'elapsed_seconds':elapsed,'microbatch':microbatch,'peak_memory_MB':torch.cuda.max_memory_allocated()/1024**2}
        save_atomic({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                     'epoch':epoch,'best_epoch':best_epoch,'best_score':best_score,'stale':stale,'microbatch':microbatch,
                     'elapsed_seconds':elapsed,'random_state':random.getstate(),'numpy_state':np.random.get_state(),
                     'torch_state':torch.get_rng_state(),'cuda_states':torch.cuda.get_rng_state_all()},last)
        # Rebuild the epoch journal after a crash between last.pt and journal writes.
        journal=read_jsonl(run/'epochs.jsonl') if (run/'epochs.jsonl').exists() else []
        journal=[r for r in journal if r['epoch']<epochs_completed]+[log]
        write_jsonl(run/'epochs.jsonl',journal)
        write_json(run/'state.json',{'status':'running','run_id':task['run_id'],**log})
        print(f'{task["run_id"]} epoch {epochs_completed}/{budget} (horizon 60): val macro-F1={result["macro_F1"]:.4f}, best={best_score:.4f}, seconds={log["epoch_seconds"]:.1f}',flush=True)
    if best_epoch<0:
        raise RuntimeError('no completed epoch available')
    snapshot=run/f'budget_{budget}'
    snapshot.mkdir(exist_ok=True)
    checkpoint=snapshot/'best.pt'
    if not checkpoint.exists():
        os.link(run/'best.pt',checkpoint)
    checkpoint_sha=digest_file(checkpoint)
    for name in ['validation_subject_predictions.jsonl','validation_vowel_predictions.jsonl']:
        predictions=read_jsonl(run/name)
        for prediction in predictions:
            prediction['checkpoint_sha256']=checkpoint_sha
        write_jsonl(snapshot/name,predictions)
    final={'run_id':task['run_id'],'trial_id':task['trial_id'],'fold':task['fold'],'resource_cap':budget,
           'epochs_completed':epochs_completed,'early_stopped':stale>=plan['early_stopping_patience'],
           'best_epoch':best_epoch+1,'validation_macro_F1':best_score,
           'wall_time_seconds':elapsed_before+time.perf_counter()-started,'checkpoint_sha256':checkpoint_sha,
           'normalization_sha256':digest_file(run/'normalization.pt'),'total_parameters':sum(p.numel() for p in model.parameters()),
           'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
           'mixed_precision':str(dtype),'final_microbatch':microbatch,'test_evaluated':False}
    write_json(result_path,final)
    write_json(run/'state.json',{'status':'early_stopped' if final['early_stopped'] else 'rung_complete',**final})
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--budget',type=int,required=True)
    args=parser.parse_args()
    run_task(args.task,args.budget)
