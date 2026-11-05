"""Development-only resumable neural training; no test evaluation entry point."""
import argparse
import contextlib
import json
import math
import os
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

from audit_waveforms import digest_file,write_json,write_jsonl
from freeze_data import ROOT,read_jsonl
from models import MultiViewModel


def aggregate(records,probabilities):
    grouped = defaultdict(list)
    for record,probability in zip(records,probabilities):
        grouped[record['subject_id']].append((record,probability))
    output = []
    classes = ['NV','SD','UVFP','BSVFL','MVFT']
    for subject,items in sorted(grouped.items()):
        assert sorted(r['vowel'] for r,p in items)==['a','e','i','o','u','v']
        probability = np.mean([p for r,p in items],axis=0)
        assert np.isfinite(probability).all() and np.isclose(probability.sum(),1.,atol=1e-5)
        output.append({'subject_id':subject,'leakage_group_id':items[0][0]['leakage_group_id'],
                       'y_true':classes.index(items[0][0]['label']),'probabilities':probability.tolist()})
    return output


def metrics(subjects):
    y = np.array([s['y_true'] for s in subjects])
    p = np.array([s['probabilities'] for s in subjects])
    return {'macro_F1':float(f1_score(y,p.argmax(1),labels=list(range(5)),average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(y,p.argmax(1))),
            'macro_OVR_AUROC':float(roc_auc_score(y,p,labels=list(range(5)),multi_class='ovr',average='macro'))}


def fit_statistics(data,indices):
    frames = sum(len(data[i]['TF']) for i in indices)
    total = sum((data[i]['TF'].double().sum(0) for i in indices),torch.zeros(128,dtype=torch.float64))
    square = sum((data[i]['TF'].double().square().sum(0) for i in indices),torch.zeros(128,dtype=torch.float64))
    mean = total/frames
    std = (square/frames-mean.square()).clamp_min(1e-12).sqrt()
    fe = torch.stack([data[i]['FE'].double() for i in indices])
    te_sum = sum(float(data[i]['TE'].double().sum()) for i in indices)
    te_square = sum(float(data[i]['TE'].double().square().sum()) for i in indices)
    te_mean = te_sum/frames
    return {'TF_mean':mean.float(),'TF_std':std.float(),'FE_mean':fe.mean(0).float(),
            'FE_std':fe.std(0,unbiased=False).clamp_min(1e-6).float(),
            'TE_mean':torch.tensor(te_mean,dtype=torch.float32),
            'TE_std':torch.tensor(max(te_square/frames-te_mean**2,1e-12)**.5,dtype=torch.float32)}


def normalize(item,stats,backbone,device='cuda'):
    values = {}
    for name in ['TF','FE','TE']:
        value = item[name]
        if name!='TF' or backbone=='resnet18_imagenet':
            value = (value-stats[name+'_mean'])/stats[name+'_std']
        values[name] = value.to(device)
    if 'frozen_tf_embedding' in item:
        values['frozen_tf_embedding'] = item['frozen_tf_embedding'].to(device)
    return values


def load_data(records):
    data=[]
    for record in records:
        path = ROOT/record['feature_path']
        if digest_file(path)!=record['feature_sha256']:
            raise RuntimeError('feature cache fingerprint mismatch')
        with np.load(path,allow_pickle=False) as cached:
            data.append({name:torch.from_numpy(cached[name].copy()) for name in ['TF','FE','TE']})
    return data


def precision_context(dtype):
    return torch.autocast('cuda',dtype=dtype)


def train_run(backbone,views,fold,fold_count=3,seed=3407,epochs=24,dropout=.3,fusion='concat',
              head_lr=.001,backbone_multiplier=1.,weight_decay=.001,run_id=None):
    torch.set_num_threads(12)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    protocol_path = ROOT/'protocol.execution.v1.2.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    lock_path = ROOT/'locks'/'data_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if digest_file(protocol_path)!=lock['protocol_sha256']:
        raise RuntimeError('protocol changed after data lock')
    for path,fingerprint in lock['files_sha256'].items():
        if digest_file(ROOT/path.replace('\\','/'))!=fingerprint:
            raise RuntimeError('locked data/split file changed')
    records = read_jsonl(ROOT/'features'/'development_manifest.jsonl')
    assert all(r['split']=='development' for r in records)
    split = {s['subject_id']:s['split'] for s in read_jsonl(ROOT/'splits'/'subject_split.jsonl')}
    assert all(split[r['subject_id']]=='development' for r in records)
    fold_map = {f['subject_id']:f['validation_fold'] for f in read_jsonl(ROOT/'splits'/'development_folds.jsonl') if f['fold_count']==fold_count}
    train_indices = [i for i,r in enumerate(records) if fold_map[r['subject_id']]!=fold]
    val_indices = [i for i,r in enumerate(records) if fold_map[r['subject_id']]==fold]
    assert not ({records[i]['leakage_group_id'] for i in train_indices}&{records[i]['leakage_group_id'] for i in val_indices})
    run_id = run_id or f'A_{backbone}_{views.replace("+","_")}_f{fold}_s{seed}'
    run = ROOT/'runs'/run_id
    run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():
        return json.loads((run/'result.json').read_text(encoding='utf-8'))
    code_files = ['train.py','models.py','features.py']
    config = {'run_id':run_id,'protocol_sha256':digest_file(protocol_path),'data_lock_sha256':digest_file(lock_path),
              'split_sha256':digest_file(ROOT/'splits'/'subject_split.jsonl'),
              'folds_sha256':digest_file(ROOT/'splits'/'development_folds.jsonl'),
              'feature_manifest_sha256':digest_file(ROOT/'features'/'development_manifest.jsonl'),
              'code_sha256':{n:digest_file(ROOT/'scripts'/n) for n in code_files},
              'code_commit':None,'code_version_basis':'not a git checkout; exact source SHA256 and per-run source snapshot',
              'environment_lock_sha256':digest_file(ROOT/'locks'/'environment.json'),
              'checkpoint_source_sha256':digest_file(ROOT/'pretrained'/('resnet18-f37072fd.pth' if backbone=='resnet18_imagenet' else 'BEATs_iter3.pt')),
              'backbone':backbone,'views':views,'fold':fold,'fold_count':fold_count,'seed':seed,'maximum_epochs':epochs,
              'head_lr':head_lr,'backbone_lr_multiplier':backbone_multiplier,'weight_decay':weight_decay,'dropout':dropout,'fusion':fusion,
              'microbatch_vowels':24,'effective_vowel_batch_size':96,'patience':10,'gradient_clip_norm':1.,
              'scheduler':'factor=(epoch_index+1)/5 if epoch_index<5; otherwise (1+cos(pi*(epoch_index-5)/(maximum_epochs-5)))/2',
              'loss':'sum class-weighted vowel CE / number of vowels in effective batch; class weights from training recording bundles',
              'train_bundles':len({records[i]['subject_id'] for i in train_indices}),'validation_bundles':len({records[i]['subject_id'] for i in val_indices})}
    if (run/'config.json').exists():
        if json.loads((run/'config.json').read_text(encoding='utf-8'))!=config:
            raise RuntimeError('resume configuration or code mismatch')
    else:
        write_json(run/'config.json',config)
        (run/'source').mkdir(exist_ok=True)
        for name in code_files:
            (run/'source'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    data = load_data(records)
    stats = fit_statistics(data,train_indices)
    torch.save(stats,run/'normalization.pt')
    model = MultiViewModel(backbone,views,dropout,fusion).cuda()
    model.train()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    if backbone=='beats_frozen' and 'TF' in views:
        cache = ROOT/'features'/f'frozen_beats_{records[0]["feature_version"]}_{config["checkpoint_source_sha256"][:12]}.pt'
        if cache.exists():
            payload = torch.load(cache,map_location='cpu',weights_only=True)
            assert payload['paths']==[r['relative_path'] for r in records]
            embeddings = payload['embeddings']
        else:
            model.eval(); vectors=[]
            with torch.no_grad(),precision_context(dtype):
                for i,item in enumerate(data):
                    vectors.append(model.tf_embedding(item['TF'].cuda())[0].float().cpu())
                    if (i+1)%500==0:
                        print(f'Immutable frozen BEATs cache {i+1}/{len(data)}',flush=True)
            embeddings = torch.stack(vectors)
            torch.save({'paths':[r['relative_path'] for r in records],'embeddings':embeddings},cache)
            model.train()
        for item,vector in zip(data,embeddings):
            item['frozen_tf_embedding'] = vector
    named = list(model.named_parameters())
    groups = [{'params':[p for n,p in named if p.requires_grad and not n.startswith('backbone.')],'lr':head_lr}]
    bparams = [p for n,p in named if p.requires_grad and n.startswith('backbone.')]
    if bparams:
        rate = (.0001 if backbone=='resnet18_imagenet' else .00001)*backbone_multiplier
        groups.append({'params':bparams,'lr':rate})
    base_lrs = [g['lr'] for g in groups]
    optimizer = torch.optim.AdamW(groups,weight_decay=weight_decay)
    train_subjects = {records[i]['subject_id']:records[i]['label'] for i in train_indices}
    counts = Counter(train_subjects.values())
    class_order = protocol['data']['class_order']
    weights = torch.tensor([len(train_subjects)/(5*counts[c]) for c in class_order],device='cuda')
    start_epoch,best_epoch,best_score,stale,elapsed_before = 0,-1,-1.,0,0.
    microbatch = 24
    last = run/'last.pt'
    if last.exists():
        payload = torch.load(last,map_location='cpu',weights_only=False) # own trusted local checkpoint, includes RNG state
        model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
        scaler.load_state_dict(payload['scaler'])
        start_epoch,best_epoch,best_score,stale = payload['epoch']+1,payload['best_epoch'],payload['best_score'],payload['stale']
        elapsed_before = payload['elapsed_seconds']; microbatch = payload['microbatch']
        random.setstate(payload['random_state']); np.random.set_state(payload['numpy_state'])
        torch.set_rng_state(payload['torch_state']); torch.cuda.set_rng_state_all(payload['cuda_states'])
    started = time.perf_counter()
    write_json(run/'state.json',{'status':'running','epoch':start_epoch,'maximum_epochs':epochs,'run_id':run_id})
    for epoch in range(start_epoch,epochs):
        epoch_start = time.perf_counter()
        factor = (epoch+1)/5 if epoch<5 else (1+math.cos(math.pi*(epoch-5)/max(1,epochs-5)))/2
        for group,base in zip(optimizer.param_groups,base_lrs):
            group['lr'] = base*factor
        model.train()
        order = np.random.default_rng(seed+epoch).permutation(train_indices).tolist()
        loss_sum = 0.
        for start in range(0,len(order),96):
            effective = order[start:start+48]
            while True:
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.
                try:
                    for offset in range(0,len(effective),microbatch):
                        indices = effective[offset:offset+microbatch]
                        inputs = [normalize(data[i],stats,backbone) for i in indices]
                        targets = torch.tensor([class_order.index(records[i]['label']) for i in indices],device='cuda')
                        with precision_context(dtype):
                            logits = model(inputs)
                            loss = torch.nn.functional.cross_entropy(logits.float(),targets,weight=weights,reduction='sum')/len(effective)
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite loss')
                        scaler.scale(loss).backward()
                        batch_loss += float(loss.detach())*len(effective)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    scaler.step(optimizer); scaler.update()
                    loss_sum += batch_loss
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if microbatch==1:
                        raise
                    microbatch //= 2
                    torch.cuda.empty_cache()
                    print(f'OOM fallback: microbatch={microbatch}, effective batch=96',flush=True)
        model.eval(); probabilities=[]
        with torch.no_grad(),precision_context(dtype):
            for start in range(0,len(val_indices),4):
                indices = val_indices[start:start+4]
                logits = model([normalize(data[i],stats,backbone) for i in indices])
                probabilities.extend(logits.float().softmax(-1).cpu().tolist())
        val_records = [records[i] for i in val_indices]
        subject_predictions = aggregate(val_records,probabilities)
        result = metrics(subject_predictions)
        score = result['macro_F1']
        improved = score>best_score
        if improved:
            best_score,best_epoch,stale = score,epoch,0
            temporary = run/'best.tmp'
            torch.save({'model':model.state_dict(),'statistics':stats,'epoch':epoch,'config':config},temporary)
            temporary.replace(run/'best.pt')
            write_jsonl(run/'validation_subject_predictions.jsonl',subject_predictions)
            write_jsonl(run/'validation_vowel_predictions.jsonl',[
                {'subject_id':r['subject_id'],'leakage_group_id':r['leakage_group_id'],'split':'development',
                 'fold':fold,'pipeline_id':run_id,'seed':seed,'vowel':r['vowel'],'y_true':class_order.index(r['label']),
                 **{'p_'+c:p[j] for j,c in enumerate(class_order)},'valid_frames':r['valid_frames'],
                 'feature_version':r['feature_version']} for r,p in zip(val_records,probabilities)])
        else:
            stale+=1
        elapsed = elapsed_before+time.perf_counter()-started
        log = {'epoch':epoch+1,'training_loss':loss_sum/len(train_indices),'validation':result,'best_epoch':best_epoch+1,
               'best_macro_F1':best_score,'epoch_seconds':time.perf_counter()-epoch_start,'elapsed_seconds':elapsed,
               'microbatch':microbatch,'peak_memory_MB':torch.cuda.max_memory_allocated()/1024**2}
        with (run/'epochs.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(log)+'\n')
        payload = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                   'epoch':epoch,'best_epoch':best_epoch,'best_score':best_score,'stale':stale,'microbatch':microbatch,
                   'elapsed_seconds':elapsed,'random_state':random.getstate(),'numpy_state':np.random.get_state(),
                   'torch_state':torch.get_rng_state(),'cuda_states':torch.cuda.get_rng_state_all()}
        torch.save(payload,run/'last.tmp'); (run/'last.tmp').replace(last)
        write_json(run/'state.json',{'status':'running','run_id':run_id,**log})
        print(f'{run_id} epoch {epoch+1}/{epochs}: val macro-F1={score:.4f}, best={best_score:.4f}, seconds={log["epoch_seconds"]:.1f}',flush=True)
        if stale>=10:
            break
    final = {'run_id':run_id,'best_epoch':best_epoch+1,'validation_macro_F1':best_score,
             'wall_time_seconds':elapsed_before+time.perf_counter()-started,'checkpoint_sha256':digest_file(run/'best.pt'),
             'normalization_sha256':digest_file(run/'normalization.pt'),'total_parameters':sum(p.numel() for p in model.parameters()),
             'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'mixed_precision':str(dtype),
             'final_microbatch':microbatch,'test_evaluated':False}
    # Stamp final best-checkpoint digest into the saved best-epoch vowel records.
    predictions = read_jsonl(run/'validation_vowel_predictions.jsonl')
    for prediction in predictions:
        prediction['checkpoint_sha256'] = final['checkpoint_sha256']
    write_jsonl(run/'validation_vowel_predictions.jsonl',predictions)
    write_json(run/'result.json',final)
    write_json(run/'state.json',{'status':'complete',**final})
    return final


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone',required=True,choices=['resnet18_imagenet','beats_frozen','beats_last2'])
    parser.add_argument('--views',required=True)
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--fold-count',type=int,default=3)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--epochs',type=int,default=24)
    args=parser.parse_args()
    train_run(args.backbone,args.views,args.fold,args.fold_count,args.seed,args.epochs)
