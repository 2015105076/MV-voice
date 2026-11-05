"""Run/resume the complete prespecified A screening queue, then stop before B."""
import json
import os
import subprocess
import sys
import time
import traceback

from audit_waveforms import digest_file,write_json
from freeze_data import ROOT


def main():
    runs=ROOT/'runs'
    runs.mkdir(exist_ok=True)
    mutex=runs/'stage_A.running'
    try:
        descriptor=os.open(mutex,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        previous=int(mutex.read_text().strip())
        try:
            os.kill(previous,0)
        except OSError:
            mutex.unlink()
            descriptor=os.open(mutex,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        else:
            raise RuntimeError(f'queue already active: process {previous}')
    with os.fdopen(descriptor,'w') as stream:
        stream.write(str(os.getpid()))
    state={'stage':'A','status':'running','pid':os.getpid(),'started_unix':time.time(),'total_runs':18,'completed_runs':0,'test_evaluated':False}
    try:
        verification=json.loads((ROOT/'locks'/'pipeline_verification.json').read_text(encoding='utf-8'))
        assert verification['checks']['all_branch_and_backbone_GPU_forward_backward']
        assert verification['development_only'] and not verification['test_evaluated']
        for name,fingerprint in verification['source_sha256'].items():
            if digest_file(ROOT/'scripts'/name)!=fingerprint:
                raise RuntimeError('training source changed since M1 verification')
        queue=[{'backbone':b,'views':v,'fold':f,'seed':3407,'epochs':24,
                'run_id':f'A_{b}_{v.replace("+","_")}_f{f}_s3407'}
               for b in ['resnet18_imagenet','beats_frozen','beats_last2']
               for f in range(3) for v in ['TF','TF+FE']]
        write_json(runs/'stage_A_queue.json',queue)
        for task in queue:
            # A user-created STOP file stops between complete runs; it never deletes results.
            if (runs/'STOP').exists():
                state['status']='stopped_between_runs'
                write_json(runs/'stage_A_state.json',state)
                return
            state['active_run']=task['run_id']
            write_json(runs/'stage_A_state.json',state)
            print(f'Starting {task["run_id"]}',flush=True)
            command=[sys.executable,'-u',str(ROOT/'scripts'/'train.py'),'--backbone',task['backbone'],
                     '--views',task['views'],'--fold',str(task['fold']),'--seed','3407','--epochs','24']
            subprocess.run(command,cwd=ROOT,check=True)
            state['completed_runs']+=1
            write_json(runs/'stage_A_state.json',state)
        results=[]
        for b in ['resnet18_imagenet','beats_frozen','beats_last2']:
            for v in ['TF','TF+FE']:
                fold_results=[json.loads((runs/f'A_{b}_{v.replace("+","_")}_f{f}_s3407'/'result.json').read_text(encoding='utf-8')) for f in range(3)]
                results.append({'backbone':b,'views':v,'mean_fold_macro_F1':sum(r['validation_macro_F1'] for r in fold_results)/3,
                                'folds':fold_results,'total_parameters':fold_results[0]['total_parameters']})
        scores=[]
        for b in ['resnet18_imagenet','beats_frozen','beats_last2']:
            matched=[r for r in results if r['backbone']==b]
            winner=max(matched,key=lambda r:r['mean_fold_macro_F1'])
            scores.append({'backbone':b,'score':winner['mean_fold_macro_F1'],'scoring_views':winner['views'],
                           'total_parameters':winner['total_parameters']})
        # Deterministic near-tie preference applied when choosing the next highest rank.
        ranked=[]
        while scores:
            maximum=max(s['score'] for s in scores)
            tied=[s for s in scores if maximum-s['score']<=.002]
            winner=min(tied,key=lambda s:(s['total_parameters'],s['backbone']))
            ranked.append(winner);scores.remove(winner)
        write_json(runs/'stage_A_summary.json',{'pipelines':results,'ranking':ranked,'retained_backbones':[s['backbone'] for s in ranked[:2]],
                  'selection_only':True,'test_evaluated':False,'next_stage':'B bounded hyperparameter selection; not dispatched by this A-only queue'})
        state.update(status='complete',active_run=None,finished_unix=time.time())
        write_json(runs/'stage_A_state.json',state)
        print('Stage A complete. Six pipeline summaries and two retained backbones saved.',flush=True)
    except BaseException as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        write_json(runs/'stage_A_state.json',state)
        raise
    finally:
        mutex.unlink(missing_ok=True)


if __name__=='__main__':
    main()
