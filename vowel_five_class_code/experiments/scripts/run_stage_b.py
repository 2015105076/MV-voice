"""Run/resume fixed B successive-halving search. Stop before C or test access."""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback

from audit_waveforms import digest_file,write_json
from freeze_data import ROOT
from search_plan import freeze_plan,rank_near_ties


def is_running(pid):
    if os.name!='nt':
        try:
            os.kill(pid,0)
            return True
        except ProcessLookupError:
            return False
    # os.kill(pid, 0) is not a read-only existence check on Windows.
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        return ctypes.get_last_error()==5  # access denied: conservatively assume live
    try:
        code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
            return True
        return code.value==259
    finally:
        kernel.CloseHandle(handle)


def state_write(state):
    temporary=ROOT/'runs'/'stage_B_state.tmp'
    write_json(temporary,state)
    temporary.replace(ROOT/'runs'/'stage_B_state.json')


def main(plan_only=False):
    plan=freeze_plan()
    plan_sha=digest_file(ROOT/'locks'/'stage_B_plan.json')
    if plan_only:
        print(f'Frozen B plan: {len(plan["pipelines"])} pipelines x 8 trials x 3 folds; caps 12,24,60.',flush=True)
        return
    a=json.loads((ROOT/'runs'/'stage_A_state.json').read_text(encoding='utf-8'))
    if a['status']!='complete' or a['completed_runs']!=18:
        raise RuntimeError('A screening must finish before B')
    verification=json.loads((ROOT/'locks'/'stage_B_verification.json').read_text(encoding='utf-8'))
    if verification['plan_sha256']!=plan_sha:
        raise RuntimeError('B plan changed since verification')
    if not verification['checks']['uninterrupted_vs_resumed_two_epochs_exact_model_optimizer_and_RNG_match']:
        raise RuntimeError('B checkpoint continuation has not passed verification')
    for name,fingerprint in verification['source_sha256'].items():
        if digest_file(ROOT/'scripts'/name)!=fingerprint:
            raise RuntimeError(f'B code changed since verification: {name}')
    mutex=ROOT/'runs'/'stage_B.running'
    try:
        descriptor=os.open(mutex,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        previous=int(mutex.read_text().strip())
        if is_running(previous):
            raise RuntimeError(f'B queue already running: {previous}')
        mutex.unlink()
        descriptor=os.open(mutex,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w') as stream:
        stream.write(str(os.getpid()))
    state={'stage':'B','status':'running','pid':os.getpid(),'started_unix':time.time(),
           'plan_sha256':plan_sha,'pipelines':len(plan['pipelines']),'initial_trials':40,
           'total_fold_budget_evaluations':210,'completed_fold_budget_evaluations':0,'test_evaluated':False}
    try:
        active={p['pipeline_id']:p['trials'] for p in plan['pipelines']}
        history=[]
        for index,budget in enumerate(plan['resource_schedule_epochs']):
            state['resource_cap']=budget
            for pipeline in plan['pipelines']:
                pipe_id=pipeline['pipeline_id']
                evaluated=[]
                for trial in active[pipe_id]:
                    folds=[]
                    for fold in range(plan['folds']):
                        if (ROOT/'runs'/'STOP').exists():
                            state.update(status='stopped_between_runs')
                            state_write(state)
                            return
                        run_id=f'{trial["trial_id"]}_f{fold}_s{plan["seed"]}'
                        run=ROOT/'runs'/run_id
                        run.mkdir(exist_ok=True)
                        task={**trial,'run_id':run_id,'fold':fold,'plan_sha256':plan_sha,'verification_only':False}
                        taskfile=run/'task.json'
                        if taskfile.exists():
                            if json.loads(taskfile.read_text(encoding='utf-8'))!=task:
                                raise RuntimeError('frozen task configuration changed')
                        else:
                            write_json(taskfile,task)
                        state.update(active_pipeline=pipe_id,active_trial=trial['trial_id'],active_run=run_id)
                        state_write(state)
                        print(f'B cap {budget}: {run_id}',flush=True)
                        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/'train_search.py'),'--task',str(taskfile),'--budget',str(budget)],cwd=ROOT,check=True)
                        result=json.loads((run/f'rung_{budget}.json').read_text(encoding='utf-8'))
                        if result['test_evaluated'] or result['resource_cap']!=budget:
                            raise RuntimeError('invalid worker result')
                        folds.append(result)
                        state['completed_fold_budget_evaluations']+=1
                        state_write(state)
                    evaluated.append({**trial,'fusion':trial['parameters']['fusion'],'resource_cap':budget,
                                      'mean_fold_macro_F1':sum(r['validation_macro_F1'] for r in folds)/len(folds),
                                      'total_parameters':folds[0]['total_parameters'],'folds':folds})
                ranked=rank_near_ties(evaluated)
                record={'pipeline_id':pipe_id,'resource_cap':budget,'ranking':ranked,'selection_only':True,'test_evaluated':False}
                if index+1<len(plan['resource_schedule_epochs']):
                    keep=plan['survivors'][index+1]
                    ids={r['trial_id'] for r in ranked[:keep]}
                    active[pipe_id]=[t for t in pipeline['trials'] if t['trial_id'] in ids]
                    record['promoted_trial_ids']=[r['trial_id'] for r in ranked[:keep]]
                write_json(ROOT/'runs'/f'B_{pipe_id}_budget_{budget}.json',record)
                history.append(record)
        winners=[r['ranking'][0] for r in history if r['resource_cap']==plan['resource_schedule_epochs'][-1]]
        eligible=[r for r in winners if r['backbone'] in plan['main_backbone_candidates']]
        chosen=rank_near_ties(eligible)[0]
        tf_fe=next(r for r in winners if r['backbone']==chosen['backbone'] and r['views']=='TF+FE')
        baseline=next(r for r in winners if r['backbone']=='resnet18_imagenet' and r['views']=='TF')
        fusion_source=chosen if chosen['views']=='TF+FE' else tf_fe
        summary={'stage':'B','plan_sha256':plan_sha,'pipeline_winners':winners,'rung_history':history,
                 'selected_backbone':chosen['backbone'],'selected_trial':chosen,
                 'common_training_hyperparameters':chosen['parameters'],'selected_fusion':fusion_source['parameters']['fusion'],
                 'fusion_source_trial_id':fusion_source['trial_id'],'resnet18_TF_baseline_trial':baseline,
                 'selection_only':True,'test_evaluated':False,
                 'next_stage':'C five-fold three-seed seven-view and baseline comparison plus planned controls; not dispatched by B-only queue'}
        write_json(ROOT/'runs'/'stage_B_summary.json',summary)
        state.update(status='complete',active_run=None,finished_unix=time.time())
        state_write(state)
        print('Stage B complete; final-rung winners and shared settings saved. No test evaluation.',flush=True)
    except BaseException as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        state_write(state)
        raise
    finally:
        mutex.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true')
    main(parser.parse_args().plan_only)
