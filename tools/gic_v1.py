"""Bounded GIC preflight and independent tmux lifecycle; no implicit formal training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from gic_v1_common import (ROOT,OUT,RUN,INIT,BASE,SERVER_MAIN,SERVER_WT,SERVER_PY,require,now,
    read_json,write_json,sha256,source_identity,binding,prepare,load_yaml,digest)
from gic_v1_lock import operation_lock,process_identity


def state(name):
    record=read_json(OUT/(name+'.json'),{'status':'NOT_STARTED'})
    if record.get('status')=='RUNNING':
        saved=record.get('process',{})
        live=process_identity(saved.get('pid',-1))
        if not live or live!=saved:
            record=dict(record,status='INTERRUPTED',reason='Recorded process is no longer alive with the same identity')
    if record.get('status')=='COMPLETE' and record.get('folder'):
        exit_record=read_json(Path(record['folder'])/'exit.json')
        if exit_record is None:
            record=dict(record,status='EXIT_PENDING',reason='Python finished; pipeline exit record not yet written')
        elif exit_record.get('python')!=0 or exit_record.get('tee')!=0:
            record=dict(record,status='FAILED',reason='Python/tee pipeline failed',pipeline_exit=exit_record)
        elif name=='finish':
            pack_exit=read_json(Path(record['folder'])/'pack_exit.json')
            if pack_exit is None:record=dict(record,status='PACK_PENDING')
            elif pack_exit.get('python')!=0 or pack_exit.get('tee')!=0:
                record=dict(record,status='FAILED',reason='Offline pack/tee failed',pack_exit=pack_exit)
    return record


def display():
    print('GIC v1 independent FP32 metrics; P=Precision; all/crack (one class)',flush=True)
    for split in ('val','test'):
        r=read_json(OUT/f'evaluation_{split}.json',{})
        if r.get('status')!='COMPLETE':
            print(split+': NOT_RUN / incomplete',flush=True);continue
        print(f"{split}: images={r['images']} GT={r['ground_truths']} exit={r['exit_code']} protocol={r['identity']['protocol']}",flush=True)
        print(' '.join(f'{k}={100*r[k]:.6f}%' for k in ('precision','recall','F1','AP50','AP75','mAP50_95'))+
              f" mother_delta={r['mother_delta_pp']:+.6f} pp",flush=True)
        print('AP50:0.05:0.95: '+', '.join(f'{100*x:.6f}%' for x in r['ap_by_iou'][0]),flush=True)
        print(f"best={r['best']} SHA256={r['best_sha256']}",flush=True)


def status():
    display()
    result=dict(run=str(RUN),evidence=str(OUT),preflight=state('preflight'),training=state('training'),finish=state('finish'),
                progress=read_json(OUT/'progress.json'),package=read_json(OUT/'package.json'),
                exits=[read_json(p) for p in sorted((OUT/'dispatches').glob('*/exit.json'))])
    result['latest_dispatch']=read_json(OUT/'latest_dispatch.json')
    if sys.platform.startswith('linux') and shutil.which('tmux'):
        result['tmux_panes']={}
        for session in ('gic-v1-training','gic-v1-finish'):
            pane=subprocess.run(['tmux','list-panes','-s','-t',session,'-F',
                                 '#{pane_id} dead=#{pane_dead} exit=#{pane_dead_status} pid=#{pane_pid} command=#{pane_current_command}'],
                                text=True,capture_output=True,timeout=10)
            result['tmux_panes'][session]=pane.stdout.splitlines() if pane.returncode==0 else ['NOT_STARTED / unavailable']
    for job in ('training','finish'):
        r=result[job]
        if r.get('folder'):
            log=Path(r['folder'])/'console.log'
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-5000));result[job+'_log_tail']=stream.read().decode('utf-8',errors='replace')
    # Read-only: no binding(), dataset scan, import of model/evaluator, or inference.
    return result


def successful_preflight(current):
    r=read_json(OUT/'preflight.json',{})
    if r.get('status')!='PASS' or r.get('binding')!=current or r.get('exit_code')!=0: return False
    return bool(r.get('artifacts')) and all(Path(p).is_file() and sha256(p)==h for p,h in r['artifacts'].items())


def bounded_preflight(args):
    if (OUT/'prepare.json').exists():
        try: current=binding(clean=not args.development)
        except (RuntimeError,OSError): current=None
        if current and successful_preflight(current):
            return dict(read_json(OUT/'preflight.json'),reused=True)
    require(state('training')['status']=='NOT_STARTED','Preflight does not alter a started experiment; inspect its existing evidence')
    folder=OUT/'preflight'/now();folder.mkdir(parents=True)
    command=[sys.executable,'-u',str(ROOT/'tools/gic_v1.py'),'_preflight','--folder',str(folder),
             '--main',args.main,'--seconds',str(args.seconds)]
    for key in ('data','mother_args'):
        if getattr(args,key): command+=['--'+key.replace('_','-'),getattr(args,key)]
    if args.development: command+=['--development']
    if args.refresh_data: command+=['--refresh-data']
    started=time.monotonic()
    with (folder/'console.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
        write_json(folder/'process.json',process_identity(child.pid))
        try: code=child.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            import psutil
            owned=psutil.Process(child.pid).children(recursive=True)
            for p in reversed(owned):
                try:p.terminate()
                except psutil.NoSuchProcess:pass
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            _,alive=psutil.wait_procs(owned,timeout=1)
            for p in alive:
                try:p.kill()
                except psutil.NoSuchProcess:pass
            code=124
    result=read_json(folder/'result.json',{})
    if code!=0:
        result.update(status='TIMEOUT' if code==124 else result.get('status','FAIL'),reason=result.get('reason',f'Preflight exit {code}; inspect console.log'))
        if result['status']=='PASS':result.update(status='FAIL',reason=f'Preflight process exit {code} overrides intermediate PASS')
    result.update(exit_code=code,folder=str(folder),elapsed_seconds=time.monotonic()-started)
    write_json(OUT/'preflight.json',result)
    return result


def preflight_child(args):
    folder=args.folder
    report=dict(status='RUNNING',time=now())
    try:
        data=Path(args.data or Path(args.main)/'configs/crack_autodl.yaml')
        source=Path(args.main)/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt'
        missing=[str(p) for p in (source,data) if not p.is_file()]
        if data.is_file():
            data_root=Path(load_yaml(data)['path'])
            missing += [str(data_root/'images'/s) for s in ('train','val','test') if not (data_root/'images'/s).is_dir()]
        if missing:
            report.update(status='NOT_RUN',reason='Required initialization/data unavailable',missing=missing)
            return report
        prepare(args)
        current=binding(clean=not args.development)
        for script,name in [('check_gic_v1.py','unit_tests'),('check_gic_v1_ops.py','operations_tests')]:
            with (folder/(name+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(ROOT/'tools'/script),'--report',str(folder/(name+'.json'))],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        from gic_v1_training import real_smoke,amp_assets
        plan=read_json(OUT/'prepare.json')
        # Separate fresh temporary models; no Trainer loop or official evaluation.
        smoke=real_smoke(Path(args.main),Path(load_yaml(plan['data_yaml'])['path']),folder/'smoke')
        if smoke['status']=='PASS' and not args.development:amp_assets(Path(args.main))
        report=dict(status=smoke['status'],binding=current,real_smoke=smoke['status'],reason=smoke.get('reason'),
                    artifacts={str(p):sha256(p) for p in [folder/'unit_tests.json',folder/'operations_tests.json',folder/'smoke/smoke.json']})
        require(binding(clean=not args.development)==current,'Identity changed during preflight')
    except BaseException as error:
        report.update(status='FAIL',reason=repr(error));raise
    finally:write_json(folder/'result.json',report)
    return report


def formal_identity():
    require(sys.platform.startswith('linux') and ROOT==Path(SERVER_WT) and Path(sys.executable)==Path(SERVER_PY),
            'Formal work requires documented server worktree and existing rtdetr Python; no environment replacement')
    plan=read_json(OUT/'prepare.json',{})
    require(plan and not plan['development'] and plan['recipe_audit']['actual_status']=='CHECKED','Need formal prepared mother recipe')
    return binding()


def ready(job):
    current=formal_identity()
    require(successful_preflight(current),'Missing/stale successful bounded preflight; run preflight once')
    if job in ('start','resume'):
        require(not (OUT/'training_completed.json').exists(),'Training already completed; use finish')
        if job=='start':
            require(not RUN.exists() and not (OUT/'training.json').exists(),f'Existing run preserved: {RUN}; inspect status/resume')
        else:
            require((RUN/'weights/last.pt').is_file(),'No same-run last checkpoint to resume')
            from ultralytics.utils.patches import torch_load
            from ultralytics.models.rtdetr.gic_loss import CONFIG
            ckpt=torch_load(RUN/'weights/last.pt',map_location='cpu')
            require(0<=ckpt.get('epoch',-1)<199 and all(ckpt.get(k) is not None for k in ('optimizer','scaler','ema')),
                    'Resume requires an unstripped, unfinished last checkpoint')
            require(getattr(ckpt['ema'],'gic_identity',None)==current and getattr(ckpt['ema'],'gic_config',None)==CONFIG,
                    'Resume checkpoint belongs to a different run/config')
            expected=read_json(OUT/'prepare.json')['args'];actual=ckpt['train_args']
            require(all(type(actual.get(k)) is type(v) and actual.get(k)==v for k,v in expected.items() if k not in ('model','resume')),
                    'Resume checkpoint training recipe differs')
    else:
        require(state('training').get('status')=='COMPLETE','Finish requires successful completed training')
        complete=read_json(OUT/'training_completed.json',{})
        require(complete.get('status')=='COMPLETE' and complete.get('binding')==current,'Incomplete/stale training completion')
        train_exit=read_json(Path(read_json(OUT/'training.json')['folder'])/'exit.json',{})
        require(train_exit.get('python')==train_exit.get('tee')==0,'Training Python/tee did not both exit successfully')
    return current


def dispatch(job):
    current=ready(job)
    session='gic-v1-finish' if job=='finish' else 'gic-v1-training'
    exists=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
    if exists:
        panes=subprocess.check_output(['tmux','list-panes','-s','-t',session,'-F','#{pane_dead}'],text=True).splitlines()
        require(panes and all(x=='1' for x in panes),f'{session} has live panes; duplicate dispatch refused')
    folder=OUT/'dispatches'/now();folder.mkdir(parents=True)
    record=dict(job=job,binding=current,folder=str(folder),session=session,time=now())
    write_json(folder/'dispatch.json',record)
    q=shlex.quote;tool=str(ROOT/'tools/gic_v1.py')
    lines=['#!/usr/bin/env bash','set +e',f'cd {q(str(ROOT))} || exit 90',
           # This executes inside the correct target pane before Python can exit.
           'tmux set-option -p -t "$TMUX_PANE" remain-on-exit on || exit 91',
           f'{q(sys.executable)} -u {q(tool)} _worker --folder {q(str(folder))} 2>&1 | tee -a {q(str(folder/"console.log"))}',
           'codes=("${PIPESTATUS[@]}")',
           f'{q(sys.executable)} {q(tool)} _record-exit --folder {q(str(folder))} --python-code "${{codes[0]}}" --tee-code "${{codes[1]}}"',
           'record_code=$?', 'result="${codes[0]}"',
           'if [[ "$result" == 0 && "${codes[1]}" != 0 ]]; then result="${codes[1]}"; fi',
           'if [[ "$result" == 0 && "$record_code" != 0 ]]; then result="$record_code"; fi',
           ]
    if job=='finish':
        lines += ['if [[ "$result" == 0 ]]; then',
                  f'  {q(sys.executable)} -u {q(tool)} pack 2>&1 | tee -a {q(str(folder/"pack.log"))}',
                  '  pack_codes=("${PIPESTATUS[@]}")',
                  f'  {q(sys.executable)} {q(tool)} _record-exit --folder {q(str(folder))} --python-code "${{pack_codes[0]}}" --tee-code "${{pack_codes[1]}}" --pack-exit',
                  '  record_code=$?', '  result="${pack_codes[0]}"',
                  '  if [[ "$result" == 0 && "${pack_codes[1]}" != 0 ]]; then result="${pack_codes[1]}"; fi',
                  '  if [[ "$result" == 0 && "$record_code" != 0 ]]; then result="$record_code"; fi','fi']
    lines += [f'printf "\\nGIC {job}: Python=%s tee=%s final_exit=%s\\nLog: %s\\nArtifacts: %s\\n" "${{codes[0]}}" "${{codes[1]}}" "$result" {q(str(folder/"console.log"))} {q(str(OUT))}',
              'exit "$result"']
    script=folder/'worker.sh';script.write_bytes(('\n'.join(lines)+'\n').encode())
    command=['tmux','new-window','-t',session,'-n',folder.name] if exists else ['tmux','new-session','-d','-s',session]
    # Old exited panes stay present; no kill-session and no global options.
    subprocess.run(command+['bash '+q(str(script))],check=True)
    write_json(OUT/'latest_dispatch.json',record)
    return dict(status='DISPATCHED',session=session,folder=str(folder),binding=current)


def worker(folder):
    dispatch_record=read_json(folder/'dispatch.json')
    require(dispatch_record and dispatch_record['binding']==ready(dispatch_record['job']),'Dispatch identity changed')
    job=dispatch_record['job'];name='finish' if job=='finish' else 'training'
    record=dict(status='RUNNING',process=process_identity(os.getpid()),folder=str(folder),job=job,
                binding=dispatch_record['binding'],git_head=source_identity()['commit'],started=now(),
                command=sys.argv,seed=42,loss=read_json(ROOT/'docs/gic_v1/gic_config.json'))
    if (OUT/(name+'.json')).exists():write_json(folder/'previous_state.json',read_json(OUT/(name+'.json')))
    write_json(OUT/(name+'.json'),record)
    try:
        if job=='finish':
            from gic_v1_eval import evaluate,offline_analysis
            from gic_v1_analysis import analyze_exports
            for split in ('val','test'):evaluate(split)
            offline_analysis();analyze_exports();display()
            from gic_v1_report import render
            render()
        else:
            from gic_v1_training import GICTrainer,amp_assets
            amp_assets(Path(SERVER_MAIN))
            args=dict(read_json(OUT/'prepare.json')['args'])
            if job=='resume':args.update(model=str(RUN/'weights/last.pt'),resume=str(RUN/'weights/last.pt'))
            (folder/'pip_freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
            trainer=GICTrainer(overrides=args,gic_binding=record['binding'])
            trainer.train()
            require(read_json(OUT/'training_completed.json',{}).get('status')=='COMPLETE','Trainer ended without completion evidence')
        record['status']='COMPLETE'
    except BaseException as error:
        record.update(status='FAILED',error=repr(error));raise
    finally:
        record['ended']=now();write_json(OUT/(name+'.json'),record)
    return record


def main():
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','start','resume','status','attach','finish','pack','_preflight','_worker','_record-exit'])
    parser.add_argument('--main',default=SERVER_MAIN);parser.add_argument('--data');parser.add_argument('--mother-args')
    parser.add_argument('--development',action='store_true');parser.add_argument('--refresh-data',action='store_true')
    parser.add_argument('--seconds',type=int,default=900);parser.add_argument('--folder',type=Path)
    parser.add_argument('--python-code',type=int);parser.add_argument('--tee-code',type=int)
    parser.add_argument('--pack-exit',action='store_true')
    args=parser.parse_args();require(0<args.seconds<=900,'Preflight budget must be <=900 seconds')
    if args.action=='status':result=status()
    elif args.action=='attach':return subprocess.call(['tmux','attach','-t','gic-v1-training'])
    elif args.action=='_record-exit':
        require(args.folder is not None and args.python_code is not None and args.tee_code is not None,'Missing exit codes')
        r=read_json(args.folder/'dispatch.json');result=dict(job=r['job'],binding=r['binding'],python=args.python_code,tee=args.tee_code,time=now())
        write_json(args.folder/('pack_exit.json' if args.pack_exit else 'exit.json'),result)
    elif args.action=='_preflight':result=preflight_child(args)
    else:
        with operation_lock(wait=args.action=='_worker'):
            if args.action=='preflight':result=bounded_preflight(args)
            elif args.action in ('start','resume','finish'):result=dispatch(args.action)
            elif args.action=='_worker':result=worker(args.folder)
            else:
                from gic_v1_pack import package
                result=package();display()
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    return 0 if result.get('status') not in ('FAIL','FAILED','OOM','NOT_RUN','PARTIAL','TIMEOUT') else 2


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        traceback.print_exc();sys.exit(1)
