"""Pinned single-experiment lifecycle; preflight once before tmux dispatch."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import traceback
import uuid
from c24_lif_v1_common import *

def token(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,ValueError,IndexError):return None

def tmux_active():
    return bool(shutil.which('tmux')) and subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode==0

def owner():
    return dict(pid=os.getpid(),start_time=token(os.getpid()),worktree=str(ROOT.resolve()),session=SESSION,token=uuid.uuid4().hex)

def live(info):
    return bool(info and info.get('start_time') and token(info.get('pid'))==info['start_time'])

def lock_path(kind):return paths()['run'].with_name(NAME+'.c24_lif_v1.'+kind+'.lock')

def acquire(kind,reservation=None):
    path=lock_path(kind);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=read_json(path/'owner.json')
        require(old and old.get('worktree')==str(ROOT.resolve()) and old.get('session')==SESSION,'Ambiguous lock owner preserved: '+str(path))
        require(not live(old) and not tmux_active(),'Existing task owner/session protected: '+str(old))
        archive=path.with_name(path.name+'.stale.'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        path.rename(archive);write_json(archive/'recovery.json',dict(reason='PID/start-time no longer live; session absent',observed=owner()))
    path.mkdir(exist_ok=False);info=reservation or owner();write_json(path/'owner.json',info);return info

def release(kind,info):
    path=lock_path(kind)
    if path.is_dir() and read_json(path/'owner.json',{}).get('token')==info['token']:
        (path/'owner.json').unlink();path.rmdir()

def state(status,**details):
    p=paths();write_json(p['launch']/'state.json',dict(status=status,commit=git('rev-parse','HEAD'),
        updated=datetime.now(timezone.utc).isoformat(),run=str(p['run']),console=str(p['launch']/'console.log'),
        preflight_log=str(p['launch']/'preflight.log'),**details))

def run_state():
    p=paths();record=read_json(p['launch']/'state.json',{});status=record.get('status','NOT_STARTED')
    exit=read_json(p['launch']/'exit_code.json',{})
    shell=p['launch']/'process_exit_code.txt'
    shell_code=int(shell.read_text().strip()) if shell.is_file() else None
    if exit.get('exit_code') not in (None,0) or shell_code not in (None,0):return 'FAILED'
    if status=='SUCCESS':
        return 'SUCCESS' if exit.get('exit_code')==shell_code==0 and all((p['run']/n).is_file() for n in ['weights/best.pt','weights/last.pt','results.csv']) else 'REQUIRES_REVIEW'
    if status=='CHECKING' and not live(read_json(lock_path('preflight')/'owner.json',{})):return 'REQUIRES_REVIEW'
    if status in ('DISPATCHED','RUNNING') and not live(read_json(lock_path('worker')/'owner.json',{})) and not tmux_active():return 'FAILED'
    return status

def verify_delivery():
    info=read_json(ROOT/'outputs/c24_lif_v1_delivery.json');require(info,'Run fixed-SHA sync first')
    require(info['branch']==BRANCH and info['commit']==git('rev-parse','HEAD'),'Delivery SHA/branch changed')
    require(Path(info['worktree']).resolve()==ROOT.resolve() and Path(info['main']).resolve()==MAIN.resolve(),'Delivery roots changed')
    require((ROOT/'.git').is_file() and ROOT.resolve()!=MAIN.resolve(),'Independent linked worktree required')
    require(not git('status','--porcelain','--untracked-files=no'),'Tracked source dirty; preserved')
    require(info['fingerprint']==fingerprint(),'Source/config fingerprint changed')
    return info

def recipe():
    from init_c24_lif_v1 import YAML
    p=paths();actual=YAML.load(p['c2_args']);expected=YAML.load(ROOT/'docs/c24_lif_v1/c2_args.yaml')
    require(len(actual)==109 and actual.keys()==expected.keys(),'109-field C2 recipe required')
    require(all(type(actual[k]) is type(expected[k]) and actual[k]==expected[k] for k in expected),'C2 recipe/value types changed')
    result=dict(actual,model=str(p['init']),name=NAME,save_dir=str(p['run']))
    require(result['data']==str(p['data']) and result['project']==str(MAIN/'runs/c_series'),'Formal data/project root changed')
    rows=[dict(field=k,C2=actual[k],candidate=result[k],type=type(result[k]).__name__,changed=actual[k]!=result[k]) for k in actual]
    require({r['field'] for r in rows if r['changed']}=={'model','name','save_dir'},'Recipe mismatch')
    return result,rows

def data_manifest(root):
    import hashlib,json
    root=Path(root).resolve();result={}
    for split in ['train','val','test']:
        images=sorted(p for p in (root/'images'/split).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
        require(images,'Empty split '+split);names=[];labels=[];instances=0
        for image in images:
            label=root/'labels'/split/image.relative_to(root/'images'/split).with_suffix('.txt')
            require(label.is_file(),'Missing label '+str(label));data=label.read_bytes();rows=[r.split() for r in data.decode().splitlines() if r.strip()]
            require(all(len(r)==5 and r[0]=='0' for r in rows),'Invalid label')
            names.append(image.relative_to(root).as_posix());labels.append([label.relative_to(root).as_posix(),hashlib.sha256(data).hexdigest()]);instances+=len(rows)
        result[split]=dict(images=len(images),instances=instances,split_paths_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
            label_contents_sha256=hashlib.sha256(json.dumps(labels,separators=(',',':')).encode()).hexdigest())
    require((result['val']['images'],result['val']['instances'])==(1728,12840),'Val counts changed')
    require((result['test']['images'],result['test']['instances'])==(864,6663),'Test counts changed')
    return dict(root=str(root),splits=result)

def verify_data():
    from init_c24_lif_v1 import YAML
    p=paths();actual=YAML.load(p['data']);expected=YAML.load(ROOT/'docs/c24_lif_v1/c2_data.yaml')
    require(actual==expected and actual['path']==str(MAIN/'datasets/crack_det'),'Data config changed')
    require(sha256(p['data'])=='ea2a922586a03526e1b4c8e64c02b2c71dce1a458f1c140f038afa744d68c6aa','Parent data byte identity changed')
    manifest=data_manifest(actual['path']);expected_manifest=read_json(ROOT/'docs/c24_lif_v1/data_manifest.json')
    require(expected_manifest and manifest['splits']==expected_manifest['splits'],'Split paths/labels changed')
    return manifest

def real_capacity_batch():
    import torch
    from init_c24_lif_v1 import RTDETRTrainer
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.torch_utils import init_seeds
    args,_=recipe();manifest=verify_data();trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args=get_cfg(overrides=args);trainer.data=check_det_dataset(args['data'],autodownload=False);trainer.stride=32
    trainer.device=torch.device('cuda:0');trainer.model=None
    init_seeds(42,deterministic=True)
    dataset=trainer.build_dataset(trainer.data['train'],mode='train',batch=16)
    # Exactly one native augmented real-data batch, never a complete data-loader pass.
    loader=torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,collate_fn=dataset.collate_fn)
    batch=trainer.preprocess_batch(next(iter(loader)))
    require(tuple(batch['img'].shape)==(16,3,640,640),'Real capacity shape changed')
    evidence=dict(manifest=manifest,images=batch['im_file'],batch=16,imgsz=640,augment=True,diagnostic_workers=0,formal_workers=8)
    return batch,evidence

def require_preflight(report):
    from c24_lif_v1_numerics import SCHEMA,CONTINUOUS,SELECTED
    required=['negative_gate','lif_original_unit','initialization','structure','history','degeneration_cpu','fusion_cpu_fp32','native_loss_cpu','ingress_cpu',
        'degeneration_cuda','fusion_cuda_fp32','fusion_cuda_amp','fusion_cuda_half','native_loss_cuda','ingress_cuda','server_B16_640']
    require(report.get('schema')==SCHEMA and report.get('status')=='PASSED','Preflight not accepted')
    rows=report.get('stages',[])
    require([s.get('name') for s in rows]==required,'Incomplete/duplicate preflight stage schema')
    for stage in rows:
        require(stage.get('status')=='PASSED' and stage.get('result'),'Empty/unaccepted stage '+stage['name'])
        result=stage['result']
        if stage['name'].startswith('fusion_'):
            require(result.get('lif_bn_state_exact') and result.get('repeat_fuse')=='EXACT' and result.get('physical_negatives')=='BLOCKED_AS_EXPECTED','Missing fusion protections')
            require([case.get('input') for case in result.get('cases',[])]==[[1,3,640,640],[1,3,160,192]],'Missing required fusion shapes')
            for case in result['cases']+[result.get('save_load',{})]:
                require(case.get('schema')==SCHEMA and case.get('status')==case.get('operator_status')=='PASSED' and
                    case.get('natural_relation') in ('IDENTICAL','PERMUTATION'),'Candidate evidence not accepted')
                for field,keys in [('continuous',CONTINUOUS),('id_aligned',SELECTED)]:
                    values=case.get(field,[]);require([v.get('key') for v in values]==list(keys),'Incomplete numerical key schema')
                    require(all(v.get('status')=='PASSED' and (v.get('finite') is True or v.get('exact') is True) for v in values),'Invalid numerical evidence')
                require(case.get('candidate_ids_a') and case.get('candidate_ids_b'),'Candidate IDs absent')
                require(all(len(ids)==len(set(ids))==300 and all(type(i) is int and i>=0 for i in ids)
                    for key in ('candidate_ids_a','candidate_ids_b') for ids in case[key]),'Malformed candidate evidence')
        if stage['name'].startswith('native_loss_') or stage['name']=='server_B16_640':
            require(result.get('steps') and result.get('coverage')==336 and len(result.get('new_tensors',[]))==10 and
                    result.get('save_load_model_optimizer')=='EXACT','Incomplete optimizer/loss evidence')
            if stage['name']=='server_B16_640':require(result.get('input')==[16,3,640,640] and all(s['amp'] for s in result['steps']),'Formal capacity missing')
    require(report.get('server_B16_640',{}).get('status')=='PASSED','No B16/640 result')
    return True

def preflight_once():
    from init_c24_lif_v1 import initialize,runtime,YAML,torch
    from check_c24_lif_v1 import SCHEMA
    p=paths();verify_delivery();require(torch.cuda.is_available(),'Server CUDA required')
    require(os.environ.get('CONDA_DEFAULT_ENV')=='rtdetr','Use existing rtdetr environment')
    require(not p['run'].exists(),'Existing formal results protected')
    args,rows=recipe();manifest=verify_data();p['launch'].mkdir(parents=True,exist_ok=True)
    for src,name in [(p['c2_args'],'authoritative_c2_args.yaml'),(p['data'],'data_config.yaml')]:shutil.copyfile(src,p['launch']/name)
    environment=runtime();environment['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
    write_json(p['launch']/'environment.json',environment);write_json(p['launch']/'data_manifest.json',manifest)
    (p['launch']/'pip_freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    write_json(p['launch']/'initialization.json',initialize(p['source'],p['init']))
    YAML.save(p['launch']/'train_args.yaml',args);write_json(p['launch']/'parameter_diff.json',rows)
    folder=p['launch']/'preflight'
    if folder.exists():folder.rename(p['launch']/('preflight_previous_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    command=[sys.executable,'-u',str(ROOT/'tools/check_c24_lif_v1.py'),'--source',str(p['source']),'--output',str(folder),'--server']
    log=p['launch']/'preflight.log';print('PREFLIGHT LOG '+str(log.resolve()),flush=True)
    with log.open('w',encoding='utf-8') as f:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=ROOT)
        for line in process.stdout:sys.stdout.write(line);sys.stdout.flush();f.write(line);f.flush()
        code=process.wait()
    report=read_json(folder/'preflight.json',{})
    require(code==0 and report.get('schema')==SCHEMA and report.get('status')=='PASSED','Preflight blocked/requires review; training not dispatched')
    require_preflight(report)
    require(report['fingerprint']==fingerprint() and report['source_sha256']==SOURCE_SHA256 and report['server_B16_640']['status']=='PASSED','Stale/capacity preflight')
    return dict(args=args,runtime=environment,fingerprint=fingerprint(),init_sha256=sha256(p['init']),data_sha256=sha256(p['data']),
        recipe_sha256=sha256(p['c2_args']),preflight_sha256=sha256(folder/'preflight.json'),command=command)

def start_direct(check_only=False):
    require(os.name=='posix','AutoDL lifecycle requires Linux; no local formal training')
    verify_delivery();p=paths();require(not tmux_active(),'Existing tmux protected');require(not p['run'].exists(),'Existing results protected')
    info=acquire('preflight');reservation=None
    try:
        # Existing active worker in any state is protected before capacity allocation.
        require(not live(read_json(lock_path('worker')/'owner.json',{})),'Existing worker protected')
        # Preserve an earlier failed attempt, including its logs and exit codes, so a
        # new attempt cannot inherit SUCCESS/FAILED from an old process.
        if p['launch'].exists():
            prior=[f for f in p['launch'].iterdir() if not f.name.startswith('previous_launch_')]
            if prior:
                archive=p['launch']/('previous_launch_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));archive.mkdir()
                for item in prior:item.rename(archive/item.name)
        state('CHECKING',owner=info);plan=preflight_once()
        if check_only:state('NOT_STARTED',preflight='PASSED',note='Optional preflight-only; start-direct runs one fresh gate');return
        require(shutil.which('tmux'),'tmux unavailable')
        reservation=acquire('worker');plan.update(reservation=reservation,session=SESSION)
        write_json(p['launch']/'plan.json',plan)
        shell=p['launch']/'worker.sh'
        shell.write_bytes(('#!/usr/bin/env bash\nset -uo pipefail\n'
            "C24_WORKER_ROOT="+shlex.quote(str(ROOT))+"\n"
            "C24_WORKER_LOG="+shlex.quote(str(p['launch']/'console.log'))+"\n"
            "bash \"$C24_WORKER_ROOT/tools/autodl_c24_lif_v1.sh\" _worker "+shlex.quote(reservation['token'])+" >> \"$C24_WORKER_LOG\" 2>&1\n"
            'rc=$?\nprintf \'%s\\n\' "$rc" > '+shlex.quote(str(p['launch']/'process_exit_code.txt'))+'\nexit "$rc"\n').encode())
        state('DISPATCHED',owner=reservation)
        subprocess.run(['tmux','new-session','-d','-s',SESSION,'bash '+shlex.quote(str(shell))],check=True)
        print('DISPATCHED; first completed training batch will change state to RUNNING.\nCONSOLE '+str(p['launch']/'console.log'))
    except BaseException as error:
        state('REQUIRES_REVIEW' if 'Preflight' in str(error) else 'FAILED',error=repr(error))
        if reservation and not tmux_active():release('worker',reservation)
        raise
    finally:release('preflight',info)

def worker(reservation_token):
    from init_c24_lif_v1 import RTDETR,RTDETRTrainer,torch,audited_rebuild,verify_model,YAML,runtime
    from train_lif_down import disable_oom_retry,ensure_amp_resources
    from check_c24_lif_v1 import optimizer_check
    p=paths();code=1;info=None
    try:
        plan=read_json(p['launch']/'plan.json');require(plan and plan['reservation']['token']==reservation_token,'Worker reservation mismatch')
        old=read_json(lock_path('worker')/'owner.json');require(old and old['token']==reservation_token,'Worker lock mismatch')
        info=owner();info['token']=reservation_token;write_json(lock_path('worker')/'owner.json',info)
        verify_delivery();require(plan['fingerprint']==fingerprint(),'Worker source changed')
        require(not p['run'].exists(),'Results appeared after dispatch; protected')
        require(sha256(p['init'])==plan['init_sha256'] and sha256(p['data'])==plan['data_sha256'] and sha256(p['c2_args'])==plan['recipe_sha256'],'Worker inputs changed')
        require(sha256(p['launch']/'preflight/preflight.json')==plan['preflight_sha256'],'Preflight changed')
        require_preflight(read_json(p['launch']/'preflight/preflight.json'))
        write_json(p['launch']/'worker_environment.json',runtime())
        ensure_amp_resources(MAIN,p['launch']/'amp_resources.json')
        class RecordingTrainer(RTDETRTrainer):
            def get_model(self,cfg=None,weights=None,verbose=True):
                model,audit=audited_rebuild(weights);write_json(p['launch']/'nc1_loading.json',audit);return model
        model=RTDETR(str(p['init']));model.add_callback('on_train_batch_start',disable_oom_retry)
        def setup(trainer):
            verify_model(trainer.model,zero=True);require(bool(trainer.amp),'Native AMP disabled; refuse recipe change')
            actual=vars(trainer.args);differences={k:[v,actual.get(k)] for k,v in plan['args'].items() if type(v) is not type(actual.get(k)) or v!=actual.get(k)}
            require(not differences,'Actual recipe changed '+str(differences))
            YAML.save(p['launch']/'actual_train_args.yaml',actual)
            ids=[id(v) for g in trainer.optimizer.param_groups for v in g['params']]
            names={id(v):n for n,v in trainer.model.named_parameters() if v.requires_grad}
            require(len(ids)==len(set(ids))==len(names) and set(ids)==set(names),'Actual optimizer missing/duplicated tensors')
            require(type(trainer.optimizer) is torch.optim.AdamW,'Actual optimizer changed')
            write_json(p['launch']/'training_setup.json',dict(amp=bool(trainer.amp),parameters=sum(v.numel() for v in trainer.model.parameters()),
                groups=[dict(group=g.get('param_group'),lr=g['lr'],weight_decay=g['weight_decay'],names=[names[id(v)] for v in g['params']]) for g in trainer.optimizer.param_groups],recipe_differences=differences))
        model.add_callback('on_train_start',setup)
        def batch_finished(trainer):
            if not (p['launch']/'first_batch.json').exists():
                write_json(p['launch']/'first_batch.json',dict(epoch=trainer.epoch,pid=os.getpid(),owner=info));state('RUNNING',owner=info)
        model.add_callback('on_train_batch_end',batch_finished)
        model.train(trainer=RecordingTrainer,**plan['args']);code=0
    except BaseException as error:
        if isinstance(error,KeyboardInterrupt):code=130
        elif isinstance(error,SystemExit):code=error.code if isinstance(error.code,int) else 1
        state('FAILED',error=repr(error));traceback.print_exc();raise
    finally:
        write_json(p['launch']/'exit_code.json',dict(exit_code=code,finished=datetime.now(timezone.utc).isoformat()))
        if code==0:state('SUCCESS')
        if info:release('worker',info)

def status():
    p=paths();record=read_json(p['launch']/'state.json',{})
    print('STATE '+run_state()+'\nSHA '+git('rev-parse','HEAD')+'\nRUN '+str(p['run'])+'\nPREFLIGHT '+str(p['launch']/'preflight.log')+'\nCONSOLE '+str(p['launch']/'console.log'))
    report=read_json(p['launch']/'preflight/preflight.json',{})
    print('STAGE '+str(report.get('stages',[{}])[-1].get('name') if report.get('stages') else 'NOT_STARTED'))
    if not (p['launch']/'console.log').is_file():print('未派发训练：console.log 尚未生成。')
    print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['start-direct','preflight-only','status','_worker']);parser.add_argument('token',nargs='?')
    a=parser.parse_args()
    if a.mode=='status':require(a.token is None,'Unexpected argument');status()
    elif a.mode=='_worker':require(a.token,'Worker token required');worker(a.token)
    else:require(a.token is None,'Unexpected argument');start_direct(a.mode=='preflight-only')
