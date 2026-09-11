"""Isolated per-variant lifecycle with pinned source and mandatory bounded preflight."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
import uuid
import tempfile
import torch
from sci_adapter import *
from init_sci_adapter import initialize, rebuild_audit
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ASSETS

MAIN=Path(os.environ.get('SCI_ADAPTER_MAIN','/root/autodl-tmp/projects/Crack_RTDETR'))


def paths(variant):
    v=VARIANTS[variant]
    return dict(variant=variant,name=v['name'],run=MAIN/'runs/c_series'/v['name'],
        launch=ROOT/v['log_dir'],init=ROOT/'weights'/f'{variant}_controlled_init.pt',
        source=MAIN/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt',
        c2_args=MAIN/'runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml',
        session=f'sci-adapter-{variant}',lock=MAIN/'runs/c_series'/f'{v["name"]}.sci-adapter.lock')


def recipe(c2_path,variant,init):
    source=YAML.load(c2_path);expected=YAML.load(ROOT/'docs/sci_adapter/c2_args.yaml')
    require(set(source)==set(expected),'C2 recipe fields changed')
    require(all(type(source[k]) is type(expected[k]) and source[k]==expected[k] for k in source),'C2 recipe value/type drift')
    target=dict(source);target.update(model=str(Path(init).resolve()),name=VARIANTS[variant]['name'],
        save_dir=str(Path(source['project'])/VARIANTS[variant]['name']))
    rows=[dict(field=k,C2=source[k],variant=target[k],type=type(source[k]).__name__,changed=source[k]!=target[k]) for k in sorted(source)]
    require({r['field'] for r in rows if r['changed']}=={'model','name','save_dir'},'Unexpected recipe changes')
    return target,rows


def process_token(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,ValueError,IndexError):return None


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).is_file() else {}


def artifacts_complete(run):
    return all((run/f).is_file() and (run/f).stat().st_size>0 for f in
               ('weights/best.pt','weights/last.pt','results.csv','args.yaml','results.png'))


def run_state(p):
    launch=read_json(p['launch']/'launch_state.json');process=read_json(p['launch']/'process.json')
    result=read_json(p['launch']/'exit_code.json');complete=read_json(p['launch']/'worker_complete.json')
    shell=p['launch']/'process_exit_code.txt';shell_code=int(shell.read_text()) if shell.is_file() else None
    if launch.get('status')=='FAILED' or result.get('exit_code',0)!=0 or (shell_code is not None and shell_code!=0):return 'FAILED'
    if result.get('exit_code')==shell_code==0:
        token=launch.get('token')
        return 'SUCCESS' if token and complete.get('token')==process.get('token')==result.get('token')==token and complete.get('artifacts_complete') and artifacts_complete(p['run']) else 'FAILED'
    if process:
        current=process_token(process.get('pid',-1))
        if current and current==process.get('process_token'):return 'RUNNING'
        # The shell can still be recording its exit after Python completion.
        if result.get('exit_code')==0 and complete.get('artifacts_complete') and shell_code is None:
            active=bool(shutil.which('tmux')) and subprocess.run(['tmux','has-session','-t',p['session']],capture_output=True).returncode==0
            return 'DISPATCHED' if active else 'FAILED'
        return 'FAILED'
    if launch:
        current=process_token(launch.get('pid',-1))
        active=bool(current and current==launch.get('process_token'))
        if launch.get('status')=='DISPATCHED' and shutil.which('tmux'):
            active=subprocess.run(['tmux','has-session','-t',p['session']],capture_output=True).returncode==0
        return 'DISPATCHED' if active else 'FAILED'
    return 'NOT_STARTED'


def verify_environment(info):
    require(os.environ.get('CONDA_DEFAULT_ENV')=='rtdetr','Activate existing rtdetr environment')
    require(Path(sys.executable).parent.name=='bin' and 'rtdetr' in Path(sys.executable).parts,'Wrong Python environment')
    require(torch.cuda.is_available(),'CUDA is required for AutoDL training')
    require(info['torch'].startswith('2.1.2') and info['cuda']=='12.1','Expected existing AutoDL PyTorch2.1.2/CUDA12.1; do not install/upgrade')
    require(not info['dirty'],'Commit/review source changes before launch')
    sync=read_json(ROOT/'outputs/sci_adapter_sync.json')
    require(sync.get('sha')==info['commit'],'Run fixed-SHA sync first; source pin missing/mismatched')
    git('merge-base','--is-ancestor',PARENT_COMMIT,info['commit'])


def amp_resources():
    """Use existing project resources; no package/environment/network installation."""
    records=[]
    for dest,candidates in [(ASSETS/'bus.jpg',[MAIN/'ultralytics-main/ultralytics/assets/bus.jpg',MAIN/'bus.jpg']),
                            (ROOT/'yolo26n.pt',[MAIN/'yolo26n.pt',MAIN/'weights/yolo26n.pt'])]:
        if not dest.is_file():
            source=next((p for p in candidates if p.is_file()),None)
            require(source is not None,f'Existing native AMP resource missing: {dest.name}')
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Publish only complete bytes; parallel variants must never observe a half-written resource.
            with tempfile.NamedTemporaryFile(dir=dest.parent,prefix='.sci_adapter_amp_',delete=False) as tmp:
                temporary=Path(tmp.name)
                with source.open('rb') as src:shutil.copyfileobj(src,tmp)
            try:
                try:os.link(temporary,dest)
                except FileExistsError:pass
            finally:temporary.unlink()
            require(sha256(dest)==sha256(source),'Concurrent AMP resource mismatch')
        records.append(dict(file=str(dest),sha256=sha256(dest)))
    return records


def record_source(folder,p,args):
    shutil.copyfile(p['c2_args'],folder/'authoritative_c2_args.yaml')
    shutil.copyfile(args['data'],folder/'data_config.yaml')
    (folder/'pip_freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(folder/'source_snapshot.tar.gz'),'HEAD',
        'tools','ultralytics-main/ultralytics','ultralytics-main/tests','configs','docs/sci_adapter','.gitattributes'],cwd=ROOT,check=True)
    (folder/'source_from_base.patch').write_text(git('diff','--binary',BASE_COMMIT,'HEAD'),encoding='utf-8')
    write_json(folder/'source_record.json',dict(runtime=runtime(),snapshot_sha256=sha256(folder/'source_snapshot.tar.gz'),
        c2_commit=BASE_COMMIT,c2_args_sha256=sha256(p['c2_args']),data_sha256=sha256(args['data'])))


def start_direct(variant):
    p=paths(variant);info=runtime();verify_environment(info)
    require(ROOT.resolve()!=MAIN.resolve(),'Use the independent SCI Adapter worktree')
    require(shutil.which('tmux'),'tmux required')
    require(not p['run'].exists() and not p['launch'].exists() and not p['init'].exists(),'Existing variant artifacts protected')
    require(subprocess.run(['tmux','has-session','-t',p['session']],capture_output=True).returncode!=0,'Variant tmux already exists')
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=proc.read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        require(not (any(a.endswith('train_sci_adapter.py') for a in argv) and 'worker' in argv and variant in argv),'Same variant worker exists')
    p['lock'].parent.mkdir(parents=True,exist_ok=True);p['lock'].mkdir(exist_ok=False)
    token=uuid.uuid4().hex
    write_json(p['lock']/'owner.json',dict(variant=variant,token=token,worktree=ROOT,runtime=info))
    p['launch'].mkdir(parents=True,exist_ok=False)
    state=dict(status='INITIALIZING',pid=os.getpid(),process_token=process_token(os.getpid()),token=token,session=p['session'])
    write_json(p['launch']/'launch_state.json',state)
    try:
        args,diff=recipe(p['c2_args'],variant,p['init'])
        require(Path(args['save_dir'])==p['run'],'C2 project differs from expected MAIN root')
        require(YAML.load(args['data'])==YAML.load(ROOT/'docs/sci_adapter/c2_data.yaml'),'Dataset/split configuration differs from archived C2')
        data=check_det_dataset(args['data'],autodownload=False);require(data['nc']==1,'Expected nc=1 dataset')
        for split in ('train','val','test'):
            locations=data.get(split);require(locations,f'Missing {split} split')
            require(all(Path(x).exists() for x in (locations if isinstance(locations,list) else [locations])),f'Missing {split} paths')
        write_json(p['launch']/'amp_resources.json',amp_resources())
        report=initialize(p['source'],p['init'],variant);write_json(p['launch']/'initialization.json',report)
        from audit_sci_adapter import run as source_audit
        write_json(p['launch']/'source_audit.json',source_audit())
        from check_sci_adapter import smoke, precision
        from check_sci_adapter_gradients import run_checks
        target=RTDETR(str(p['init'])).model
        preflight=dict(status='passed',gradient_edge=run_checks(),AMP_loss_DN=smoke(target,variant,'cuda',True),
            precision=precision(target),topology=topology(target,variant),parameters=VARIANTS[variant]['parameters'],runtime=runtime())
        write_json(p['launch']/'preflight.json',preflight)
        del target
        record_source(p['launch'],p,args)
        shutil.copyfile(MODEL_DIR/VARIANTS[variant]['yaml'],p['launch']/'model.yaml')
        YAML.save(p['launch']/'train_args.yaml',args);write_json(p['launch']/'parameter_diff.json',diff)
        plan=dict(variant=variant,token=token,runtime=info,args=args,init_sha256=sha256(p['init']),
            c2_args_sha256=sha256(p['c2_args']),session=p['session'],preflight='passed',created=datetime.now(timezone.utc).isoformat())
        write_json(p['launch']/'plan.json',plan)
        worker=p['launch']/'worker.sh';console=p['launch']/'console.log'
        command=[sys.executable,'-u',str(ROOT/'tools/train_sci_adapter.py'),'worker',variant,'--token',token]
        env={k:os.environ.get(k) for k in ('PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','OMP_NUM_THREADS','MKL_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','CONDA_DEFAULT_ENV')}
        env.update(PYTHONPATH=str(ROOT/'ultralytics-main'),YOLO_AUTOINSTALL='false',SCI_ADAPTER_MAIN=str(MAIN))
        exports=''.join(('unset '+k if v is None else 'export '+k+'='+shlex.quote(v))+'\n' for k,v in env.items())
        worker.write_text('#!/usr/bin/env bash\nset -uo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+exports+
            shlex.join(command)+' >> '+shlex.quote(str(console))+' 2>&1\nrc=$?\nprintf \'%s\\n\' "$rc" > '+
            shlex.quote(str(p['launch']/'process_exit_code.txt'))+'\nexit "$rc"\n',encoding='utf-8')
        # Record dispatch BEFORE spawn so a fast worker cannot be overwritten by stale state.
        state['status']='DISPATCHED';write_json(p['launch']/'launch_state.json',state)
        subprocess.run(['tmux','new-session','-d','-s',p['session'],'bash '+shlex.quote(str(worker))],check=True)
        print(f'DISPATCHED {variant}: {console}')
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));write_json(p['launch']/'launch_state.json',state)
        raise


def disable_oom_retry(trainer):
    trainer._oom_retries=3  # original trainer retries by reducing batch; formal C2 batch must remain 16


def worker(variant,token):
    p=paths(variant);code=1;claimed=False
    try:
        plan=read_json(p['launch']/'plan.json');require(plan['token']==token,'Wrong worker token')
        require(runtime()['commit']==plan['runtime']['commit'] and not runtime()['dirty'],'Source changed after dispatch')
        require(sha256(p['init'])==plan['init_sha256'],'Initialization changed')
        frozen=read_json(p['launch']/'source_record.json')
        require(sha256(plan['args']['data'])==frozen['data_sha256'] and sha256(p['c2_args'])==plan['c2_args_sha256'],'Data/recipe changed')
        require(not p['run'].exists(),'Run directory appeared after reservation')
        # Exclusive claim prevents direct duplicate worker invocation with the same token.
        with (p['launch']/'worker_claim.json').open('x',encoding='utf-8') as f:json.dump(dict(token=token,pid=os.getpid()),f)
        claimed=True
        write_json(p['launch']/'process.json',dict(pid=os.getpid(),process_token=process_token(os.getpid()),token=token))
        class RecordingTrainer(RTDETRTrainer):
            def get_model(self,cfg=None,weights=None,verbose=True):
                m=super().get_model(cfg,weights,verbose)
                write_json(p['launch']/'nc1_loading.json',rebuild_audit(weights,m,variant));return m
        model=RTDETR(str(p['init']));model.add_callback('on_train_batch_start',disable_oom_retry)
        def setup(trainer):
            topology(trainer.model,variant);verify_zero(trainer.model)
            require(bool(trainer.amp),'Native AMP check disabled AMP; refusing recipe drift')
            actual=vars(trainer.args);diff={k:[v,actual.get(k)] for k,v in plan['args'].items() if type(v)!=type(actual.get(k)) or v!=actual.get(k)}
            require(not diff,f'Actual recipe drift: {diff}')
            ids=[id(p) for g in trainer.optimizer.param_groups for p in g['params']]
            require(type(trainer.optimizer) is torch.optim.AdamW and all(ids.count(id(p))==1 for p in trainer.model.parameters()),'Optimizer drift/coverage gap')
            YAML.save(p['launch']/'actual_train_args.yaml',actual)
            names={id(p):n for n,p in trainer.model.named_parameters()}
            groups=[dict(decay=g['weight_decay'],group=g.get('param_group'),parameters=[names[id(p)] for p in g['params']]) for g in trainer.optimizer.param_groups]
            write_json(p['launch']/'training_setup.json',dict(amp=True,recipe_diff=diff,optimizer_groups=groups,parameters=VARIANTS[variant]['parameters']))
        model.add_callback('on_train_start',setup)
        from sci_adapter_stats import install
        install(model,p['launch'])
        model.train(trainer=RecordingTrainer,**plan['args'])
        require(artifacts_complete(p['run']),'Training returned without complete artifacts')
        write_json(p['launch']/'worker_complete.json',dict(token=token,artifacts_complete=True,
            best_sha256=sha256(p['run']/'weights/best.pt'),last_sha256=sha256(p['run']/'weights/last.pt')))
        code=0
    except BaseException:
        traceback.print_exc();raise
    finally:
        if claimed:write_json(p['launch']/'exit_code.json',dict(exit_code=code,token=token,finished=datetime.now(timezone.utc).isoformat()))


def status(variant):
    p=paths(variant);print(json.dumps(dict(variant=variant,state=run_state(p),run=str(p['run']),log=str(p['launch']/'console.log')),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('start-direct','worker','status'))
    p.add_argument('variant',choices=VARIANTS);p.add_argument('--token')
    a=p.parse_args();torch.set_num_threads(4)
    if a.mode=='worker':worker(a.variant,a.token)
    elif a.mode=='start-direct':start_direct(a.variant)
    else:status(a.variant)
