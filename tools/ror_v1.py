"""ROR v1 分阶段入口：prepare/preflight/diagnose 均不启动训练；仅 start/resume 显式启动。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import io
import traceback

import torch
from init_c19_lif_v1 import (ROOT, SOURCE_SHA256, initialize, runtime, require, sha256, write_json,
                             verify_model, build_training_model)
from c19_lif_v1_data import dataset_inventory
from ultralytics import RTDETR
from ultralytics.utils import YAML, ASSETS
from ultralytics.utils.patches import torch_load
from ultralytics.models.utils.ror import CONFIG
from ror_v1_training import RORTrainer, install
from ror_v1_diagnose import diagnose, TRAINED_SHA

BASE = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
BRANCH = 'exp-rtdetr-r18-lite-ror-v1'
RUN = 'ror_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug'
SESSION = 'ror-v1-training'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def info():
    result=runtime()
    import ultralytics.models.utils.loss as loss_module
    result.update(loss_file=loss_module.__file__,base_commit=BASE,branch=git('branch','--show-current'))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def environment(server=False):
    result=info()
    expected=('3.10.13','2.1.2+cu121') if server else ('3.9.25','2.7.1+cu118')
    actual=(platform.python_version(),str(torch.__version__))
    require(actual==expected, f'环境不同：实际 {actual}，记录 {expected}。未升级；须先审查兼容影响。')
    if server: require(torch.cuda.is_available(),'Formal server CUDA unavailable')
    return result


def paths(args):
    main=args.main.resolve()
    return dict(main=main,output=ROOT/'outputs/ror_v1',init=ROOT/'weights/ror_v1_controlled_init.pt',
                source=main/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt',
                data=args.data or main/'configs/crack_autodl.yaml',dataset=main/'datasets/crack_det',
                c2=args.c2_args or main/'runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml',
                run=main/'runs/c_series'/RUN)


def verify_data(p):
    actual=YAML.load(p['data']);expected=YAML.load(ROOT/'docs/c19_lif_v1/c2_data.yaml')
    expected['path']=str(p['dataset'].resolve())
    actual['path']=str(Path(actual['path']).resolve())
    require(actual==expected,'Data config split/path/class semantics differ')
    inventory=dataset_inventory(p['dataset'])
    require(inventory==read(ROOT/'docs/ror_v1/parent_dataset_inventory.json'),
            'Data identity differs from successful parent (paths/labels/counts); no re-splitting')
    return inventory


def recipe(p):
    source=YAML.load(p['c2']);expected=YAML.load(ROOT/'docs/c19_lif_v1/c2_args.yaml')
    require(len(source)==len(expected)==109 and set(source)==set(expected),'Incomplete 109-field C2 recipe')
    require(all(type(source[k]) is type(v) and source[k]==v for k,v in expected.items()),'Authoritative C2 recipe differs')
    result=dict(source)
    result.update(model=str(p['init']),data=str(p['data'].resolve()),project=str(p['run'].parent),name=RUN,save_dir=str(p['run']))
    rows=[dict(field=k,parent=source[k],ror=result[k],changed=source[k]!=result[k],
               reason='verified same data, absolute path relocation' if k=='data' else 'experiment model/output identity') for k in source]
    require({r['field'] for r in rows if r['changed']}<={'model','data','project','name','save_dir'},'Unexpected recipe change')
    return result,rows


def prepare(args):
    p=paths(args);p['output'].mkdir(parents=True,exist_ok=True)
    env=environment(server=os.name!='nt')
    require(git('branch','--show-current')==BRANCH,'Wrong ROR branch')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
    require(p['source'].is_file() and sha256(p['source'])==SOURCE_SHA256,'PENDING_ASSET: missing/incorrect unified source')
    inventory=verify_data(p);train_args,rows=recipe(p)
    if p['init'].exists():
        prior=read(p['output']/'initialization.json')
        require(prior['source_sha256']==SOURCE_SHA256 and prior['output_sha256']==sha256(p['init']),'Existing init identity changed')
    else:
        prior=initialize(p['source'],p['init']);write_json(p['output']/'initialization.json',prior)
    identity=dict(config=CONFIG,base=BASE,code=git('rev-parse','HEAD'),run=RUN,source_sha256=SOURCE_SHA256,
                  init_sha256=sha256(p['init']),recipe_sha256=digest_json(train_args),data_sha256=sha256(p['data']),
                  inventory_sha256=digest_json(inventory))
    plan=dict(status='PREPARED',identity=identity,args=train_args,output=str(p['output']),dataset=str(p['dataset']),
              source=str(p['source']),c2_args=str(p['c2']),runtime=env)
    plan_file=p['output']/'plan.json'
    if plan_file.exists(): require(read(plan_file)['identity']==identity,'Existing preparation differs; preserve it and inspect')
    write_json(p['output']/'dataset_inventory.json',inventory);write_json(p['output']/'recipe_diff.json',rows)
    YAML.save(p['output']/'train_args.yaml',train_args);write_json(p['output']/'ror_config.json',CONFIG)
    write_json(plan_file,plan)
    print('PREPARED; training NOT_RUN. '+str(plan_file))


def checked_plan(args, preflight=False):
    p=paths(args);plan=read(p['output']/'plan.json');identity=plan['identity']
    require(identity['code']==git('rev-parse','HEAD') and identity['config']==CONFIG,'Code/config changed since prepare')
    require(not git('status','--porcelain','--untracked-files=no'),'Tracked changes invalidate formal gates')
    require(sha256(p['init'])==identity['init_sha256'] and sha256(p['source'])==SOURCE_SHA256,'Initialization changed')
    require(sha256(p['data'])==identity['data_sha256'],'Data YAML changed')
    require(digest_json(plan['args'])==identity['recipe_sha256'],'Recipe changed')
    current_recipe,_=recipe(p)
    require(current_recipe==plan['args'],'Main root, authoritative C2 args or selected paths changed')
    require(digest_json(verify_data(p))==identity['inventory_sha256'],'Dataset changed')
    if preflight:
        check=read(p['output']/'preflight.json')
        require(check['status']=='PASS' and check['identity']==identity,'Missing/mismatched passed preflight')
        require(check['capacity']['status']=='PASS' and check['capacity']['batch']==16 and check['capacity']['imgsz']==640,
                'Missing B16/640 capacity check')
        require(check['runtime']['executable']==sys.executable and check['runtime']['torch']==str(torch.__version__),
                'Preflight environment differs')
    return p,plan


def amp_assets(p):
    """Reuse local resources needed by the unchanged native AMP self-check, no downloads."""
    rows=[]
    for destination,candidates in ((ASSETS/'bus.jpg',[p['main']/'ultralytics-main/ultralytics/assets/bus.jpg',p['main']/'bus.jpg']),
                                   (ROOT/'yolo26n.pt',[p['main']/'yolo26n.pt',p['main']/'weights/yolo26n.pt'])):
        if not destination.is_file():
            found=next((v for v in candidates if v.is_file()),None)
            require(found is not None,'Missing local native AMP check resource: '+str(destination))
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(found,destination)
        rows.append(dict(path=str(destination),sha256=sha256(destination)))
    return rows


def capacity(p, plan):
    """One isolated native-AMP B16/640 forward/backward; no optimizer update."""
    from ultralytics.cfg import get_cfg
    from ultralytics.models.rtdetr.val import RTDETRDataset
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(42,deterministic=True)
    dataset=RTDETRDataset(img_path=str(p['dataset']/'images/train'),imgsz=640,batch_size=16,augment=True,
                         hyp=get_cfg(overrides=plan['args']),rect=False,cache=False,
                         data=check_det_dataset(str(p['data']),autodownload=False))
    loader=build_dataloader(dataset,batch=16,workers=0,shuffle=False)
    batch=next(iter(loader));batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    batch['img']=batch['img'].float()/255
    initial=RTDETR(str(p['init'])).model
    model,_=build_training_model(initial.yaml,initial,dict(nc=1,channels=3))
    install(model,plan['identity']);model.cuda().train();model.criterion.set_epoch(20)
    free,total=torch.cuda.mem_get_info();torch.cuda.reset_peak_memory_stats()
    with torch.cuda.amp.autocast(enabled=True): loss,shown=model(batch)
    require(torch.isfinite(loss),'Nonfinite native AMP capacity loss')
    loss.backward()  # memory capacity only: no optimizer or artificial scaling overflow
    require(all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None),'Nonfinite capacity gradient')
    torch.cuda.synchronize()
    report=dict(status='PASS',batch=16,imgsz=640,amp=True,optimizer_update=False,loss_scaling=False,
                scope='one native online-augmentation batch, isolated model; no formal initialization mutation',
                loss=float(loss),shown=shown.tolist(),free_before=free,total_memory=total,
                peak_allocated=torch.cuda.max_memory_allocated(),images=batch['im_file'],gt_count=len(batch['bboxes']))
    del model,initial,batch,loader,dataset;torch.cuda.empty_cache()
    return report


def preflight(args):
    p,plan=checked_plan(args)
    report=dict(status='FAIL',identity=plan['identity'],runtime=environment(server=True))
    try:
        from check_ror_v1 import run
        report['cpu_checks']=run(p['source'],'cpu')
        report['checks']=run(p['source'],'cuda:0')
        report['amp_resources']=amp_assets(p)
        from ultralytics.utils.checks import check_amp
        initial=RTDETR(str(p['init'])).model.cuda()
        require(check_amp(initial),'Native AMP equivalence check failed; not disabling AMP')
        del initial;torch.cuda.empty_cache()
        report['capacity']=capacity(p,plan)
        direct=report['checks']['integration']['cuda_direct_step_variation_check']
        report['status']='PASS' if direct is True else 'REVIEW_REQUIRED'
        if direct is not True:
            report['reason']='CUDA direct optimizer-update comparison did not pass its recorded native-variation bound; CPU exactness/Adam reference do not relabel that check as passed.'
    except BaseException as error:
        report['error']=repr(error);raise
    finally: write_json(p['output']/'preflight.json',report)
    print('PREFLIGHT '+report['status']+'; training NOT_RUN')


def parent_weights(p, explicit=None):
    if explicit: return Path(explicit)
    # Archived successful run location; diagnose verifies content hash, never name alone.
    candidate=p['main']/'runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt'
    return candidate


def diagnostic(args):
    p=paths(args);weights=parent_weights(p,args.weights)
    inventory=verify_data(p);output=args.output or p['output']/'diagnose'
    report=diagnose(weights,p['dataset'],output,args.device,args.count,args.batch)
    report.update(code=git('rev-parse','HEAD'),data_sha256=sha256(p['data']),inventory_sha256=digest_json(inventory))
    write_json(output/'diagnosis.json',report)
    return report


def launch(args, resume=False):
    require(os.name!='nt','tmux launcher is for the verified Linux server')
    p,plan=checked_plan(args,preflight=True);environment(server=True)
    require(shutil.which('tmux') is not None,'tmux unavailable')
    require(subprocess.run(['tmux','has-session','-t','='+SESSION],capture_output=True).returncode!=0,'This ROR run already has a session')
    diagnosis=read(p['output']/'diagnose/diagnosis.json')
    require(diagnosis['status']=='OBSERVED_SUPPORT','Diagnosis is missing or NO_OBSERVED_SUPPORT; do not directly invest in long training')
    require(diagnosis['weights_sha256']==TRAINED_SHA,'Unverified diagnostic parent')
    require(diagnosis['code']==plan['identity']['code'] and diagnosis['inventory_sha256']==plan['identity']['inventory_sha256'],
            'Diagnostic code/data identity differs from prepared run')
    last=p['run']/'weights/last.pt'
    if resume:
        require(last.is_file(),'Only this experiment last.pt is resumable')
        ckpt=torch_load(last,map_location='cpu');saved=ckpt.get('ema') or ckpt.get('model')
        require(getattr(saved,'ror_v1',None)==plan['identity'],'Not this ROR checkpoint/config/code')
        require(0<=ckpt['epoch']<199 and ckpt.get('optimizer') is not None,'Finished/stripped checkpoint cannot resume')
    else: require(not p['run'].exists(),'Existing run preserved; use resume for this run')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    launch_dir=p['output']/'launches'/stamp;launch_dir.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(ROOT/'tools/ror_v1.py'),'worker','--main',str(p['main']),
             '--data',str(p['data']),'--c2-args',str(p['c2']),'--launch-dir',str(launch_dir)]
    if resume:command.append('--resume-worker')
    script=launch_dir/'run.sh'
    script.write_text('#!/usr/bin/env bash\nset -uo pipefail\ncd '+shlex.quote(str(ROOT))+ '\n'
                      'export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false\n'+shlex.join(command)+
                      ' >> '+shlex.quote(str(launch_dir/'console.log'))+' 2>&1\nror_rc=$?\nprintf "%s\\n" "$ror_rc" > '+
                      shlex.quote(str(launch_dir/'exit_code.txt'))+'\nexit "$ror_rc"\n',encoding='utf-8')
    subprocess.run(['tmux','new-session','-d','-s',SESSION,'bash '+shlex.quote(str(script))],check=True)
    write_json(launch_dir/'dispatch.json',dict(resume=resume,identity=plan['identity'],command=command,session=SESSION))
    print('DISPATCHED '+SESSION+'; log: '+str(launch_dir/'console.log'))


def worker(args):
    # Kernel file lock serializes only this exact ROR run; unrelated GPU jobs are allowed.
    import fcntl
    p,plan=checked_plan(args,preflight=True);environment(server=True)
    lock_path=p['run'].with_name(p['run'].name+'.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.resume_worker:require(not p['run'].exists(),'Another worker already created this ROR run')
        overrides=dict(plan['args'])
        if args.resume_worker:overrides.update(model=str(p['run']/'weights/last.pt'),resume=str(p['run']/'weights/last.pt'))
        code=1;error_text=None
        try:
            trainer=RORTrainer(overrides=overrides,ror_plan=plan)
            trainer.train();code=0
        except BaseException:
            error_text=traceback.format_exc();traceback.print_exc();raise
        finally:
            write_json(args.launch_dir/'result.json',dict(exit_code=code,identity=plan['identity'],
                       traceback=error_text,finished=datetime.now(timezone.utc).isoformat()))


def evaluate(args, split):
    from c19_lif_v1_results import evaluate as parent_evaluate
    p,plan=checked_plan(args)
    selection=p['output']/'val_selection.json'
    if split=='val':
        weights=(args.weights or p['run']/'weights/best.pt').resolve()
        require(weights.is_file(),'Missing selected weight')
        require(not selection.exists(),'Val choice already frozen; this entry does not scan checkpoints')
    else:
        chosen=read(selection);weights=Path(chosen['weights'])
        require(sha256(weights)==chosen['weights_sha256'] and chosen['code']==git('rev-parse','HEAD'),'Frozen val selection changed')
        require(args.weights is None or args.weights.resolve()==weights,'Test cannot choose another weight')
    model=RTDETR(str(weights)).model
    require(getattr(model,'ror_v1',None)==plan['identity'],'Evaluation checkpoint is not this experiment')
    del model
    output=args.output or p['output']/split
    report=parent_evaluate(weights,p['data'],split,output,device='0',
                           val_report=read(selection)['val_report'] if split=='test' else None)
    if split=='val':
        write_json(selection,dict(weights=str(weights),weights_sha256=sha256(weights),code=git('rev-parse','HEAD'),
                                 val_report=str(output/'metrics.json'),metrics=report,
                                 reason='Original training val fitness selects best.pt; single independent val, then freeze SHA for test'))
    print(split+' completed: '+str(output/'metrics.json'))


def pack(args):
    p=paths(args);p['output'].mkdir(parents=True,exist_ok=True)
    destination=args.output or p['output']/('ror_v1_LIGHT_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.tar.gz')
    require(not destination.exists(),'Preserve previous LIGHT package')
    rows=[];payload=[]
    changed=git('diff','--name-only',BASE,'HEAD').splitlines()
    for rel in changed:
        path=ROOT/rel
        if path.is_file(): payload.append(('source/'+rel,path.read_bytes(),str(path),None))
    for folder,label in ((p['output'],'outputs'),(p['run'],'training')):
        if not folder.exists():continue
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix not in {'.json','.jsonl','.yaml','.csv','.log','.txt','.sh'}:continue
            if path.stat().st_size>2*1024*1024 and path.suffix not in {'.log','.txt','.jsonl'}:continue
            size=path.stat().st_size;tail=path.suffix in {'.log','.txt','.jsonl'}
            with path.open('rb') as stream:
                start=max(0,size-32768) if tail else 0;stream.seek(start);raw=stream.read()
            payload.append((label+'/'+path.relative_to(folder).as_posix(),raw,str(path),dict(start=start,end=size,total=size)))
    for name,raw,origin,span in payload:
        rows.append(dict(member=name,source=origin,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),byte_range=span))
    manifest=dict(code=git('rev-parse','HEAD'),base=BASE,branch=BRANCH,files=rows,
                  scope='LIGHT: no weights/data/full predictions. Logs include at most last 32KiB; explicit byte ranges retained.',
                  formal_weights=[dict(path=str(f),sha256=sha256(f)) for f in (p['run']/'weights/best.pt',p['run']/'weights/last.pt') if f.is_file()])
    with tarfile.open(destination,'w:gz') as archive:
        for name,raw,_,_ in payload+[('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2).encode(),None,None)]:
            entry=tarfile.TarInfo(name);entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
    with tarfile.open(destination,'r:gz') as archive:
        for row in rows:require(hashlib.sha256(archive.extractfile(row['member']).read()).hexdigest()==row['sha256'],'Package verification failed')
    print(json.dumps(dict(path=str(destination),bytes=destination.stat().st_size,sha256=sha256(destination),within_8MiB=destination.stat().st_size<=8*1024*1024)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','preflight','diagnose','start','resume','val','test','pack','worker'))
    parser.add_argument('--main',type=Path,default=Path('/root/autodl-tmp/projects/Crack_RTDETR'),help='已核验的服务器主根；不是另一个实验 worktree')
    parser.add_argument('--data',type=Path,help='等价数据配置；仍核验路径/标签指纹')
    parser.add_argument('--c2-args',type=Path,help='权威完整 C2 args.yaml')
    parser.add_argument('--weights',type=Path,help='diagnose 的已训练母版，或 val 的唯一选定权重')
    parser.add_argument('--output',type=Path,help='诊断/评估目录或 LIGHT 文件')
    parser.add_argument('--device',default='0',help='仅诊断设备；正式训练固定 0')
    parser.add_argument('--count',type=int,default=128,help='每个 train/val split 最多 128 张')
    parser.add_argument('--batch',type=int,default=4,help='仅诊断 batch；正式固定 16')
    parser.add_argument('--launch-dir',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--resume-worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();torch.set_num_threads(4)
    operations=dict(prepare=prepare,preflight=preflight,diagnose=diagnostic,start=lambda a:launch(a,False),
                    resume=lambda a:launch(a,True),val=lambda a:evaluate(a,'val'),test=lambda a:evaluate(a,'test'),pack=pack,worker=worker)
    operations[args.operation](args)


if __name__=='__main__':main()
