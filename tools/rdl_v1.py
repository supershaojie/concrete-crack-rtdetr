"""RDL v1 分阶段执行：prepare/preflight/diagnose/start/resume/val/test/pack；绝不隐式长训。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import traceback

import torch
from init_c19_lif_v1 import ROOT, SOURCE_SHA256, require, sha256, write_json, initialize, source_contract
from init_c19_lif_v1 import verify_model
from ultralytics import RTDETR
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.models.utils.rdl import CONFIG
from rdl_v1_training import RDLTrainer
from c19_lif_v1_data import dataset_inventory, real_batch

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-rdl-v1"
RUN = "rdl_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
OUT = ROOT / "outputs/rdl_v1"
TRAINED_SHA = "24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa"
SESSION = "rdl-v1-training"


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT, text=True).strip()


def runtime(server=False):
    import ultralytics
    from ultralytics.models.utils import loss
    require(Path(ultralytics.__file__).resolve() == ROOT/"ultralytics-main/ultralytics/__init__.py", "Wrong Ultralytics import")
    info = dict(executable=sys.executable, python=platform.python_version(), torch=str(torch.__version__),
                ultralytics=ultralytics.__file__, loss_file=loss.__file__, base_commit=BASE, commit=git("rev-parse","HEAD"),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    info['source_lf_sha256']={name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for name in ('ultralytics-main/ultralytics/models/utils/loss.py','ultralytics-main/ultralytics/models/utils/rdl.py',
                     'ultralytics-main/ultralytics/nn/tasks.py','tools/rdl_v1_training.py')}
    if server:
        require(info["python"] == "3.10.13" and info["torch"] == "2.1.2+cu121" and torch.cuda.is_available(),
                f"Server environment differs from verified recipe; no automatic upgrade: {info}")
    source_contract()
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
    return info


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_identity(paths):
    return {str(Path(p).resolve()):sha256(p) for p in paths}


def verify_files(identity):
    require(all(Path(p).is_file() and sha256(p)==digest for p,digest in identity.items()), "Frozen asset/config changed")


def prepare(args):
    info=runtime(server=not args.local)
    main=args.main.resolve(); source=main/"weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
    data=args.data or main/"configs/crack_autodl.yaml"
    c2=args.c2_args or main/"runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
    require(main != ROOT and (main/".git").is_dir(), "MAIN must be the primary repository, not another worktree")
    require(source.is_file() and sha256(source)==SOURCE_SHA256,"Unified source missing or hash differs")
    expected=YAML.load(ROOT/"docs/c19_lif_v1/c2_args.yaml");actual=YAML.load(c2)
    require(len(expected)==109 and actual.keys()==expected.keys(),"All 109 authoritative recipe fields required")
    differences={k:[expected[k],actual[k]] for k in expected if type(expected[k]) is not type(actual[k]) or expected[k]!=actual[k]}
    require(not differences,f"Authoritative server C2 recipe differs: {differences}")
    config=YAML.load(data);ref=YAML.load(ROOT/"docs/c19_lif_v1/c2_data.yaml")
    dataset=main/"datasets/crack_det"
    require(Path(config['path']).resolve()==dataset.resolve(),"Data root mismatch")
    require({k:v for k,v in config.items() if k!='path'}=={k:v for k,v in ref.items() if k!='path'},"Split/class mapping changed")
    inventory=dataset_inventory(dataset)
    require(inventory==read(ROOT/"docs/rdl_v1/dataset_reference.json"),"Dataset split paths/label identity differs from successful mother")
    OUT.mkdir(parents=True,exist_ok=True)
    init=ROOT/"weights/rdl_v1_controlled_init.pt"
    report_path=OUT/"initialization.json"
    if init.exists():
        require(report_path.is_file() and read(report_path)["output_sha256"]==sha256(init),"Existing init has no matching audit")
    else:
        write_json(report_path,initialize(source,init))
    model=RTDETR(str(init)).model;verify_model(model,zero=True)
    require(model.model[-1].nc==80,"Controlled artifact must retain nc80 until native Trainer adaptation")
    run=main/"runs/c_series"/RUN
    resolved=dict(actual,model=str(init),data=str(data.resolve()),project=str(run.parent),name=RUN,save_dir=str(run))
    rows=[dict(field=k,C2=actual[k],RDL=resolved[k],changed=actual[k]!=resolved[k]) for k in sorted(actual)]
    require({r['field'] for r in rows if r['changed']} <= {'model','data','project','name','save_dir'},"Unexpected recipe changes")
    YAML.save(OUT/"train_args.yaml",resolved);write_json(OUT/"recipe_diff.json",rows)
    write_json(OUT/"dataset_inventory.json",inventory);write_json(OUT/"loss_config.json",CONFIG)
    write_json(OUT/"environment.json",info)
    (OUT/"pip_freeze.txt").write_bytes(subprocess.check_output([sys.executable,"-m","pip","freeze"]))
    identities=file_identity([source,init,data,c2,OUT/"train_args.yaml",OUT/"loss_config.json",OUT/"dataset_inventory.json"])
    write_json(OUT/"prepared.json",dict(status="PASS",runtime=info,local=args.local,main=str(main),run=str(run),
               dataset=str(dataset),source=str(source),init=str(init),data=str(data.resolve()),c2_args=str(c2.resolve()),
               files=identities,config=CONFIG,training_started=False))
    print("prepare PASS; 未启动训练")


def prepared(server=False):
    p=read(OUT/"prepared.json");verify_files(p["files"])
    require(p["config"]==CONFIG and p["runtime"]["commit"]==git("rev-parse","HEAD"),"Prepared code/config identity changed")
    if server:
        require(not p["local"],"Local preparation cannot authorize formal server training")
        info=runtime(server=True)
        require(info['executable']==p['runtime']['executable'],"Interpreter changed")
        require(info['source_lf_sha256']==p['runtime']['source_lf_sha256'],'Prepared source content changed')
        require(not git("status","--porcelain","--untracked-files=no"),"Tracked source has uncommitted changes")
        delivery=read(ROOT/"outputs/rdl_v1_delivery.json")
        require(delivery["commit"]==info["commit"] and delivery["branch"]==BRANCH,"sync delivery mismatch")
    return p


def capacity(p):
    """Exactly one isolated B16/640 native AMP forward/backward, no optimizer update."""
    from rdl_v1_fusion import precision_settings
    precision_before_amp = precision_settings()
    trainer=RDLTrainer.__new__(RDLTrainer);trainer.data=dict(nc=1,channels=3);trainer.resume=False
    init=RTDETR(p['init']).model
    model=trainer.get_model(init.yaml,init,False).cuda().train();model.nc=1;model.rdl_epoch=20
    batch,records=real_batch(Path(p['dataset']),size=640,count=16)
    batch={k:v.cuda() for k,v in batch.items()}
    torch.cuda.reset_peak_memory_stats();free_before,total=torch.cuda.mem_get_info()
    # Reuse the mother's bounded-check scale; formal Trainer default is unchanged.
    scaler=torch.cuda.amp.GradScaler(init_scale=128.)
    optimizer=trainer.build_optimizer(model,name="AdamW",lr=.0005,momentum=.937,decay=.0001)
    with torch.cuda.amp.autocast():
        precision_inside_amp = precision_settings()
        loss,_=model.loss(batch)
    precision_after_amp = precision_settings()
    require(precision_after_amp == precision_before_amp, "Capacity AMP settings were not restored")
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    require(torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None),
            "B16 native AMP nonfinite")
    require(model.criterion.last_stats['active'] and model.criterion.last_stats['M']>0,"Capacity did not exercise active RDL")
    torch.cuda.synchronize()
    result=dict(status="PASS",batch=16,imgsz=640,amp=True,optimizer_update=False,diagnostic_scaler_init=128.,loss=float(loss),
                max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved(),
                free_before=free_before,total=total,samples=records,rdl=model.criterion.last_stats,
                precision_before_amp=precision_before_amp,precision_inside_amp=precision_inside_amp,
                precision_after_amp=precision_after_amp)
    del model,init,trainer,batch,optimizer,loss;torch.cuda.empty_cache()
    return result


def preflight(args):
    from check_rdl_v1 import run_checks
    from check_rdl_v1_ops import run as check_operations
    from rdl_v1_fusion import precision_settings
    p=prepared(server=not args.local)
    report=dict(status="FAIL",runtime=runtime(server=not args.local),prepared_sha256=sha256(OUT/"prepared.json"),local=args.local)
    report_path=OUT/('preflight_local.json' if args.local else 'preflight.json')
    if report_path.exists():
        previous=OUT/(report_path.stem+'_history_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'.json')
        shutil.copy2(report_path,previous)
        report['previous_report']=dict(path=str(previous),sha256=sha256(previous))
    try:
        report['checks']=run_checks(Path(p['source']),args.device)
        require(all(r['status']=='PASS' for r in report['checks'].values()),"Required engineering checks did not pass")
        require(report['checks']['real_model']['fusion_acceptance']['accepted'] is True,
                "Independent fusion acceptance did not pass")
        report['operations']=check_operations()
        if args.local:
            report['capacity']=dict(status="SKIPPED",reason="Formal server B16/640/native AMP capacity remains required")
        else:
            require(args.device=='cuda:0',"Formal native AMP preflight must use original device 0")
            fusion_precision=report['checks']['real_model']['fusion_precision']
            report['precision_before_capacity']=precision_settings()
            require(fusion_precision['restored'] and report['precision_before_capacity']==fusion_precision['before'],
                    "Capacity must run with the restored pre-fusion precision settings")
            report['capacity']=capacity(p)
        report['status']='LOCAL_PASS' if args.local else 'PASS'
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        write_json(report_path,report)


def gate():
    p=prepared(server=True);report=read(OUT/"preflight.json")
    require(report['status']=='PASS' and report['capacity']['status']=='PASS' and not report['local'],"Formal preflight incomplete")
    require(report['prepared_sha256']==sha256(OUT/'prepared.json'),"Preflight belongs to another preparation")
    require(report['runtime']['commit']==git('rev-parse','HEAD'),"Preflight source changed")
    require(dataset_inventory(Path(p['dataset']))==read(OUT/'dataset_inventory.json'),"Data paths/labels changed")
    return p


def check_last(p):
    last=Path(p['run'])/'weights/last.pt'
    require(last.is_file(),"Only this RDL run's last.pt may be resumed")
    ckpt=torch_load(last,map_location='cpu');saved=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt.get('model')
    require(getattr(saved,'rdl_config',None)==CONFIG,"Checkpoint has wrong loss version/config")
    require(0<=ckpt.get('epoch',-1)<199 and getattr(saved,'rdl_epoch',None)==ckpt['epoch'],"Cannot resume finished/stripped/epoch-mismatched checkpoint")
    require(ckpt.get('optimizer') is not None and ckpt.get('scaler') is not None,"Resume state missing")
    expected=YAML.load(OUT/'train_args.yaml');actual=ckpt['train_args']
    # Native resume changes model/resume identity; every recipe field remains locked.
    differences={k:[v,actual.get(k)] for k,v in expected.items() if k not in {'model','resume'} and (type(v) is not type(actual.get(k)) or v!=actual.get(k))}
    require(not differences,f"Checkpoint training recipe differs: {differences}")
    require(getattr(saved,'rdl_code_commit',None)==git('rev-parse','HEAD'),"Checkpoint code identity differs")
    return dict(path=str(last),sha256=sha256(last),completed_epochs=ckpt['epoch']+1)


def dispatch(args,resume=False):
    p=gate();require(shutil.which('tmux'),"tmux unavailable")
    require(subprocess.run(['tmux','has-session','-t','='+SESSION],capture_output=True).returncode!=0,"This RDL session is already active")
    checkpoint=check_last(p) if resume else None
    if not resume: require(not Path(p['run']).exists(),"RDL run exists; use explicit resume for valid last.pt")
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    plan=OUT/f"dispatch_{stamp}.json"
    log=OUT/f"console_{stamp}.log"
    write_json(plan,dict(resume=resume,checkpoint=checkpoint,preflight_sha256=sha256(OUT/'preflight.json'),
                        prepared_sha256=sha256(OUT/'prepared.json'),code=git('rev-parse','HEAD'),log=str(log)))
    worker=OUT/f"worker_{stamp}.sh"
    command=[sys.executable,str(ROOT/'tools/rdl_v1.py'),'worker','--plan',str(plan)]
    worker.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+
                      'source /root/miniconda3/etc/profile.d/conda.sh\nconda activate rtdetr\n'+
                      'cd '+shlex.quote(str(ROOT))+'\nexport PYTHONUNBUFFERED=1\n'+
                      'export PYTHONPATH='+shlex.quote(str(ROOT/'ultralytics-main'))+'\n'+
                      'exec '+shlex.join(command)+' >> '+shlex.quote(str(log))+' 2>&1\n',encoding='utf-8')
    subprocess.run(['tmux','new-session','-d','-s',SESSION,'bash '+shlex.quote(str(worker))],check=True)
    print(f"仅派发本实验 {'resume' if resume else 'start'}；tmux={SESSION}；日志={log}")


def worker(args):
    import fcntl
    plan=read(args.plan);p=prepared(server=True)
    lock=Path(p['run']).with_suffix('.rdl.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(status='FAIL',plan=str(args.plan),pid=os.getpid())
        try:
            gate()
            require(plan['code']==git('rev-parse','HEAD') and plan['preflight_sha256']==sha256(OUT/'preflight.json')
                    and plan['prepared_sha256']==sha256(OUT/'prepared.json'),'Dispatch evidence changed')
            overrides=YAML.load(OUT/'train_args.yaml')
            if plan['resume']:
                require(check_last(p)==plan['checkpoint'],'Resume checkpoint changed after dispatch')
                overrides.update(model=plan['checkpoint']['path'],resume=plan['checkpoint']['path'])
            else:
                require(not Path(p['run']).exists(),'RDL run appeared after dispatch')
            # Native check_amp may use these reference assets; only copy existing originals.
            from ultralytics.utils import ASSETS
            for name,dest,candidates in (
                ('bus.jpg',ASSETS/'bus.jpg',[Path(p['main'])/'ultralytics-main/ultralytics/assets/bus.jpg']),
                ('yolo26n.pt',ROOT/'yolo26n.pt',[Path(p['main'])/'yolo26n.pt',Path(p['main'])/'weights/yolo26n.pt'])):
                if not dest.is_file():
                    existing=next((q for q in candidates if q.is_file()),None)
                    require(existing is not None,f"Native AMP check resource missing: {name}; no substitute downloaded")
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(existing,dest)
            trainer=RDLTrainer(overrides=overrides)
            def setup(t):
                require(bool(t.amp),'Native AMP was disabled; stopping without changing recipe')
                require(type(t.optimizer) is torch.optim.AdamW and all(v.requires_grad for v in t.model.parameters()),'Optimizer/freeze changed')
                diffs={k:[v,vars(t.args).get(k)] for k,v in overrides.items()
                       if type(vars(t.args).get(k)) is not type(v) or vars(t.args).get(k)!=v}
                require(not diffs,f'Actual recipe changed: {diffs}')
                t.model.rdl_code_commit=plan['code'];t.ema.ema.rdl_code_commit=plan['code']
                write_json(OUT/'actual_setup.json',dict(loading=t.rdl_loading_audit,amp=bool(t.amp),args=vars(t.args),
                           added_parameters=0,validation_loss='L0 only',resume=plan['resume'],start_epoch=t.start_epoch))
            trainer.add_callback('on_train_start',setup)
            trainer.train();status['status']='PASS'
        except BaseException as error:
            status['error']=repr(error);traceback.print_exc();raise
        finally:
            write_json(Path(args.plan).with_suffix('.exit.json'),status)


def evaluate(args,split):
    from c19_lif_v1_results import evaluate as native_evaluate
    p=prepared(server=True)
    if split=='val':
        weight=Path(p['run'])/'weights/best.pt'
        output=OUT/'val';require(not (OUT/'selection.json').exists(),'Frozen selection already exists')
        require(weight.is_file(),'Formal best.pt is missing')
        model=RTDETR(str(weight)).model
        require(getattr(model,'rdl_config',None)==CONFIG,'Wrong experiment weights')
        require(getattr(model,'rdl_code_commit',None)==git('rev-parse','HEAD'),'Weight code identity differs')
        report=native_evaluate(weight,p['data'],'val',output,device='0')
        write_json(OUT/'selection.json',dict(weight=str(weight.resolve()),sha256=sha256(weight),code=git('rev-parse','HEAD'),
                   reason='Native training validation fitness selected best.pt; frozen after independent val',
                   val_metrics=str(output/'metrics.json'),val_metrics_sha256=sha256(output/'metrics.json'),
                   metrics={k:report[k] for k in ('mAP50_95','mAP50','AP75','precision','recall')}))
    else:
        selection=read(OUT/'selection.json')
        require(selection['code']==git('rev-parse','HEAD') and selection['sha256']==sha256(selection['weight']), 'Frozen selection changed')
        require(selection['val_metrics_sha256']==sha256(selection['val_metrics']),'Val selection evidence changed')
        native_evaluate(selection['weight'],p['data'],'test',OUT/'test',device='0',val_report=selection['val_metrics'])


def pack(args):
    destination=OUT/('RDL_v1_LIGHT_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'.tar.gz')
    OUT.mkdir(parents=True,exist_ok=True);entries={};excluded=[]
    tracked=git('diff','--name-only',BASE,'HEAD').splitlines()
    for name in tracked:
        path=ROOT/name
        if path.is_file(): entries['source/'+name]=(path.read_bytes(),dict(source=str(path),scope='complete'))
    for folder,prefix in ((ROOT/'docs/rdl_v1','docs'),(OUT,'evidence')):
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix not in {'.json','.jsonl','.yaml','.txt','.log','.csv','.md'}:continue
            size=path.stat().st_size
            is_log=path.suffix in {'.log','.jsonl'}
            with path.open('rb') as stream:
                start=max(0,size-32768) if is_log else 0;stream.seek(start);raw=stream.read()
            if len(raw)>2*1024*1024 and path.name!='pairs.json':
                excluded.append(dict(path=str(path),bytes=len(raw),reason='LIGHT single-file size limit'));continue
            entries[prefix+'/'+path.relative_to(folder).as_posix()]=(raw,dict(source=str(path),byte_start=start,total_bytes=size))
    prepared_path=OUT/'prepared.json'
    if prepared_path.is_file():
        run=Path(read(prepared_path)['run'])
        for name in ('args.yaml','results.csv','rdl_epochs.jsonl'):
            path=run/name
            if path.is_file():
                size=path.stat().st_size;start=max(0,size-32768) if name.endswith('jsonl') else 0
                with path.open('rb') as stream:stream.seek(start);raw=stream.read()
                entries['training/'+name]=(raw,dict(source=str(path),byte_start=start,total_bytes=size))
    manifest={};
    with tarfile.open(destination,'w:gz') as archive:
        for name,(raw,scope) in entries.items():
            member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
            manifest[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),**scope)
        raw=json.dumps(dict(commit=git('rev-parse','HEAD'),branch=BRANCH,files=manifest,excluded=excluded,
                            note='LIGHT: logs/jsonl tail <=32KiB with byte ranges; excludes weights/data/large predictions; errors also in structured exit/preflight reports'),ensure_ascii=False,indent=2).encode()
        member=tarfile.TarInfo('MANIFEST.json');member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
    require(destination.stat().st_size<8*1024*1024,'LIGHT exceeds 8MiB; preserve artifact and inspect')
    with tarfile.open(destination,'r:gz') as archive:
        for name,row in manifest.items():require(hashlib.sha256(archive.extractfile(name).read()).hexdigest()==row['sha256'],'Archive verification failed')
    print(destination,sha256(destination))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='action',required=True)
    p=subs.add_parser('prepare',help='核验109字段配方、统一源、数据指纹并生成初值；不训练')
    p.add_argument('--main',type=Path,default=Path('/root/autodl-tmp/projects/Crack_RTDETR'))
    p.add_argument('--data',type=Path);p.add_argument('--c2-args',type=Path);p.add_argument('--local',action='store_true')
    p=subs.add_parser('preflight',help='有界数学/真实模型/恢复检查，服务器含一次B16/640容量检查')
    p.add_argument('--local',action='store_true');p.add_argument('--device',default='cuda:0')
    p=subs.add_parser('diagnose',help='已训练母版固定train/val样本，只读，不评test、不训练')
    p.add_argument('--weights',type=Path);p.add_argument('--dataset',type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch',type=int,default=2);p.add_argument('--limit',type=int,default=128)
    subs.add_parser('start',help='显式派发正式200epoch配方至独立tmux')
    subs.add_parser('resume',help='只恢复本实验已核验的last.pt')
    subs.add_parser('val',help='独立val正式best.pt并冻结选择SHA')
    subs.add_parser('test',help='只评估val已冻结选择，不扫描权重')
    subs.add_parser('pack',help='生成并回读验证LIGHT包，保留原始指标')
    p=subs.add_parser('worker',help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.action=='prepare':prepare(args)
    elif args.action=='preflight':preflight(args)
    elif args.action=='diagnose':
        from diagnose_rdl_v1 import diagnose
        if args.weights is None or args.dataset is None:
            p=prepared()
            args.weights=args.weights or Path(p['main'])/'runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt'
            args.dataset=args.dataset or Path(p['dataset'])
        diagnose(args.weights,args.dataset,args.device,args.batch,args.limit)
    elif args.action in {'start','resume'}:dispatch(args,resume=args.action=='resume')
    elif args.action in {'val','test'}:evaluate(args,args.action)
    elif args.action=='pack':pack(args)
    elif args.action=='worker':worker(args)


if __name__=='__main__':
    main()
