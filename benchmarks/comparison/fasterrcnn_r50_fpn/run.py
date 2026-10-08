"""Configurable Faster R-CNN: preview/preflight/train/resume/export/evaluate/pack."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import uuid
from support import HERE,RUNTIME,RUN_ROOT,canonical,code_identity,environment,local_lock,read_json,run_identity,status,write_json
from configuration import candidate,freeze_config,frozen_config,safe_run_id

def record_invocation(run):
    from datetime import datetime,timezone
    value={'argv':[os.path.abspath(sys.executable),*sys.argv],'cwd':str(Path.cwd()),
        'shell':os.environ.get('FRCNN_LAUNCH_COMMAND'),'created_utc':datetime.now(timezone.utc).isoformat()}
    if (RUNTIME/'tmux_latest.json').exists(): value['tmux']=read_json(RUNTIME/'tmux_latest.json')
    write_json(Path(run)/'stage_invocations'/(uuid.uuid4().hex+'.json'),value)

def summary(run,save=True):
    training=read_json(run/'train_status.json') if (run/'train_status.json').exists() else {}
    progress=read_json(run/'training_progress.json') if (run/'training_progress.json').exists() else {}
    rows={}; raw={}
    for split in ('val','test'):
        seal=run/'metrics'/(split+'_complete.json')
        if seal.is_file():
            seal=read_json(seal); value=read_json(seal['metrics_path'])
            rows[split]=value['display_percent']; raw[split]=value['raw_0_1']
        else: rows[split]=None; raw[split]=None
    complete=training.get('status')=='completed' and all(rows.values())
    result={'status':'completed' if complete else 'incomplete','run':str(run),'model':'Faster R-CNN (ResNet-50-FPN)',
        'initialization':frozen_config(run,check_code=False)['initialization'],'scope':read_json(run/'run_id.json')['scope'],
        'completed_epoch':training.get('completed_epoch',progress.get('epoch')),
        'best_epoch':training.get('best_epoch',progress.get('best_epoch')),'best_sha256':training.get('best_sha256'),
        'stop_reason':training.get('stop_reason',training.get('status')),'metric_policy':'corrected_sorted_conf_mask_v1',
        'raw_0_1':raw,'display_percent':rows,'units':'displayed values are percent; JSON raw values are 0-1'}
    if save: write_json(run/'summary.json',result)
    print('SPLIT       P%       R%     AP50%     AP75%   mAP50-95%',flush=True)
    for split in ('val','test'):
        if rows[split]: print(f"{split.upper():5s} "+' '.join(f'{rows[split][k]:9.4f}' for k in ('P','R','AP50','AP75','mAP50-95')),flush=True)
        else: print(split.upper()+'  pending',flush=True)
    print(f"epochs={result['completed_epoch']} best_epoch={result['best_epoch']} status={result['status']}",flush=True)
    return result

def strict_identity(run):
    from data import verify_inputs
    from model import prepare_weights
    cfg=frozen_config(run); manifest=verify_inputs(run,('train','val'))
    info=read_json(run/'run_id.json')
    if info['scope']!='FORMAL': raise ValueError('Formal commands refuse smoke/data-only runs')
    env=environment(strict=True); probe=read_json(run/'preflight_model.json')
    if probe['status']!='PASSED_ACTUAL_CONFIGURED_CAPACITY' or probe['environment']!=env: raise ValueError('Missing/stale environment/capacity preflight')
    if read_json(run/'preflight_identity.json')!={'code_sha256':code_identity(),'python':os.path.abspath(sys.executable)}:
        raise ValueError('Code/interpreter differs from isolated preflight')
    if (RUNTIME/'python_path.txt').read_text().strip()!=os.path.abspath(sys.executable): raise ValueError('Use the recorded bootstrap interpreter')
    current=prepare_weights(cfg)
    if current!=read_json(run/'initialization_identity.json'): raise ValueError('Initialization file/mode changed')
    identity=run_identity(manifest,info['run_uuid'],cfg,env,current)
    return manifest,identity,cfg,env

def preflight(args,run):
    from data import preflight as data_preflight
    from model import prepare_weights
    cfg,raw,overrides,origin=candidate(args.config,args.set,args.clone_config_from)
    print(json.dumps(cfg,ensure_ascii=False,indent=2),flush=True)
    freeze_config(run,cfg,raw,overrides,origin,scope='LOCAL_DATA_CHECK_ONLY' if args.data_only else 'FORMAL')
    record_invocation(run)
    if (RUNTIME/'tmux_latest.json').exists(): write_json(run/'tmux_context.json',read_json(RUNTIME/'tmux_latest.json'))
    status(run,'preflight','running')
    manifest=data_preflight(cfg['data_yaml'],run,cfg['data_root'],cfg['public_coco'],reuse_manifest=cfg['reuse_manifest'])
    if args.data_only:
        status(run,'preflight','completed',exit_code=0,scope='LOCAL_DATA_CHECK_ONLY'); return
    env=environment(strict=True)
    if not (RUNTIME/'source_correspondence.json').is_file(): raise ValueError('Run bootstrap first for source/wheel verification')
    write_json(run/'source_correspondence.json',read_json(RUNTIME/'source_correspondence.json'))
    if (RUNTIME/'python_path.txt').read_text().strip()!=os.path.abspath(sys.executable): raise ValueError('Use bootstrap interpreter')
    initialization=prepare_weights(cfg); write_json(run/'initialization_identity.json',initialization)
    subprocess.run([sys.executable,str(HERE/'probe.py'),'--run',str(run),'--config',str(run/'resolved_config.yaml'),
        '--output',str(run/'preflight_model.json')],check=True)
    write_json(run/'environment.json',env)
    write_json(run/'preflight_identity.json',{'code_sha256':code_identity(),'python':os.path.abspath(sys.executable)})
    status(run,'preflight','completed',exit_code=0,counts=manifest['splits'])

def train_run(run,resume):
    from engine import train
    manifest,identity,cfg,env=strict_identity(run)
    return train(run,manifest,identity,cfg,env,resume=resume)

def export_run(run,split,batch=None):
    from data import verify_inputs
    from export import export_split
    manifest,identity,cfg,_=strict_identity(run)
    verify_inputs(run,(split,))
    if read_json(run/'identity.json')!=identity: raise ValueError('Export run identity differs')
    settings={'eval_batch_size':batch or cfg['eval_batch_size'],'eval_workers':cfg['eval_workers']}
    file=run/'predictions'/(split+'_inference_settings.json')
    if file.exists() and read_json(file)!=settings: raise ValueError('Existing prediction settings differ; original cache protected')
    write_json(file,settings)
    cfg={**cfg,**settings}
    return export_split(run,split,manifest,identity,cfg)

def stage(run,name,operation):
    status(run,name,'running')
    value=operation()
    status(run,name,'completed',exit_code=0,**{k:value[k] for k in ('metrics','path','images') if isinstance(value,dict) and k in value})
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ('print-config','preflight','full','train','export','evaluate','pack','summary','redraw'):
        p=sub.add_parser(cmd)
        if cmd!='print-config': p.add_argument('--run-id',required=True,type=safe_run_id)
        if cmd in ('print-config','preflight','full'):
            p.add_argument('--config',type=Path); p.add_argument('--set',action='append',default=[])
            p.add_argument('--clone-config-from'); p.add_argument('--data-only',action='store_true')
        if cmd in ('train','full'): p.add_argument('--resume',action='store_true')
        if cmd in ('export','evaluate','redraw'):
            p.add_argument('--split',choices=('val','test','both'),default='both')
        if cmd=='export': p.add_argument('--inference-batch',type=int,choices=range(1,129))
        if cmd=='pack': p.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.command=='print-config':
        cfg,*_=candidate(args.config,args.set,args.clone_config_from); print(json.dumps(cfg,ensure_ascii=False,indent=2)); return
    run=(RUN_ROOT/args.run_id).absolute(); active_stage=args.command
    def interrupt(number,frame): raise KeyboardInterrupt('signal '+str(number))
    signal.signal(signal.SIGINT,interrupt); signal.signal(signal.SIGTERM,interrupt)
    with local_lock(RUN_ROOT/'.locks'/(args.run_id+'.lock')):
        try:
            if args.command in ('preflight','full') and not getattr(args,'resume',False):
                preflight(args,run)
                if args.command=='preflight' or args.data_only: return
            elif not run.is_dir(): raise FileNotFoundError('Run missing: '+str(run))
            if args.command!='preflight': record_invocation(run)
            if args.command=='full' and args.resume and (args.config or args.set or args.clone_config_from or args.data_only):
                raise ValueError('Resume uses the immutable frozen run; config/--set/clone/data-only are forbidden')
            if args.command in ('full','train'):
                active_stage='train'; train_run(run,args.resume)
                if args.command=='train': return summary(run)
            if args.command in ('full','export','evaluate','redraw'):
                from export import evaluate_split
                splits=('val','test') if args.command=='full' or args.split=='both' else (args.split,)
                for split in splits:
                    if args.command in ('full','export'):
                        active_stage='export_'+split
                        stage(run,active_stage,lambda:export_run(run,split,getattr(args,'inference_batch',None)))
                    if args.command in ('full','evaluate','redraw'):
                        active_stage='evaluate_'+split
                        stage(run,active_stage,lambda:evaluate_split(run,split))
                if args.command=='export': return summary(run)
                if args.command=='redraw':
                    from curves import training_curves
                    history=[json.loads(line) for line in (run/'epoch_trace.jsonl').read_text().splitlines() if line.strip()]
                    training_curves(run,history)
                if args.command in ('evaluate','redraw'): return summary(run)
            if args.command in ('full','pack'):
                from packaging_run import pack
                active_stage='pack'; stage(run,active_stage,lambda:pack(run,getattr(args,'out',None)))
            summary(run)
        except BaseException as exc:
            code=130 if isinstance(exc,KeyboardInterrupt) else 1
            if run.is_dir():
                error={'exit_code':code,'error':str(exc),'traceback':traceback.format_exc(),'argv':sys.argv}
                old_path=run/(active_stage+'_status.json')
                if not old_path.exists() or read_json(old_path).get('status')!='completed':
                    status(run,active_stage,'interrupted' if code==130 else 'failed',**error)
                write_json(run/('error_'+uuid.uuid4().hex+'.json'),error)
            traceback.print_exc(); raise SystemExit(code)
if __name__=='__main__': main()
