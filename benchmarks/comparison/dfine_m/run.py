"""Official D-FINE-M configurable project runner. No SSH or automatic recipe changes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from configuration import (candidate,freeze,frozen_config,paths,RUN_ROOT,safe_run_id)
from support import (HERE,ROOT,LOCK,checked_source,checked_weight,local_lock,read_json,status,write_json)


def prepare(run,run_id,cfg,raw,overrides,smoke=False):
    from data import prepare_inputs
    from lifecycle import identity
    checked_source(paths(cfg)['source']); checked_weight(paths(cfg)['weights'])
    freeze(run,run_id,cfg,raw,overrides,smoke=smoke)
    status(run,'prepare','running')
    manifest=prepare_inputs(cfg,run)
    write_json(Path(run)/'identity.json',identity(run,cfg,manifest))
    status(run,'prepare','completed',splits=manifest['splits'])
    from support import ROOT,atomic_bytes
    atomic_bytes(ROOT/'.runtime/dfine-m-configurable/runs'/ (run_id+'.path'),(str(Path(run).absolute())+'\n').encode())
    return cfg,manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('preview','bootstrap','prepare','preflight','train','final','evaluate','pack','all','_capacity','smoke'))
    parser.add_argument('--config',type=Path); parser.add_argument('--set',action='append',default=[])
    parser.add_argument('--clone-config-from',type=Path); parser.add_argument('--run-id'); parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--resume',type=Path); parser.add_argument('--split',choices=('val','test','both'),default='both')
    parser.add_argument('--new-environment',action='store_true')
    args=parser.parse_args()
    if args.resume and args.stage not in ('train','all'): parser.error('--resume is only implemented for train/all')
    if args.new_environment and args.stage!='bootstrap': parser.error('--new-environment is only implemented for bootstrap')
    if args.split!='both' and args.stage not in ('final','evaluate','all'): parser.error('--split is only implemented for final/evaluate/all')
    if args.stage=='smoke' and (args.config or args.set or args.clone_config_from or args.run_id): parser.error('smoke uses its recorded isolated synthetic recipe')
    if args.stage=='preview':
        cfg,raw,overrides=candidate(args.config,args.set,args.clone_config_from)
        print(json.dumps({'resolved_config':cfg,'precedence':['committed_defaults','YAML_or_clone','CLI'],
            'group_target_lrs':{'backbone':cfg['backbone_lr'],'encoder_decoder':cfg['lr0']},
            'no_implicit_batch_scaling':True,'mixing_close_zero_based_epoch':cfg['epochs']-cfg['close_mosaic'] if cfg['close_mosaic'] else None,
            'source_lock':{k:v for k,v in read_json(LOCK).items() if k!='files_lf_sha256'},'paths':paths(cfg)},indent=2,ensure_ascii=False)); return
    if args.stage=='bootstrap':
        from bootstrap import bootstrap
        cfg,_,_=candidate(args.config,args.set,args.clone_config_from)
        with local_lock(ROOT/'.runtime/dfine-m-configurable/bootstrap.lock'): bootstrap(cfg,args.new_environment)
        return
    if args.stage=='smoke':
        from smoke import run_smoke
        run_smoke(args.run_dir); return
    if args.stage=='_capacity':
        if not args.run_dir: parser.error('_capacity requires --run-dir')
        from lifecycle import capacity_probe
        capacity_probe(args.run_dir); return
    if args.run_dir and args.run_id: parser.error('Choose --run-id or --run-dir')
    if not args.run_dir and not args.run_id: parser.error('Provide --run-id or --run-dir')
    if args.stage in ('prepare','all') and not args.resume:
        if args.run_dir: parser.error('New prepare/all requires --run-id')
        cfg,raw,overrides=candidate(args.config,args.set,args.clone_config_from)
        run=Path(paths(cfg)['run_root'])/safe_run_id(args.run_id)
    else:
        if args.config or args.set or args.clone_config_from: parser.error('Frozen run stages do not accept configuration changes; create a new run')
        run=(args.run_dir or RUN_ROOT/safe_run_id(args.run_id)).absolute()
        cfg=frozen_config(run)
    lock=run.parent/('.'+run.name+'.lock')
    with local_lock(lock):
        try:
            if args.stage in ('prepare','all') and not args.resume:
                cfg,manifest=prepare(run,args.run_id,cfg,raw,overrides)
            else:
                from light_data import verify_inputs
                manifest=verify_inputs(run/'data',splits=('train','val') if args.stage in ('train','preflight') else ('val','test'))
            if args.stage in ('preflight','all') and not args.resume:
                from lifecycle import preflight_process
                preflight_process(run)
            if args.stage in ('train','all'):
                from lifecycle import train
                train(run,args.resume)
            splits=('val','test') if args.split=='both' else (args.split,)
            if args.stage in ('final','all'):
                from export import export_split,evaluate_split,show_summary
                for split in splits:
                    export_split(run,split,cfg,manifest); evaluate_split(run,split,cfg,manifest)
                if all((run/f'metrics/{s}.json').exists() for s in ('val','test')): show_summary(run)
            if args.stage=='evaluate':
                from export import evaluate_split,show_summary
                for split in splits: evaluate_split(run,split,cfg,manifest)
                if args.split=='both': show_summary(run)
            if args.stage in ('pack','all'):
                from packaging_run import package
                package(run)
        except Exception as exc:
            if run.exists(): status(run,args.stage,'failed',exit_code=1,error=repr(exc))
            raise

if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
