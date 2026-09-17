"""SRE plan/start/resume with immutable original recipe and strict bounded-preflight gate."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import traceback

from sre_common import *
from init_sre import build_training_model, verify_model
import torch
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.patches import torch_load


def plan(variant,data,output_dir=None):
    p=paths(variant,output_dir)
    args,rows=recipe(variant,p['init'],data)
    result=dict(status='PLANNED',variant=variant,identity=code_identity(),args=args,parent_args_sha256=sha256(ROOT/'docs/sre/parent_args.yaml'),
                recipe_differences=rows,formal_training='NOT_STARTED',final_test='NOT_RUN')
    p['folder'].mkdir(parents=True,exist_ok=True)
    YAML.save(p['folder']/'planned_args.yaml',args)
    write_json(p['folder']/'recipe_diff.json',rows)
    write_json(p['plan'],result)
    print(json.dumps(dict(status='PLANNED',variant=variant,run=str(p['run']),preflight=str(p['preflight'])),indent=2))
    return result


def process_live(pid):
    try: os.kill(int(pid),0); return True
    except ProcessLookupError: return False
    except (OSError,ValueError,TypeError): return True  # unknown owner remains protected


def run(variant,data,output_dir=None,checkpoint=None):
    p=paths(variant,output_dir)
    preflight=gate(variant,data,output_dir)
    require(torch.cuda.is_available(),'CUDA required; no CPU formal fallback')
    require(ROOT.resolve()!=MAIN.resolve(),'Use independent SRE worktree')
    require(os.environ.get('CONDA_DEFAULT_ENV')=='rtdetr','Activate existing rtdetr environment')
    args,rows=recipe(variant,p['init'],data)
    resume=checkpoint is not None
    previous=read_json(p['state']) if p['state'].exists() else {}
    if resume:
        checkpoint=Path(checkpoint).resolve()
        require(checkpoint==(p['run']/'weights/last.pt').resolve(),'Resume only this run actual last.pt')
        require(previous and previous.get('identity')==code_identity(),'Resume requires same recorded execution commit/code')
        require(previous.get('status') not in {'COMPLETED_200','EARLY_STOPPED'},'Completed training must not restart')
        saved=torch_load(checkpoint,map_location='cpu')
        require(0<=saved.get('epoch',-1)<199 and saved.get('optimizer') and saved.get('scaler') and saved.get('ema'),
                'Checkpoint is completed/stripped or missing native resume states')
        verify_model(saved['ema'],variant,zero=False)
        require(torch.count_nonzero(saved['ema'].model[19].sre.W_o.weight)>0,'Resume requires genuinely learned SRE')
        saved_args=saved['train_args']
        require(all(saved_args.get(k)==v for k,v in args.items() if k not in {'model','resume'}),'Checkpoint recipe differs')
        args.update(model=str(checkpoint),resume=str(checkpoint))
        del saved
    else:
        require(not p['run'].exists() and not previous,'Existing formal run/state protected; inspect and use resume if unfinished')
    lock=p['run'].with_name(p['run'].name+'.sre.lock.json')
    lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.exists():
        owner=read_json(lock)
        require(resume and not process_live(owner.get('pid')),'Existing/shared worker reservation protected')
    else:
        with lock.open('x',encoding='utf-8') as stream: stream.write('{}\n')
    write_json(lock,dict(pid=os.getpid(),variant=variant,worktree=str(ROOT),identity=code_identity(),resume=resume))
    state=dict(status='RUNNING',variant=variant,pid=os.getpid(),identity=code_identity(),
               preflight_sha256=sha256(p['preflight']),init_sha256=sha256(p['init']),
               completed_training_epochs=previous.get('completed_training_epochs',0),final_eval='NOT_RUN',
               started=datetime.now(timezone.utc).isoformat(),resume=resume,final_test='NOT_RUN')
    write_json(p['state'],state)
    def update(**values): state.update(values);write_json(p['state'],state)
    class RecordingTrainer(RTDETRTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            model,audit=build_training_model(cfg,weights,self.data,variant,zero=not resume)
            write_json(p['folder']/('resume_loading.json' if resume else 'native_loading.json'),audit)
            return model
        def final_eval(self):
            update(final_eval='RUNNING')
            try:
                result=super().final_eval()
                update(final_eval='PASSED')
                return result
            except BaseException as error:
                update(final_eval='FAILED',final_eval_error=repr(error))
                raise
    def on_start(trainer):
        verify_model(trainer.model,variant,zero=not resume)
        require(trainer.amp is True,'Native AMP check disabled AMP; stopping without recipe change')
        differences={k:[v,vars(trainer.args).get(k)] for k,v in args.items() if vars(trainer.args).get(k)!=v}
        require(not differences,'Actual training args changed: '+repr(differences))
        require(Path(trainer.save_dir).resolve()==p['run'].resolve(),'Output name auto-incremented; refusing')
        write_json(p['folder']/'optimizer_audit.json',optimizer_audit(trainer.model,trainer.optimizer))
        YAML.save(p['folder']/'actual_args.yaml',vars(trainer.args))
        update(start_epoch=trainer.start_epoch)  # epoch need not exist on_train_start
    def on_batch_start(trainer): trainer._oom_retries=3  # native OOM must not reduce B16
    def on_epoch_end(trainer): update(completed_training_epochs=trainer.epoch+1)
    exit_code=1
    try:
        model=RTDETR(str(checkpoint if resume else p['init']))
        model.add_callback('on_train_start',on_start)
        model.add_callback('on_train_batch_start',on_batch_start)
        model.add_callback('on_train_epoch_end',on_epoch_end)
        model.train(trainer=RecordingTrainer,**args)
        update(status='COMPLETED_200' if state['completed_training_epochs']>=200 else 'EARLY_STOPPED')
        exit_code=0
    except KeyboardInterrupt:
        update(status='COMPLETED_200' if state['completed_training_epochs']>=200 else 'INTERRUPTED')
        exit_code=130
        raise
    except BaseException as error:
        update(status='COMPLETED_200' if state['completed_training_epochs']>=200 else 'FAILED',error=repr(error))
        raise
    finally:
        update(exit_code=exit_code,finished=datetime.now(timezone.utc).isoformat())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','start','resume'])
    parser.add_argument('--variant',choices=VARIANTS,default='cbr_lif_sre_v1')
    parser.add_argument('--data',type=Path,default=MAIN/'configs/crack_autodl.yaml')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--checkpoint',type=Path,help='Only resume: actual unfinished run weights/last.pt')
    args=parser.parse_args();torch.set_num_threads(4)
    require((args.mode=='resume')==(args.checkpoint is not None),'--checkpoint is required only for resume')
    if args.mode=='plan': plan(args.variant,args.data,args.output_dir)
    else: run(args.variant,args.data,args.output_dir,args.checkpoint)
