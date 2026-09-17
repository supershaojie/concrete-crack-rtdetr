"""Bounded SRE checks; real B16/640 native-AMP updates never launch formal training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

from sre_common import *
from init_sre import build_training_model, verify_model
import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import ASSETS
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import autocast, ModelEMA


class PendingDependency(RuntimeError): pass


def prepare_amp_resources():
    """Reuse resources used by the parent's native AMP check, without installation/download."""
    rows=[]
    for name,dest,candidates in [
        ('yolo26n.pt',ROOT/'yolo26n.pt',[MAIN/'yolo26n.pt',MAIN/'weights/yolo26n.pt']),
        ('bus.jpg',ASSETS/'bus.jpg',[MAIN/'ultralytics-main/ultralytics/assets/bus.jpg',MAIN/'bus.jpg'])]:
        if not dest.is_file():
            found=next((v for v in candidates if v.is_file()),None)
            if not found: raise PendingDependency('Missing existing native AMP resource: '+name)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with found.open('rb') as source, dest.open('xb') as output: shutil.copyfileobj(source,output)
        rows.append(dict(name=name,path=str(dest),sha256=sha256(dest)))
    return rows


def exact_nested(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(exact_nested(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a) is type(b) and len(a)==len(b) and all(exact_nested(x,y) for x,y in zip(a,b))
    return a==b


def native_resume_audit(trainer,folder):
    """Native save_model + native get_model + resume_training, including quantized optimizer/EMA."""
    require(torch.count_nonzero(trainer.model.model[19].sre.W_o.weight)>0,'Resume needs learned nonzero SRE')
    trainer.epoch=0; trainer.fitness=0.; trainer.best_fitness=0.
    trainer.save_model()
    saved=torch_load(trainer.last,map_location='cpu')
    require(saved['epoch']==0 and saved['optimizer'] and saved['scaler'] and saved['updates']>0,'Native checkpoint missing state')
    restored=RTDETRTrainer.__new__(RTDETRTrainer)
    restored.data=trainer.data
    restored.model=RTDETRTrainer.get_model(restored,cfg=saved['ema'].yaml,weights=saved['ema'],verbose=False).to(trainer.device)
    restored.args=deepcopy(trainer.args)
    restored.optimizer=restored.build_optimizer(restored.model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    restored.scaler=torch.amp.GradScaler('cuda') if hasattr(torch,'amp') and hasattr(torch.amp,'GradScaler') else torch.cuda.amp.GradScaler()
    restored.ema=ModelEMA(restored.model)
    restored.resume=True; restored.epochs=200; restored.start_epoch=0
    expected_ema={k:v.detach().float().clone() for k,v in saved['ema'].state_dict().items()}
    RTDETRTrainer.resume_training(restored,saved)
    require(restored.start_epoch==1,'Native resume epoch mismatch')
    require(exact_nested(restored.scaler.state_dict(),saved['scaler']),'Native scaler restore differs')
    # load_state_dict casts FP16 saved optimizer moments to parameter dtype by design.
    expected_opt=deepcopy(saved['optimizer'])
    for state in expected_opt['state'].values():
        for k,v in state.items():
            if isinstance(v,torch.Tensor) and k!='step': state[k]=v.float()
    actual_opt=restored.optimizer.state_dict()
    require(exact_nested(actual_opt,expected_opt),'Native optimizer restoration differs at same quantization')
    require(all(torch.equal(v,restored.model.state_dict()[k].float().cpu()) for k,v in expected_ema.items()),'Native learned weights reset')
    require(all(torch.equal(v,restored.ema.ema.state_dict()[k].float().cpu()) for k,v in expected_ema.items()),'Native EMA reload differs')
    require(restored.ema.updates==saved['updates'] and torch.count_nonzero(restored.model.model[19].sre.W_o.weight)>0,
            'Native EMA/nonzero SRE lost')
    result=dict(status='PASSED',start_epoch=restored.start_epoch,checkpoint_sha256=sha256(trainer.last),
                optimizer_exact_at_native_half_quantization=True,scaler_exact=True,ema_exact=True,learned_sre_preserved=True,
                scope='Native save_model/get_model/resume_training; disposable epoch0 checkpoint, not a formal resume source')
    del restored,saved
    return result


def capacity(initialized,variant,data,device,max_batches,folder,persist,max_seconds=600):
    require(2<=max_batches<=64,'Bounded batch budget must be 2..64')
    require(30<=max_seconds<=3600,'Wall-time budget must be 30..3600 seconds')
    if not torch.cuda.is_available(): raise PendingDependency('CUDA unavailable; B16/640/native AMP requires server')
    resources=prepare_amp_resources()
    args,_=recipe(variant,initialized,data)
    # Only experiment output identity changes; all 109 recipe fields otherwise retained.
    args.update(project=str(folder),name='native_capacity',save_dir=str(folder/'native_capacity'),device=str(device))
    class AuditTrainer(RTDETRTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            model,audit=build_training_model(cfg,weights,self.data,variant,zero=True)
            write_json(folder/'native_loading.json',audit)
            return model
    record=dict(status='RUNNING',batch=16,imgsz=640,native_amp=True,max_batches=max_batches,
                max_seconds=max_seconds,wall_time_guard='POSIX SIGALRM' if hasattr(signal,'SIGALRM') else
                'per-batch elapsed check; Windows startup additionally requires external process timeout',
                effective_updates=0,skipped_steps=0,batches=[],amp_resources=resources,
                original_online_augmentation=True,formal_training='NOT_STARTED',final_test='NOT_RUN')
    persist(record)
    trainer=None
    started=time.perf_counter()
    old_handler=old_timer=None
    if hasattr(signal,'SIGALRM'):
        def wall_timeout(signum,frame): raise PendingDependency('WALL_TIME_BUDGET: native setup/batches exceeded configured time')
        old_handler=signal.getsignal(signal.SIGALRM)
        old_timer=signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM,wall_timeout)
        signal.setitimer(signal.ITIMER_REAL,max_seconds)
    try:
        trainer=AuditTrainer(overrides=args)
        # Native label scanning caches are derived artifacts, even when image cache=False.
        # Keep writes inside this disposable preflight, away from shared datasets.
        from ultralytics.data.utils import save_dataset_cache_file
        cache_folder=folder/'label_cache';cache_folder.mkdir()
        cache_writes=[]
        def disposable_label_cache(prefix,path,x,version):
            destination=cache_folder/Path(path).name
            save_dataset_cache_file(prefix,destination,x,version)
            cache_writes.append(dict(original=str(path),disposable=str(destination)))
        with patch('ultralytics.data.dataset.save_dataset_cache_file',disposable_label_cache):
            trainer._setup_train()
        record['disposable_label_cache_writes']=cache_writes
        require(trainer.amp is True and trainer.args.batch==16 and trainer.args.imgsz==640,'Native capacity conditions changed')
        record['optimizer']=optimizer_audit(trainer.model,trainer.optimizer)
        record['native_scaler_initial']=trainer.scaler.state_dict()
        record['actual_recipe_differences']={k:[v,vars(trainer.args).get(k)] for k,v in args.items()
                                           if vars(trainer.args).get(k)!=v}
        require(not record['actual_recipe_differences'],'Native preflight recipe drift')
        trainer.epoch=0; trainer._model_train(); trainer.optimizer.zero_grad()
        torch.cuda.reset_peak_memory_stats(trainer.device)
        decoder_records={}
        def decoder_hook(module,inputs,outputs):
            meta=outputs[-1]
            decoder_records.update(total_queries=int(outputs[0].shape[2]),dn_num_split=meta['dn_num_split'] if meta else None)
        hook=trainer.model.model[26].register_forward_hook(decoder_hook)
        from check_c19_lif_v1 import native_warmup
        before_nonzero=False; upstream_success=False; first_output_gradient=False; last_step=-1
        nb=len(trainer.train_loader)
        for index,batch in enumerate(trainer.train_loader):
            if index>=max_batches: break
            if time.perf_counter()-started>max_seconds: raise PendingDependency('WALL_TIME_BUDGET: bounded native batches')
            tick=time.perf_counter()
            warmup=native_warmup(trainer.optimizer,index,nb)
            trainer.accumulate=warmup['accumulate']
            scale_before=float(trainer.scaler.get_scale())
            with autocast(trainer.amp):
                batch=trainer.preprocess_batch(batch)
                require(tuple(batch['img'].shape)==(16,3,640,640),'Real preflight changed B16/640')
                loss,trainer.loss_items=trainer.model(batch)
                trainer.loss=loss.sum()
            require(torch.isfinite(trainer.loss),'Nonfinite real detection loss')
            require(batch['cls'].numel()>0 and decoder_records.get('dn_num_split'),'Real GT/DN not exercised')
            trainer.scaler.scale(trainer.loss).backward()
            raw_gradients={n:float(p.grad.detach().float().norm()/scale_before) if p.grad is not None else None
                           for n,p in trainer.model.named_parameters() if '.sre.' in n}
            # Serialize scaled-gradient overflows honestly without changing any tensor.
            gradients={n:v if v is not None and math.isfinite(v) else None for n,v in raw_gradients.items()}
            nonfinite_gradients=[n for n,v in raw_gradients.items() if v is not None and not math.isfinite(v)]
            finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.model.parameters())
            old_w=trainer.model.model[19].sre.W_o.weight.detach().clone()
            stepped=False; scale_after=scale_before
            if index-last_step>=trainer.accumulate:
                trainer.optimizer_step()  # unchanged native unscale/clip/scaler.step/update/EMA
                scale_after=float(trainer.scaler.get_scale())
                stepped=scale_after>=scale_before
                last_step=index
                if stepped:
                    require(finite,'Native optimizer accepted nonfinite gradients')
                    record['effective_updates']+=1
                else: record['skipped_steps']+=1
            changed=not torch.equal(old_w,trainer.model.model[19].sre.W_o.weight)
            output_grad=gradients.get('model.19.sre.W_o.weight')
            upstream={k:v for k,v in gradients.items() if k!='model.19.sre.W_o.weight'}
            if stepped and output_grad is not None and math.isfinite(output_grad) and output_grad>0:
                first_output_gradient=True
            if stepped and before_nonzero and all(v is not None and math.isfinite(v) and v>0 for v in upstream.values()):
                upstream_success=True
            before_nonzero=bool(torch.count_nonzero(trainer.model.model[19].sre.W_o.weight))
            torch.cuda.synchronize(trainer.device)
            record['batches'].append(dict(index=index,gt=int(batch['cls'].numel()),dn=deepcopy(decoder_records),
                   loss=float(trainer.loss.detach()),scale_before=scale_before,scale_after=scale_after,
                   scaler_skipped=stepped is False and scale_after<scale_before,effective_optimizer_update=stepped,
                   gradients_finite=bool(finite),sre_grad_norms=gradients,sre_nonfinite_gradient_names=nonfinite_gradients,
                   W_o_changed=changed,warmup=warmup,
                   seconds=time.perf_counter()-tick))
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(trainer.device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(trainer.device),
                          elapsed_seconds=time.perf_counter()-started,
                          upstream_gradient_after_nonzero_output=upstream_success,
                          first_output_gradient_nonzero=first_output_gradient)
            persist(record)
            del loss,batch,old_w
            if record['effective_updates']>=2 and upstream_success and before_nonzero: break
        hook.remove()
        require(record['effective_updates']>=2 and upstream_success and first_output_gradient,
                'Finite budget exhausted before two native updates and gradient startup; do not retry indefinitely')
        record['resume']=native_resume_audit(trainer,folder)
        record['status']='PASSED'
        persist(record)
        return record
    except PendingDependency as error:
        record.update(status='PENDING',reason=str(error),elapsed_seconds=time.perf_counter()-started)
        persist(record)
        return record
    except torch.cuda.OutOfMemoryError as error:
        record.update(status='PENDING',reason='RESOURCE_CAPACITY: B16/640/native AMP exhausted this GPU; rerun unchanged on server',
                      error=str(error),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.perf_counter()-started)
        persist(record)
        return record
    except BaseException as error:
        record.update(status='FAILED',error=repr(error),elapsed_seconds=time.perf_counter()-started)
        persist(record)
        raise
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,old_handler)
            if old_timer and old_timer[0]>0: signal.setitimer(signal.ITIMER_REAL,*old_timer)
        del trainer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def preflight(args):
    p=paths(args.variant,args.output_dir)
    missing=[]
    cuda_missing=args.device!='cpu' and not torch.cuda.is_available()
    if cuda_missing: missing.append('CUDA unavailable: required native B16/640/AMP capacity remains PENDING')
    core_device='cpu' if args.device=='cpu' or cuda_missing else 'cuda'
    existing=read_json(p['initialization']) if p['initialization'].is_file() else {}
    if not p['init'].is_file() or not existing:
        missing.append('Controlled init/report unavailable: run init_sre.py')
    else:
        require(existing['variant']==args.variant and sha256(p['init'])==existing['output_sha256'],'Init identity changed')
        if not Path(existing['source']).is_file(): missing.append('Unified source unavailable: '+existing['source'])
        else: require(sha256(existing['source'])==SOURCE_SHA256,'Unified source changed')
    if p['preflight'].exists():
        # Preserve old report as evidence; report is regenerated, never hand-edited to PASSED.
        archive=p['folder']/('preflight-'+sha256(p['preflight'])[:16]+'.json')
        if not archive.exists(): shutil.copyfile(p['preflight'],archive)
    report=dict(status='RUNNING',variant=args.variant,runtime=runtime(),identity=code_identity(),
                init_sha256=sha256(p['init']) if p['init'].is_file() else None,source_sha256=SOURCE_SHA256,
                formal_training='NOT_STARTED',final_test='NOT_RUN',
                dataset=dict(status='PENDING'),core=dict(status='PENDING'),capacity=dict(status='PENDING'),
                budget=dict(max_batches=args.max_batches,max_seconds=args.max_seconds,batch=16,imgsz=640,native_amp=True))
    write_json(p['preflight'],report)
    def save_capacity(value):
        report['capacity']=value;write_json(p['preflight'],report)
    try:
        try: report['dataset']=dataset_identity(args.data)
        except FileNotFoundError as error:
            missing.append(str(error));report['dataset']=dict(status='PENDING',reason=str(error))
        if args.core_report:
            core=read_json(args.core_report)
            require(core.get('identity')==report['identity'],'Core report code identity changed; rerun check_sre.py')
            require(core.get('status') in {'PASSED','PENDING'},'Core checks failed')
            require(args.variant in core.get('variants',{}),'Core report lacks requested variant')
            require(core.get('device')==core_device,'Core report device differs from available requested execution')
            report['core']=dict(status=core['status'],path=str(args.core_report.resolve()),sha256=sha256(args.core_report),results=core)
        else:
            core_path=p['folder']/('core-'+core_device+'.json')
            command=[sys.executable,str(ROOT/'tools/check_sre.py'),'--output',str(core_path),'--device',
                     core_device,'--variant',args.variant]
            subprocess.run(command,cwd=ROOT,check=True)
            core=read_json(core_path)
            require(core.get('status') in {'PASSED','PENDING'} and core.get('identity')==report['identity'],'Core check identity/status mismatch')
            report['core']=dict(status=core['status'],path=str(core_path),sha256=sha256(core_path),results=core)
        if missing:
            report['capacity']=dict(status='PENDING',reason='Required dependencies unavailable',dependencies=missing)
        elif args.device=='cpu':
            report['capacity']=dict(status='PENDING',reason='CPU local preflight; server B16/640/native CUDA AMP not run')
        else:
            with tempfile.TemporaryDirectory(prefix='sre-preflight-',dir=p['folder']) as tmp:
                report['capacity']=capacity(p['init'],args.variant,args.data,args.device,args.max_batches,Path(tmp),save_capacity,args.max_seconds)
        report['status']='PASSED' if report['capacity']['status']==report['core']['status']=='PASSED' else 'PENDING'
    except (PendingDependency,FileNotFoundError) as error:
        report.update(status='PENDING',pending_reason=str(error))
    except BaseException as error:
        report.update(status='FAILED',error=repr(error))
        write_json(p['preflight'],report)
        raise
    write_json(p['preflight'],report)
    print(json.dumps(dict(status=report['status'],report=str(p['preflight']),capacity=report['capacity']['status']),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS,default='cbr_lif_sre_v1')
    parser.add_argument('--data',type=Path,default=MAIN/'configs/crack_autodl.yaml')
    parser.add_argument('--device',default='0',help='cpu for local evidence only; 0 for required CUDA capacity')
    parser.add_argument('--max-batches',type=int,default=24,help='Fixed finite budget, at most64; no batch reduction')
    parser.add_argument('--max-seconds',type=int,default=600,help='POSIX setup+batch wall budget; Windows also requires external startup timeout')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--core-report',type=Path,help='Reuse PASSED check_sre report only when code identity matches')
    args=parser.parse_args();torch.set_num_threads(4)
    result=preflight(args)
    sys.exit(0 if result['status']=='PASSED' else 2)
