"""4090 readiness: real data, full model, 640/batch16, native optimizer/AMP/EMA.

Independent process/run only. No formal checkpoints, test evaluation or recipe fallback.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid
from support import (ROOT, HERE, SOURCE, ASSET, canonical, configure, digest, git,
                     native_recipe, read_json, run_identity, sha256, status, write_json)
from configuration import candidate, DEFAULT_DATA, DEFAULT_DATA_ROOT


def gpu_snapshot():
    import torch
    torch.cuda.synchronize()
    commands = (['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version','--format=csv,noheader'],
                ['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
    rows = [subprocess.run(c,capture_output=True,text=True) for c in commands]
    return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
            'mem_get_info_free_total':list(torch.cuda.mem_get_info()),
            'gpu':rows[0].stdout.strip(),'processes':rows[1].stdout.strip(),
            'nvidia_smi_errors':[r.stderr.strip() for r in rows if r.returncode]}


class LimitedValidationLoader:
    """One real batch through the unmodified native validator; never formal metrics."""
    def __init__(self,loader):
        self.loader=loader;self.dataset=loader.dataset;self.batch_size=loader.batch_size;self.num_workers=loader.num_workers
    def __len__(self):
        return 1
    def __iter__(self):
        yield next(iter(self.loader))


class SmokeBudgetCompleted(Exception):
    pass


def check(args):
    import torch
    from backend import (attention_scope, flash_evidence, flash_forward_count,
                         native_fp32, optional_flash_parity, resolve_train_backend, ArithmeticAudit)
    from bootstrap import verify_frozen_environment
    from data import preflight
    from run import load_initial_model
    from adapters import ComparisonTrainer, capture_rng, restore_rng, tensor_digest, strict_amp_probe
    config,_,_,_=candidate(args.config,args.set)
    if config['imgsz']!=640 or config['amp'] is not True or config['deterministic'] is not True:
        raise ValueError('Readiness requires exact 640 / requested physical batch / AMP / deterministic; never reduce on OOM')
    if config['train_attention_backend']=='native':
        raise ValueError('Flash readiness requires a flash/auto candidate')
    source=configure(ROOT/'.runtime/yolov13l-configurable/flash-readiness-runtime')
    resolution=resolve_train_backend(config['train_attention_backend'],config['amp'],config['deterministic'])
    if resolution['resolved']!='flash':
        raise RuntimeError('4090 startup readiness resolved native: '+resolution['reason'])
    if git('status','--porcelain','--untracked-files=normal'):
        raise ValueError('Startup readiness requires the fixed clean implementation worktree')
    source=configure(ROOT/'.runtime/yolov13l-configurable/flash-readiness-runtime',backend_resolution=resolution)
    environment=verify_frozen_environment()
    run=ROOT/'outputs/yolov13l-configurable-validation'/('SMOKE_ONLY_flash640_'+uuid.uuid4().hex[:10])
    run.mkdir(parents=True)
    report={'scope':'SMOKE_ONLY_REAL_DATA_640_BATCH'+str(config['batch']),'formal_training_started':False,
            'run':str(run),'config':config,'config_sha256':digest(canonical(config)),
            'code_sha':git('rev-parse','HEAD'),'adapter_sha256':source['adapter_sha256'],
            'patch_sha256':source['patch_sha256'],'patched_source_sha256':source['patched_source_sha256'],
            'source':source,'backend_resolution':resolution,
            'environment_identity_sha256':digest(canonical(environment)),
            'data':str(args.data.absolute()),'data_root':str(args.data_root.absolute()),
            'profiler':False,'layer_observer':False,
            'precision_policy':'author AMP with Flash train; native validator with original precision; independent native FP32'}
    stage='data'
    try:
        manifest=preflight(args.data,run,args.data_root)
        report['dataset_identity_sha256']=manifest['dataset_identity_sha256']
        stage='operator_and_AAttn_parity'
        report['parity']=optional_flash_parity()
        if report['parity']['status']!='PASSED':
            raise RuntimeError('AAttn area1/4 output/input/parameter gradient parity failed; tolerance is fixed')
        stage='actual_model_AMP_reference'
        model,initialization=load_initial_model(config=config)
        torch.cuda.reset_peak_memory_stats()
        before=capture_rng();weights={k:tensor_digest(v) for k,v in model.state_dict().items()}
        probe=deepcopy(model).float().cuda()
        strict_amp_probe(probe)
        del probe
        restore_rng(before)
        after=capture_rng()
        import numpy as np
        if (before['python']!=after['python'] or before['numpy'][0]!=after['numpy'][0]
                or not np.array_equal(before['numpy'][1],after['numpy'][1]) or before['numpy'][2:]!=after['numpy'][2:]
                or not torch.equal(before['torch_cpu'],after['torch_cpu'])
                or len(before['torch_cuda'])!=len(after['torch_cuda'])
                or any(not torch.equal(a,b) for a,b in zip(before['torch_cuda'],after['torch_cuda']))):
            raise RuntimeError('AMP reference changed RNG state')
        if weights!={k:tensor_digest(v) for k,v in model.state_dict().items()}:
            raise RuntimeError('AMP reference altered original COCO/BN state')
        report['actual_model_AMP_reference']={'status':'PASSED','native_reference':True,'weights_BN_RNG_preserved':True,
                                              'no_optimizer_EMA_scaler_created_on_original':True,'memory':gpu_snapshot()}
        del model
        torch.cuda.empty_cache()
        identity=run_identity(manifest,source,run_id=run.name,config=config,environment=environment,require_clean=True)
        overrides=native_recipe(config)
        overrides.update(model=str(ASSET),data=str(run/'data.yaml'),project=str(run),name='train',exist_ok=False)
        trainer=ComparisonTrainer(overrides=overrides,run=run,manifest=manifest,identity=identity,config=config)
        batches={'count':0};rows=[]
        stage='640_batch16_training'
        def setup(t):
            if t.batch_size!=config['batch'] or t.train_loader.batch_size!=config['batch'] or t.test_loader.batch_size!=config['batch'] or t.test_loader.num_workers!=0 or not t.amp:
                raise RuntimeError('Actual smoke setup differs from physical batch16 / native AMP / val workers0')
            state=t.model.state_dict()
            transferred=read_json(run/'initialization.json')
            if any(tensor_digest(state[k])!=h for k,h in transferred['matching_tensor_sha256'].items()):
                raise RuntimeError('COCO tensors/gates changed before smoke optimizer step')
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            report['native_setup']={'optimizer':type(t.optimizer).__name__,'batch':t.batch_size,
                'nbs':t.args.nbs,'scaler_initial_scale':t.scaler.get_scale(),
                'initialization_tensors_verified':transferred['transferred_tensors'],
                'protected_tensors':transferred['protected_tensor_count'],
                'groups':[{k:g.get(k) for k in ('lr','initial_lr','momentum','betas','eps','weight_decay','nesterov')} for g in t.optimizer.param_groups]}
        def batch_end(t):
            batches['count']+=1
            rows.append({'batch':batches['count'],'accumulate':t.accumulate,'loss':t.loss.detach().cpu().tolist(),
                         'optimizer_updates':t.comparison_optimizer_updates,'optimizer_steps':t.comparison_optimizer_steps,
                         'overflow_skips':t.comparison_overflow_skips,'EMA_updates':t.ema.updates,'memory':gpu_snapshot()})
            if batches['count']>=3 and t.comparison_optimizer_updates>=1:
                raise SmokeBudgetCompleted()
            if batches['count']>=16:
                raise RuntimeError('16 real AMP batches produced no optimizer update; keep native scaler evidence, do not lower it')
        trainer.add_callback('on_pretrain_routine_end',setup)
        trainer.add_callback('on_train_batch_end',batch_end)
        try:
            with attention_scope('flash','readiness_training',True):
                trainer.train()
        except SmokeBudgetCompleted:
            pass
        if batches['count']<3 or trainer.comparison_optimizer_updates<1 or trainer.ema.updates<1:
            raise RuntimeError('Smoke did not actually update optimizer and EMA')
        report['training_batches']=rows;report['training_memory']=gpu_snapshot()
        stage='native_training_validation'
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        from ultralytics.nn.modules import block
        with attention_scope('flash','readiness_training',True):
            trainer.validator.dataloader=LimitedValidationLoader(trainer.test_loader)
            calls=flash_forward_count()
            trainer.validator(trainer)
            if not block.USE_FLASH_ATTN or flash_forward_count()!=calls:
                raise RuntimeError('Validation Flash/native scope restoration or zero-Flash audit failed')
            report['training_validation']={'status':'PASSED_ONE_REAL_VAL_BATCH_NOT_FORMAL_METRICS',
                'precision':trainer.validator.actual_precision,'restored_training_flash':True,'flash_calls':0,
                'memory':gpu_snapshot()}
        # Release the training optimizer/EMA GPU allocation before public FP32 smoke.
        public_model=deepcopy(trainer.ema.ema).cpu().float().eval()
        val_images=[str(Path(manifest['data_root'])/r['image']) for r in manifest['records'] if r['split']=='val'][:16]
        for loader in (trainer.train_loader,trainer.test_loader):
            iterator=getattr(loader,'iterator',None)
            if iterator and hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        del trainer
        import gc
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        stage='public_native_FP32'
        from ultralytics import YOLO
        from adapters import SquarePredictor
        from export import SETTINGS
        wrapper=YOLO(str(ASSET),task='detect')
        wrapper.model=public_model  # Smoke selected EMA, not the original COCO80 model.
        wrapper.overrides.update(task='detect')
        with native_fp32(),torch.no_grad(),ArithmeticAudit(require_fp32=True) as audit:
            results=list(wrapper.predict(source=val_images,predictor=SquarePredictor,stream=True,
                         **SETTINGS,project=str(run/'public_smoke'),name='val',exist_ok=True))
        if (len(results)!=16 or wrapper.predictor.model.fp16
                or next(wrapper.predictor.model.model.parameters()).dtype!=torch.float32):
            raise RuntimeError('Actual public predictor omitted images or used non-FP32 parameters')
        arithmetic=audit.report()
        if arithmetic['flash_forward_calls']!=0:
            raise RuntimeError('Public FP32 executed Flash')
        report['public_native_FP32']={'status':'PASSED_REAL_VAL_IMAGE_BATCH_NOT_PUBLIC_METRICS',
            'arithmetic':arithmetic,'input_dtype':'torch.float32; actual SquarePredictor asserts every batch',
            'actual_images':len(results),'predictor':'SquarePredictor with original export.SETTINGS',
            'parameter_dtype':str(next(wrapper.predictor.model.model.parameters()).dtype),'autocast':{'cpu':False,'cuda':False},
            'flash_calls':0,'internal_half_casts':arithmetic['internal_half_cast_calls'],'memory':gpu_snapshot()}
        report['flash_evidence']=flash_evidence()
        if not any(r['phase']=='readiness_training' and r['calls']>0 for r in report['flash_evidence']['calls']):
            raise RuntimeError('No observed real author flash_attn_func during 640/batch16 training')
        report.update(status='PASSED_640_STARTUP_READINESS',exit_code=0,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      scope_note='No formal epochs/checkpoints or test metrics; timings under shared GPU are not paper speed')
        write_json(args.out,report)
        write_json(ROOT/'.runtime/yolov13l-configurable/flash_readiness.json',
                   {'status':report['status'],'report':str(args.out.absolute()),'report_sha256':sha256(args.out),
                    **{k:report[k] for k in ('config_sha256','code_sha','adapter_sha256','patch_sha256','patched_source_sha256','environment_identity_sha256','dataset_identity_sha256','data','data_root')}})
        status(run,'readiness','completed',exit_code=0)
        return report
    except BaseException as exc:
        report.update(status='FAILED_STARTUP_READINESS',failed_stage=stage,error=str(exc),
                      traceback=traceback.format_exc(),exit_code=130 if isinstance(exc,KeyboardInterrupt) else 1,
                      training_batches=locals().get('rows',[]),flash_evidence=flash_evidence())
        if torch.cuda.is_available():
            try:report['failure_memory']=gpu_snapshot()
            except Exception as memory_error:report['failure_memory_error']=str(memory_error)
        write_json(args.out,report)
        status(run,'readiness','failed',exit_code=report['exit_code'],failed_stage=stage,error=str(exc))
        raise


def verify_readiness(config, data, data_root):
    from bootstrap import environment_identity
    from support import adapter_hash, LOCK
    lock=read_json(LOCK)
    pointer=read_json(ROOT/'.runtime/yolov13l-configurable/flash_readiness.json')
    if (pointer['status']!='PASSED_640_STARTUP_READINESS' or pointer['code_sha']!=git('rev-parse','HEAD')
            or pointer['config_sha256']!=digest(canonical(config))
            or pointer['adapter_sha256']!=adapter_hash() or pointer['patch_sha256']!=lock['patch_sha256']
            or pointer['patched_source_sha256']!=lock['patched_source_sha256']
            or pointer['environment_identity_sha256']!=digest(canonical(environment_identity()))
            or pointer['data']!=str(Path(data).absolute()) or pointer['data_root']!=str(Path(data_root).absolute())
            or sha256(pointer['report'])!=pointer['report_sha256']):
        raise ValueError('640/batch16 startup readiness missing/changed or belongs to another recipe/code/environment/data; rerun flash_checks.py')
    return pointer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path);p.add_argument('--set',action='append',default=[])
    p.add_argument('--data',type=Path,default=DEFAULT_DATA);p.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Preserve earlier readiness evidence; use a new --out')
    try:
        result=check(args)
        print(json.dumps({'status':result['status'],'report':str(args.out)}))
    except BaseException as exc:
        if not args.out.exists():
            torch=__import__('torch')
            compatible=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]>=8
            write_json(args.out,{'status':'FAILED_BEFORE_SMOKE' if compatible else 'NOT_RUN',
                       'error':str(exc),'traceback':traceback.format_exc(),'formal_training_started':False})
        raise


if __name__=='__main__':main()
