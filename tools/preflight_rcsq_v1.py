"""RCS-Q finite preflight. CPU/local CUDA never substitute for the real B16/640 gate."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
from io import StringIO
import logging
import os
from pathlib import Path
import time
import traceback
import warnings

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from init_rcsq_v1 import (ROOT, VARIANTS, ORIGINAL, build, build_training_model, controlled_models,
    code_fingerprint, initialized_path, require, runtime, sha256, write_json, verify_model)
from check_rcsq_v1 import (module_checks,dn_checks,initial_equivalence,native_loss_checks,reload_checks,
    fusion_checks,complexity,optimizer_check,synthetic_batch,targets,grad_norms)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML, LOGGER, ASSETS
from ultralytics.utils.patches import torch_load


def native_trainer_setup(initialized,expected,variant,output):
    """Construct the real Trainer and call native setup_model; stop before epochs."""
    import numpy as np
    import cv2
    fixture=output/'fixture'; (fixture/'images').mkdir(parents=True)
    (fixture/'labels').mkdir()
    for i in range(2):
        require(cv2.imwrite(str(fixture/'images'/f'{i}.png'),np.full((160,192,3),80+i*40,dtype=np.uint8)), 'Fixture image write')
        (fixture/'labels'/f'{i}.txt').write_text('0 0.5 0.5 0.3 0.2\n',encoding='utf-8')
    config=fixture/'data.yaml'
    YAML.save(config,dict(path=str(fixture),train='images',val='images',names={0:'crack'}))
    args=dict(model=str(initialized),data=str(config),device='cpu',batch=2,imgsz=160,workers=0,
              project=str(output),name='trainer_setup',exist_ok=False,optimizer='AdamW',lr0=.0005,
              weight_decay=.0001,seed=42,amp=True,plots=False,save=False)
    trainer=RTDETRTrainer(overrides=args)
    trainer.setup_model();trainer.set_model_attributes()
    state=trainer.model.state_dict()
    require(all(torch.equal(v,state[k]) for k,v in expected.state_dict().items()),'Real Trainer setup nc1 values changed')
    verify_model(trainer.model,variant,zero=True)
    _,groups=optimizer_check(trainer.model)
    return dict(status='PASSED',native_constructor=True,native_setup_model=True,common_state_exact=True,
                optimizer=groups,epochs_started=0,optimizer_steps=0,fixture_only=True)


def amp_resources_check(main,model,output):
    from train_c19_lif_v1 import ensure_amp_resources
    from ultralytics.utils.checks import check_amp
    ensure_amp_resources(Path(main),output/'amp_resources.json')
    require((ROOT/'yolo26n.pt').is_file() and (ASSETS/'bus.jpg').is_file(),'Actual native AMP resources absent')
    # check_amp historically returns True when a check is skipped; that is not a pass here.
    log=StringIO();handler=logging.StreamHandler(log);LOGGER.addHandler(handler)
    cwd=Path.cwd()
    try:
        os.chdir(ROOT)
        ok=check_amp(model)
    finally:
        os.chdir(cwd);LOGGER.removeHandler(handler)
    transcript=log.getvalue()
    (output/'native_amp.log').write_text(transcript,encoding='utf-8')
    require(ok and 'checks passed' in transcript and 'checks skipped' not in transcript,'Native AMP check failed or skipped')
    return dict(status='PASSED',resources=str(output/'amp_resources.json'),log=str(output/'native_amp.log'))


def actual_capacity(initialized,variant,data,output):
    """One real augmented training batch via native Trainer; no optimizer.step."""
    require(torch.cuda.is_available(),'CUDA required')
    class CapacityTrainer(RTDETRTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            result,self.adaptation=build_training_model(cfg,weights,self.data,variant)
            return result
    args=YAML.load(ROOT/'docs/rcsq_v1/cbr_lif_authoritative_args.yaml')
    args.update(model=str(initialized),data=str(data),project=str(output),name='capacity_setup',
                save_dir=str(output/'capacity_setup'),exist_ok=False)
    require(args['batch']==16 and args['imgsz']==640 and args['amp'] is True,'Formal capacity recipe changed')
    # Outputs are diagnostic-only; native data transform/model/loss/scaler/optimizer paths stay in force.
    trainer=CapacityTrainer(overrides=args)
    trainer._setup_train()
    require(trainer.amp and trainer.batch_size==16 and trainer.args.imgsz==640,'Native setup changed capacity recipe')
    require(torch.count_nonzero(trainer.model.model[-1].rcsq.out_proj.weight)==0,'Capacity initial RCS-Q changed')
    batch=trainer.preprocess_batch(next(iter(trainer.train_loader)))
    require(batch['img'].shape==(16,3,640,640),'Capacity loader did not produce full B16/640')
    # Native AMP scale calibration on one fixed batch; no parameter updates.
    import random
    import numpy as np
    model=trainer.model.train()
    require(type(trainer.optimizer) is torch.optim.AdamW and
            not getattr(trainer.optimizer, '_step_supports_amp_scaling', False),
            'Capacity calibration requires the original non-fused AdamW')
    require(trainer.scaler.is_enabled(), 'Native AMP scaler must remain enabled')
    buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    versions={k:p._version for k,p in model.named_parameters()}
    step_attempts=[0]
    def forbid_optimizer_update(*unused, **kwargs):
        step_attempts[0]+=1
        raise RuntimeError('Capacity probe attempted a real optimizer update')
    guard=trainer.optimizer.register_step_pre_hook(forbid_optimizer_update)
    scale_audit=dict(status='RUNNING',initial_scale=float(trainer.scaler.get_scale()),
                     backoff_factor=float(trainer.scaler.get_backoff_factor()),
                     max_attempts=17,optimizer_steps=0,same_batch_and_rng=True,trials=[])
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        for attempt in range(1,18):
            trainer.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for k,v in model.named_buffers(): v.copy_(buffers[k])
            random.setstate(rng[0]);np.random.set_state(rng[1])
            torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            scale=float(trainer.scaler.get_scale())
            row=dict(attempt=attempt,scale=scale)
            scale_audit['trials'].append(row)
            with torch.autocast('cuda',dtype=torch.float16):
                predictions=model.predict(batch['img'],batch=targets(batch))
                loss=model.loss(batch,preds=predictions)[0]
            finite_loss=bool(torch.isfinite(loss).all())
            row.update(loss=float(loss.detach()) if finite_loss else str(float(loss.detach())),
                       loss_finite=finite_loss)
            require(finite_loss, 'Capacity forward loss is nonfinite; scale backoff cannot fix it')
            trainer.scaler.scale(loss).backward()
            trainer.scaler.unscale_(trainer.optimizer)
            gradients=[(n,p.grad) for n,p in model.named_parameters() if p.grad is not None]
            require(gradients, 'Capacity backward produced no parameter gradients')
            bad=[n for n,g in gradients if not torch.isfinite(g).all()]
            row.update(gradients_finite=not bad,bad_gradients=bad)
            if not bad:
                g=model.model[-1].rcsq.out_proj.weight.grad
                require(g is not None and bool(torch.isfinite(g).all()) and
                        bool(torch.count_nonzero(g)>0), 'Capacity first RCS-Q gradient missing')
                row['rcsq_out_grad_norm']=float(g.detach().double().norm())
                require(versions=={k:p._version for k,p in model.named_parameters()},
                        'Capacity probe changed parameters')
                scale_audit['status']='FINITE_GRADIENTS'
                print(f'Capacity AMP finite at scale={scale}; optimizer updates=0',flush=True)
                break
            print(f'Capacity AMP overflow at scale={scale}; native backoff',flush=True)
            require(attempt<17, 'Capacity AMP remained nonfinite after bounded scale calibration')
            # Native GradScaler must skip the actual AdamW update on overflow.
            # The pre-hook aborts before any update if that assumption is violated.
            trainer.scaler.step(trainer.optimizer)
            require(step_attempts[0]==0 and
                    versions=={k:p._version for k,p in model.named_parameters()},
                    'Overflow trial attempted an optimizer update')
            trainer.scaler.update()
            row.update(native_step_skipped=True,next_scale=float(trainer.scaler.get_scale()))
            require(0<row['next_scale']<scale, 'Native GradScaler did not lower the scale')
            write_json(output/'amp_scale_calibration.json',scale_audit)
            del predictions,loss,gradients,bad
        require(scale_audit['status']=='FINITE_GRADIENTS', 'Capacity scale probe did not finish')
    except BaseException as error:
        scale_audit.update(status='FAILED',error=repr(error))
        raise
    finally:
        guard.remove()
        scale_audit['optimizer_step_attempts']=step_attempts[0]
        scale_audit['parameters_unchanged']=versions=={k:p._version for k,p in model.named_parameters()}
        write_json(output/'amp_scale_calibration.json',scale_audit)
    torch.cuda.synchronize()
    report=dict(status='PASSED',batch=16,imgsz=640,AMP=True,loss=float(loss),optimizer_steps=0,
                gt_groups=targets(batch)['gt_groups'],dn_num_split=predictions[-1]['dn_num_split'] if predictions[-1] else [0,300],
                total_queries=predictions[0].shape[2],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.perf_counter()-started,
                amp_scale_calibration=scale_audit,
                native_online_augmentation=True,native_trainer_setup=True,rcsq_gradients=grad_norms(model.model[-1].rcsq),
                samples=[str(p) for p in batch.get('im_file',[])])
    require(report['rcsq_gradients']['out_proj.weight']>0,'Capacity first gradient missing')
    del trainer,model,predictions,loss,batch;gc.collect();torch.cuda.empty_cache()
    return report


def diagnostic_summary(model,device):
    """Opt-in aggregate JSON only, no feature maps saved."""
    head=deepcopy(model.model[-1]).to(device).train()
    with torch.no_grad():head.rcsq.out_proj.weight.normal_(std=.01)
    features=[torch.randn(2,256,h,w,device=device) for h,w in ((20,24),(10,12),(5,6))]
    batch=targets(synthetic_batch(device,groups=(0,3)))
    with torch.no_grad():out,details=head.forward_with_diagnostics(features,batch)
    d=details.get('rcsq',details);D=out[-1]['dn_num_split'][0]
    def stats(t):
        t=t.float().flatten()
        return dict(min=float(t.min()),mean=float(t.mean()),max=float(t.max()),
                    quantiles=torch.quantile(t,torch.tensor([.1,.5,.9],device=t.device)).cpu().tolist())
    rows={}
    for name,section in (('DN',slice(0,D)),('normal',slice(D,None))):
        rows[name]=dict(null_weights=stats(d['null_weights'][:,section]),
                        valid_points=stats(d['valid_count'][:,section]),
                        correction_to_query_norm=stats(d['delta'][:,section].norm(dim=-1)/d['query'][:,section].float().norm(dim=-1).clamp_min(1e-12)))
    return dict(groups=rows,out_proj_norm=float(head.rcsq.out_proj.weight.norm()),
                scope='isolated nonzero diagnostic copy; aggregate statistics only',formal_updates=0)


def run_checks(source,init_dir,output,variant='both',device='cpu',data=None,capacity=False,main=None):
    source,init_dir,output=Path(source).resolve(),Path(init_dir).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    chosen=list(VARIANTS) if variant=='both' else [variant]
    require(device in ('cpu','cuda') and (device!='cuda' or torch.cuda.is_available()),'Requested device unavailable')
    require(not capacity or (device=='cuda' and data is not None),'Capacity requires CUDA and real data config')
    saved=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
           torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,
           torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.get_num_threads())
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True);torch.set_num_threads(4)
    report=dict(status='FAILED',runtime=runtime(),source_sha256=sha256(source),code_fingerprint=code_fingerprint()['sha256'],
                device=device,formal_training='NOT_STARTED',final_test='NOT_RUN',formal_optimizer_steps=0,
                variants={},capacity={'status':'PENDING','reason':'Requires actual server CUDA, B16/640 and real training loader'},
                native_amp={'status':'PENDING','reason':'Run with --capacity on server'},
                tolerance=dict(initial=[2e-6,2e-5],fuse_fp32=[2e-5,2e-4],fuse_half_amp=[3e-3,3e-2]),
                tf32_diagnostic=dict(matmul=False,cudnn=False,restored_on_exit=True))
    write_json(output/'checks.json',report)
    caught=[]
    try:
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            report['module']=module_checks(device);report['DN']=dn_checks(device)
            print('Module / DN probes passed',flush=True)
            models,mapping=controlled_models(source);report['initialization']=mapping
            adapted={}
            for name in chosen:
                path=initialized_path(init_dir,name)
                checkpoint=torch_load(path,map_location='cpu');weights=checkpoint['model'].float()
                provenance=checkpoint.get('rcsq_v1_provenance',{})
                require(provenance.get('source_sha256')==sha256(source) and provenance.get('variant')==name,
                        'Initialization provenance mismatch')
                require(provenance.get('code_fingerprint')==report['code_fingerprint'],'Initialization code fingerprint stale; generate a new init directory')
                require(all(torch.equal(v,weights.state_dict()[k]) for k,v in models[name].state_dict().items()),'Init differs from fixed-source construction')
                torch.manual_seed(42)
                model,adapt=build_training_model(weights.yaml,weights,dict(nc=1,channels=3),name)
                parent=build(ORIGINAL[name],nc=1)
                parent.load_state_dict({k:model.state_dict()[k] for k in parent.state_dict()},strict=True)
                folder=output/name;folder.mkdir()
                row=report['variants'][name]=dict(initialization_sha256=sha256(path),adaptation=adapt,
                    topology=verify_model(model,name,zero=True),status='RUNNING')
                row['native_trainer']=native_trainer_setup(path,model,name,folder)
                row['initial_equivalence']=initial_equivalence(parent,model,device)
                row['native_FP32_loss']=native_loss_checks(model,device)
                if device=='cuda':row['native_AMP_loss']=native_loss_checks(model,device,amp=True)
                row['reload_EMA']=reload_checks(model,device)
                row['fusion']=fusion_checks(model,device,folder/'fusion')
                row['diagnostics']=diagnostic_summary(model,device)
                row['status']='PASSED'; adapted[name]=model;adapted[ORIGINAL[name]]=parent
                write_json(output/'checks.json',report)
                print(name+' initial/train/loss/reload/fusion passed',flush=True)
            del models,weights;gc.collect()
            report['complexity']=complexity(adapted,device)
            if data is not None:
                from train_rcsq_v1 import verify_data_config
                report['data']=verify_data_config(Path(data))
            if capacity:
                require(main is not None,'--main required for trusted native AMP resources')
                probe=deepcopy(adapted[chosen[0]]).cuda()
                report['native_amp']=amp_resources_check(main,probe,output);del probe
                report['capacity']={}
                for name in chosen:report['capacity'][name]=actual_capacity(initialized_path(init_dir,name),name,Path(data),output/name)
                report['status']='PASSED'
            else:report['status']='LOCAL_PASSED_SERVER_PENDING'
            caught=[str(w.message) for w in observed]
    except BaseException as error:
        report['failure']=dict(error=repr(error),traceback=traceback.format_exc(),detail=getattr(error,'detail',None))
        raise
    finally:
        if 'observed' in locals():caught=[str(w.message) for w in observed]
        report['warnings']=caught
        report['determinism']='deterministic=True, warn_only=True; CUDA grid_sample backward is not guaranteed bitwise deterministic'
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_json(output/'checks.json',report)
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=saved[:2]
        torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=saved[2:4]
        torch.use_deterministic_algorithms(saved[4],warn_only=saved[5]);torch.set_num_threads(saved[6])
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','init-dir','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--variant',choices=['both',*VARIANTS],default='both')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--data',type=Path)
    parser.add_argument('--capacity',action='store_true')
    parser.add_argument('--main',type=Path,help='Existing main repository with trusted AMP resources')
    args=parser.parse_args()
    result=run_checks(args.source,args.init_dir,args.output,args.variant,args.device,args.data,args.capacity,args.main)
    print(result['status'],str(args.output/'checks.json'))
