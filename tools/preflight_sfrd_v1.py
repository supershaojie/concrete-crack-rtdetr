"""Finite SFR-D preflight, on disposable models only; no epochs or full val/test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from init_sfrd_v1 import (ROOT, VARIANTS, SOURCE_SHA256, require, sha256, write_json, runtime,
    controlled_models, build_training_model, verify_model, new_keys, tensor_hash, RTDETR)
from check_sfrd_v1 import module_checks, real_blocks, activate_gate, ema_check
from check_c19_lif_v1 import optimizer, activate
from c19_lif_v1_data import real_batch
from c19_lif_v1_probe import capture, targets
from c19_lif_v1_diagnostic import fusion_protocol
from c19_lif_v1_cutoff import fusion_accepted
from train_sfrd_v1 import SFRDTrainer, source_hashes, verify_data
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load


def detection_step(model, batch, device, amp=False, opt=None):
    """Native criterion with matching/path hooks; no artificial parameter losses."""
    model.train(); model.nc=1
    if opt is None: opt,_=optimizer(model)
    opt.zero_grad(set_to_none=True)
    batch={k:v.to(device) for k,v in batch.items()}
    require(batch['bboxes'].numel()>0,'Detection check needs valid targets')
    model.criterion=model.init_criterion(); matched=[]; calls={}
    def match_hook(module,args,result): matched.append(sum(len(a) for a,b in result))
    handles=[model.criterion.matcher.register_forward_hook(match_hook)]
    def path_hook(name):
        def hook(module,args,out): calls[name]=calls.get(name,0)+1
        return hook
    for i in (6,7,20):handles.append(model.model[i].register_forward_hook(path_hook(str(i))))
    if hasattr(model.model[-1],'cbr'): handles.append(model.model[-1].cbr.register_forward_hook(path_hook('cbr')))
    try:
        with torch.autocast(device_type=device,enabled=amp,dtype=torch.float16):
            loss=model.loss(batch)[0]
        loss.backward()
    finally:
        for h in handles:h.remove()
    gradients={n:float(p.grad.float().norm()) if p.grad is not None else None for n,p in model.named_parameters() if n in new_keys(model.sfrd_variant)}
    require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Nonfinite detection loss/gradient')
    require(all(v is not None and v>0 for v in gradients.values()),'A/B/gate detection gradients absent')
    require(matched and min(matched)>0 and all(calls.get(k,0)>0 for k in ('6','7','20','cbr')),'Matching/CBR/LIF path missing')
    return dict(status='PASSED',loss=float(loss.detach()),matched_targets=matched,path_calls=calls,
                gradient_norms=gradients,AMP=amp,device=device,batch=len(batch['img']),imgsz=batch['img'].shape[-1],optimizer_steps=0)


def actual_trainer(initialized,dataset,folder):
    """Full native Trainer setup and one real batch backward; stop before the epoch loop."""
    fixture=folder/'trainer_fixture'
    for part in ('images/train','labels/train','images/val','labels/val'):(fixture/part).mkdir(parents=True)
    files=sorted((dataset/'images/train').glob('*.jpg'))[:2]
    for image in files:
        for split in ('train','val'):
            shutil.copyfile(image,fixture/'images'/split/image.name)
            label=dataset/'labels/train'/image.with_suffix('.txt').name
            shutil.copyfile(label,fixture/'labels'/split/label.name)
    data=fixture/'data.yaml';YAML.save(data,dict(path=str(fixture.resolve()),train='images/train',val='images/val',names={0:'crack'}))
    observed={}
    class Done(Exception):pass
    class AuditTrainer(SFRDTrainer):
        def train(self):
            self._setup_train()
            expected=self.sfrd_initial_report['state_sha256']
            require({k:tensor_hash(v) for k,v in self.model.state_dict().items()} == expected,'Native setup changed initial tensors')
            ids=[id(p) for g in self.optimizer.param_groups for p in g['params']]
            require(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in self.model.parameters()},'Native optimizer inventory differs')
            require({k:tensor_hash(v) for k,v in self.ema.ema.state_dict().items()} == expected,'Native EMA changed initial state')
            batch=self.preprocess_batch(next(iter(self.train_loader)))
            observed.update(detection_step(self.model,{k:batch[k] for k in ('img','cls','bboxes','batch_idx')},'cpu',opt=self.optimizer))
            observed.update(actual_model_train=True,native_setup_train=True,native_optimizer=type(self.optimizer).__name__,
                initialized_exact_before_backward=True,native_ema_initial_exact=True,parameter_tensors=len(ids),
                bounded_override_reason='CPU 2 training-image engineering fixture; does not substitute CUDA B16',
                formal_recipe_unchanged=True,train_report=self.sfrd_initial_report)
            raise Done()
    config=YAML.load(ROOT/'docs/c19_lif_v1/c2_args.yaml')
    config.update(model=str(initialized),data=str(data),device='cpu',batch=2,imgsz=160,workers=0,plots=False,
                  project=str(folder/'native_trainer'),name='setup_only',save_dir=str(folder/'native_trainer/setup_only'))
    with patch('ultralytics.engine.model.checks.check_pip_update_available'):
        try:RTDETR(str(initialized)).train(trainer=AuditTrainer,**config)
        except Done:pass
    require(observed,'Native Trainer not reached')
    return observed


def reload_probe(checkpoint,model,folder,label):
    image=torch.rand(1,3,160,192,generator=torch.Generator().manual_seed(814))
    model=deepcopy(model).eval().cpu()
    with torch.no_grad():output=model(image)[0]
    payload=folder/(label+'_expected.pt');result=folder/(label+'_reload.json')
    torch.save(dict(image=image,output=output,state_sha256={k:tensor_hash(v) for k,v in model.state_dict().items()}),payload)
    with (folder/(label+'_reload.log')).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(ROOT/'tools/sfrd_v1_reload.py'),'--checkpoint',str(checkpoint),
            '--expected',str(payload),'--output',str(result)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads(result.read_text(encoding='utf-8'))


def fusion_checks(target,device,folder):
    results={}
    for nonzero in (False,True):
        model=deepcopy(target).eval()
        if nonzero:
            activate_gate(model);activate(model,lif=True,cbr=True,bn=True)
        model.to(device)
        fused=deepcopy(model).fuse(verbose=False)
        for i in (6,7):
            path=fused.model[i].blocks[1].branch2b
            require(not hasattr(path,'norm') and path.B_3x1.bias is not None,'Whole-model fuse did not fold B/BN2')
            require(hasattr(path,'gate'),'Fusion removed dynamic gate')
            for name,v in model.model[i].blocks[1].branch2b.gate.state_dict().items():
                require(torch.equal(v,path.gate.state_dict()[name]),'Fusion changed gate')
        image=torch.rand(1,3,160,192,generator=torch.Generator().manual_seed(42)).to(device)
        state='nonzero' if nonzero else 'initial'
        # Use the committed candidate-aware capture/replay and acceptance rules unchanged.
        diagnostic=fusion_protocol(model,fused,image,folder/(device+'_'+state+'_fusion'),device,'fp32')
        require(fusion_accepted(diagnostic,device,'fp32'),'Fusion requires review; no tolerance/selection changes allowed')
        before={k:tensor_hash(v) for k,v in fused.state_dict().items()}
        fused.fuse(verbose=False)
        require(before=={k:tensor_hash(v) for k,v in fused.state_dict().items()},'Repeat fusion changed state')
        results[state]=diagnostic
        if device=='cuda':
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):out=fused(image)[0]
            require(torch.isfinite(out).all(),'Fused AMP inference nonfinite')
            results[state+'_AMP_inference']=dict(status='PASSED',dtype=str(out.dtype))
        del model,fused;gc.collect()
    return results


def complexity(models,pair):
    import thop
    rows={}
    image=torch.zeros(1,3,640,640)
    for name,model in {'cbr_lif':pair,**models}.items():
        model=deepcopy(model).eval()
        unfused=sum(p.numel() for p in model.parameters())
        fused=deepcopy(model).fuse(verbose=False)
        with torch.no_grad():
            macs,_=thop.profile(deepcopy(fused),inputs=(image,),verbose=False)
            unfused_macs,_=thop.profile(deepcopy(model),inputs=(image,),verbose=False)
        rows[name]=dict(parameters_unfused=unfused,parameters_fused=sum(p.numel() for p in fused.parameters()),
            GFLOPs_fused_actual_640=2*macs/1e9,GFLOPs_unfused_actual_640=2*unfused_macs/1e9,
            parameter_delta_by_top_layer={str(i):sum(p.numel() for p in model.model[i].parameters())-sum(p.numel() for p in fused.model[i].parameters()) for i in range(27) if sum(p.numel() for p in model.model[i].parameters()) != sum(p.numel() for p in fused.model[i].parameters())})
    return dict(status='PASSED',input=[1,3,640,640],dtype='FP32',device='cpu',tool='thop '+str(thop.__version__),
        convention='2 FLOPs/MAC; actual full 640 forward; THOP module hooks omit tanh, elementwise calibration, functional attention/grid sampling and other unregistered operations',
        models=rows,analytic_convolution_GFLOPs_reduction=1.5616,latency='NOT_MEASURED',FPS=None,
        fusion_note='Native BaseModel does not fuse original ConvNormLayer; only new B+BN2 support is added. Each new BN2 fusion reduces C parameters (768 total), beyond the supplied estimate.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','cache','initialized','data','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--variant',choices=list(VARIANTS),default='cbr_lif_sfrd_v1')
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--capacity-b16',action='store_true')
    args=parser.parse_args()
    require(args.variant=='cbr_lif_sfrd_v1','This complete preflight targets the first-round main combination; other variants receive construction/init/output checks')
    folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    report=dict(status='RUNNING',variant=args.variant,runtime=runtime(),source_sha256=SOURCE_SHA256,
        initialized_sha256=sha256(args.initialized),source_hashes=source_hashes(),formal_training='NOT_RUN',full_val_test='NOT_RUN',
        capacity=dict(status='PENDING_CUDA_B16'),cuda=dict(status='NOT_RUN'))
    try:
        report['algebra_and_gate']=module_checks();print('Algebra and gate checks passed',flush=True)
        base,pair,models,audit=controlled_models(args.source,args.cache)
        report['real_blocks']=real_blocks(pair,models)
        report['controlled_common_states']=audit['cross_variant_shared_states_exact']
        nc1={};report['variants']={}
        for variant,initial in models.items():
            torch.manual_seed(42)
            nc1[variant],adapt=build_training_model(initial.yaml,initial,dict(nc=1,channels=3),variant,initial=True)
            nc1[variant].eval()
            path=args.initialized.parent/(variant+'_init.pt');loaded=RTDETR(str(path)).model
            require(all(torch.equal(v,loaded.state_dict()[k]) for k,v in initial.state_dict().items()),'Saved init differs from controlled mapping')
            with torch.no_grad():output=nc1[variant](torch.rand(1,3,640,640))[0]
            require(tuple(output.shape)==(1,300,5) and torch.isfinite(output).all(),'640 output interface')
            report['variants'][variant]=dict(trainer=adapt,structure=verify_model(nc1[variant],variant,zero=True),output_shape=list(output.shape),
                reload=reload_probe(path,nc1[variant],folder,variant))
        common=set.intersection(*(set(m.state_dict()) for m in nc1.values()))
        require(all(torch.equal(nc1['sfrd_v1'].state_dict()[k],m.state_dict()[k]) for m in nc1.values() for k in common),'nc1 cross-variant common states differ')
        report['cross_variant_nc1_exact']=len(common)
        torch.manual_seed(42)
        pair1,_=__import__('init_c19_lif_v1').build_training_model(pair.yaml,pair,dict(nc=1,channels=3))
        target=nc1[args.variant]
        require(all(torch.equal(v,target.state_dict()[k]) for k,v in pair1.state_dict().items() if k not in __import__('init_sfrd_v1').REMOVED),'Original pair nc1 public values differ')
        report['parent_nc1_public_exact']=True
        write_json(folder/'complexity.json',complexity(nc1,pair1));print('Three variants, reload and profile passed',flush=True)
        report['ema']=ema_check(target)
        trained=deepcopy(target).eval();activate_gate(trained)
        nonzero=folder/'disposable_nonzero.pt';torch.save(dict(model=trained,epoch=0,train_args={'task':'detect'}),nonzero)
        report['nonzero_reload']=reload_probe(nonzero,trained,folder,'nonzero')
        dataset,report['dataset']=verify_data(args.data)
        report['cpu']=dict(fuse=fusion_checks(target,'cpu',folder))
        report['actual_trainer']=actual_trainer(args.initialized,dataset,folder)
        batch,records=real_batch(dataset,160,2);report['samples']=records
        report['cpu']['loss']=detection_step(deepcopy(target),batch,'cpu')
        write_json(folder/'preflight.json',report);print('CPU Trainer, loss and whole-model fusion passed',flush=True)
        if args.cuda:
            require(torch.cuda.is_available(),'CUDA requested but unavailable')
            report['cuda']=dict(fuse=fusion_checks(target,'cuda',folder))
            for amp,key in ((False,'loss'),(True,'AMP_loss')):
                disposable=deepcopy(target).cuda()
                report['cuda'][key]=detection_step(disposable,batch,'cuda',amp)
                del disposable;gc.collect();torch.cuda.empty_cache()
                print('CUDA '+key+' passed',flush=True)
        if args.capacity_b16:
            require(args.cuda,'B16 requires --cuda')
            disposable=deepcopy(target).cuda();large,samples=real_batch(dataset,640,16)
            try:
                torch.cuda.reset_peak_memory_stats()
                report['capacity']=detection_step(disposable,large,'cuda',True)
                report['capacity'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),samples=samples)
            except torch.cuda.OutOfMemoryError as error:
                report['capacity']=dict(status='PENDING_CUDA_B16',reason='CUDA_OUT_OF_MEMORY',error=str(error),batch=16,imgsz=640,AMP=True,optimizer_steps=0)
            finally:
                del disposable,large;gc.collect();torch.cuda.empty_cache()
        report['status']='PASSED' if report['capacity']['status']=='PASSED' and args.cuda else 'LOCAL_CHECKS_PASSED_PENDING_CUDA_B16'
    except BaseException as error:
        report.update(status='FAILED',error=repr(error));raise
    finally:write_json(folder/'preflight.json',report)


if __name__=='__main__':main()
