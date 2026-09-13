"""Finite engineering acceptance only; never an epoch loop or full split evaluation."""
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import contextmanager,nullcontext
import gc
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import traceback
import warnings
from types import SimpleNamespace
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from init_c24_lif_v1 import *
from c24_lif_v1_numerics import *
from check_lif_down import module_checks
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from c24_lif_v1_acceptance import ACCEPTED, aggregate, fusion_status, stage_status, blocking_summary
from c24_lif_v1_loss import diagnostic_loss

def tree_exact(a,b):
    if torch.is_tensor(a):require(a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu()),'Historical/save-load tensor mismatch')
    elif isinstance(a,dict):
        require(a.keys()==b.keys(),'Tree keys differ')
        for k in a:tree_exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        require(len(a)==len(b),'Tree lengths differ')
        for x,y in zip(a,b):tree_exact(x,y)
    else:require(a==b,'Tree values differ')

def history(models,folder):
    from audit_c24_lif_v1 import COMMITS
    reports={};torch.manual_seed(444);image=torch.rand(1,3,160,192)
    for kind in ('c2','c24','lif'):
        with tempfile.TemporaryDirectory(dir=folder,prefix='history_'+kind+'_') as temp:
            temp=Path(temp).resolve();blob=subprocess.check_output(['git','archive',COMMITS[kind],'ultralytics-main/ultralytics'],cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(blob)) as t:
                for m in t.getmembers():
                    if m.isfile():
                        p=temp/m.name;require(p.resolve().is_relative_to(temp.resolve()),'Unsafe historical path')
                        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
            model=deepcopy(models[kind]);activate(model);model.eval()
            payload=dict(kind=kind,yaml=CONFIGS[kind],state=model.state_dict(),image=image)
            torch.save(payload,temp/'input.pt')
            values=[]
            for label,root in [('historical',temp),('current',ROOT)]:
                output=temp/(label+'.pt')
                result=subprocess.run([sys.executable,str(ROOT/'tools/c24_lif_v1_history_worker.py'),str(root),str(temp/'input.pt'),str(output)],
                    cwd=temp,env={**os.environ,'PYTHONPATH':str(root/'ultralytics-main')},capture_output=True,text=True)
                require(result.returncode==0,result.stdout+result.stderr)
                values.append(torch_load(output,map_location='cpu'))
            a,b=values;source=a.pop('import_path');b.pop('import_path');tree_exact(a,b)
            require(str(temp.resolve()) in source,'Historical import cache pollution')
            reports[kind]=dict(commit=COMMITS[kind],isolated_source='git archive '+COMMITS[kind],
                parameters=a['parameters'],states=a['states'],forward='EXACT',
                nonzero_input_parameter_gradients='EXACT' if kind!='c2' else 'NOT_APPLICABLE',
                pre_post_norm='EXACT' if kind=='c24' else 'NOT_APPLICABLE',
                original_constructor_initial_values={k:tensor_hash(v) for k,v in a['initial_new'].items()})
    return reports

def degeneration(models,device):
    rows=[]
    for shape in [(640,640),(160,192)]:
        torch.manual_seed(404);x=torch.rand(1,3,*shape,device=device)
        for kind,scca,lif in [('c2',False,False),('c24',True,False),('lif',False,True)]:
            a,b=deepcopy(models[kind]).to(device).eval(),deepcopy(models['combo']).to(device).eval()
            activate(a,scca,lif);activate(b,scca,lif)
            # Preserve the exact active parent's complete nonzero state (activation RNG
            # positions differ when a parent lacks the other module).
            b.load_state_dict({**b.state_dict(),**a.state_dict()},strict=True)
            with torch.no_grad():left,right=a(x)[0],b(x)[0]
            rows.append(dict(parent=kind,device=device,input=list(x.shape),comparison=require_compare(left,right,'final_prediction')))
            del a,b,left,right;gc.collect()
    return rows

def optimizer_check(model):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    opt=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    names={id(p):n for n,p in model.named_parameters() if p.requires_grad};ids=[id(p) for g in opt.param_groups for p in g['params']]
    require(len(ids)==len(set(ids))==len(names) and set(ids)==set(names),'Optimizer coverage mismatch')
    rows=[dict(group=g.get('param_group'),lr=g['lr'],weight_decay=g['weight_decay'],names=[names[id(p)] for p in g['params']]) for g in opt.param_groups]
    innovations=[dict(name=names[id(p)],group=g.get('param_group'),lr=g['lr'],weight_decay=g['weight_decay']) for g in opt.param_groups for p in g['params'] if added(names[id(p)])]
    require(len(innovations)==10 and all(v['group']=='weight' for v in innovations),'New optimizer group mismatch')
    # Verify native grouping for all parameters by inspecting their owning modules.
    norm=tuple(v for n,v in torch.nn.__dict__.items() if 'Norm' in n)
    expected={}
    for module_name,module in model.named_modules():
        for name,p in module.named_parameters(recurse=False):
            expected[id(p)]='bias' if 'bias' in name else 'bn' if isinstance(module,norm) or 'logit_scale' in name else 'weight'
    for group in opt.param_groups:
        require(all(expected[id(p)]==group['param_group'] for p in group['params']),'Native norm/bias classification changed')
    return opt,dict(coverage=len(ids),groups=rows,new_tensors=innovations,initial_accumulate=4,effective_decay=.0001)

def synthetic_batch():
    torch.manual_seed(902)
    return dict(img=torch.rand(2,3,160,192),cls=torch.zeros(5,1),bboxes=torch.tensor([[.5,.5,.2,.3],[.3,.4,.1,.2],[.7,.6,.1,.4],[.4,.4,.2,.2],[.6,.5,.3,.1]]),batch_idx=torch.tensor([0,0,1,1,1]))

def loss_checks(model,device,folder):
    from c24_lif_v1_loss import loss_suite
    return loss_suite(model,device,folder)


def ingress(model,device,half,folder):
    """Run one forward inside the real validator setup and real predictor backend."""
    sample=torch.rand(1,3,160,192,device=device);rows={}
    class ProbeDone(Exception):pass
    class ProbeValidator(RTDETRValidator):
        def init_metrics(self,actual):
            x=sample.half() if self.args.half else sample.float()
            out=actual(x);out=out[0] if isinstance(out,(tuple,list)) else out
            require(bool(torch.isfinite(out).all()),'Validator ingress nonfinite')
            net=actual.model if isinstance(actual,AutoBackend) else actual
            require(hasattr(net.model[locate(net.yaml)['p3_to_p4']['downsample']],'bn'),'Validator lost LIF BN')
            rows['ema_validator']=dict(actual_half=self.args.half,output_dtype=str(out.dtype),training=self.training,finite=True)
            raise ProbeDone()
    args=dict(model=str(MODEL_DIR/MODEL),data=str(ROOT/'docs/c24_lif_v1/c2_data.yaml'),imgsz=640,device='0' if device=='cuda' else 'cpu',plots=False,workers=0,half=half)
    validator=ProbeValidator(dataloader=[None],save_dir=Path(folder)/('entry_'+device),args=args)
    trainer=SimpleNamespace(device=torch.device(device),data=dict(nc=1,names={0:'crack'},channels=3),amp=half,
        ema=SimpleNamespace(ema=deepcopy(model).to(device)),model=None,args=SimpleNamespace(compile=False),
        loss_items=torch.zeros(3,device=device),stopper=SimpleNamespace(possible_stop=False),epoch=0,epochs=200)
    try:validator(trainer=trainer)
    except ProbeDone:pass
    require('ema_validator' in rows,'Validator probe not reached')
    predictor=RTDETRPredictor(overrides=args);predictor.setup_model(model=deepcopy(model),verbose=False)
    with torch.no_grad():out=predictor.model(sample.half() if half else sample)
    out=out[0] if isinstance(out,(tuple,list)) else out
    require(bool(torch.isfinite(out).all()),'Predict AutoBackend nonfinite')
    require(hasattr(predictor.model.model.model[20],'bn'),'AutoBackend lost LIF BN')
    rows['predict_AutoBackend']=dict(fp16=predictor.model.fp16,output_dtype=str(out.dtype),finite=True,fused=True)
    return rows

def fusion(model,device,mode,folder):
    a=deepcopy(model).to(device).eval();activate(a,bn=True)
    b=deepcopy(a).fuse(verbose=False);top=locate(a.yaml);index=top['p3_to_p4']['downsample']
    require(hasattr(b.model[index],'bn') and not hasattr(b.model[top['p4_to_p5']['downsample']],'bn'),'Normal fusion/LIF exclusion changed')
    tree_exact(a.model[index].state_dict(),b.model[index].state_dict())
    count=sum(isinstance(m,torch.nn.BatchNorm2d) for m in b.modules())
    state=deepcopy(b.state_dict());b.fuse(verbose=False);tree_exact(state,b.state_dict())
    if mode=='half':a.half();b.half()
    reports=[]
    for shape in [(640,640),(160,192)]:
        torch.manual_seed(120);x=torch.rand(1,3,*shape,device=device)
        if mode=='half':x=x.half()
        with torch.autocast('cuda',dtype=torch.float16) if mode=='amp' else nullcontext():row=compare_pair(a,b,x,mode)
        row.update(device=device,input=list(x.shape));reports.append(row)
    with tempfile.TemporaryDirectory(dir=folder,prefix='fuse_reload_') as tmp:
        path=Path(tmp)/'state.pt';torch.save(dict(model=b),path);loaded=torch_load(path,map_location=device)['model'];tree_exact(b.state_dict(),loaded.state_dict())
        with torch.autocast('cuda',dtype=torch.float16) if mode=='amp' else nullcontext():reload=compare_pair(b,loaded,x,mode)
    # Physical negative controls must visibly change the activated LIF result.
    m=deepcopy(a.model[index]).float();sample=torch.rand(1,256,11,13,device=device)
    with torch.no_grad():
        correct=m(sample);missing=m.act(m.bn(m.conv(sample)));wrong=m.act(m.bn(m.conv(sample))+m.residual(sample))
    require(compare(correct,missing,'missing_residual',2e-5,2e-4)['status']=='BLOCKED','Residual negative not detected')
    require(compare(correct,wrong,'wrong_bn',2e-5,2e-4)['status']=='BLOCKED','BN negative not detected')
    result=dict(mode=mode,device=device,remaining_bn=count,lif_bn_state_exact=True,repeat_fuse='EXACT',save_load=reload,
        cases=reports,physical_negatives='BLOCKED_AS_EXPECTED')
    result['status']=fusion_status(result)
    return result

def original_parent_cutoff(models,device,mode,folder):
    from audit_c24_lif_v1 import COMMITS,HASHES
    from c24_lif_v1_amp import model_evidence
    rows={}
    for kind in ('c24','lif'):
        a=deepcopy(models[kind]).to(device).eval()
        initial=model_evidence(a)
        activate(a,bn=True)
        active=model_evidence(a)
        b=deepcopy(a).fuse(verbose=False)
        if mode=='half':a.half();b.half()
        torch.manual_seed(120);x=torch.rand(1,3,640,640,device=device)
        if mode=='half':x=x.half()
        with torch.autocast('cuda',dtype=torch.float16) if mode=='amp' else nullcontext():
            result=compare_pair(a,b,x,mode)
        rows[kind]=dict(original_commit=COMMITS[kind],module_sha256=HASHES[kind],public_source_sha256=SOURCE_SHA256,
            initial_native_model=initial,nonzero_pressure_model=active,numerics=result,comparability='DIAGNOSTIC_ONLY',
            reason='Original parent architecture and mapped public/seed-42 weights; no combo prefix copied. Similar cutoff drift does not satisfy the strict exception.')
        write_json(Path(folder)/('parent_'+kind+'_'+mode+'.json'),rows[kind])
        del a,b;gc.collect()
    return dict(status='BLOCKED' if any(r['numerics']['status']=='BLOCKED' for r in rows.values()) else 'PASSED',
        scope='Original parent diagnostic audit, not combo cutoff acceptance',parents=rows)

@contextmanager
def diagnostic_settings():
    import random
    import numpy as np
    other=(random.getstate(),np.random.get_state(),torch.get_default_dtype(),torch.is_grad_enabled())
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,
           torch.backends.cudnn.deterministic,torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
    environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG']}
    with torch.random.fork_rng(devices=devices):
        try:
            torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
            yield
        finally:
            random.setstate(other[0]);np.random.set_state(other[1]);torch.set_default_dtype(other[2]);torch.set_grad_enabled(other[3])
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=flags[:4]
            torch.use_deterministic_algorithms(flags[4],warn_only=flags[5])
            for key,value in environment.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value

def preflight(source,output,server=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    report=dict(schema=SCHEMA,status='CHECKING',runtime=runtime(),fingerprint=fingerprint(),source_sha256=sha256(source),stages=[],
        formal_training='NOT_RUN',full_val_test='NOT_RUN',server_B16_640='NOT_RUN',scope='server' if server else 'local')
    def stage(name,fn,fatal=False):
        print('STAGE '+name+' | report '+str((output/'preflight.json').resolve()),flush=True);start=time.monotonic()
        row=dict(name=name,status='RUNNING');report['stages'].append(row);write_json(output/'preflight.json',report)
        try:
            result=fn();row.update(status=stage_status(name,result),result=result)
            if isinstance(result,dict) and result.get('error'):report['terminal_exception_stage']=name
            return result
        except BaseException as error:
            row.update(status='BLOCKED',error=repr(error),traceback=traceback.format_exc())
            report['terminal_exception_stage']=name
            if fatal or isinstance(error,(KeyboardInterrupt,SystemExit)):raise
            return dict(status='BLOCKED',error=repr(error))
        finally:
            row['seconds']=round(time.monotonic()-start,3);write_json(output/(name+'.json'),row);write_json(output/'preflight.json',report)
            write_json(output/'blocking_summary.json',blocking_summary(report))
            print('END '+name+' '+row['status']+' '+str(row['seconds'])+'s',flush=True)
    try:
        with diagnostic_settings():
            stage('negative_gate',negative_checks);stage('lif_original_unit',module_checks)
            with tempfile.TemporaryDirectory(dir=output,prefix='init_') as tmp:
                init=stage('initialization',lambda:initialize(source,Path(tmp)/'init.pt'),fatal=True)
                models80,_=controlled_models(source);models={}
                for kind in CONFIGS:
                    with torch.random.fork_rng(devices=[]):torch.manual_seed(42);models[kind]=native_rebuild(models80['combo'],kind)
                counts={k:dict(parameters=sum(p.numel() for p in m.parameters()),states=len(m.state_dict())) for k,m in models.items()}
                require(all(counts[k]['parameters']==COUNTS[k] for k in COUNTS),'Four model count mismatch')
                stage('structure',lambda:dict(counts=counts,topology=verify_model(models['combo'],zero=True)))
                del models80;gc.collect()
                if server:
                    from audit_c24_lif_v1 import HASHES,COMMITS
                    evidence=read_json(ROOT/'docs/c24_lif_v1/history.json')
                    require(evidence and evidence['status']=='PASSED','Missing committed local historical regression')
                    require(all(evidence['result'][k]['commit']==COMMITS[k] for k in COMMITS),'Historical source identity changed')
                    require(all(sha256(ROOT/'ultralytics-main/ultralytics/nn/modules'/name)==HASHES[k] for k,name in [('c24','scca_aifi.py'),('lif','lif_down.py')]),'Original module bytes changed')
                    stage('history',lambda:dict(scope='VERIFIED_COMMITTED_LOCAL_DYNAMIC_REGRESSION',source=str(ROOT/'docs/c24_lif_v1/history.json'),sha256=sha256(ROOT/'docs/c24_lif_v1/history.json')))
                else:stage('history',lambda:history(models,output))
                for device in ['cpu','cuda']:
                    if device=='cuda' and not torch.cuda.is_available():
                        stage('cuda_unavailable',lambda:dict(status='NOT_RUN',reason='CUDA unavailable'));continue
                    stage('degeneration_'+device,lambda:degeneration(models,device))
                    for mode in (['fp32'] if device=='cpu' else ['fp32','amp','half']):
                        stage('fusion_'+device+'_'+mode,lambda:fusion(models['combo'],device,mode,output))
                    if device=='cuda':
                        for mode in ('amp','half'):
                            stage('original_parent_cutoff_cuda_'+mode,lambda:original_parent_cutoff(models,device,mode,output))
                    stage('native_initialization_small_'+device,lambda:diagnostic_loss(models['combo'],device,output,label='native_initialization_small_'+device))
                    stage('native_loss_'+device,lambda:loss_checks(models['combo'],device,output))
                    stage('ingress_'+device,lambda:ingress(models['combo'],device,device=='cuda',output))
                    gc.collect()
                    if device=='cuda':torch.cuda.empty_cache()
                if server:
                    require(torch.cuda.is_available(),'Server CUDA required')
                    from train_c24_lif_v1 import real_capacity_batch
                    holder={}
                    def get_batch():
                        batch,evidence=real_capacity_batch();holder['batch']=batch
                        return dict(status='PASSED',data=evidence)
                    stage('real_capacity_batch',get_batch)
                    for label,stress in [('native_initialization_B16_640',False),('nonzero_branch_stress_B16_640',True)]:
                        def capacity():
                            require('batch' in holder,'Real capacity batch unavailable')
                            return diagnostic_loss(models['combo'],'cuda',output,label=label,batch=holder['batch'],stress=stress)
                        stage(label,capacity)
                        gc.collect();torch.cuda.empty_cache()
                    report['server_B16_640']=dict(status=aggregate([r['status'] for r in report['stages'] if r['name'] in
                        ('real_capacity_batch','native_initialization_B16_640','nonzero_branch_stress_B16_640')]))
            report['status']=aggregate([r['status'] for r in report['stages']])
    except BaseException as error:
        report.update(status='BLOCKED',error=repr(error))
        raise
    finally:
        report['first_failure']=next((dict(name=r['name'],error=r.get('error'),status=r['status']) for r in report['stages'] if r['status'] not in ACCEPTED),None)
        write_json(output/'preflight.json',report)
        write_json(output/'blocking_summary.json',blocking_summary(report))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--server',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    result=preflight(args.source,args.output,args.server)
    require(result['status'] in ACCEPTED,'Preflight '+result['status']+'; inspect '+str(args.output/'blocking_summary.json'))
