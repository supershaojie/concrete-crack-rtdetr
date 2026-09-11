"""Bounded local validation of the three fixed variants; no dataset loop or formal training."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import io
from pathlib import Path
import time
import torch
from sci_adapter import *
from init_sci_adapter import controlled_models, initialize, rebuild_audit
from check_sci_adapter_gradients import run_checks, activate
from ultralytics import RTDETR
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import CSCEFv51, SCCAAIFI, CrackBoundaryRefinement
from triad_compat import baseline_roles
from ultralytics.utils.patches import torch_load


def tensors(value, path="output"):
    if isinstance(value,torch.Tensor): yield path,value
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value): yield from tensors(v,f"{path}.{i}")
    elif isinstance(value,dict):
        for k,v in value.items(): yield from tensors(v,f"{path}.{k}")


def errors(a,b,exact=False):
    require(a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(), "Invalid equivalence tensors")
    d=(a.detach().float()-b.detach().float()).abs()
    result=dict(max_abs_error=float(d.max()) if d.numel() else 0,
                max_rel_error=float((d/a.detach().float().abs().clamp_min(1e-8)).max()) if d.numel() else 0)
    if exact: require(torch.equal(a,b),f"Non-exact values: {result}")
    return result


def nested_errors(a,b,exact=True):
    ta,tb=dict(tensors(a)),dict(tensors(b)); require(ta.keys()==tb.keys(),"Nested output structure changed")
    return {k:errors(v,tb[k],exact) for k,v in ta.items()}


def feature_forward(model,image,roles,retain=False,batch=None):
    captured={}; handles=[]; order=[]
    def hook(role):
        def record(module,args,out):
            order.append(role)
            captured[role]=out
            if retain and out.requires_grad: out.retain_grad()
        return record
    for role in ("P3_ref","P4_ref","P3_base","P4_base","P5_base","P3_enh"):
        handles.append(model.model[roles[role]].register_forward_hook(hook(role)))
    def decoder_in(m,args): captured["decoder_inputs"]=args[0]
    handles.append(model.model[-1].register_forward_pre_hook(decoder_in))
    try: output=model.predict(image,batch=batch)
    finally:
        for h in handles:h.remove()
    return output,captured,order


def equivalence(reference,target,variant):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    reference.to(device).eval();target.to(device).eval()
    for m in (reference,target):m.model[-1].shapes=None
    roles=topology(target,variant)['roles']; br=baseline_roles(reference.yaml);br['P3_enh']=br['P3_ref']
    report={}
    for height,width in ((640,640),(160,192)):
        torch.manual_seed(51);image=torch.randn(1,3,height,width,device=device)
        with torch.no_grad():
            a,af,_=feature_forward(reference,image,br); b,bf,order=feature_forward(target,image,roles)
        result=nested_errors(a,b)
        for name in ('P3_ref','P4_ref','P3_base','P3_enh','P4_base','P5_base'):
            result[name]=dict(**errors(af[name],bf[name],True),shape=list(bf[name].shape))
        for slot,role in enumerate(('P3_base','P4_base','P5_base')):
            require(bf['decoder_inputs'][slot] is bf[role], f'Decoder does not use {role}')
        if VARIANTS[variant]['cbr']:require(len(bf['decoder_inputs'])==3,'CBR must use final P3, not a fourth reference')
        require(order.index('P3_enh')<order.index('P3_base')<order.index('P4_base')<order.index('P5_base'),'Original CSCEF/PAN propagation changed')
        require([list(bf[k].shape[-2:]) for k in ('P3_base','P4_base','P5_base')]==[[height//s,width//s] for s in (8,16,32)],'Bad pyramid shape')
        report[f'{height}x{width}']=result
    reference.cpu();target.cpu()
    for m in (reference,target):m.model[-1].shapes=None
    return report


def optimizer_coverage(model):
    trainer=object.__new__(BaseTrainer)
    optimizer=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    require(len(ids)==len(set(ids))==len(list(model.parameters())), 'Duplicate/missing optimizer parameter')
    require(set(ids)=={id(p) for p in model.parameters()},'Optimizer coverage gap')
    bn=tuple(v for k,v in torch.nn.__dict__.items() if 'Norm' in k)
    expected={}
    for name,m in model.named_modules():
        for leaf,p in m.named_parameters(recurse=False):
            full=f'{name}.{leaf}' if name else leaf
            expected[id(p)]=0. if 'bias' in full or isinstance(m,bn) or 'logit_scale' in full else .0001
    names={id(p):n for n,p in model.named_parameters()}; added=new_keys(model)
    rows=[]
    for g in optimizer.param_groups:
        require(all(g['weight_decay']==expected[id(p)] for p in g['params']), 'Native bias/norm decay rule drift')
        rows.append(dict(group=g.get('param_group'),decay=g['weight_decay'],count=len(g['params']),
            new_parameters=[names[id(p)] for p in g['params'] if names[id(p)] in added]))
    return optimizer,dict(all_parameters_once=True,groups=rows)


def smoke(target,variant,device='cpu',amp=False):
    model=deepcopy(target).to(device).train(); model.nc=1; model.model[-1].shapes=None
    optimizer,coverage=optimizer_coverage(model)
    scaler=torch.cuda.amp.GradScaler(enabled=amp,init_scale=128.)  # diagnostic only
    rows=[]; roles=topology(model,variant)['roles']
    # Original CSCEF output zero delays SCI restore learning until the second
    # update; SCI's own zero restore delays GN/reduce learning one more update.
    for step,(bs,counts) in enumerate(((2,[2,1]),(1,[3]),(2,[1,4]))):
        torch.manual_seed(60+step)
        image=torch.rand(bs,3,160,160,device=device)
        total=sum(counts)
        batch=dict(img=image,cls=torch.zeros(total,1,device=device),
            bboxes=torch.rand(total,4,device=device)*.3+.3,
            batch_idx=torch.repeat_interleave(torch.arange(bs,device=device),torch.tensor(counts,device=device)))
        targets=dict(cls=batch['cls'].long().flatten(),bboxes=batch['bboxes'],batch_idx=batch['batch_idx'],gt_groups=counts)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext():
            predictions,features,_=feature_forward(model,image,roles,retain=True,batch=targets)
            loss,components=model.loss(batch,preds=predictions)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        require(torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), 'Nonfinite native loss/gradient')
        added={n:None if p.grad is None else float(p.grad.float().abs().sum()) for n,p in model.named_parameters() if n in new_keys(model)}
        require(all(v is not None for v in added.values()), 'New parameter missing native loss gradient')
        heads={n:v for n,v in added.items() if any(k in n for k in ('output_projection','scca_o','offset_out','restore'))}
        original_heads={n:v for n,v in heads.items() if 'restore' not in n}
        require(all(v>0 for v in original_heads.values()),'Original zero-init output head did not learn')
        if step==0:require(all(v==0 for n,v in added.items() if n.startswith('model.18.')),'SCI gradient should wait for original CSCEF output')
        if step>=1:require(all(v>0 for n,v in heads.items() if 'restore' in n),'SCI restore did not learn after CSCEF opened')
        if step>=2:require(all(v>0 for v in added.values()),f'Third-step new parameter gradient missing: {added}')
        main={k:float(features[k].grad.float().abs().sum()) for k in ('P3_ref','P4_ref','P3_base','P4_base','P5_base')}
        require(all(v>0 for v in main.values()), 'Whole model main feature gradient absent')
        query_count=predictions[0].shape[2];dn=predictions[-1]
        require(query_count==sum(dn['dn_num_split']) and dn['dn_num_split'][1]==300 and query_count>300,'Native DN layout changed')
        scaler.step(optimizer);scaler.update()
        rows.append(dict(step=step,batch=bs,gt_groups=counts,loss=float(loss),components=components.tolist(),
                         query_count=query_count,dn_split=dn['dn_num_split'],new_gradients=added,main_feature_gradients=main))
    # Nonzero full model serialization; fresh constructors must not zero loaded state.
    model.eval().cpu();model.model[-1].shapes=None
    buffer=io.BytesIO();torch.save(dict(model=model,optimizer=optimizer.state_dict()),buffer);buffer.seek(0)
    checkpoint=torch_load(buffer,map_location='cpu');loaded=checkpoint['model'];loaded.model[-1].shapes=None
    reloaded_optimizer,_=optimizer_coverage(loaded)
    reloaded_optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer_tensors=nested_errors(checkpoint['optimizer'],reloaded_optimizer.state_dict())
    optimizer_errors=dict(tensors_compared=len(optimizer_tensors),all_exact=True,
        max_abs_error=max(r['max_abs_error'] for r in optimizer_tensors.values()),
        max_rel_error=max(r['max_rel_error'] for r in optimizer_tensors.values()))
    require(checkpoint['optimizer']['param_groups']==reloaded_optimizer.state_dict()['param_groups'],'Optimizer group reload drift')
    require(all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items()),'Learned model state reload mismatch')
    probe=torch.rand(1,3,160,192)
    with torch.no_grad(): reload_errors=nested_errors(model(probe),loaded(probe))
    rebuilt=__import__('init_sci_adapter').rebuild_nc1(loaded)
    rebuild_audit(loaded,rebuilt,variant,zero=False)
    result=dict(status='passed',device=device,amp=amp,debug_scaler_initial=128 if amp else None,steps=rows,
        optimizer_coverage=coverage,save_load=reload_errors,optimizer_save_load=optimizer_errors,
        native_rebuild_preserves_learned=True)
    del model,loaded,rebuilt,optimizer,reloaded_optimizer,checkpoint;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def precision(target):
    if not torch.cuda.is_available():return dict(status='NOT_RUN',reason='CUDA unavailable')
    model=deepcopy(target).cuda().eval();model.model[-1].shapes=None;activate(model)
    image=torch.rand(1,3,160,192,device='cuda')
    with torch.no_grad():
        fp=model(image)
        with torch.autocast('cuda',dtype=torch.float16):mixed=model(image)
    comparison=nested_errors(fp,mixed,False)
    model.half();model.model[-1].shapes=None
    with torch.no_grad():half=model(image.half())
    require(all(torch.isfinite(t).all() for _,t in tensors(half)),'True half nonfinite')
    return dict(status='passed',AMP_vs_FP32=comparison,true_half={k:str(v.dtype) for k,v in tensors(half)},shape=list(half[0].shape))


def routing_interventions(target,variant):
    if not VARIANTS[variant]['cscef']:return dict(status='not_applicable')
    model=deepcopy(target).eval();roles=topology(model,variant)['roles'];activate(model)
    image=torch.rand(1,3,160,192)
    with torch.no_grad():
        _,before,_=feature_forward(model,image,roles)
        with torch.no_grad():model.model[roles['P3_enh']].output_projection.weight.mul_(3)
        _,after,_=feature_forward(model,image,roles)
    for name in ('P3_ref','P4_ref'):
        require(torch.equal(before[name],after[name]),f'CSCEF unexpectedly changed upstream {name}')
    differences={name:float((before[name]-after[name]).abs().max()) for name in ('P3_enh','P3_base','P4_base','P5_base')}
    require(all(v>0 for v in differences.values()),'Original CSCEF residual no longer propagates through PAN')
    return dict(status='passed',original_PAN_propagation=True,max_abs_changes=differences)



def legacy_regression():
    results={}
    for label,file in [('c2',BASE_YAML),('c17','rtdetr-resnet18-lite-cscef-v51.yaml'),('c19','rtdetr-resnet18-lite-cbr.yaml'),('c24','rtdetr-resnet18-lite-scca.yaml')]:
        m=RTDETRDetectionModel(str(MODEL_DIR/file),nc=1,verbose=False).eval()
        with torch.no_grad():out=m(torch.rand(1,3,160,192))
        require(all(torch.isfinite(t).all() for _,t in tensors(out)),'Legacy regression nonfinite')
        results[label]=dict(parameters=sum(p.numel() for p in m.parameters()),output_shape=list(out[0].shape))
    return results


def run(source,output,variants=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=dict(runtime=runtime(),scope='bounded synthetic; no formal training/full dataset val/test',variants={})
    report['gradient_edge']=run_checks();report['legacy_regression']=legacy_regression()
    write_json(output/'gradient_edge.json',report['gradient_edge'])
    for variant in variants or VARIANTS:
        print('VALIDATING',variant,flush=True)
        reference,target,init=controlled_models(source,variant)
        write_json(output/f'{variant}_initialization.json',init)
        result=dict(parameters=sum(p.numel() for p in target.parameters()),topology=topology(target,variant))
        result['zero_init_C2']=equivalence(reference,target,variant)
        from check_sci_adapter_gradients import equivalence as sci_equivalence, routing
        if not VARIANTS[variant]['cbr']:
            result['zero_init_legacy']=sci_equivalence(variant,'cuda' if torch.cuda.is_available() else 'cpu')
            result['nonzero_routing']=routing(variant,'cuda' if torch.cuda.is_available() else 'cpu')
        result['routing']=routing_interventions(target,variant)
        result['FP32_loss_DN']=smoke(target,variant,'cpu',False)
        result['AMP_loss_DN']=smoke(target,variant,'cuda',True) if torch.cuda.is_available() else {'status':'NOT_RUN'}
        result['precision']=precision(target)
        from sci_adapter_stats import statistics
        from check_sci_adapter_gradients import capture
        probe=deepcopy(target).eval();activate(probe)
        with torch.no_grad():_,f=capture(probe,torch.rand(1,3,160,192))
        result['adapter_stats']=statistics(probe.model[18],f['semantic'],f['compat'])
        del probe
        # Exercise actual serialized RTDETR and YAML.load paths, not only state_dict.
        file=output/variant/'init.pt'
        result['checkpoint']=initialize(source,file,variant)
        from_yaml=RTDETR(str(MODEL_DIR/VARIANTS[variant]['yaml'])).load(str(file)).model
        require(from_yaml.model[-1].nc==80, 'YAML API unexpected nc')
        loaded_states=from_yaml.state_dict();source_states=target.state_dict()
        skipped=[k for k,v in source_states.items() if v.shape!=loaded_states[k].shape]
        require(len(skipped)==9 and all(torch.equal(v,loaded_states[k]) for k,v in source_states.items() if k not in skipped),
                'RTDETR(YAML).load failed to preserve shape-compatible common/new values')
        result['checkpoint']['yaml_load_class_adaptation']=skipped
        result['checkpoint']['yaml_load_exact_states']=len(source_states)-len(skipped)
        report['variants'][variant]=result
        write_json(output/f'{variant}_validation.json',result)
        write_json(output/'validation.json',report)
        del reference,target,from_yaml;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variant',choices=VARIANTS,action='append')
    a=p.parse_args();torch.set_num_threads(4);run(a.source,a.output,a.variant)
