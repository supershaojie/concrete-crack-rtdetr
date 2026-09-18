"""Bounded DCC network audit on real train images; never epochs, val or test."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import gc
import hashlib
from pathlib import Path
from types import SimpleNamespace
import time

import torch
from dcc_common import (ROOT, controlled_models, native_rebuild, verify_model,
                        require, sha256, write_json, runtime, build, DCC_KEYS)
from c19_lif_v1_data import dataset_inventory, real_batch
from c19_lif_v1_probe import capture, targets
from c19_lif_v1_diagnostic import rng_state, restore_rng, selection_report
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import DCCConv
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA

ATOL, RTOL = 2e-5, 2e-4  # Declared before any run; never tuned to outcomes.


def metric(a, b, atol=ATOL, rtol=RTOL):
    a, b = a.detach().cpu(), b.detach().cpu()
    require(a.shape == b.shape, 'Shape mismatch')
    if not a.is_floating_point():
        return dict(allclose=bool(torch.equal(a,b)), max_abs=0 if torch.equal(a,b) else 1)
    finite = torch.isfinite(a) & torch.isfinite(b)
    special = torch.equal(a[~finite], b[~finite]) and not torch.isnan(a).any() and not torch.isnan(b).any()
    aa, bb = a[finite].double(), b[finite].double()
    diff = (aa-bb).abs()
    bad = diff > atol + rtol*bb.abs()
    return dict(allclose=bool(special and not bad.any()), max_abs=float(diff.max()) if diff.numel() else 0.,
                relative_L2=float(torch.linalg.vector_norm(aa-bb)/torch.linalg.vector_norm(bb).clamp_min(1e-12)),
                exceeded_fraction=float(bad.double().mean()) if bad.numel() else 0., atol=atol,rtol=rtol)


def assert_state(a,b):
    require(set(a)==set(b),'State keys differ')
    require(all(torch.equal(v.detach().cpu(), b[k].detach().cpu()) for k,v in a.items()),'State values differ')


@contextmanager
def strict_precision():
    old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old


def wiring(model, image):
    values={}; handles=[]
    def save(name):
        def hook(m,a,y): values[name]=y.detach().clone()
        return hook
    for i in (5,16,17,18,19,20,22,25):handles.append(model.model[i].register_forward_hook(save(str(i))))
    handles.append(model.model[17].act.register_forward_hook(save('projection')))
    handles.append(model.model[17].dcc.register_forward_pre_hook(lambda m,a:values.update(dcc_input=a[0].detach().clone())))
    handles.append(model.model[20].register_forward_pre_hook(lambda m,a:values.update(down_input=a[0].detach().clone())))
    handles.append(model.model[26].register_forward_pre_hook(lambda m,a:values.update(decoder_inputs=[v.detach().clone() for v in a[0]])))
    try:
        with torch.no_grad(): model.eval()(image)
    finally:
        for h in handles:h.remove()
    require(torch.equal(values['projection'],values['dcc_input']),'DCC must follow complete Conv/BN/act')
    require(torch.equal(values['18'],torch.cat([values['16'],values['17']],1)),'Concat order changed')
    require(torch.equal(values['19'],values['down_input']),'P3 downsample source changed')
    for i,v in zip((19,22,25),values['decoder_inputs']):require(torch.equal(values[str(i)],v),'Decoder input changed')
    return dict(status='PASSED',nodes=len(model.model),shapes={k:list(v.shape) for k,v in values.items() if isinstance(v,torch.Tensor)},
                dcc_after_complete_projection=True,concat=[16,17],p3_downsample=20,decoder_from=[19,22,25])


def trace_compare(a,b,keys):
    for k in keys:
        if k not in ('anchors','selected_anchors','reference_boxes'):
            require(torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all(),'Nonfinite continuous tensor '+k)
    return {k:metric(a[k],b[k]) for k in keys}


def equivalence(left,right,image,enforce=True):
    with torch.no_grad(): _,a=capture(left.eval(),image);_,b=capture(right.eval(),image)
    keys=['scale_0','scale_1','scale_2','projection_0','projection_1','projection_2','encoder_features','candidate_scores']
    continuous=trace_compare(a,b,keys)
    if enforce:require(all(v['allclose'] for v in continuous.values()),'Continuous pre-selection mismatch')
    selection=selection_report(a,b)
    outputs=trace_compare(a,b,['boxes','scores','raw_boxes','raw_scores'])
    same=torch.equal(a['candidate_indices'],b['candidate_indices'])
    replay={}
    if not same:
        # Diagnostic only: no model source or production selection changes.
        with torch.no_grad(): _,aa=capture(left,image,fixed_ids=a['candidate_indices']);_,bb=capture(right,image,fixed_ids=a['candidate_indices'])
        replay=trace_compare(aa,bb,['selected_features','reference_boxes','boxes','scores','raw_boxes','raw_scores'])
        if enforce:require(all(v['allclose'] for v in replay.values()),'Fixed-candidate continuous path mismatch')
    elif enforce:require(all(v['allclose'] for v in outputs.values()),'Same-candidate output mismatch')
    passed=all(v['allclose'] for v in continuous.values()) and all(v['allclose'] for v in (outputs if same else replay).values())
    return dict(status='PASSED' if same and passed else 'PRECISION_NOTE',continuous=continuous,selection=selection,
                candidate_indices_a=a['candidate_indices'].tolist(),candidate_indices_b=b['candidate_indices'].tolist(),
                native_outputs=outputs,fixed_candidate_diagnostic_only=replay)


def optimizer(model):
    t=RTDETRTrainer.__new__(RTDETRTrainer)
    t.args=SimpleNamespace(lr0=.0005,warmup_bias_lr=.1,weight_decay=.0001)
    opt=t.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    require(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad},'Optimizer coverage mismatch')
    require(all(p.requires_grad for p in model.parameters()),'Unexpected frozen public parameter')
    return opt


def scaler(enabled):
    # PyTorch 2.1.2-compatible signature, default native scale (65536).
    return torch.cuda.amp.GradScaler(enabled=enabled)


def do_step(t,batch,amp,gradient_capture=None):
    t.model.train();t.model.nc=1;t.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=batch['img'].device.type,enabled=amp):
        pred, taps=capture(t.model,batch['img'],targets(batch));loss=t.model.loss(batch,preds=pred)[0]
    require(torch.isfinite(loss).all(),'Nonfinite forward loss')
    t.scaler.scale(loss).backward()
    scale=t.scaler.get_scale()
    grads={n:float(p.grad.detach().float().norm()/scale) if p.grad is not None and torch.isfinite(p.grad).all() else None
           for n,p in t.model.named_parameters() if n.startswith('model.17.dcc.')}
    public=t.model.model[0].conv.weight.grad
    require(public is not None,'Missing public backbone gradient')
    if gradient_capture is not None:
        gradient_capture.update({n:p.grad.detach().float().cpu().clone()/scale
                                 for n,p in t.model.named_parameters() if p.grad is not None})
    before={n:p.detach().clone() for n,p in t.model.named_parameters() if n.startswith('model.17.dcc.')}
    t.optimizer_step()
    after=t.scaler.get_scale();effective=after>=scale
    changed=any(not torch.equal(v,dict(t.model.named_parameters())[n]) for n,v in before.items())
    if effective:require(changed and all(v is not None for v in grads.values()),'Effective step lost DCC or finite gradients')
    return dict(loss=float(loss.detach()),scale_before=scale,scale_after=after,skipped=not effective,
                dcc_grad_norms=grads,dcc_changed=changed,gt=int(batch['bboxes'].shape[0]),dn_split=taps['dn_split'],
                forward_hashes={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()
                                for k,v in taps.items() if isinstance(v,torch.Tensor)} if gradient_capture is not None else None)


def make_trainer(model,amp,device):
    t=RTDETRTrainer.__new__(RTDETRTrainer);t.model=model.to(device);t.optimizer=optimizer(t.model)
    t.scaler=scaler(amp);t.ema=ModelEMA(t.model);return t


def lifecycle(t,batch,amp,folder,variant):
    folder.mkdir(parents=True,exist_ok=False)
    m=t.model.eval();require(torch.count_nonzero(m.model[17].dcc.W_o.weight)>0,'Lifecycle needs actually updated W_o')
    torch.save(m.state_dict(),folder/'state.pt')
    with torch.random.fork_rng(devices=[]): rebuilt=native_rebuild(m.yaml,m,1,3).to(batch['img'].device)
    rebuilt.load_state_dict(torch_load(folder/'state.pt',map_location=batch['img'].device),strict=True)
    assert_state(m.state_dict(),rebuilt.state_dict())
    torch.save(dict(model=deepcopy(m).cpu(),epoch=-1,train_args=dict(task='detect')),folder/'model.pt')
    loaded=RTDETR(str(folder/'model.pt')).model.to(batch['img'].device)
    assert_state(m.state_dict(),loaded.state_dict());verify_model(loaded,variant,zero=False)
    wide=build(variant,nc=80)
    wide.load_state_dict({**wide.state_dict(),**{k:v.detach().cpu() for k,v in m.state_dict().items()
                         if v.shape==wide.state_dict()[k].shape}},strict=True)
    adapted=native_rebuild(wide.yaml,wide,nc=1)
    require(all(torch.equal(adapted.state_dict()[k],m.state_dict()[k].cpu()) for k in DCC_KEYS),
            'Learned DCC lost during actual nc80-to1 reconstruction')
    del wide,adapted
    ema_copy=deepcopy(t.ema.ema).eval()
    with strict_precision(),torch.no_grad():
        ema_comparison=metric(t.ema.ema.eval()(batch['img'])[0],ema_copy(batch['img'])[0])
    require(ema_comparison['allclose'],'EMA own-copy comparison failed')
    # Execute native checkpoint serializer, setup_model, and resume_training.
    # Parent saves EMA/optimizer moments in half. Direct control uses that same
    # quantized checkpoint state, not the pre-serialization FP32 live model.
    t.args=SimpleNamespace(**{**DEFAULT_CFG_DICT,'model':str(folder/'last.pt'),'epochs':200,'close_mosaic':10,
                             'data':'diagnostic_only','task':'detect'})
    t.epoch=0;t.best_fitness=0.;t.fitness=0.;t.metrics={};t.save_period=-1
    t.wdir=folder;t.last=folder/'last.pt';t.best=folder/'best.pt';t.csv=folder/'absent_results.csv'
    t.save_model();ckpt=torch_load(t.last,map_location='cpu')
    moment_underflow=sum(int(((v['exp_avg_sq'].detach().cpu()!=0)&(ckpt['optimizer']['state'][k]['exp_avg_sq'].cpu()==0)).sum())
                         for k,v in t.optimizer.state_dict()['state'].items() if 'exp_avg_sq' in v)
    resumed=RTDETRTrainer.__new__(RTDETRTrainer);resumed.args=deepcopy(t.args)
    resumed.model=str(t.last);resumed.data=dict(nc=1,channels=3);resumed.epochs=200;resumed.resume=True
    loaded_ckpt=resumed.setup_model();resumed.model.to(batch['img'].device)
    resumed.optimizer=optimizer(resumed.model);resumed.scaler=scaler(amp);resumed.ema=ModelEMA(resumed.model)
    resumed.resume_training(loaded_ckpt)
    require(resumed.start_epoch==1 and resumed.scaler.state_dict()==ckpt['scaler'],'Native epoch/scaler restore mismatch')
    require(resumed.ema.updates==ckpt['updates'],'EMA updates restore mismatch')
    assert_state(resumed.ema.ema.state_dict(),ckpt['ema'].float().state_dict())
    # Both controls need identical ephemeral decoder caches as well as state_dict.
    # A serialized EMA can carry half-quantized, nonpersistent anchor caches;
    # native setup_model constructs fresh ones. Copy that structure, then restore
    # the independently loaded checkpoint weights/optimizer without resume logic.
    direct_model=deepcopy(resumed.model).requires_grad_(True)
    direct_model.load_state_dict(ckpt['ema'].float().state_dict(),strict=True)
    direct=make_trainer(direct_model,amp,batch['img'].device)
    direct.optimizer.load_state_dict(deepcopy(ckpt['optimizer']));direct.scaler.load_state_dict(ckpt['scaler'])
    direct.ema.ema.load_state_dict(ckpt['ema'].float().state_dict());direct.ema.updates=ckpt['updates']
    assert_state(direct.model.state_dict(),resumed.model.state_dict())
    for k,values in direct.optimizer.state_dict()['state'].items():
        for name,v in values.items():
            other=resumed.optimizer.state_dict()['state'][k][name]
            require(torch.equal(v,other) if isinstance(v,torch.Tensor) else v==other,'Optimizer restore mismatch')
    states=[deepcopy(direct.optimizer.state_dict()),deepcopy(resumed.optimizer.state_dict())]
    ga,gb={},{}
    rng=rng_state();a=do_step(direct,batch,amp,ga);restore_rng(rng);b=do_step(resumed,batch,amp,gb)
    errors={k:metric(v,resumed.model.state_dict()[k]) for k,v in direct.model.state_dict().items()}
    grad_errors={k:metric(v,gb[k]) for k,v in ga.items()}
    raw_equal=all(v['allclose'] for v in errors.values())
    write_json(folder/'resume_comparison.json',dict(direct=a,resumed=b,errors=errors,gradient_errors=grad_errors))
    require(all(torch.isfinite(v).all() for v in direct.model.state_dict().values()) and
            all(torch.isfinite(v).all() for v in resumed.model.state_dict().values()),'Nonfinite resumed state')
    require(set(ga)==set(gb) and all(torch.isfinite(v).all() for v in list(ga.values())+list(gb.values())),
            'Missing/nonfinite resumed gradient')
    require(a['forward_hashes']==b['forward_hashes'] and a['loss']==b['loss'],
            'Same-state resumed forward or loss differs')
    # Keep the predeclared FP32 gradient comparison result, including AMP failures.
    # Exact full forward identity isolates any remaining difference to backward.
    # Do not relabel raw CUDA trajectories as equal or widen their tolerance.
    if not batch['img'].is_cuda:
        require(all(v['allclose'] for v in grad_errors.values()),'CPU resumed gradient mismatch')
    replay_exact=False
    if not raw_equal:
        require(batch['img'].is_cuda,'CPU native resume update mismatch')
        # A strict diagnostic separates native CUDA backward variation from state
        # restoration. Replay identical unscaled gradients through the same native
        # optimizer_step on both independently restored optimizer states. Never
        # used in formal training. Do not relax the raw comparison tolerance.
        replays=[]
        for state in states:
            replica=make_trainer(deepcopy(resumed.model).requires_grad_(True),False,batch['img'].device)
            replica.model.load_state_dict(ckpt['ema'].float().state_dict(),strict=True)
            replica.optimizer.load_state_dict(state)
            for n,p in replica.model.named_parameters():p.grad=ga[n].to(p.device,p.dtype).clone() if n in ga else None
            replica.optimizer_step()
            replays.append({k:v.detach().cpu().clone() for k,v in replica.model.state_dict().items()})
            del replica
        assert_state(replays[0],replays[1]);replay_exact=True
        del replays
    require(not a['skipped'] and not b['skipped'],'Resume comparison must include effective updates')
    return dict(status='PASSED' if raw_equal else 'PRECISION_NOTE',state_dict_exact=True,full_model_exact=True,native_get_model_nonzero_preserved=True,
                ema_own_copy=ema_comparison,native_save_model=True,native_setup_model=True,native_resume_training=True,
                diagnostic_epoch_metadata=0,completed_training_epochs=0,start_epoch=resumed.start_epoch,
                optimizer_scaler_ema_restored=True,quantization='native EMA half + optimizer moments half; same quantized direct control',
                learned_nc80_to1=True,ephemeral_caches='same fresh native reconstruction in both controls',
                raw_next_update_allclose=raw_equal,fixed_gradient_replay_exact=replay_exact,
                gradient_allclose=all(v['allclose'] for v in grad_errors.values()),gradient_max_abs=max(v['max_abs'] for v in grad_errors.values()),
                captured_forward_hashes_exact=True,loss_exact=True,optimizer_second_moment_nonzero_to_half_zero=moment_underflow,
                precision_note=None if raw_equal else 'Exact native restored state, forward tensors and loss; CUDA backward gradients differ (original tolerance results retained). Native serializer underflow is counted. Amplification by quantized moments is a numerical inference. Raw next weights NOT allclose; identical-gradient native optimizer replay is exact. This proves restore consistency, not native trajectory equality. No formal policy changed.',
                direct_step=a,resumed_step=b,max_state_abs=max(v['max_abs'] for v in errors.values()),
                max_state_relative_L2=max(v.get('relative_L2',0.) for v in errors.values()),
                checkpoint_sha256=sha256(t.last))


def fusion(model,image,folder):
    model=deepcopy(model).eval();fused=deepcopy(model).fuse(verbose=False)
    require(type(fused.model[17]) is DCCConv and fused.model[17].forward.__func__ is DCCConv.forward_fuse,'Fuse bypasses subclass')
    assert_state(model.model[17].dcc.state_dict(),fused.model[17].dcc.state_dict())
    calls=[];h=fused.model[17].dcc.register_forward_hook(lambda m,a,y:calls.append(1))
    try:
        with torch.no_grad():fused(image)
    finally:h.remove()
    require(len(calls)==1,'DCC must execute exactly once after fusion')
    default=equivalence(model,fused,image,enforce=False)
    with strict_precision(): strict=equivalence(model,fused,image)
    result=dict(status='PASSED',dcc_calls=1,dcc_state_exact=True,default=default,strict_fp32=strict,
                unfused_parameters=sum(p.numel() for p in model.parameters()),fused_parameters=sum(p.numel() for p in fused.parameters()))
    if image.is_cuda:
        half=deepcopy(model).half();half_fused=deepcopy(fused).half()
        with torch.no_grad():
            ho=half(image.half())[0];hf=half_fused(image.half())[0]
            with torch.autocast('cuda',dtype=torch.float16):ao=model(image)[0]
        require(torch.isfinite(ho).all() and torch.isfinite(hf).all() and torch.isfinite(ao).all(),'Nonfinite half/AMP inference')
        result['cuda_half']=dict(status='PASSED',shape=list(ho.shape),same_precision_natural_output=metric(ho,hf,3e-3,3e-2),
                                 note='Output comparison alone is not operator equivalence; candidate ordering may change')
    return result


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    previous=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    report=dict(status='FAILED',variant=args.variant,runtime=runtime(),initialization_sha256=sha256(args.initialized),
                formal_training='NOT_STARTED',final_test='NOT_RUN',tolerances=dict(atol=ATOL,rtol=RTOL),devices={},
                precision=dict(deterministic_warn_only=True,cudnn_deterministic=True,cudnn_benchmark=False,
                               matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32))
    try:
        parent,target,report['controlled_initialization']=controlled_models(args.source,args.variant)
        initial=RTDETR(str(args.initialized)).model
        assert_state(initial.state_dict(),target.state_dict());verify_model(initial,args.variant,zero=True)
        report['data_identity']=dataset_inventory(args.real_dataset)
        batch,report['samples']=real_batch(args.real_dataset,160,2)
        require(batch['bboxes'].shape[0]>0,'Real train batch has no GT')
        report['wiring_640']=wiring(deepcopy(initial).eval(),torch.rand(1,3,640,640))
        report['parameters']={}
        for name,m in [('parent',parent),('target',target)]:
            report['parameters'][name]=dict(unfused=sum(p.numel() for p in m.parameters()),fused=sum(p.numel() for p in deepcopy(m).fuse(verbose=False).parameters()))
        require(report['parameters']['target']['unfused']-report['parameters']['parent']['unfused']==18432,'DCC count mismatch')
        with strict_precision():report['initial_equivalence']=equivalence(parent,target,batch['img'])
        for device,amp,label in [('cpu',False,'cpu_fp32'),('cuda',False,'cuda_fp32'),('cuda',True,'cuda_native_amp')]:
            if args.device!='all' and args.device!=device:continue
            if device=='cuda' and not torch.cuda.is_available():report['devices'][label]=dict(status='PENDING',reason='No CUDA');continue
            print('BEGIN '+label,flush=True);started=time.perf_counter()
            t=make_trainer(deepcopy(initial),amp,device);data={k:v.to(device) for k,v in batch.items()}
            rows=[];effective=0;upstream=False
            for i in range(16 if amp else 4):
                row=do_step(t,data,amp);rows.append(row)
                if not row['skipped']:effective+=1
                if effective>=2:
                    upstream=all(v is not None and v>0 for k,v in row['dcc_grad_norms'].items())
                if effective>=3 and upstream:break
            require(effective>=3 and upstream,'No finite gradient startup within bounded updates')
            info=dict(status='RUNNING',steps=rows,effective_updates=effective,optimizer_every_parameter_once=True,
                      scope='B2/160 real train images, native detection loss; fixed-lr smoke is NOT B16/640 capacity')
            report['devices'][label]=info
            info['lifecycle']=lifecycle(t,data,amp,args.output/label,args.variant)
            # Original model after updates, never the formal initialization.
            info['fusion']=fusion(t.model,data['img'][:1],args.output/label)
            info['status']='PASSED' if info['lifecycle']['status']=='PASSED' else 'PRECISION_NOTE'
            info['seconds']=time.perf_counter()-started
            write_json(args.output/'checks.json',report)
            del t,data;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            print(info['status']+' '+label,flush=True)
        report['status']='PRECISION_NOTE' if any(v['status']=='PRECISION_NOTE' for v in report['devices'].values()) else 'PASSED'
        report['server_torch_2_1_2']=dict(status='PASSED' if str(torch.__version__).startswith('2.1.2') else 'PENDING',
                                      reason='Actual executed runtime recorded; rerun on server when it differs')
        report['capacity']=dict(status='PENDING',reason='Separate native online-augmentation B16/640 preflight required')
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        torch.use_deterministic_algorithms(previous[0],warn_only=previous[1])
        torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark=previous[2:]
        from train_dcc import code_identity
        report['code_identity']=code_identity()
        write_json(args.output/'checks.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','initialized','real-dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['cbr_lif_dcc_v1','dcc_v1'],default='cbr_lif_dcc_v1')
    p.add_argument('--device',choices=['cpu','cuda','all'],default='all')
    run(p.parse_args())
