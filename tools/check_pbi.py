"""Bounded PBI network audit on real train images; never epochs, val or test."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
from tempfile import TemporaryDirectory

import torch
from init_pbi import (ROOT, controlled_models, native_rebuild, verify_model,
                        require, sha256, write_json, runtime, build, PBI_KEYS, build_training_model)
from c19_lif_v1_data import dataset_inventory, real_batch
from pbi_probe import capture, targets
from c19_lif_v1_diagnostic import rng_state, restore_rng, selection_report
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import PBIConv
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA

from pbi_checkpoint import PBICheckpointTrainer
from pbi_acceptance import CONTRACT_VERSION, evaluate_mode
from pbi_resume_audit import (audit_live_control, attach_mode_acceptance, bounded_updates,
                              make_audit_trainer, seed42)

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
    handles.append(model.model[17].pbi.register_forward_pre_hook(lambda m,a:values.update(pbi_input=a[0].detach().clone())))
    handles.append(model.model[20].register_forward_pre_hook(lambda m,a:values.update(down_input=a[0].detach().clone())))
    handles.append(model.model[26].register_forward_pre_hook(lambda m,a:values.update(decoder_inputs=[v.detach().clone() for v in a[0]])))
    try:
        with torch.no_grad(): model.eval()(image)
    finally:
        for h in handles:h.remove()
    require(torch.equal(values['projection'],values['pbi_input']),'PBI must follow complete Conv/BN/act')
    require(torch.equal(values['18'],torch.cat([values['16'],values['17']],1)),'Concat order changed')
    require(torch.equal(values['19'],values['down_input']),'P3 downsample source changed')
    for i,v in zip((19,22,25),values['decoder_inputs']):require(torch.equal(values[str(i)],v),'Decoder input changed')
    return dict(status='PASSED',nodes=len(model.model),shapes={k:list(v.shape) for k,v in values.items() if isinstance(v,torch.Tensor)},
                pbi_after_complete_projection=True,concat=[16,17],p3_downsample=20,decoder_from=[19,22,25])


def trace_compare(a,b,keys):
    for k in keys:
        if k not in ('anchors','selected_anchors','reference_boxes'):
            require(torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all(),'Nonfinite continuous tensor '+k)
    return {k:metric(a[k],b[k]) for k in keys}


def equivalence(left,right,image,enforce=True):
    with torch.no_grad(): _,a=capture(left.eval(),image);_,b=capture(right.eval(),image)
    keys=['scale_0','scale_1','scale_2','projection_0','projection_1','projection_2','encoder_features','candidate_scores']
    keys += [key for key in ('pbi_input','pbi_u','pbi_v','pbi_output','pbi_residual') if key in a and key in b]
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
    return dict(status=('PASSED' if same else 'PRECISION_NOTE') if passed else 'FAILED',continuous=continuous,selection=selection,
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
    require(all(torch.isfinite(value).all() for key,value in taps.items() if isinstance(value,torch.Tensor)
                and key not in {'anchors','selected_anchors','reference_boxes'}),'Nonfinite native forward tensor')
    t.scaler.scale(loss).backward()
    scale=t.scaler.get_scale()
    grads={n:float(p.grad.detach().float().norm()/scale) if p.grad is not None and torch.isfinite(p.grad).all() else None
           for n,p in t.model.named_parameters() if n.startswith('model.17.pbi.')}
    public=t.model.model[0].conv.weight.grad
    require(public is not None,'Missing public backbone gradient')
    if gradient_capture is not None:
        gradient_capture.update({n:p.grad.detach().float().cpu().clone()/scale
                                 for n,p in t.model.named_parameters() if p.grad is not None})
    before={n:p.detach().clone() for n,p in t.model.named_parameters() if n.startswith('model.17.pbi.')}
    steps_before=sum(float(state.get('step',0)) for state in t.optimizer.state.values())
    t.optimizer_step()
    after=t.scaler.get_scale()
    steps_after=sum(float(state.get('step',0)) for state in t.optimizer.state.values())
    effective=steps_after>steps_before
    require(effective == (after>=scale),'Native scaler and actual optimizer counters disagree')
    changed=any(not torch.equal(v,dict(t.model.named_parameters())[n]) for n,v in before.items())
    if effective:require(changed and all(v is not None for v in grads.values()),'Effective step lost PBI or finite gradients')
    require(all(torch.isfinite(p).all() for p in t.model.parameters()),'Nonfinite weight after native update')
    return dict(loss=float(loss.detach()),scale_before=scale,scale_after=after,skipped=not effective,effective_update=effective,
                optimizer_steps_before=steps_before,optimizer_steps_after=steps_after,
                pbi_grad_norms=grads,pbi_changed=changed,gt=int(batch['bboxes'].shape[0]),dn_split=taps['dn_split'],
                forward_hashes={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()
                                for k,v in taps.items() if isinstance(v,torch.Tensor)} if gradient_capture is not None else None)


def make_trainer(model,amp,device):
    t=PBICheckpointTrainer.__new__(PBICheckpointTrainer);t.model=model.to(device);t.optimizer=optimizer(t.model)
    t.scaler=scaler(amp);t.ema=ModelEMA(t.model);return t


def lifecycle(t,batch,amp,folder,variant):
    from pbi_resume_audit import audit_checkpoint_resume

    folder.mkdir(parents=True,exist_ok=False)
    m=t.model.eval();require(torch.count_nonzero(m.model[17].pbi.Wo.weight)>0,'Lifecycle needs actually updated Wo')
    with torch.no_grad(): _,learned_taps=capture(m,batch['img'])
    require(torch.count_nonzero(learned_taps['pbi_residual'])>0,'Learned PBI output residual is zero')
    # Only these newly created disposable weights are removed. Historical reports
    # and earlier output directories remain untouched.
    with TemporaryDirectory(prefix='lifecycle_',dir=folder) as temporary:
        scratch=Path(temporary)
        torch.save(m.state_dict(),scratch/'state.pt')
        with torch.random.fork_rng(devices=[]): rebuilt=native_rebuild(m.yaml,m,1,3).to(batch['img'].device)
        rebuilt.load_state_dict(torch_load(scratch/'state.pt',map_location=batch['img'].device),strict=True)
        assert_state(m.state_dict(),rebuilt.state_dict())
        del rebuilt
        torch.save(dict(model=deepcopy(m).cpu(),epoch=-1,train_args=dict(task='detect')),scratch/'model.pt')
        loaded=RTDETR(str(scratch/'model.pt')).model.to(batch['img'].device)
        assert_state(m.state_dict(),loaded.state_dict());verify_model(loaded,variant,zero=False)
        del loaded
        wide=build(variant,nc=80)
        wide.load_state_dict({**wide.state_dict(),**{k:v.detach().cpu() for k,v in m.state_dict().items()
                             if v.shape==wide.state_dict()[k].shape}},strict=True)
        adapted=native_rebuild(wide.yaml,wide,nc=1)
        require(all(torch.equal(adapted.state_dict()[k],m.state_dict()[k].cpu()) for k in PBI_KEYS),
                'Learned PBI lost during actual nc80-to1 reconstruction')
        del wide,adapted
        ema_copy=deepcopy(t.ema.ema).eval()
        with strict_precision(),torch.no_grad():
            ema_comparison=metric(t.ema.ema.eval()(batch['img'])[0],ema_copy(batch['img'])[0])
        require(ema_comparison['allclose'],'EMA own-copy comparison failed')
        del ema_copy
    # The same PBI-specific serializer is used by the actual training entry.
    # Restore correctness is the hard requirement. Independent CUDA trajectories
    # retain their original measurements and receive a separate mode assessment.
    report=audit_checkpoint_resume(t,batch,amp,folder)
    report.update(state_dict_exact=True,full_model_exact=True,native_get_model_nonzero_preserved=True,
                  ema_own_copy=ema_comparison,learned_nc80_to1=True,pbi_calls=1,nonzero_residual=True)
    return report


def fusion(model,image,folder):
    model=deepcopy(model).eval();fused=deepcopy(model).fuse(verbose=False)
    require(type(fused.model[17]) is PBIConv and fused.model[17].forward.__func__ is PBIConv.forward_fuse,'Fuse bypasses subclass')
    assert_state(model.model[17].pbi.state_dict(),fused.model[17].pbi.state_dict())
    with torch.no_grad(): _,taps=capture(fused,image)
    require(torch.count_nonzero(taps['pbi_residual'])>0,'Learned fused PBI residual unexpectedly zero')
    default=equivalence(model,fused,image,enforce=False)
    with strict_precision(): strict=equivalence(model,fused,image)
    result=dict(status='PASSED',pbi_calls=1,pbi_state_exact=True,nonzero_residual=True,default=default,strict_fp32=strict,
                unfused_parameters=sum(p.numel() for p in model.parameters()),fused_parameters=sum(p.numel() for p in fused.parameters()))
    if image.is_cuda:
        half=deepcopy(model).half();half_fused=deepcopy(fused).half()
        with torch.no_grad():
            hv,ht=capture(half,image.half());hfv,hft=capture(half_fused,image.half())
            ho,hf=hv[0],hfv[0]
            with torch.autocast('cuda',dtype=torch.float16):ao=model(image)[0]
        require(torch.isfinite(ho).all() and torch.isfinite(hf).all() and torch.isfinite(ao).all(),'Nonfinite half/AMP inference')
        require(torch.count_nonzero(ht['pbi_residual'])>0 and torch.count_nonzero(hft['pbi_residual'])>0,
                'Half/fused-half learned PBI residual unexpectedly zero')
        assert_state(half.model[17].pbi.state_dict(),half_fused.model[17].pbi.state_dict())
        half_equivalence=equivalence(half,half_fused,image.half(),enforce=False)
        # Isolate PBI math from earlier FP16 Conv/BN fusion rounding. The
        # reference explicitly evaluates all three projections, including the
        # functional Wo call; a Wo Conv2d hook would observe nothing.
        references={}
        with torch.no_grad():
            for label,current,tap in (('unfused',half,ht),('fused',half_fused,hft)):
                x=tap['pbi_input'].to(image.device)
                pbi=current.model[17].pbi
                u=torch.nn.functional.conv2d(x,pbi.W1.weight)
                v=torch.nn.functional.conv2d(x,pbi.W2.weight)
                z=u.float()*v.float()
                expected=(x.float()+torch.nn.functional.conv2d(z,pbi.Wo.weight.float())).to(x.dtype)
                actual=tap['pbi_output'].to(image.device)
                require(torch.equal(expected,actual),'Half PBI differs from independent explicit formula')
                references[label]=dict(exact=True,comparison=metric(expected,actual),
                                       nonzero_residual=bool(torch.count_nonzero(expected.float()-x.float())))
            common_input=ht['pbi_input'].to(image.device)
            same_input=metric(half.model[17].pbi(common_input),half_fused.model[17].pbi(common_input))
        require(same_input['allclose'],'Half fused PBI changes under identical projected input')
        half_status='PASSED' if half_equivalence['status']=='PASSED' else 'PRECISION_NOTE'
        result['cuda_half']=dict(status=half_status,shape=list(ho.shape),pbi_calls=1,nonzero_residual=True,
                                 pbi_state_exact=True,pbi_same_precision={key:metric(ht[key],hft[key])
                                 for key in ('pbi_input','pbi_u','pbi_v','pbi_output','pbi_residual')},
                                 candidate_selection=selection_report(ht,hft),continuous_and_candidate_diagnostic=half_equivalence,
                                 same_precision_natural_output=metric(ho,hf),
                                 explicit_formula=references,identical_input_pbi=same_input,
                                 strict_fp32_continuous_passed=True,
                                 note='Raw half errors retain original tolerances. Learned PBI weights and identical-input computation are exact; '
                                      'differences entering PBI arise before the branch from half Conv/BN fusion rounding, with native candidate changes recorded. '
                                      'This is finite half inference with localized precision evidence, not a claim of raw half-output equivalence.')
    return result


def prerequisite_records(args):
    records = {}
    for key, supplied in (("initialization", args.initialization_report), ("math", args.math_report)):
        path = supplied.resolve()
        require(path.is_file(), "Missing prerequisite report: " + str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        require(data.get("status") == "PASSED", key + " prerequisite is not PASSED")
        if key == "initialization":
            require(data.get("variant") == args.variant and data.get("output_sha256") == sha256(args.initialized),
                    "Initialization prerequisite variant/checkpoint differs")
        else:
            from train_pbi import _math_pass
            _math_pass(data)
        records[key] = dict(path=str(path), sha256=sha256(path))
    return records


def run(args):
    from train_pbi import code_identity, git_head
    initial_code_identity=code_identity()
    initial_git_head=git_head()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    previous=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    report=dict(status='FAILED',report_kind='full_preflight_engineering',contract_version=CONTRACT_VERSION,
                code_identity=initial_code_identity,git_head=initial_git_head,
                variant=args.variant,runtime=runtime(),initialization_sha256=sha256(args.initialized),
                formal_training='NOT_STARTED',final_test='NOT_RUN',tolerances=dict(atol=ATOL,rtol=RTOL),devices={},
                precision=dict(deterministic_warn_only=True,cudnn_deterministic=True,cudnn_benchmark=False,
                               cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                               matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32))
    try:
        report['prerequisites']=prerequisite_records(args)
        parent,target,report['controlled_initialization']=controlled_models(args.source,args.variant)
        initial=RTDETR(str(args.initialized)).model
        assert_state(initial.state_dict(),target.state_dict());verify_model(initial,args.variant,zero=True)
        # Source/checkpoint nc=80 is retained for formal native construction.
        # Loss, shape, parameter-count and continuation probes must use nc=1,
        # with identical CPU RNG for each parent's nine classification states.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            parent=native_rebuild(parent.yaml,parent,nc=1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            initial,report['native_nc1_rebuild']=build_training_model(initial.yaml,initial,dict(nc=1,channels=3),args.variant)
        report['controlled_initialization']['native_nc1_rebuild']=report['native_nc1_rebuild']
        require(all(torch.equal(value,initial.state_dict()[name]) for name,value in parent.state_dict().items()),
                'Actual nc1 parent/candidate public state differs')
        target=initial
        verify_model(initial,args.variant,zero=True)
        report['data_identity']=dataset_inventory(args.real_dataset)
        batch,report['samples']=real_batch(args.real_dataset,160,2)
        require(batch['bboxes'].shape[0]>0,'Real train batch has no GT')
        report['wiring_640']=wiring(deepcopy(initial).eval(),torch.rand(1,3,640,640))
        report['parameters']={}
        for name,m in [('parent',parent),('target',target)]:
            report['parameters'][name]=dict(unfused=sum(p.numel() for p in m.parameters()),fused=sum(p.numel() for p in deepcopy(m).fuse(verbose=False).parameters()))
        require(report['parameters']['target']['unfused']-report['parameters']['parent']['unfused']==24576,'PBI count mismatch')
        with strict_precision():report['initial_equivalence']=equivalence(parent,target,batch['img'])
        for device,amp,label in [('cpu',False,'cpu_fp32'),('cuda',False,'cuda_fp32'),('cuda',True,'cuda_native_amp')]:
            if args.device!='all' and args.device!=device:
                report['devices'][label]=dict(status='PENDING',reason='Excluded by requested --device subset')
                continue
            if device=='cuda' and not torch.cuda.is_available():report['devices'][label]=dict(status='PENDING',reason='No CUDA');continue
            print('BEGIN '+label,flush=True);started=time.perf_counter()
            seed42()
            t=make_trainer(deepcopy(initial),amp,device);data={k:v.to(device) for k,v in batch.items()}
            rows=[];effective=0;upstream=False
            for i in range(16 if amp else 4):
                row=do_step(t,data,amp);rows.append(row)
                if not row['skipped']:
                    effective+=1
                    if effective==1:
                        require(row['pbi_grad_norms']['model.17.pbi.Wo.weight']>0,'First effective Wo gradient must be nonzero')
                        require(all(row['pbi_grad_norms'][key]==0 for key in ('model.17.pbi.W1.weight','model.17.pbi.W2.weight')),
                                'Zero Wo must give exactly zero first-step W1/W2 gradients')
                if effective>=2:
                    upstream=all(v is not None and v>0 for k,v in row['pbi_grad_norms'].items())
                if effective>=3 and upstream:break
            require(effective>=3 and upstream,'No finite gradient startup within bounded updates')
            info=dict(status='RUNNING',steps=rows,effective_updates=effective,optimizer_every_parameter_once=True,
                      scope='B2/160 real train images, native detection loss; fixed-lr smoke is NOT B16/640 capacity')
            report['devices'][label]=info
            parent_name='parent_cbr_lif' if args.variant=='cbr_lif_pbi_v1' else 'parent_c2'
            info['parent_variant']=parent_name
            info['lifecycle']=lifecycle(t,data,amp,args.output/label,args.variant)
            info['live_controls']={'pbi':audit_live_control(t,data,amp,args.output/label/'pbi_live','pbi')}
            # Original model after updates, never the formal initialization.
            info['fusion']=fusion(t.model,data['img'][:1],args.output/label)
            del t;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
            seed42()
            parent_trainer=make_audit_trainer(deepcopy(parent),amp,device)
            info['parent_updates']=bounded_updates(parent_trainer,data,amp,max_batches=16 if amp else 4,target_updates=3)
            info['live_controls']['parent']=audit_live_control(parent_trainer,data,amp,args.output/label/'parent_live',parent_name)
            del parent_trainer;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
            info['acceptance']=evaluate_mode(label,info['lifecycle'],info['live_controls']['parent'],info['live_controls']['pbi'])
            attach_mode_acceptance(info['lifecycle'],info['acceptance'])
            write_json(args.output/label/'resume_comparison.json',info['lifecycle'])
            info['status']=info['acceptance']['status']
            if info['status']=='PASSED' and info['fusion'].get('cuda_half',{}).get('status')=='PRECISION_NOTE':
                info['status']='PRECISION_NOTE'
            info['seconds']=time.perf_counter()-started
            write_json(args.output/'checks.json',report)
            del data;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            print(info['status']+' '+label,flush=True)
        statuses=[value['status'] for value in report['devices'].values()]
        report['status']=('FAILED' if 'FAILED' in statuses else 'PENDING' if 'PENDING' in statuses
                          else 'PRECISION_NOTE' if 'PRECISION_NOTE' in statuses else 'PASSED')
        report['server_torch_2_1_2']=dict(status='PASSED' if str(torch.__version__).startswith('2.1.2') else 'PENDING',
                                      reason='Actual executed runtime recorded; rerun on server when it differs')
        report['capacity']=dict(status='PENDING',reason='Separate native online-augmentation B16/640 preflight required')
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        torch.use_deterministic_algorithms(previous[0],warn_only=previous[1])
        torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark=previous[2:]
        report['code_identity_unchanged_during_run']=(code_identity()==initial_code_identity and git_head()==initial_git_head)
        if not report['code_identity_unchanged_during_run']:
            report['status']='FAILED'
            report['identity_error']='Code or HEAD changed during execution; retained raw evidence cannot grant a launch permit'
        write_json(args.output/'checks.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','initialized','real-dataset','output','initialization-report','math-report'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['cbr_lif_pbi_v1','pbi_v1'],default='cbr_lif_pbi_v1')
    p.add_argument('--device',choices=['cpu','cuda','all'],default='all')
    try:
        report=run(p.parse_args())
    except Exception:
        traceback.print_exc()
        return 3
    return 0 if report['status'] in {'PASSED','PRECISION_NOTE'} else 3


if __name__=='__main__':
    raise SystemExit(main())
