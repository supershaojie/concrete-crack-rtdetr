"""Finite BSC-Rep engineering gates; formal training and final test never run.

Local runs deliberately leave the real B16/640/AMP online-augmentation capacity
gate PENDING. Only --server runs that gate; no batch/image/AMP fallback exists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ultralytics-main'))
import torch

from init_bscrep_v1 import controlled_models, build_training_model, verify_model, require, sha256
from bscrep_v1_common import DEFAULT_VARIANT, VARIANTS, fingerprint, optimizer_coverage, recipe, verified_data
from c19_lif_v1_diagnostic import (atomic_json, rng_state, restore_rng, compare_records,
    fusion_protocol, selection_report, align_to_ids, PRE_KEYS, schema)
from c19_lif_v1_probe import capture, targets
from c19_lif_v1_cutoff import fusion_accepted
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import RepC3
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA


@contextmanager
def strict_fp32():
    """Strict fusion comparison owns, and restores, its TF32/determinism flags."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
           torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[:2]
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[2:4]
        torch.use_deterministic_algorithms(old[4], warn_only=old[5])


def exact(a, b, message):
    require(set(a) == set(b) and all(torch.equal(v.cpu(), b[k].cpu()) for k, v in a.items()), message)
    return len(a)


def parent_from(target):
    """Remove only BSC from an isolated nc=1 copy, preserving all shared state."""
    from init_bscrep_v1 import build
    variant = DEFAULT_VARIANT if hasattr(target.model[20], 'O_proj') else 'bscrep_v1'
    model = build(variant, nc=1, parent=True).float()
    model.load_state_dict({k:v for k,v in target.state_dict().items() if '.bsc.' not in k}, strict=True)
    return model


def activate_bsc(model):
    weight = model.model[19].bsc.out_proj.weight
    generator = torch.Generator(device=weight.device).manual_seed(917)
    with torch.no_grad():
        weight.copy_(torch.randn(weight.shape, generator=generator, device=weight.device, dtype=weight.dtype)*.003)


def native_optimizer(model):
    owner = RTDETRTrainer.__new__(RTDETRTrainer)
    owner.args = SimpleNamespace(warmup_bias_lr=.1, lr0=.0005, weight_decay=.0001)
    opt = owner.build_optimizer(model, name='AdamW', lr=.0005, momentum=.937, decay=.0001)
    return opt, optimizer_coverage(model, opt)


def native_scaler(enabled):
    from ultralytics.utils.torch_utils import TORCH_2_4
    return torch.amp.GradScaler('cuda', enabled=enabled) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=enabled)


def synthetic_batch():
    """Small diagnostic only, explicitly not a capacity or dataset result."""
    return dict(img=torch.rand(2, 3, 160, 192, generator=torch.Generator().manual_seed(873)),
                bboxes=torch.tensor([[.42, .53, .3, .07], [.26, .38, .12, .25], [.72, .67, .25, .09]]),
                cls=torch.zeros(3, 1), batch_idx=torch.tensor([0, 1, 1]))


def actual_train_api(initialized, expected, variant, folder):
    """Exercise public RTDETR.train -> production AuditedTrainer.get_model at nc=1."""
    from train_bscrep_v1 import AuditedTrainer
    observed = {}
    class Stopped(Exception):
        pass
    class Audit(AuditedTrainer):
        def __init__(self, overrides, _callbacks):
            self.args = SimpleNamespace(**{k:v for k,v in overrides.items() if k != 'session'})
            self.data = dict(nc=1, channels=3, names={0:'crack'})
            self.variant = variant
            self.audit_dir = folder/'actual_api'
            self.audit_dir.mkdir(exist_ok=False)
            self.fresh = True
            torch.manual_seed(42)
        def train(self):
            count = exact(expected.state_dict(), self.model.state_dict(), 'Actual Trainer nc=1 reconstruction differs')
            opt, groups = native_optimizer(self.model)
            observed.update(status='PASSED', shared_and_new_states_exact=count, nc=self.model.model[-1].nc,
                            get_model='production AuditedTrainer.get_model', optimizer=groups,
                            optimizer_steps=0, data_scope='synthetic class metadata; no dataloader')
            raise Stopped()
    args = YAML.load(ROOT/'docs/bscrep_v1/parent_args.yaml')
    args.update(model=str(initialized), name='isolated_api_probe', project=str(folder), save_dir=str(folder/'actual_api'))
    with patch('ultralytics.engine.model.checks.check_pip_update_available'):
        try:
            RTDETR(str(initialized)).train(trainer=Audit, **args)
        except Stopped:
            pass
    require(observed and observed['nc'] == 1, 'Real public train API did not reach nc=1 audit')
    return observed


def lifecycle(target, folder, active, variant):
    """Actual serialization, controlled resume rebuild, native EMA/resume state."""
    model = deepcopy(target).cpu().float().train()
    if active:
        activate_bsc(model)
    opt, groups = native_optimizer(model)
    scaler = native_scaler(False)
    # A bounded isolated optimizer update creates real AdamW state for resume.
    opt.zero_grad(set_to_none=True)
    sum(p.square().sum() for p in model.model[19].bsc.parameters()).backward()
    opt.step()
    ema = ModelEMA(model)
    ema.update(model)
    bsc_before = {k:v.clone() for k,v in model.state_dict().items() if '.bsc.' in k}
    path = folder/('lifecycle_nonzero.pt' if active else 'lifecycle_zero.pt')
    torch.save(dict(model=model, ema=ema.ema, updates=ema.updates, optimizer=opt.state_dict(),
                    scaler=scaler.state_dict(), epoch=0, best_fitness=.1), path)
    loaded = torch_load(path, map_location='cpu')
    exact(model.state_dict(), loaded['model'].state_dict(), 'Checkpoint reload changed state')
    resumed, mapping = build_training_model(model.yaml, loaded['model'], dict(nc=1, channels=3),
                                            variant, fresh=False)
    exact(model.state_dict(), resumed.state_dict(), 'Resume model rebuild changed learned parameters')
    holder = RTDETRTrainer.__new__(RTDETRTrainer)
    holder.model = resumed
    holder.optimizer, _ = native_optimizer(resumed)
    holder.scaler = native_scaler(False)
    holder.ema = ModelEMA(resumed)
    holder.args = SimpleNamespace(model=str(path), close_mosaic=10)
    holder.resume = True
    holder.epochs = 200
    holder.resume_training(loaded)
    exact(ema.ema.state_dict(), holder.ema.ema.state_dict(), 'Native resume EMA state mismatch')
    require(holder.start_epoch == 1 and holder.ema.updates == ema.updates, 'Native resume epoch/EMA lost')
    for pid, values in opt.state_dict()['state'].items():
        for name, value in values.items():
            restored = holder.optimizer.state_dict()['state'][pid][name]
            require(torch.equal(value, restored) if isinstance(value, torch.Tensor) else value == restored,
                    'Native resume optimizer state mismatch')
    for k,v in bsc_before.items():
        require(torch.equal(v, resumed.state_dict()[k]), 'Resume BSC changed')
    require(bool(torch.count_nonzero(resumed.model[19].bsc.out_proj.weight)) == active, 'Zero/nonzero branch lost')
    return dict(status='PASSED', nonzero=active, reload_exact=True, native_ema=True, native_resume=True,
                resumed_epoch=holder.start_epoch, optimizer_state_entries=len(opt.state), optimizer=groups,
                diagnostic_optimizer_steps=1, formal_optimizer_steps=0, checkpoint_sha256=sha256(path), mapping=mapping)


def loss_backward(target, batch, device, amp=False, active=False, *, parent_diagnostic=True, max_attempts=16, copy_model=True):
    """Default native GradScaler, finite backoff, identical batch/RNG on retries."""
    model = (deepcopy(target) if copy_model else target).to(device).float().train()
    model.nc = 1
    if active:
        activate_bsc(model)
    batch = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    opt, groups = native_optimizer(model)
    scaler = native_scaler(amp)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    rng = rng_state()
    rows = []
    initial_scale = float(scaler.get_scale())
    for attempt in range(max_attempts if amp else 1):
        model.load_state_dict(initial, strict=True)
        restore_rng(rng)
        opt.zero_grad(set_to_none=True)
        scale = float(scaler.get_scale())
        with (torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext()):
            prediction = model.predict(batch['img'], batch=targets(batch))
            dn = prediction[-1]
            loss = model.loss(batch, preds=prediction)[0]
        require(torch.isfinite(loss).all(), 'Nonfinite original RTDETR loss before scaling')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        bad = [n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        grads = {n:float(p.grad.float().norm()) if p.grad is not None and torch.isfinite(p.grad).all() else None
                 for n,p in model.named_parameters() if '.bsc.' in n}
        row = dict(attempt=attempt, scale_before=scale, loss=float(loss.detach()), unscaled_all_finite=not bad,
                   nonfinite_parameter_names=bad, bsc_gradient_norms=grads,
                   gt_groups=targets(batch)['gt_groups'], dn_split=dn['dn_num_split'] if dn else None,
                   total_queries=int(prediction[0].shape[2]))
        if not bad:
            if grads:
                require(grads['model.19.bsc.out_proj.weight'] is not None and grads['model.19.bsc.out_proj.weight'] > 0,
                        'BSC output projection lost nonzero finite gradient')
                if active:
                    require(all(v is not None and v > 0 for v in grads.values()), 'Activated BSC upstream lost gradients')
            require(next(model.parameters()).grad is not None, 'Backbone gradient missing')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        # Native scaler.step skips bad updates and update performs real backoff.
        if amp or not bad:
            scaler.step(opt)
            scaler.update()
        row.update(scale_after=float(scaler.get_scale()), optimizer_step_skipped=bool(bad))
        rows.append(row)
        del prediction, loss
        if not bad:
            return dict(status='PASSED', device=device, AMP=amp, nonzero_bsc=active, native_default_initial_scale=initial_scale,
                        attempts=rows, batch=len(batch['img']), input_shape=list(batch['img'].shape), optimizer=groups,
                        diagnostic_optimizer_steps=1, formal_optimizer_steps=0)
    detail = dict(status='FAILED', reason='Persistent nonfinite unscaled gradients after bounded native backoff', attempts=rows)
    if parent_diagnostic:
        # Same tensors/settings, no different loss, batch size or precision fallback.
        model.load_state_dict(initial, strict=True)
        parent_snapshot = parent_from(model)
        del model, opt, scaler
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        try:
            restore_rng(rng)
            detail['same_batch_parent'] = loss_backward(parent_snapshot, batch, device, amp, False,
                parent_diagnostic=False, max_attempts=max_attempts)
        except Exception as error:
            detail['same_batch_parent'] = getattr(error, 'detail', {'status':'FAILED', 'error':repr(error)})
    error = RuntimeError(detail['reason'])
    error.detail = detail
    raise error


def fusion_check(target, folder, device, active, precision='fp32'):
    model = deepcopy(target).float().eval()
    if active:
        activate_bsc(model)
    # Exercise original nonzero LIF/CBR residuals and nontrivial retained BN.
    if hasattr(model.model[20], 'O_proj'):
        from check_c19_lif_v1 import activate
        activate(model, bn=True)
    model.to(device)
    before = {k:v.clone() for k,v in model.model[19].bsc.state_dict().items()}
    fused = deepcopy(model).fuse(verbose=False)
    exact(before, fused.model[19].bsc.state_dict(), 'Fusion changed BSC learned state')
    float_source = deepcopy(model).cpu().float() if hasattr(model.model[20], 'O_proj') else None
    x = torch.rand(1, 3, 160, 192, device=device)
    if precision == 'half':
        model.half(); fused.half(); x = x.half()
    with strict_fp32():
        if hasattr(model.model[20], 'O_proj'):
            # Existing reviewed cutoff protocol handles candidate ordering/drift.
            def baseline_factory():
                from init_c19_lif_v1 import build
                baseline = build('C2', nc=1)
                baseline.load_state_dict({k:v for k,v in float_source.state_dict().items() if k in baseline.state_dict()}, strict=True)
                return baseline
            result = fusion_protocol(model, fused, x, folder, device, precision,
                                     parent_factory=baseline_factory, parent_source=float_source)
            require(fusion_accepted(result, device, precision), 'Candidate-aware fusion rejected')
        else:
            folder.mkdir(parents=True, exist_ok=False)
            context = lambda: torch.autocast(device_type=device, dtype=torch.float16) if precision == 'amp' else nullcontext()
            def run(m, ids=None):
                with torch.no_grad(), context():
                    return capture(m, x, fixed_ids=ids)[1]
            a,b = run(model),run(fused)
            atol,rtol = (2e-5,2e-4) if precision == 'fp32' else (3e-3,3e-2)
            selected = selection_report(a,b)
            pre = compare_records(a,b,atol,rtol,device=device,precision=precision,keys=PRE_KEYS)
            replay = compare_records(run(model,a['candidate_indices']),run(fused,a['candidate_indices']),atol,rtol,
                                     device=device,precision=precision)
            require(selected['kind'] != 'SET_DRIFT', 'Baseline fusion candidate set drift: inspect fixed replay, do not silently pass')
            aligned = compare_records(a,align_to_ids(a,b),atol,rtol,device=device,precision=precision)
            result = dict(status='PASSED', selection=selected, pre_selection=pre, fixed_query_replay=replay,
                          common_id_alignment=aligned, tf32=False, precision=precision)
            atomic_json(folder/'fuse_diagnostic.json',result)
    return dict(status='PASSED', nonzero_bsc=active, precision=precision, bsc_state_exact=True,
                tf32_matmul=False, tf32_cudnn=False, settings_restored=True, diagnostic=str(folder/'fuse_diagnostic.json'),
                candidate_status=result['status'])


def real_capacity(initialized, variant, data, folder, target):
    require(torch.cuda.is_available(), 'Server capacity requires CUDA; no CPU or reduced-batch substitute')
    require(data is not None, 'Server capacity requires verified --data')
    from train_bscrep_v1 import AuditedTrainer
    from train_c19_lif_v1 import ensure_amp_resources
    main = Path(os.environ.get('BSCREP_V1_MAIN', '/root/autodl-tmp/projects/Crack_RTDETR')).resolve()
    ensure_amp_resources(main, folder/'amp_resources.json')
    args, diff = recipe(variant, initialized, data, folder/'disposable_trainer')
    class CapacityTrainer(AuditedTrainer):
        pass
    CapacityTrainer.variant = variant
    CapacityTrainer.audit_dir = folder/'capacity_trainer_audit'
    CapacityTrainer.audit_dir.mkdir(exist_ok=False)
    CapacityTrainer.fresh = True
    CapacityTrainer.expected_args = args
    trainer = CapacityTrainer(overrides=args)
    trainer._setup_train()
    require(trainer.amp and trainer.batch_size == 16 and trainer.args.imgsz == 640, 'Formal capacity settings changed')
    require(trainer.train_loader.dataset.augment, 'Capacity train dataset has no online augmentation')
    exact(target.state_dict(), trainer.model.state_dict(), 'Actual capacity Trainer nc=1 initial state mismatch')
    groups = optimizer_coverage(trainer.model, trainer.optimizer)
    batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
    require(list(batch['img'].shape) == [16,3,640,640], 'Capacity actual batch/shape changed')
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # Native Trainer model/loader; loss updates take place only on a disposable copy.
    result = loss_backward(trainer.model, batch, 'cuda', amp=True, active=False, copy_model=False)
    torch.cuda.synchronize()
    result.update(batch=16, imgsz=640, online_augmentation=True, data_scope='verified real train split',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  seconds=time.perf_counter()-started, optimizer=groups, native_setup_train=True,
                  augmentation={k:getattr(trainer.args,k) for k in ('mosaic','mixup','hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective','flipud','fliplr')},
                  samples=[str(p) for p in batch.get('im_file',[])], formal_optimizer_steps=0)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','initialized','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--variant',choices=tuple(VARIANTS),default=DEFAULT_VARIANT)
    p.add_argument('--data',type=Path)
    p.add_argument('--server',action='store_true')
    args=p.parse_args()
    args.output=args.output.resolve();args.source=args.source.resolve();args.initialized=args.initialized.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    report=dict(status='FAILED',variant=args.variant,formal_training='NOT_STARTED',test='NOT_RUN',
                formal_optimizer_steps=0,server=args.server,
                runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                    cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                capacity=dict(status='PENDING',reason='Requires --server verified real train online augmentation B16/640/native AMP'))
    before_sha=sha256(args.initialized)
    def persist():
        atomic_json(args.output/'checks.json',report)
    try:
        report['identity']=fingerprint(args.source,args.initialized,args.variant,args.data)
        print(json.dumps(dict(identity=report['identity'],output=str(args.output)),ensure_ascii=False),flush=True)
        from check_bscrep_v1_math import run_checks, profile_models
        report['mathematics']=run_checks()
        _, fresh, mapping=controlled_models(args.source,args.variant)
        weights=torch_load(args.initialized,map_location='cpu')['model'].float()
        exact(fresh.state_dict(),weights.state_dict(),'Initialized checkpoint differs from controlled fresh source')
        report['controlled_initialization']=mapping
        torch.manual_seed(42)
        target,adaptation=build_training_model(weights.yaml,weights,dict(nc=1,channels=3),args.variant,fresh=True)
        report['trainer_rebuild']=adaptation
        report['topology']=verify_model(target,args.variant,zero=True)
        report['actual_train_api']=actual_train_api(args.initialized,target,args.variant,args.output)
        del fresh,weights
        report['lifecycle']={str(active):lifecycle(target,args.output,active,args.variant) for active in (False,True)}
        persist()
        parent=parent_from(target).eval();new=deepcopy(target).eval()
        with strict_fp32(),torch.no_grad():
            image=synthetic_batch()['img'][:1]
            _,left=capture(parent,image);_,right=capture(new,image)
            report['zero_parent_equivalence']=dict(status='PASSED',comparison=compare_records(left,right),scope='nc=1 shared-state parent')
        del parent,new,left,right
        report['cost']=profile_models(device='cpu')
        batch=synthetic_batch()
        for device in ('cpu','cuda'):
            if device=='cuda' and not torch.cuda.is_available():
                report[device]=dict(status='PENDING',reason='CUDA unavailable');continue
            print('Small synthetic engineering checks: '+device,flush=True)
            checks=report[device]=dict(scope='synthetic B2 160x192, not capacity or validation',status='RUNNING')
            with strict_fp32():
                checks['FP32_loss']={str(a):loss_backward(target,batch,device,active=a) for a in (False,True)}
                checks['fusion']={str(a):fusion_check(target,args.output/f'{device}_fuse_{a}',device,a) for a in (False,True)}
                if device=='cuda':
                    checks['AMP_loss']={str(a):loss_backward(target,batch,device,amp=True,active=a) for a in (False,True)}
                    checks['half_checkpoint_fusion']=fusion_check(target,args.output/'cuda_half_fuse',device,True,'half')
            checks['status']='PASSED';persist();gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        if args.data:
            _,inventory=verified_data(args.data)
            report['dataset']=dict(status='PASSED',inventory=inventory)
            report['dataset_inventory']=inventory
        else:
            report['dataset']=dict(status='PENDING',reason='No verified --data supplied')
        if args.server:
            report['capacity']=real_capacity(args.initialized,args.variant,args.data,args.output,target)
        require(sha256(args.initialized)==before_sha,'Formal initialization file was modified')
        require(fingerprint(args.source,args.initialized,args.variant,args.data)==report['identity'],
                'Source/config/weights/recipe/data identity changed during preflight; rerun after edits finish')
        pending=report['capacity']['status']!='PASSED' or report['cuda']['status']!='PASSED' or report['dataset']['status']!='PASSED'
        report['status']='PENDING' if pending else 'PASSED'
        require(not args.server or not pending,'Server preflight must complete all mandatory checks')
    except BaseException as error:
        report['status']='FAILED'
        report['failure']=getattr(error,'detail',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        report['formal_initialization_unchanged']=sha256(args.initialized)==before_sha
        persist()
        print('BSC preflight '+report['status']+': '+str(args.output/'checks.json'),flush=True)


if __name__=='__main__':
    main()
