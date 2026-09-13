"""Separate cold native initialization, nonzero stress and explicitly warmed smoke."""
from copy import deepcopy
import gc
from pathlib import Path
import torch
from c24_lif_v1_common import require
from c24_lif_v1_amp import calibration, rtdetr_forward, model_evidence, Evidence
from c24_lif_v1_acceptance import aggregate


def branch_gradients(model, stress):
    from init_c24_lif_v1 import added
    values = {n: float(p.grad.float().norm()) if p.grad is not None else None
              for n, p in model.named_parameters() if added(n)}
    require(len(values) == 10 and all(v is not None for v in values.values()), 'Missing innovation gradients')
    if stress:
        require(all(v > 0 for v in values.values()), 'Activated branch gradient missing')
    main = {n: float(p.grad.float().norm()) if p.grad is not None else None for n, p in model.named_parameters()
            if n in ('model.9.ma.in_proj_weight', 'model.19.cv1.conv.weight', 'model.20.conv.weight')}
    require(len(main) == 3 and all(v is not None and v > 0 for v in main.values()), 'Related main gradient missing')
    return dict(new_tensors=values, related_main=main,
                zero_upstream_allowed=not stress, scope='nonzero_branch_stress' if stress else 'native_initialization')


def diagnostic_loss(model, device, folder, *, label, batch=None, stress=False, amp=None, warmed=False):
    from check_c24_lif_v1 import optimizer_check, synthetic_batch
    from init_c24_lif_v1 import verify_model
    from c24_lif_v1_numerics import activate
    folder = Path(folder) / label
    evidence = Evidence(folder)
    report = dict(status='RUNNING', label=label, initialization='nonzero_branch_stress' if stress else 'native_zero_initialization',
                  amp_start='FP32_WARMED' if warmed else 'COLD', training_dispatched=False)
    source_before = model_evidence(model)
    initial_rng = None
    b = None
    try:
        b = deepcopy(model).to(device).train()
        b.nc = 1
        verify_model(b, zero=True)
        if stress:
            activate(b)
        batch = batch if batch is not None else synthetic_batch()
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        opt, groups = optimizer_check(b)
        report.update(coverage=groups['coverage'], new_tensors=groups['new_tensors'],
                      initial_model=model_evidence(b), input=list(batch['img'].shape),
                      original_native_model_sha256=source_before['parameters']['sha256'])
        evidence.write('stage', report)
        amp = device == 'cuda' if amp is None else amp
        require(not warmed or stress and amp, 'Warmed diagnostic must be explicitly labelled stress AMP')
        initial_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if device == 'cuda' else [])
        if warmed:
            report['fp32_warmup'] = calibration(b, opt, batch, rtdetr_forward, folder / 'fp32_warmup', amp=False,
                budget=2, label='nonzero_branch_stress_FP32_warmup', gradient_check=lambda m: branch_gradients(m, True))
            if report['fp32_warmup']['status'] != 'PASSED':
                report.update(status='BLOCKED', error='FP32 warmup blocked')
                return report
        result = calibration(b, opt, batch, rtdetr_forward, folder / 'calibration', amp=amp,
            budget=12 if amp else 2, initial_scale=128. if warmed else None,
            gradient_check=lambda m: branch_gradients(m, stress), optimizer_factory=lambda m: optimizer_check(m)[0], label=label)
        report.update(status=result['status'], calibration=result)
        if result['status'] == 'PASSED' and device == 'cuda' and stress and not amp:
            from c24_lif_v1_numerics import compare_pair
            from c24_lif_v1_acceptance import numerical_status
            require(all(bool(torch.count_nonzero(layer.layers[-1].weight)) for layer in b.model[-1].dec_bbox_head),
                    'Disposable bbox head did not update')
            a = b.eval()
            fused = deepcopy(a).fuse(verbose=False)
            report['updated_nonzero_bbox_fusion'] = compare_pair(a, fused, torch.rand(1, 3, 160, 192, device=device), 'fp32')
            report['status'] = numerical_status(report['updated_nonzero_bbox_fusion'])
            del a, fused
        if result.get('error'):
            report['error'] = result['error']
        if result['status'] == 'BLOCKED' and amp and not warmed:
            # Controlled cold-start FP32 replay: identical initial parameters, BN,
            # augmented batch and first-forward RNG. This is NOT a claim that the
            # last failing attempt's evolved state was replayed.
            del opt, b
            b = None
            gc.collect()
            torch.cuda.empty_cache()
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.set_rng_state(initial_rng[0])
                torch.cuda.set_rng_state_all(initial_rng[1])
                control = deepcopy(model).to(device).train()
                control.nc = 1
                if stress:
                    activate(control)
                control_opt, _ = optimizer_check(control)
                report['cold_initial_state_FP32_control'] = calibration(control, control_opt, batch, rtdetr_forward,
                    folder / 'fp32_control', amp=False, budget=1, required_updates=1, label='same_initial_state_batch_and_first_forward_RNG_FP32',
                    gradient_check=lambda m: branch_gradients(m, stress))
                report['cold_initial_state_FP32_control']['causal_scope'] = 'First attempt only; later AMP states are not reproduced by this control'
                del control, control_opt
    except Exception as error:
        report.update(status='BLOCKED', error=repr(error))
    finally:
        report['source_model_unchanged'] = source_before == model_evidence(model)
        if not report['source_model_unchanged']:
            report.update(status='BLOCKED', error='Disposable diagnostic polluted source model')
        evidence.write('stage', report)
    return report


def loss_suite(model, device, folder):
    checks = {}
    checks['nonzero_branch_stress_fp32'] = diagnostic_loss(model, device, folder,
        label='nonzero_branch_stress_' + device + '_fp32', stress=True, amp=False)
    if device == 'cuda':
        checks['nonzero_branch_stress_cold_amp'] = diagnostic_loss(model, device, folder,
            label='nonzero_branch_stress_cuda_cold_amp', stress=True, amp=True)
        checks['nonzero_branch_stress_fp32_warmed_amp'] = diagnostic_loss(model, device, folder,
            label='nonzero_branch_stress_cuda_fp32_warmed_amp', stress=True, amp=True, warmed=True)
    return dict(status=aggregate([v['status'] for v in checks.values()]), checks=checks,
                error=next((v.get('error') for v in checks.values() if v.get('error')), None),
                scope='Separate disposable nonzero branch checks; no capacity claim')
