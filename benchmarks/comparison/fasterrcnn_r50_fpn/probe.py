"""Disposable-process CUDA native ops + FP32/AMP forward/loss/backward checks."""
from __future__ import annotations

from copy import deepcopy
import json

from support import environment, recipe, write_json


def check(output, local_compat=False):
    import torch
    import torchvision
    from model import build_model, finite_gradients, finite_losses, seed_all
    env = environment(strict=not local_compat)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available for required NMS/RoIAlign/AMP preflight')
    torch.set_num_threads(2)
    seed_all()
    device = 'cuda:0'
    boxes = torch.tensor([[0., 0., 8., 8.], [1., 1., 7., 7.]], device=device)
    nms = torchvision.ops.nms(boxes, torch.tensor([.9, .8], device=device), .7)
    if nms.tolist() != [0, 1]:  # IoU=36/64 < 0.7
        raise ValueError('CUDA NMS mismatch')
    features = torch.randn(1, 2, 16, 16, device=device, requires_grad=True)
    rois = torch.tensor([[0., 1., 1., 10., 10.]], device=device)
    aligned = torchvision.ops.roi_align(features, rois, (7, 7), sampling_ratio=2)
    aligned.sum().backward()
    if not torch.isfinite(features.grad).all():
        raise FloatingPointError('CUDA RoIAlign backward failed')
    cfg = recipe()
    model, initialization = build_model(cfg)
    model.to(device).eval()
    with torch.inference_mode():
        output640 = model([torch.zeros(3, 640, 640, device=device)])
    if model.backbone.comparison_observed_size != [640, 640] or output640[0]['boxes'].dtype != torch.float32:
        raise ValueError('Actual 640 backbone/FP32 inference check failed')
    del output640, model
    torch.cuda.empty_cache()
    small = deepcopy(cfg)
    small.update(input_height=64, input_width=64, model_transform_min_size=64, model_transform_max_size=64)
    checks = {}
    for amp in (False, True):
        seed_all()
        model, _ = build_model(small)
        model.cuda().train()
        images = [torch.rand(3, 64, 64, device=device) for _ in range(2)]
        targets = [{'boxes': torch.tensor([[10., 10., 45., 50.]], device=device),
                    'labels': torch.tensor([1], dtype=torch.int64, device=device)} for _ in range(2)]
        optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9, weight_decay=.0001)
        scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=cfg['amp_init_scale'],
                                         growth_interval=cfg['amp_growth_interval'])
        with torch.cuda.amp.autocast(enabled=amp):
            losses = model(images, targets)
            total = finite_losses(losses)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        try:
            gradients = finite_gradients(model)
        except FloatingPointError as exc:
            write_json(output, {'status': 'FAILED_NONFINITE_GRADIENTS', 'environment': env,
                'prior_checks': checks, 'failing_amp_enabled': amp, 'scale': scaler.get_scale(),
                'losses': {k: float(v.detach()) for k, v in losses.items()}, 'error': str(exc),
                'formal_training_started': False})
            raise
        checks['AMP' if amp else 'FP32'] = {'losses': {k: float(v.detach()) for k, v in losses.items()},
            'gradient_tensors': gradients, 'batch': 2, 'input_size': 64, 'amp_enabled': scaler.is_enabled(),
            'initial_scale': scaler.get_scale(), 'max_absolute_unscaled_gradient': max(
                float(p.grad.detach().abs().max()) for p in model.parameters() if p.grad is not None)}
        del model, optimizer, scaler, losses, total, images, targets
        torch.cuda.empty_cache()
    report = {'status': 'PASSED_LOCAL_COMPATIBILITY_ONLY' if local_compat else 'PASSED_TARGET_VERSION_ISOLATED_PROBE',
        'environment': env, 'initialization': initialization, 'cuda_nms': 'passed', 'cuda_roi_align_forward_backward': 'passed',
        'backbone_640_observed': [640, 640], 'FP32_and_AMP': checks,
        'isolation': 'dedicated disposable process; formal process reseeds/rebuilds all states',
        'formal_batch16_640_parallel_capacity': 'NOT_TESTED', 'formal_training_started': False}
    write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('environment', 'initialization')}, indent=2), flush=True)
    return report
