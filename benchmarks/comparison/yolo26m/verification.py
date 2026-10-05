"""Bounded preflight checks on a copy of the actual model; preserve weights and RNG."""
from __future__ import annotations
from copy import copy, deepcopy
from pathlib import Path
import random
from support import native_recipe


def preflight_model_probe(model, config, run, manifest, device='cuda:0'):
    import cv2
    import numpy as np
    import torch
    from ultralytics.cfg import get_cfg
    from adapters import IsolatedDataset, capture_rng, restore_rng, strict_amp_probe
    from augment_b19 import snapshot
    before = capture_rng()
    try:
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA preflight NOT_RUN: no CUDA device; formal recipe was not changed')
        hyp = get_cfg(overrides=native_recipe(config))
        ds = IsolatedDataset(img_path=str(Path(run)/'train.txt'), imgsz=64, batch_size=2,
            augment=True, hyp=copy(hyp), rect=False, cache=False, data={'names':{0:'crack'}},
            manifest=manifest, split='train', cache_root=Path(run)/'cache', cutmix_probability=config['cutmix'])
        raw = ds.get_image_and_label(0)
        image = cv2.imread(ds.im_files[0])
        h,w = image.shape[:2]
        if raw['img'].shape[:2] != (64,64):
            raise ValueError('Actual training sample did not use square stretch')
        boxes = raw['instances']
        boxes.convert_bbox('xyxy')
        boxes.denormalize(64,64)
        original = np.asarray(ds.records[0]['boxes'],dtype=np.float32).reshape(-1,5)[:,1:]
        expected = np.column_stack((original[:,0]-original[:,2]/2, original[:,1]-original[:,3]/2,
                                    original[:,0]+original[:,2]/2, original[:,1]+original[:,3]/2))*64
        if not np.allclose(boxes.bboxes,expected,rtol=0,atol=1e-4):
            raise ValueError('Square stretch changed normalized label geometry')
        samples = [ds[i] for i in range(2)]
        batch = ds.collate_fn(samples)
        probe = deepcopy(model).float().to(device).train()
        probe.args = hyp
        amp_accuracy = strict_amp_probe(probe) if config['amp'] and device.startswith('cuda') else None
        batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        batch['img'] = batch['img'].float()/255
        # First check the unscaled FP32 backward independently of dynamic AMP overflow.
        fp_loss,fp_items = probe(batch)
        fp_loss.sum().backward()
        fp_gradients = [p.grad for p in probe.parameters() if p.grad is not None]
        if not torch.isfinite(fp_loss).all() or not all(torch.isfinite(g).all() for g in fp_gradients):
            raise ValueError('Actual-model FP32 preflight loss/gradients are nonfinite')
        probe.zero_grad(set_to_none=True)
        # This copy-only numerical check uses unit scaling; formal trainer retains native dynamic scaling.
        scaler = torch.cuda.amp.GradScaler(init_scale=1,enabled=config['amp'] and device.startswith('cuda'))
        with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',
                            dtype=torch.float16 if device.startswith('cuda') else torch.bfloat16,
                            enabled=config['amp'] and device.startswith('cuda')):
            loss,items = probe(batch)
        if not torch.isfinite(loss).all():
            raise ValueError('Actual-model preflight loss is nonfinite')
        scaler.scale(loss.sum()).backward()
        gradients = [p.grad for p in probe.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Actual-model preflight gradients are missing/nonfinite')
        from adapters import ComparisonTrainer, criterion_state, restore_criterion
        from optimizer import optimizer_identity, verify_buffers
        trainer = ComparisonTrainer.__new__(ComparisonTrainer)
        trainer.args = hyp
        trainer.comparison_config = config
        optimizer = trainer.build_optimizer(probe,name=config['optimizer'],lr=config['lr0'],
            momentum=config['momentum'],decay=config['weight_decay']*config['batch']*
            max(round(config['nbs']/config['batch']),1)/config['nbs'])
        for group in optimizer.param_groups: group['initial_lr'] = group['lr']
        observed = {name:p.detach().clone() for name,p in probe.named_parameters()
                    if p.grad is not None and p.ndim>=2 and name.startswith('model.23.')}
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(probe.parameters(),max_norm=10.)
        scaler.step(optimizer)
        scaler.update()
        updated = [name for name,p in probe.named_parameters() if name in observed and not torch.equal(p,observed[name])]
        if not updated or not any('one2one' in n for n in updated) or not any('.cv2.' in n and 'one2one' not in n for n in updated):
            raise ValueError('Preflight must update matching parameters in both native training branches')
        opt_report = optimizer_identity(optimizer,probe,config)
        buffers = verify_buffers(optimizer)
        if config['optimizer']=='MuSGD' and not buffers['populated_mixed_tensors']:
            raise ValueError('Actual MuSGD preflight did not populate its two mixed momentum buffers')
        schedule_before = criterion_state(probe.criterion)
        probe.criterion.update()
        schedule_after = criterion_state(probe.criterion)
        restore_criterion(probe.criterion,schedule_before)
        if criterion_state(probe.criterion) != schedule_before:
            raise ValueError('Native E2ELoss state restoration failed')
        result = {'scope':'SMOKE_ONLY_PREFLIGHT', 'batch':2, 'imgsz':64, 'device':device,
            'loss':float(loss.sum()), 'loss_items':items.detach().cpu().tolist(),
            'gradient_tensors':len(gradients), 'amp_requested':config['amp'],
            'cuda_amp_accuracy':'passed' if amp_accuracy else 'NOT_RUN',
            'fp32_backward':'passed', 'fp32_loss':float(fp_loss.sum()), 'probe_scaler_initial_scale':1,
            'sample':{'original_width':w,'original_height':h, 'square_stretch':[64,64],
                      'box_geometry':'normalized xywh -> independent x/y stretch verified'},
            'transforms':ds.transform_report, 'augmentation_counts':snapshot(ds),
            'model_copy_only':True, 'rng_restored':True, 'formal_optimizer_updates':0,
            'copy_only_optimizer_updates':1,'optimizer':opt_report,'optimizer_buffers':buffers,
            'both_branches_updated':updated,'E2ELoss':{'before':schedule_before,'after_copy_only_update':schedule_after,
                'restored':True,'formal_update_calls':0},
            'native_bfloat16_matrix_operations':'passed with actual MuSGD update' if config['optimizer']=='MuSGD' else 'not_used'}
        del probe, batch, ds
        return result
    finally:
        restore_rng(before)
