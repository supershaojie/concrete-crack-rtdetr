"""Disposable finite AMP diagnostics using the public torch 2.1.2 GradScaler API.

No training recipe, module, gradient rule or production Trainer is modified.
Each attempt is a fresh graph with one unscale, and at most 12 attempts run.
"""
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback
import uuid
import torch
from c24_lif_v1_common import require, write_json

MAX_ATTEMPTS = 12
REQUIRED_UPDATES = 2
RECORD_LIMIT = 64000


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def tensor_digest(tensor):
    raw = tensor.detach().cpu().contiguous()
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def tensors(value, prefix=''):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, prefix + '/' + str(key))
    elif isinstance(value, (list, tuple)):
        for key, item in enumerate(value):
            yield from tensors(item, prefix + '/' + str(key))


def state_evidence(value):
    digest = hashlib.sha256()
    types = Counter()
    nonfinite = []
    count = 0
    for name, tensor in tensors(value):
        count += 1
        types[str(tensor.dtype)] += 1
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype), tensor_digest(tensor)]).encode())
        if not bool(torch.isfinite(tensor).all()):
            nonfinite.append(name)
    return dict(sha256=digest.hexdigest(), tensors=count, dtype_distribution=dict(types),
                finite=not nonfinite, nonfinite_names=nonfinite)


def model_evidence(model):
    return dict(parameters=state_evidence(dict(model.named_parameters())),
                buffers=state_evidence(dict(model.named_buffers())),
                output_projections={n: dict(zero=not bool(torch.count_nonzero(p)), sha256=tensor_digest(p))
                                    for n, p in model.named_parameters()
                                    if n.endswith(('scca_o.weight', 'O_proj.weight'))})


class Evidence:
    """Atomic JSON records with explicit, lossless structured shards below 64 KB."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def write(self, name, value):
        value = deepcopy(value)
        encode = lambda v: json.dumps(v, ensure_ascii=False, allow_nan=False, default=str).encode()
        # Split large immediate children recursively. References include all pieces;
        # unlike tail-only evidence, a gradient parameter cannot silently disappear.
        if len(encode(value)) > 54000:
            if isinstance(value, list):
                if len(value) == 1:
                    value = [dict(structured_shard=self.write(name + '_item_0', value[0]))]
                else:
                    middle = len(value) // 2
                    require(middle > 0, 'Unshardable evidence record')
                    value = dict(structured_shards=[self.write(name + '_a', value[:middle]),
                                                   self.write(name + '_b', value[middle:])], items=len(value))
            elif isinstance(value, dict):
                for key in sorted(value, key=lambda k: len(encode(value[k])), reverse=True):
                    if len(encode(value)) <= 54000:
                        break
                    value[key] = dict(structured_shard=self.write(name + '_' + key, value[key]))
            else:
                raise RuntimeError('Evidence scalar exceeds record budget')
        path = self.folder / (name + '.json')
        payload = encode(value) + b'\n'
        require(len(payload) <= RECORD_LIMIT, 'Evidence exceeds 64 KB')
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        return dict(path=path.name, bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def batch_evidence(batch):
    image = batch['img']
    row = dict(shape=list(image.shape), dtype=str(image.dtype), device=str(image.device),
               finite=bool(torch.isfinite(image).all()), min=finite_number(image.min()),
               max=finite_number(image.max()), augmented_tensor_sha256=tensor_digest(image),
               label_tensors={k: dict(shape=list(batch[k].shape), dtype=str(batch[k].dtype), sha256=tensor_digest(batch[k]))
                              for k in ('cls', 'bboxes', 'batch_idx') if k in batch},
               images=[str(p).replace('\\', '/').split('/datasets/crack_det/')[-1]
                       for p in batch.get('im_file', [])])
    if all(k in batch for k in ('cls', 'bboxes', 'batch_idx')):
        cls, boxes, indices = (batch[k] for k in ('cls', 'bboxes', 'batch_idx'))
        row['targets'] = dict(count=len(cls), boxes_count=len(boxes), indices_count=len(indices),
            finite=all(bool(torch.isfinite(v).all()) for v in (cls, boxes, indices)),
            classes_valid=bool((cls == 0).all()), boxes_valid=boxes.ndim == 2 and boxes.shape[1] == 4 and
            bool(((boxes >= 0) & (boxes <= 1)).all()) and bool((boxes[:, 2:] > 0).all()),
            batch_indices_valid=bool(((indices >= 0) & (indices < len(image)) & (indices == indices.long())).all()),
            gt_groups=[int((indices == i).sum()) for i in range(len(image))])
    return row


def coverage(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    return dict(parameters=len(names), optimizer_entries=len(ids), duplicates=len(ids)-len(set(ids)),
                missing=[n for i, n in names.items() if i not in ids], extra=len(set(ids)-set(names)),
                valid=len(ids) == len(set(ids)) == len(names) and set(ids) == set(names))


def gradient_evidence(model, evidence, name):
    bad, missing, types = [], [], Counter()
    for key, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            missing.append(key)
            continue
        types[str(grad.dtype)] += 1
        finite = torch.isfinite(grad)
        if not bool(finite.all()):
            good = grad[finite]
            bad.append(dict(name=key, nan=int(torch.isnan(grad).sum()), inf=int(torch.isinf(grad).sum()),
                            finite_abs_max=float(good.abs().max()) if good.numel() else None))
    return dict(finite=not bad, nonfinite_parameter_count=len(bad),
                nonfinite_parameters=evidence.write(name, bad), missing=missing, dtype_distribution=dict(types),
                ordering='parameter enumeration; does not identify the earliest faulty operator')


def calibration(model, optimizer, batch, forward, folder, *, amp=True, budget=MAX_ATTEMPTS,
                required_updates=REQUIRED_UPDATES, initial_scale=None, gradient_check=None,
                optimizer_factory=None, label='native_initialization'):
    require(1 <= budget <= MAX_ATTEMPTS and 1 <= required_updates <= budget, 'Finite calibration budget invalid')
    device = batch['img'].device.type
    evidence = Evidence(folder)
    scaler = torch.cuda.amp.GradScaler(enabled=amp, **({} if initial_scale is None else {'init_scale': initial_scale}))
    report = dict(status='RUNNING', label=label, amp=amp, initial_scale=scaler.get_scale(), budget=budget,
                  required_consecutive_updates=required_updates, actual_optimizer_steps=0,
                  consecutive_updates=0, attempts=[], training_dispatched=False,
                  trajectory='ordinary finite diagnostic evolution; fresh graph and zero gradients per attempt; BN/RNG advance normally',
                  diagnostic_optimizer='native AdamW grouping, lr=0.0005; no production warmup or accumulation simulated')
    original_step = optimizer.step
    prior_step = optimizer.__dict__.get('step')
    count = [0]
    def counted_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        count[0] += 1
        return result
    optimizer.step = counted_step
    current = None
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        report.update(input=batch_evidence(batch), initial_model=model_evidence(model),
                      initial_optimizer=state_evidence(optimizer.state_dict()), coverage=coverage(model, optimizer))
        evidence.write('summary', report)
        require(report['coverage']['valid'], 'Missing/duplicate/foreign optimizer parameter')
        inp = report['input']
        require(inp['finite'] and 0 <= inp['min'] <= inp['max'] <= 1, 'Invalid/nonfinite normalized input')
        if 'targets' in inp:
            t = inp['targets']
            require(t['count'] == t['boxes_count'] == t['indices_count'] and
                    all(t[k] for k in ('finite', 'classes_valid', 'boxes_valid', 'batch_indices_valid')), 'Invalid targets')
        for attempt in range(budget):
            optimizer.zero_grad(set_to_none=True)
            current = dict(attempt=attempt, status='FORWARD_PENDING', scale_before=scaler.get_scale(),
                           optimizer_steps_before=count[0], model_before=model_evidence(model),
                           optimizer_before=state_evidence(optimizer.state_dict()))
            def observe(**values):
                current.update(values)
                evidence.write('attempt_%02d' % attempt, current)
            observe()
            require(current['model_before']['parameters']['finite'] and current['model_before']['buffers']['finite'] and
                    current['optimizer_before']['finite'], 'Nonfinite state before forward')
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp) if device == 'cuda' else nullcontext():
                loss = forward(model, batch, observe).sum()
            observe(loss=finite_number(loss.detach()), loss_finite=bool(torch.isfinite(loss)), status='FORWARD_RECORDED')
            require(current.get('predictions_finite') is True and current['loss_finite'], 'Forward prediction/loss nonfinite')
            scaled_loss = scaler.scale(loss)
            observe(scaled_loss_finite=bool(torch.isfinite(scaled_loss)), scaled_loss=finite_number(scaled_loss.detach()), status='BACKWARD_PENDING')
            scaled_loss.backward()
            observe(gradients_scaled=gradient_evidence(model, evidence, 'attempt_%02d_scaled_nonfinite' % attempt))
            scaler.unscale_(optimizer)
            observe(gradients_unscaled=gradient_evidence(model, evidence, 'attempt_%02d_unscaled_nonfinite' % attempt), unscale_calls=1)
            finite = current['gradients_unscaled']['finite'] and current['gradients_scaled']['finite']
            if finite:
                if gradient_check:
                    observe(branch_gradients=gradient_check(model))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                observe(clip_norm=finite_number(norm))
                require(bool(torch.isfinite(norm)), 'Nonfinite gradient norm before optimizer')
            else:
                require(amp, 'FP32 gradient nonfinite')
            # Both paths run the public native scaler sequence. On overflow do not
            # clip invalid gradients; scaler.step must skip the underlying optimizer.
            scaler.step(optimizer)
            scaler.update()
            observe(scale_after=scaler.get_scale(), optimizer_steps_after=count[0],
                    actual_step_delta=count[0]-current['optimizer_steps_before'],
                    model_after=model_evidence(model), optimizer_after=state_evidence(optimizer.state_dict()))
            require(current['model_after']['parameters']['finite'] and current['model_after']['buffers']['finite'] and
                    current['optimizer_after']['finite'], 'Post-update parameter/buffer/optimizer pollution')
            if not finite:
                require(current['actual_step_delta'] == 0 and
                        current['model_before']['parameters']['sha256'] == current['model_after']['parameters']['sha256'] and
                        current['optimizer_before'] == current['optimizer_after'] and
                        current['scale_after'] == current['scale_before'] * scaler.get_backoff_factor(),
                        'Overflow did not skip unchanged optimizer/parameters with native backoff')
                report['consecutive_updates'] = 0
                observe(status='OVERFLOW_SKIPPED', optimizer_step_skipped=True)
            else:
                require(current['actual_step_delta'] == 1, 'Finite gradients did not cause one actual optimizer step')
                report['consecutive_updates'] += 1
                observe(status='UPDATED', optimizer_step_skipped=False)
            optimizer.zero_grad(set_to_none=True)
            report['actual_optimizer_steps'] = count[0]
            report['attempts'].append(dict(attempt=attempt, status=current['status'], loss=current['loss'],
                loss_finite=current['loss_finite'], scale_before=current['scale_before'], scale_after=current['scale_after'],
                actual_step_delta=current['actual_step_delta'], evidence='attempt_%02d.json' % attempt))
            print('AMP_DIAGNOSTIC %s attempt=%d status=%s loss=%s scale=%s->%s actual_steps=%d' %
                  (label, attempt, current['status'], current['loss'], current['scale_before'], current['scale_after'], count[0]), flush=True)
            evidence.write('summary', report)
            del loss, scaled_loss
            if report['consecutive_updates'] >= required_updates:
                break
        require(report['consecutive_updates'] >= required_updates, 'AMP calibration budget exhausted without consecutive real updates')
        if optimizer_factory:
            # Model + optimizer + public scaler state are reloaded together.
            from check_c24_lif_v1 import tree_exact
            with tempfile.TemporaryDirectory(dir=folder, prefix='update_') as tmp:
                path = Path(tmp) / 'state.pt'
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict()), path)
                saved = torch.load(path, map_location=device, weights_only=False)
                clone = deepcopy(model)
                clone.load_state_dict(saved['model'], strict=True)
                other = optimizer_factory(clone)
                other.load_state_dict(saved['optimizer'])
                other_scaler = torch.cuda.amp.GradScaler(enabled=amp)
                other_scaler.load_state_dict(saved['scaler'])
                tree_exact(model.state_dict(), clone.state_dict())
                tree_exact(optimizer.state_dict(), other.state_dict())
                tree_exact(scaler.state_dict(), other_scaler.state_dict())
                report['save_load_model_optimizer_scaler'] = 'EXACT'
        report['status'] = 'PASSED'
    except BaseException as error:
        report.update(status='BLOCKED', error=repr(error), traceback=traceback.format_exc())
        print('AMP_DIAGNOSTIC %s BLOCKED %r | %s' % (label, error, evidence.folder/'summary.json'), flush=True)
        if current is not None:
            current.update(status='BLOCKED', error=repr(error))
            report['failed_attempt'] = evidence.write('attempt_%02d' % current['attempt'], current)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if prior_step is None:
            optimizer.__dict__.pop('step', None)
        else:
            optimizer.step = prior_step
        report['actual_optimizer_steps'] = count[0]
        optimizer.zero_grad(set_to_none=True)
        if device == 'cuda':
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        evidence.write('summary', report)
    return report


def rtdetr_forward(model, batch, observe):
    target = dict(cls=batch['cls'].long().flatten(), bboxes=batch['bboxes'], batch_idx=batch['batch_idx'].long(),
                  gt_groups=[int((batch['batch_idx'] == i).sum()) for i in range(len(batch['img']))])
    predictions = model.predict(batch['img'], batch=target)
    finite = all(bool(torch.isfinite(t).all()) for _, t in tensors(predictions))
    dn = predictions[-1]
    observe(predictions_finite=finite, prediction_shapes={n: list(t.shape) for n, t in tensors(predictions)},
            dn_num_split=dn['dn_num_split'] if dn is not None else None,
            total_queries=int(predictions[0].shape[-2]))
    require(finite, 'Forward prediction nonfinite')
    require(dn is not None and dn['dn_num_split'][-1] == 300 and sum(dn['dn_num_split']) == predictions[0].shape[-2],
            'Native DN/query contract changed')
    if not hasattr(model, 'criterion'):
        model.criterion = model.init_criterion()
    def loss_terms(module, args, output):
        observe(native_loss_terms={k: finite_number(v.detach()) for k, v in output.items()},
                native_loss_terms_finite=all(bool(torch.isfinite(v).all()) for v in output.values()))
    handle = model.criterion.register_forward_hook(loss_terms)
    try:
        loss, components = model.loss(batch, preds=predictions)
    finally:
        handle.remove()
    observe(loss_components=[finite_number(x) for x in components.detach().flatten()],
            loss_component_names=['giou', 'cls', 'l1'], loss_reduction='native total loss.sum()')
    return loss
