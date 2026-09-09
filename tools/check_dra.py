"""Bounded synthetic correctness checks, never formal training or full val/test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from init_dra import (ROOT, build, controlled_models, initialize, is_added, require, runtime, sha256,
                      verify_model, write_json)
from train_dra import rebuild_audit, recipe
from dra_results import export_queries, image_record, verify_archive
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.modules import AIFI, DRAAIFI


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensors(v)]
    return []


def compare(a, b, atol=2e-5, rtol=2e-5):
    aa, bb = tensors(a), tensors(b)
    require(len(aa) == len(bb), "Tensor structure differs")
    maximum = 0.
    for x, y in zip(aa, bb):
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
        if x.numel():
            maximum = max(maximum, float((x.detach() - y.detach()).abs().max()))
    return maximum


def rebuild(source):
    trainer = object.__new__(RTDETRTrainer)
    trainer.data = dict(nc=1, channels=3)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return trainer.get_model(cfg=source.yaml, weights=source, verbose=False)


def module_checks(device):
    rows = []
    for pre in (False, True):
        for h, w in ((20, 20), (5, 7), (7, 5), (1, 1)):
            torch.manual_seed(42)
            original = AIFI(256, 1024, 8, normalize_before=pre).to(device)
            dra = DRAAIFI(256, 1024, 8, normalize_before=pre).to(device)
            dra.load_state_dict({**dra.state_dict(), **original.state_dict()}, strict=True)
            x = torch.rand(2, 256, h, w, device=device)
            row = dict(device=device, hw=[h, w], pre_norm=pre)
            for training in (False, True):
                original.train(training); dra.train(training)
                with torch.no_grad():
                    y = dra(x)
                    row['train' if training else 'eval'] = compare(original(x), y)
                    require(y.shape == x.shape, "Module shape changed")
                    require(torch.count_nonzero(dra.directional_bias(x)) == 0, "Zero bias not zero")
            with torch.no_grad():
                dra.dra_predictor[-1].weight.normal_(0, .4)
                dra.dra_predictor[-1].bias.normal_(0, .4)
                bias = dra.directional_bias(x)
                require(torch.isfinite(bias).all() and bias.abs().max() <= .5, "Invalid bias bound")
                require(torch.count_nonzero(bias[:, 4:]) == 0, "Unmodified heads changed")
                require(torch.count_nonzero(bias.diagonal(dim1=-2, dim2=-1)) == 0, "Nonzero diagonal")
                row['bias_symmetry_max_abs'] = compare(bias, bias.transpose(-1, -2), atol=0, rtol=0)
                row['nonzero_bias_max_abs'] = float(bias.abs().max())
                axis, decay = dra.geometry(h, w, x.device, x.dtype)
                if w > 1:
                    torch.testing.assert_close(axis[0, 1], x.new_tensor([1., 0.]))
                if h > 1:
                    torch.testing.assert_close(axis[0, w], x.new_tensor([-1., 0.]))
                if w > 1 and h > 1:
                    torch.testing.assert_close(axis[0, w + 1], x.new_tensor([0., 1.]))
                    expected = .5 * torch.exp(x.new_tensor(-1 / max(h, w) ** 2 / (2 * .35 ** 2)))
                    torch.testing.assert_close(decay[0, 1], expected)
                require(not any('_axis' in k or '_decay' in k for k in dra.state_dict()), "Persistent geometry")
                dra.half().float()
                require(dra._axis is None, "Stale geometry after dtype conversion")
                dra.directional_bias(x)
                require(dra._axis.dtype == torch.float32, "Geometry must remain FP32")
            rows.append(row)
    return rows


def batch(device):
    return dict(img=torch.rand(2, 3, 160, 192, device=device), cls=torch.zeros(2, 1, device=device),
                batch_idx=torch.arange(2, device=device),
                bboxes=torch.tensor([[.5, .5, .2, .3], [.4, .6, .1, .4]], device=device))


def predict_train(model, sample):
    return model.predict(sample['img'], batch=dict(cls=sample['cls'].flatten().long(), bboxes=sample['bboxes'],
                         batch_idx=sample['batch_idx'].long(), gt_groups=[1, 1]))


def network_checks(reference, target, device, amp, temp):
    a, b = deepcopy(reference).to(device), deepcopy(target).to(device)
    a.nc = b.nc = 1
    row = dict(device=device, amp=amp, synthetic_batch=2, image_hw=[160, 192], steps=[])
    sample = batch(device)
    atol, rtol = ((.005, .005) if amp else (2e-5, 2e-5))
    row['comparison_tolerance'] = dict(atol=atol, rtol=rtol)
    a.eval(); b.eval()
    with torch.no_grad(), torch.autocast(device_type=device, enabled=amp):
        row['eval_max_abs'] = compare(a(sample['img']), b(sample['img']), atol, rtol)
    a.train(); b.train()
    torch.manual_seed(71)
    with torch.autocast(device_type=device, enabled=amp):
        pa = predict_train(a, sample)
        la = a.loss(sample, pa)
    torch.manual_seed(71)
    with torch.autocast(device_type=device, enabled=amp):
        pb = predict_train(b, sample)
        lb = b.loss(sample, pb)
    row['train_output_max_abs'] = compare(pa, pb, atol, rtol)
    row['native_loss_max_abs'] = compare(la, lb, atol, rtol)
    require(pb[0].shape[:2] == (3, 2) and pb[-1]['dn_num_split'][1] == 300 and pb[-1]['dn_num_split'][0] > 0,
            'Decoder/DN output mismatch')
    row['dn_num_split'] = pb[-1]['dn_num_split']
    # The original eval decoder selects exactly its configured final layer.
    with torch.no_grad(), torch.autocast(device_type=device, enabled=amp):
        b.eval()
        result = b(sample['img'])
        raw = result[1]
        expected = torch.cat((raw[0].squeeze(0), raw[1].squeeze(0).sigmoid()), -1)
        row['eval_final_selection_max_abs'] = compare(result[0], expected, 0, 0)
    del a, pa, pb, la, lb, result, raw, expected
    gc.collect()
    b.train()
    trainer = object.__new__(RTDETRTrainer)
    opt = trainer.build_optimizer(b, name='AdamW', lr=.0005, momentum=.937, decay=.0001)
    added = {n: p for n, p in b.named_parameters() if is_added(n)}
    ids = [id(p) for g in opt.param_groups for p in g['params']]
    row['added_optimizer_occurrences'] = {n: ids.count(id(p)) for n, p in added.items()}
    require(all(ids.count(id(p)) == 1 for p in b.parameters() if p.requires_grad), 'Optimizer missing/duplicate')
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.)
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp):
            pred = predict_train(b, sample)
            loss, items = b.loss(sample, pred)
        scaler.scale(loss.sum()).backward()
        scaler.unscale_(opt)
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters()), 'Nonfinite gradient')
        grads = {n: float(p.grad.abs().max()) for n, p in added.items()}
        require(all(v > 0 for n, v in grads.items() if '.4.' in n), 'Last predictor not learning')
        require(all(v == 0 if step == 0 else v > 0 for n, v in grads.items() if '.4.' not in n), 'Staged early gradients failed')
        scaler.step(opt); scaler.update()
        row['steps'].append(dict(step=step, loss=float(loss.sum().detach()), gradients=grads))
    b.eval()
    with torch.no_grad():
        bias = b.model[9].directional_bias(torch.rand(1, 256, 5, 7, device=device))
        row['learned_bias_max_abs'] = float(bias.abs().max())
        require(0 < row['learned_bias_max_abs'] <= .5, 'Bias failed to leave zero')
        before = b(sample['img'])
    checkpoint = temp / f'{device}_{amp}_debug.pt'
    torch.save(dict(epoch=-1, model=deepcopy(b).cpu().float(), train_args={'task': 'detect'}, debug_only=True), checkpoint)
    restored = RTDETR(str(checkpoint)).model.to(device).eval()
    verify_model(restored)
    with torch.no_grad():
        row['reload_max_abs'] = compare(before, restored(sample['img']))
        half = deepcopy(restored).half()
        out = half(sample['img'].half())
        require(all(torch.isfinite(t).all() for t in tensors(out)), 'Actual model.half failed')
        row['actual_half_forward'] = dict(status='passed', output_dtype=str(out[0].dtype), shape=list(out[0].shape))
    return row


def export_check():
    # Unsorted scores, ties, below-threshold and rectangular original images.
    raw = torch.rand(2, 300, 5)
    raw[..., 4] = torch.arange(300).remainder(3) / 2
    validator = object.__new__(RTDETRValidator)
    validator.args = SimpleNamespace(imgsz=640, conf=.3)
    full = export_queries(raw, 640, .3)
    selected = validator.postprocess(raw.clone())
    for pred, actual in zip(full, selected):
        for k in ('bboxes', 'conf', 'cls'):
            torch.testing.assert_close(pred[k][pred['metric_mask']], actual[k], atol=0, rtol=0)
        row = image_record(pred, dict(ori_shape=(480, 800), imgsz=(640, 640), bboxes=torch.tensor([[1., 2., 3., 4.]]),
                           cls=torch.tensor([0]), im_file=str(ROOT / 'fake.jpg')), ROOT, .3)
        require(len(row['predictions']) == 300 and sum(p['used_for_metrics'] for p in row['predictions']) == len(actual['conf']),
                'Same-pass export subset differs')
    return 'Native sorted order/mask exactly preserved; same-pass flags and pixel scaling checked'


def run(source, output):
    require(not output.exists(), 'Preserve prior evidence')
    output.mkdir(parents=True)
    report = dict(status='failed', runtime=runtime(), full_server_preflight='NOT_RUN', formal_training='NOT_RUN',
                  full_val_test='NOT_RUN', server_4090_torch212='NOT_RUN')
    try:
        a80, b80, mapping = controlled_models(source)
        write_json(output / 'mapping.json', mapping)
        a, b = rebuild(a80), rebuild(b80)
        report['nc1_loading'] = rebuild_audit(b80, b, 'dra_aifi')
        require(all(torch.equal(v, b.state_dict()[k]) for k,v in a.state_dict().items()), 'nc1 common initialization differs')
        report['parameters_unfused_nc1'] = dict(c2=sum(p.numel() for p in a.parameters()), dra_aifi=sum(p.numel() for p in b.parameters()), added=4376)
        del a80, b80
        args, rows = recipe(ROOT / 'docs/dra/c2_args.yaml', 'dra_aifi', ROOT / 'weights/dra_aifi_controlled_init.pt')
        write_json(output / 'parameter_diff.json', rows)
        report['export'] = export_check()
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        report['module'] = []; report['network'] = []
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            temp = Path(tmp)
            init = temp / 'clean_init.pt'
            report['checkpoint_initialization'] = initialize(source, init, 'dra_aifi')
            # Native public train API reconstruction, stopped before dataset/training setup.
            probe = {}
            class StopProbe(Exception):
                pass
            class Probe(RTDETRTrainer):
                def __init__(self, overrides, _callbacks):
                    self.data = dict(nc=1, channels=3)
                def get_model(self, cfg=None, weights=None, verbose=True):
                    model = super().get_model(cfg, weights, False)
                    probe.update(rebuild_audit(weights, model, 'dra_aifi'))
                    return model
                def train(self):
                    verify_model(self.model, zero=True)
                    raise StopProbe()
            with patch('ultralytics.utils.checks.check_pip_update_available'):
                try:
                    RTDETR(str(init)).train(trainer=Probe, data='unused-probe', epochs=200)
                except StopProbe:
                    report['native_train_api_rebuild'] = probe
                else:
                    raise RuntimeError('Probe failed to stop')
            for device in devices:
                report['module'] += module_checks(device)
                for amp in ([False, True] if device == 'cuda' else [False]):
                    print(f'Checking {device=} {amp=}', flush=True)
                    report['network'].append(network_checks(a, b, device, amp, temp))
                    write_json(output / 'report.json', report)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        report['status'] = 'passed'
        return report
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(output / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    run(args.source, args.output)
