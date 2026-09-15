"""Bounded CPU, CUDA and real B16/640 AMP checks; never launches formal training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import torch
from init_nbrg_v1 import (ROOT, VARIANTS, WIDTHS, SOURCE_SHA256, require, write_json, sha256, runtime, pair,
    RTDETR, verify_model, build_training_model, select_channels, selection_from_source, slice_state, tensor_hash, gate_key)
from ultralytics.nn.modules.nbr_g import InputGuidedGroupGate, NarrowBasicBlock, NBRGBlock
from ultralytics.nn.modules import BasicBlock, ConvNormLayer
from ultralytics.utils.torch_utils import init_seeds, ModelEMA, fuse_conv_and_bn
from ultralytics.utils.patches import torch_load
from c19_lif_v1_data import real_batch
from c19_lif_v1_diagnostic import fusion_protocol
from c19_lif_v1_cutoff import fusion_accepted
from check_c19_lif_v1 import optimizer, activate
from train_nbrg_v1 import trainer_class, source_fingerprint, verify_data


def close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    return dict(max_abs=float((a - b).abs().max()), rtol=1e-5, atol=1e-6)


def module_checks(source):
    src = torch_load(source, map_location='cpu')['model'].float()
    selection = selection_from_source(src)
    rows = {}
    for stage, (channels, hidden) in WIDTHS.items():
        original = deepcopy(src.model[stage].blocks[1]).eval()
        index = torch.tensor(selection['stages'][str(stage)]['indices'])
        narrow, full, gated = NarrowBasicBlock(channels, hidden).eval(), NarrowBasicBlock(channels, channels).eval(), NBRGBlock(channels, hidden).eval()
        local = slice_state(src.state_dict(), selection)
        prefix = f'model.{stage}.blocks.1.'
        selected = {k[len(prefix):]: v for k, v in local.items() if k.startswith(prefix)}
        narrow.load_state_dict(selected, strict=True)
        full.load_state_dict(original.state_dict(), strict=True)
        gated.load_state_dict({**gated.state_dict(), **selected}, strict=True)
        for block in (full, narrow, gated):
            for name in ('branch2a', 'branch2b'):
                a, b = getattr(original, name).norm, getattr(block, name).norm
                b.eps, b.momentum = a.eps, a.momentum
        x = torch.randn(2, channels, 9, 11)
        with torch.no_grad():
            h = original.branch2a(x)
            mask = torch.zeros(channels); mask[index] = 1
            masked = original.act(x + original.branch2b(h * mask[None, :, None, None]))
            a = narrow(x)
            error = (a - masked).abs()
            fp32 = dict(max_abs=float(error.max()), elements_outside_suggested_tolerance=int((error > 1e-6 + 1e-5 * masked.abs()).sum()),
                        note='Different dense reduction widths change FP32 accumulation; verified with the same source/input in FP64 below, no tolerance widening')
            wide64, narrow64 = deepcopy(original).double(), deepcopy(narrow).double()
            h64 = wide64.branch2a(x.double()) * mask.double()[None, :, None, None]
            reference64 = wide64.act(x.double() + wide64.branch2b(h64))
            torch.testing.assert_close(narrow64(x.double()), reference64, rtol=1e-10, atol=1e-11)
            rows[str(stage)] = dict(full_width=close(full(x), original(x)), masked_wide_fp32_diagnostic=fp32,
                masked_wide_fp64=dict(max_abs=float((narrow64(x.double())-reference64).abs().max()),rtol=1e-10,atol=1e-11),
                zero_gate=close(narrow(x), gated(x)), gate_shape=list(gated.gate(x, torch.randn_like(x)).shape))
    # Unequal channel magnitudes distinguish mean vs max, and negative values exercise abs.
    gate = InputGuidedGroupGate(16)
    x = torch.arange(1., 17.).reshape(1, 16, 1, 1).expand(1, 16, 3, 4).clone().requires_grad_()
    f = (-x.detach() * 2 - 1).requires_grad_()
    before = (x.detach().clone(), f.detach().clone())
    gate(x, f).sum().backward()
    require(x.grad is not None and f.grad is not None and x.grad.count_nonzero() == 0 and f.grad.count_nonzero() == 0,
            'Zero gate weights should give zero first-step descriptor input gradient')
    x.grad, f.grad = None, None
    desc = gate.descriptors(x, f)
    for group in range(8):
        xx, ff = x[:, 2*group:2*group+2].abs(), f[:, 2*group:2*group+2].abs()
        for j, expected in enumerate((xx.mean(1), xx.amax(1), ff.mean(1), ff.amax(1))):
            close(desc[:, group*4+j], expected)
    with torch.no_grad():
        gate.gate_conv.bias.copy_(torch.linspace(-1., 1., 8))
    g = gate(x, f)
    result = (f.reshape(1, 8, 2, 3, 4) * g.unsqueeze(2)).reshape_as(f)
    for i in range(16):
        close(result[:, i], f[:, i] * (1 + .5 * torch.tanh(gate.gate_conv.bias[i//2])))
    # Nonzero weights exercise descriptor -> X/F gradients; zero weights correctly give no extra path.
    with torch.no_grad():
        gate.gate_conv.weight.fill_(.001)
    gate(x, f).sum().backward()
    require(x.grad.abs().sum() > 0 and f.grad.abs().sum() > 0, 'Descriptor path detached')
    require(torch.equal(x, before[0]) and torch.equal(f, before[1]), 'Inputs modified in place')
    block = NBRGBlock(16, 8)
    xx = torch.randn(2, 16, 5, 7, requires_grad=True)
    (block(xx) * torch.randn_like(xx)).sum().backward()
    gradients = {name: float(p.grad.norm()) for name, p in block.named_parameters()}
    require(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                for p in block.gate.parameters()), 'Gate has no finite learning signal')
    require(torch.isfinite(xx.grad).all() and block.branch2a.conv.weight.grad.abs().sum() > 0, 'Narrow gradient failure')
    # Controlled positive FP32 values also exercise masked-wide equivalence
    # without the cancellation in the real pretrained S5 BN statistics.
    controlled_rows = {}
    for stage, (channels, hidden) in WIDTHS.items():
        wide = BasicBlock(channels, channels, 1, True).eval()
        small = NarrowBasicBlock(channels, hidden).eval()
        with torch.no_grad():
            wide.branch2a.conv.weight.fill_(1 / 4096)
            wide.branch2b.conv.weight.fill_(1 / 4096)
        selected = dict(wide.state_dict())
        for suffix in ('conv.weight', 'norm.weight', 'norm.bias', 'norm.running_mean', 'norm.running_var'):
            selected['branch2a.' + suffix] = selected['branch2a.' + suffix][:hidden]
        selected['branch2b.conv.weight'] = selected['branch2b.conv.weight'][:, :hidden]
        small.load_state_dict(selected, strict=True)
        positive = torch.rand(1, channels, 3, 5)
        with torch.no_grad():
            hh = wide.branch2a(positive)
            hh[:, hidden:] = 0  # independent reference tensor, never a production branch view
            controlled_rows[str(stage)] = close(small(positive), wide.act(positive + wide.branch2b(hh)))
    # Exact tie order and gamma=0,beta>0 constant channel scoring regression.
    special = BasicBlock(8, 8, 1, True)
    with torch.no_grad():
        special.branch2a.conv.weight.fill_(1)
        special.branch2b.conv.weight.fill_(1)
        special.branch2a.norm.weight.zero_()
        special.branch2a.norm.bias.fill_(1)
        special.branch2a.norm.bias[0] = -1
    scored = select_channels(special, 3)
    require(scored['indices'] == [1, 2, 3] and scored['scores'][0] == 0 and scored['scores'][1] > 0, 'Constant-channel/tie score rule')
    with torch.no_grad():
        special.branch2a.norm.running_var[0] = -1
    try:
        select_channels(special, 3)
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Illegal variance accepted')
    return dict(status='PASSED', relations=rows, descriptors_interleaved=True, contiguous_broadcast=True,
                no_inplace=True, descriptor_gradients=True, zero_first_step_descriptor_gradient=True,
                controlled_fp32_masked_wide=controlled_rows, gate_gradient_norms=gradients, score_edge_cases=True)


def api_probe(initial, expected, variant, source):
    seen = {}
    class StopProbe(Exception):
        pass
    class ProbeTrainer(trainer_class(variant, source, None)):
        def __init__(self, overrides, _callbacks):
            self.args = SimpleNamespace(**{k: v for k, v in overrides.items() if k != 'session'})
            self.data = dict(nc=1, channels=3)
            init_seeds(42, deterministic=True)
        def train(self):
            require(all(torch.equal(v, self.model.state_dict()[k]) for k, v in expected.state_dict().items()), 'Actual API initial state differs')
            opt = self.build_optimizer(self.model, name='AdamW', lr=.0005, momentum=.937, decay=.0001)
            seen.update(status='PASSED', real_train_api_get_model=True, real_recording_trainer_optimizer=True,
                optimizer_parameter_tensors=sum(len(g['params']) for g in opt.param_groups),
                gates=[dict(name=k, trainable=p.requires_grad, zero=bool(p.count_nonzero() == 0))
                       for k, p in self.model.named_parameters() if gate_key(k)], optimizer_steps=0)
            raise StopProbe()
    from ultralytics.utils import YAML
    args = YAML.load(ROOT / 'docs/c19_lif_v1/c2_args.yaml')
    args.update(model=str(initial), name='nbrg_api_probe', save_dir=str(ROOT / 'outputs/nbrg_api_probe'))
    with patch('ultralytics.engine.model.checks.check_pip_update_available'):
        try:
            RTDETR(str(initial)).train(trainer=ProbeTrainer, **args)
        except StopProbe:
            pass
    require(seen, 'Train API probe never reached')
    return seen


def serialization(model, variant, folder, zero):
    copy = deepcopy(model).cpu().eval()
    if not zero:
        activate(copy, bn=True)
        with torch.no_grad():
            for name, p in copy.named_parameters():
                if gate_key(name):
                    p.fill_(.002 if name.endswith('weight') else .1)
    image = torch.rand(1, 3, 160, 192)
    with torch.no_grad():
        prediction = copy(image)[0]
    tag = 'zero' if zero else 'learned_gate'
    checkpoint = folder / (tag + '.pt')
    torch.save(dict(model=copy, epoch=-1, train_args=dict(task='detect')), checkpoint)
    fixture = folder / (tag + '_fixture.pt')
    torch.save(dict(checkpoint=str(checkpoint.resolve()), variant=variant, zero=zero, image=image, prediction=prediction,
                    state_hashes={k: tensor_hash(v) for k, v in copy.state_dict().items()}), fixture)
    output = folder / (tag + '_reload.json')
    subprocess.run([sys.executable, str(ROOT / 'tools/nbrg_v1_reload_probe.py'), '--fixture', str(fixture), '--output', str(output)],
                   cwd=ROOT, check=True)
    return json.loads(output.read_text(encoding='utf-8'))


def safe_full_fuse(model):
    """Explicit deployment copy: native fusion plus sequential ConvNormLayer BN folding."""
    model.eval().fuse(verbose=False)
    for module in model.modules():
        if isinstance(module, ConvNormLayer) and hasattr(module, 'norm'):
            module.conv = fuse_conv_and_bn(module.conv, module.norm)
            del module.norm
            module.forward = module.forward_fuse
    return model


def fusion_checks(target, device, folder):
    model = deepcopy(target).eval().float().to(device)
    activate(model, bn=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if gate_key(name):
                p.fill_(.002 if name.endswith('weight') else .1)
    x = torch.rand(1, 3, 160, 192, device=device)
    full = safe_full_fuse(deepcopy(model))
    native = deepcopy(model).fuse(verbose=False)
    def parent_factory():
        parent = pair.build('C2', nc=1).eval()
        for stage in WIDTHS:
            parent.model[stage] = deepcopy(model.model[stage]).cpu().float()
        parent.load_state_dict({k: model.state_dict()[k].cpu().float() for k in parent.state_dict()}, strict=True)
        return parent
    rows = {}
    checks = [('native_fp32', model, native, x, 'fp32'), ('full_fp32', model, full, x, 'fp32')]
    if device == 'cuda':
        checks += [('native_amp', model, native, x, 'amp'), ('native_half', deepcopy(model).half(), deepcopy(native).half(), x.half(), 'half')]
    for name, a, b, image, precision in checks:
        result = fusion_protocol(a, b, image, folder / name, device, precision,
                                 parent_factory=parent_factory, parent_source=model)
        require(fusion_accepted(result, device, precision), 'Fusion blocked; inspect inherited cutoff/replay evidence')
        rows[name] = dict(status='PASSED', diagnostic=str(folder / name / 'fuse_diagnostic.json'), selection=result['selection'])
    return rows


def backward(model, batch, device, amp):
    model = deepcopy(model).to(device).train()
    model.nc = 1
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer(model)
    with torch.autocast(device_type=device, enabled=amp):
        loss = model.loss(batch)[0]
    loss.backward()
    require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), 'Nonfinite criterion/backward')
    gates = {k: float(p.grad.norm()) if p.grad is not None else None for k, p in model.named_parameters() if gate_key(k)}
    require(all(v is not None and v > 0 for v in gates.values()), 'Gate learning signal missing in criterion')
    return dict(status='PASSED', loss=float(loss), batch=len(batch['img']), image_shape=list(batch['img'].shape),
                amp=amp, device=device, gate_gradients=gates, optimizer_steps=0, disposable=True)


def complexity(model, variant):
    import thop
    model = deepcopy(model).cpu().eval()
    image = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        macs, _ = thop.profile(deepcopy(model), inputs=(image,), verbose=False)
    native = deepcopy(model).fuse(verbose=False)
    full = safe_full_fuse(deepcopy(model))
    return dict(variant=variant, nc=1, imgsz=640, batch=1, params_unfused=sum(p.numel() for p in model.parameters()),
        params_native_fused=sum(p.numel() for p in native.parameters()), params_all_sequential_bn_fused=sum(p.numel() for p in full.parameters()),
        thop_version=thop.__version__, thop_gmacs=macs / 1e9, thop_gflops=2 * macs / 1e9,
        scope='Actual 640 forward, unfused THOP supported operators; dynamic abs/mean/amax/stack/tanh/broadcast and functional attention/grid_sample are not fully counted.',
        latency='NOT_MEASURED', native_fusion='Existing BaseModel.fuse; ConvNormLayer.norm is retained',
        extended_fusion='Independent copy; fold all sequential ConvNormLayer; original LIF shared BN retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'cuda', 'capacity'), default='cpu')
    parser.add_argument('--variant', choices=['all', *VARIANTS], default='all')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--initial-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, help='Original data YAML with environment root mapping')
    parser.add_argument('--output', type=Path, required=True, help='New disposable preflight directory')
    args = parser.parse_args()
    require(not args.output.exists(), 'Existing diagnostic directory preserved')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    init_seeds(42, deterministic=True)
    variants = list(VARIANTS) if args.variant == 'all' else [args.variant]
    report = dict(status='RUNNING', mode=args.mode, runtime=runtime(), source_sha256=sha256(args.source),
        source_fingerprint=source_fingerprint(), initial_sha256={}, variants={}, formal_training='NOT_RUN', final_test='NOT_RUN')
    def persist():
        write_json(args.output / 'preflight.json', report)
    try:
        require(report['source_sha256'] == SOURCE_SHA256, 'Wrong source hash')
        if args.mode == 'cpu':
            report['modules'] = module_checks(args.source)
        else:
            if not torch.cuda.is_available():
                report.update(status='PENDING_CUDA_B16' if args.mode == 'capacity' else 'PENDING_CUDA')
                return
            if args.data is None:
                report.update(status='PENDING_DATA')
                return
            report['dataset_inventory'] = verify_data(args.data)
        hashes = {}
        for variant in variants:
            folder = args.output / variant
            folder.mkdir()
            initial = args.initial_dir / (variant + '.pt')
            report['initial_sha256'][variant] = sha256(initial)
            row = report['variants'][variant] = dict(status='RUNNING')
            persist()
            weights = RTDETR(str(initial)).model
            init_seeds(42, deterministic=True)
            target, row['nc1'] = build_training_model(weights.yaml, weights, dict(nc=1, channels=3), variant, args.source)
            hashes[variant] = {k: tensor_hash(v) for k, v in target.state_dict().items()}
            target.eval()
            if args.mode == 'cpu':
                row['train_api'] = api_probe(initial, target, variant, args.source)
                outputs = {}
                handles = [target.model[i].register_forward_hook(lambda m, a, out, i=i: outputs.update({str(i): list(out.shape)})) for i in (4, 5, 6, 7)]
                try:
                    with torch.no_grad():
                        pred = target(torch.rand(1, 3, 640, 640))[0]
                    require(torch.isfinite(pred).all() and list(pred.shape) == [1, 300, 5], 'nc1 output contract')
                    require(outputs == {'4': [1,64,160,160], '5': [1,128,80,80], '6': [1,256,40,40], '7': [1,512,20,20]}, 'Backbone interfaces')
                finally:
                    for handle in handles:
                        handle.remove()
                row['backbone_outputs'] = outputs
                row['serialization_zero'] = serialization(target, variant, folder, True)
                row['serialization_nonzero'] = serialization(target, variant, folder, False)
                row['complexity'] = complexity(target, variant)
                if VARIANTS[variant][1]:
                    batch = dict(img=torch.rand(2, 3, 160, 160), cls=torch.zeros(3, 1),
                        bboxes=torch.tensor([[.4,.4,.2,.3],[.7,.6,.1,.2],[.3,.2,.1,.1]]), batch_idx=torch.tensor([0,0,1]))
                    row['criterion'] = backward(target, batch, 'cpu', False)
                    row['fusion'] = fusion_checks(target, 'cpu', folder)
            else:
                from ultralytics.utils import YAML
                dataset = Path(YAML.load(args.data)['path'])
                batch, records = real_batch(dataset, size=640 if args.mode == 'capacity' else 160, count=16 if args.mode == 'capacity' else 2)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                try:
                    row['criterion'] = backward(target, batch, 'cuda', True)
                    if args.mode == 'cuda' and VARIANTS[variant][1]:
                        row['fusion'] = fusion_checks(target, 'cuda', folder)
                    torch.cuda.synchronize()
                    row.update(samples=records, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), seconds=time.perf_counter()-started)
                except torch.cuda.OutOfMemoryError as error:
                    row.update(status='PENDING_CUDA_B16' if args.mode == 'capacity' else 'PENDING_CUDA_MEMORY', error=str(error),
                               batch_requested=len(batch['img']), imgsz_requested=batch['img'].shape[-1], lowered_batch=False)
                    persist()
                    continue
            row['status'] = 'PASSED'
            persist()
            del target, weights
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        control, candidate = 'cbr_lif_nbr_control_v1', 'cbr_lif_nbrg_v1'
        if control in hashes and candidate in hashes:
            require(all(v == hashes[candidate][k] for k, v in hashes[control].items()), 'nc1 control/candidate common tensors differ')
            report['nc1_pair_common_exact'] = len(hashes[control])
        report['status'] = 'PASSED' if all(row['status'] == 'PASSED' for row in report['variants'].values()) else 'PENDING'
    except BaseException as error:
        report.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        persist()
        print('Preflight:', report['status'], args.output / 'preflight.json', flush=True)


if __name__ == '__main__':
    main()
