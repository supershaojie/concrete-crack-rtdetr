"""Strict C2 -> original CBR/LIF -> synchronized NBR/NBR-G initialization."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import torch
import init_c19_lif_v1 as pair
from init_c19_lif_v1 import ROOT, MODEL_DIR, SOURCE_SHA256, require, sha256, write_json, runtime
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import BasicBlock, LIFDown, RTDETRDecoderCBR, NBRStage, NBRGBlock, NarrowBasicBlock
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE_SHA = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
BRANCH = 'exp-rtdetr-r18-lite-nbrg-v1'
VARIANTS = {
    'cbr_lif_nbr_control_v1': ('rtdetr-resnet18-lite-cbr-lif-nbr-control-v1.yaml', True, False),
    'cbr_lif_nbrg_v1': ('rtdetr-resnet18-lite-cbr-lif-nbrg-v1.yaml', True, True),
    'nbrg_v1': ('rtdetr-resnet18-lite-nbrg-v1.yaml', False, True),
}
WIDTHS = {6: (256, 192), 7: (512, 256)}
ALGORITHM = 'bn-folded-two-sided-l2-product-cpu-f64-stable-index-v1'


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def gate_key(key):
    return any(key.startswith(f'model.{stage}.blocks.1.gate.') for stage in WIDTHS)


def verify_model(model, variant, zero=False):
    yaml, combined, gated = VARIANTS[variant]
    expected = YAML.load(MODEL_DIR / yaml)
    require(all(model.yaml[k] == expected[k] for k in ('backbone', 'head', 'scales')), 'Unexpected model YAML')
    require(type(model) is RTDETRDetectionModel and len(model.model) == 27, 'Native topology changed')
    for stage, (channels, hidden) in WIDTHS.items():
        first, second = model.model[stage].blocks
        require(type(model.model[stage]) is NBRStage and type(first) is BasicBlock, 'Wrong stage/first block')
        require(first.branch2a.conv.stride == (2, 2) and not first.shortcut, 'Downsampling block changed')
        require(type(first.short.pool) is torch.nn.AvgPool2d, 'Projection shortcut changed')
        require(type(second) is (NBRGBlock if gated else NarrowBasicBlock) and second.shortcut, 'Wrong second block')
        for layer, dims in ((second.branch2a, (channels, hidden)), (second.branch2b, (hidden, channels))):
            conv = layer.conv
            require((conv.in_channels, conv.out_channels) == dims, 'Narrow dimensions changed')
            require(conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1) and conv.groups == 1, 'Dense 3x3 contract')
        if gated:
            conv = second.gate.gate_conv
            require(sum(p.numel() for p in conv.parameters()) == 296 and conv.groups == 8, 'Gate count/groups')
    gates = {k: p for k, p in model.named_parameters() if gate_key(k)}
    require(sum(p.numel() for p in gates.values()) == (592 if gated else 0), 'Pure control has gate or wrong gate count')
    if zero:
        require(all(p.requires_grad and torch.count_nonzero(p) == 0 for p in gates.values()), 'Fresh gates must be trainable zero')
    require(isinstance(model.model[20], LIFDown) == combined and isinstance(model.model[-1], RTDETRDecoderCBR) == combined, 'CBR/LIF variant mismatch')
    require(model.model[-1].f == [19, 22, 25], 'Decoder inputs changed')
    if combined:
        pair.source_contract()
        lif, cbr = model.model[20], model.model[-1].cbr
        require(hasattr(lif, 'bn') and lif.forward.__func__ is LIFDown.forward, 'LIF fusion protection lost')
        require(cbr.rho == cbr.normal_fraction == .1, 'Original CBR changed')
        if zero:
            require(torch.count_nonzero(lif.O_proj.weight) == 0, 'LIF initialization changed')
            require(all(torch.count_nonzero(p) == 0 for p in cbr.offset_out.parameters()), 'CBR initialization changed')
    return dict(variant=variant, nc=model.model[-1].nc, gates=len(gates), gate_parameters=sum(p.numel() for p in gates.values()))


def select_channels(block, keep):
    require(type(block) is BasicBlock and block.shortcut, 'Source is not identity BasicBlock')
    a, b = block.branch2a, block.branch2b
    require(a.conv.bias is None and b.conv.bias is None, 'Scoring assumes bias-free convolution')
    values = [a.conv.weight, b.conv.weight] + [v for layer in (a.norm, b.norm) for v in
              (layer.weight, layer.bias, layer.running_mean, layer.running_var)]
    require(all(torch.isfinite(v).all() for v in values), 'Nonfinite source')
    require(all(torch.isfinite(torch.tensor(layer.eps)) and layer.eps > 0 and (layer.running_var >= 0).all()
                for layer in (a.norm, b.norm)), 'Invalid source BN variance/eps')
    w1, w2, g1, beta1, mean1, var1, g2, _, _, var2 = [v.detach().cpu().double() for v in values]
    alpha1, alpha2 = g1 / (var1 + a.norm.eps).sqrt(), g2 / (var2 + b.norm.eps).sqrt()
    k1 = alpha1[:, None, None, None] * w1
    bias1 = beta1 - alpha1 * mean1
    k2 = alpha2[:, None, None, None] * w2
    score = (k1.square().sum((1, 2, 3)) + bias1.square()).sqrt() * k2.square().sum((0, 2, 3)).sqrt()
    dead = (k1 == 0).flatten(1).all(1) & (bias1 <= 0)
    score[dead] = 0
    require(torch.isfinite(score).all() and 0 < keep <= len(score), 'Invalid score/keep')
    chosen = sorted(sorted(range(len(score)), key=lambda i: (-score[i].item(), i))[:keep])
    return dict(channels=len(score), keep=keep, scores=score.tolist(), indices=chosen,
                eps1=a.norm.eps, eps2=b.norm.eps, dead_relu_indices=dead.nonzero().flatten().tolist())


def selection_from_source(base):
    return dict(status='PASSED', algorithm=ALGORITHM, source_sha256=SOURCE_SHA256,
                scoring_data='checkpoint parameters and BN buffers only; no images',
                stages={str(i): select_channels(base.model[i].blocks[1], m) for i, (_, m) in WIDTHS.items()})


def slice_state(source_state, selection):
    result = dict(source_state)
    for stage, row in selection['stages'].items():
        prefix = f'model.{stage}.blocks.1.'
        indices = torch.tensor(row['indices'], dtype=torch.long)
        keys = [prefix + 'branch2a.conv.weight'] + [prefix + 'branch2a.norm.' + k for k in
                 ('weight', 'bias', 'running_mean', 'running_var')]
        for key in keys:
            result[key] = source_state[key].index_select(0, indices.to(source_state[key].device))
        key = prefix + 'branch2b.conv.weight'
        result[key] = source_state[key].index_select(1, indices.to(source_state[key].device))
    return result


def copy_bn_settings(source, target):
    src = dict(source.named_modules())
    rows = {}
    for key, module in target.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            old = src[key]
            require(module.affine == old.affine and module.track_running_stats == old.track_running_stats, 'BN behavior mismatch')
            module.eps, module.momentum = old.eps, old.momentum
            rows[key] = dict(eps=module.eps, momentum=module.momentum, affine=module.affine,
                             track_running_stats=module.track_running_stats)
    return rows


def convert(reference, variant, selection):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        target = RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][0]), nc=reference.model[-1].nc, verbose=False)
    sliced, state = slice_state(reference.state_dict(), selection), target.state_dict()
    extra = set(state) - set(sliced)
    require(extra == {k for k in state if gate_key(k)} and not set(sliced) - set(state), 'Unknown missing/unexpected state')
    require(all(sliced[k].shape == state[k].shape for k in sliced), 'Unknown shape mismatch')
    target.load_state_dict({**sliced, **{k: state[k] for k in extra}}, strict=True)
    bn_settings = copy_bn_settings(reference, target)
    verify_model(target, variant, zero=True)
    require(all(torch.equal(v, target.state_dict()[k]) for k, v in sliced.items()), 'Synchronized transfer failed')
    return target, dict(shared_and_sliced_tensors=len(sliced), shared_and_sliced_exact=True,
                        state_hashes={k: tensor_hash(v) for k, v in target.state_dict().items()},
                        NEW_TRAINABLE=sorted(extra), MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[], bn_settings=bn_settings)


def controlled_models(source):
    base, combined, parent_audit = pair.controlled_models(source)
    original = torch_load(source, map_location='cpu')['model'].float()
    for stage in WIDTHS:
        for name in ('branch2a', 'branch2b'):
            raw = getattr(original.model[stage].blocks[1], name).norm
            ref = getattr(base.model[stage].blocks[1], name).norm
            require((raw.eps, raw.momentum, raw.affine, raw.track_running_stats) ==
                    (ref.eps, ref.momentum, ref.affine, ref.track_running_stats), 'Source/reference BN behavior differs')
    selection = selection_from_source(original)
    models, reports = {}, {}
    for variant, (_, is_pair, _) in VARIANTS.items():
        models[variant], reports[variant] = convert(combined if is_pair else base, variant, selection)
    public = models['cbr_lif_nbr_control_v1'].state_dict()
    gated = models['cbr_lif_nbrg_v1'].state_dict()
    require(all(torch.equal(v, gated[k]) for k, v in public.items()), 'Control/candidate public state differs')
    return models, selection, dict(status='PASSED', source=str(Path(source).resolve()), source_sha256=SOURCE_SHA256,
        source_nc=80, serialized_nc=80, trainer_nc=1, parent_initialization=parent_audit,
        variants=reports, pair_common_tensors=len(public), pair_common_exact=True, seed=42,
        class_adaptation='Native original C19/LIF Trainer timing; nine exact C2 class tensors only')


def build_training_model(cfg, weights, data, variant, source):
    """Run original nc=1 adaptation at the same RNG state, then verify every tensor."""
    require(data['nc'] == 1 and data['channels'] == 3, 'Formal task is nc1 RGB')
    verify_model(weights, variant, zero=True)
    # Entire reference check is RNG-isolated. The real target then consumes the
    # same wide constructor RNG as original C2/C19+LIF at the native Trainer time.
    with torch.random.fork_rng(devices=[]):
        base, combined, _ = pair.controlled_models(source)
        selection = selection_from_source(base)
        reference80 = combined if VARIANTS[variant][1] else base
        reference1 = (pair.build_training_model(reference80.yaml, reference80, data)[0] if VARIANTS[variant][1]
                      else pair.native_rebuild(reference80.yaml, reference80, 1, 3))
        expected80 = slice_state(reference80.state_dict(), selection)
        require(all(torch.equal(v, weights.state_dict()[k]) for k, v in expected80.items()), 'Not a clean controlled initial state')
        expected1 = slice_state(reference1.state_dict(), selection)
    target = pair.native_rebuild(cfg, weights, 1, 3)
    copy_bn_settings(reference1, target)
    before, after = weights.state_dict(), target.state_dict()
    allowed = {'model.26.' + key for key in ('denoising_class_embed.weight', 'enc_score_head.weight', 'enc_score_head.bias')}
    allowed |= {f'model.26.dec_score_head.{i}.{suffix}' for i in range(3) for suffix in ('weight', 'bias')}
    changed = {k for k in before if before[k].shape != after[k].shape}
    require(set(before) == set(after) and changed == (allowed if weights.model[-1].nc == 80 else set()), 'Unknown class adaptation')
    require(all(torch.equal(v, after[k]) for k, v in expected1.items()), 'Native nc1 public/sliced values changed')
    require(all(torch.equal(v, after[k]) for k, v in before.items() if k not in changed), 'Trainer lost initialized values')
    verify_model(target, variant, zero=True)
    return target, dict(status='PASSED', native_get_model=True, before_optimizer=True,
        public_and_sliced_nc1_exact=len(expected1), gate_trainable_zero=True,
        ALLOWED_CLASS_ADAPTATION=[dict(name=k, source_shape=list(before[k].shape), target_shape=list(after[k].shape),
            rule='original native C2/C19+LIF RNG and adaptation timing') for k in sorted(changed)],
        MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[])


def initialize(source, output):
    output = Path(output)
    require(not output.exists(), f'Existing initialization directory preserved: {output}')
    models, selection, report = controlled_models(source)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'channel_selection.json', selection)
    for variant, model in models.items():
        dest = output / (variant + '.pt')
        model.eval()
        model.args = {**DEFAULT_CFG_DICT, 'model': str(MODEL_DIR / VARIANTS[variant][0]), 'task': 'detect'}
        model.task, model.pt_path = 'detect', str(dest.resolve())
        ckpt = dict(epoch=-1, best_fitness=None, model=model.float(), ema=None, updates=None, optimizer=None, scaler=None,
            train_args=model.args, train_metrics=None, train_results=None, date=datetime.now(timezone.utc).isoformat(),
            nbrg_v1_provenance=dict(variant=variant, source_sha256=SOURCE_SHA256,
                selection_sha256=sha256(output / 'channel_selection.json'), algorithm=ALGORITHM))
        with dest.open('xb') as stream:
            torch.save(ckpt, stream)
        loaded = RTDETR(str(dest)).model
        require(all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items()), 'Serialization altered state')
        report['variants'][variant].update(checkpoint=str(dest.resolve()), sha256=sha256(dest), reload_exact=True)
    report.update(selection_sha256=sha256(output / 'channel_selection.json'), runtime=runtime())
    write_json(output / 'initialization_audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for all three clean initial checkpoints and audits')
    args = parser.parse_args()
    torch.set_num_threads(4)
    initialize(args.source, args.output)
