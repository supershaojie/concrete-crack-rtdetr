"""Strict unified C2 initialization and native nc=1 Trainer audit for CSR-P3."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import torch
from init_lif_down import ROOT, MODEL_DIR, SOURCE_SHA256, C2_COMMIT, require, sha256, write_json
import init_c19_lif_v1 as parent_pair
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import CSRConv, CurvedSamplingResidual, Conv, LIFDown, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE_COMMIT = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
VARIANTS = {
    'cbr_lif_csr_p3_v1': ('rtdetr-resnet18-lite-cbr-lif-csr-p3-v1.yaml',
                          'rtdetr-resnet18-lite-cbr-lif-down.yaml',
                          'cbr_lif_csr_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
    'csr_p3_v1': ('rtdetr-resnet18-lite-csr-p3-v1.yaml', 'rtdetr-resnet18-lite.yaml',
                 'csr_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
}
COUNTS = {'cbr_lif_csr_p3_v1': (20167313, 19962513), 'csr_p3_v1': (20100320, 19895264)}
PREFIX = 'model.17.csr.'


def is_added(key):
    return key.startswith(PREFIX)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def runtime():
    info = parent_pair.runtime()
    import ultralytics.nn.modules.csr_p3 as module
    require(Path(module.__file__).resolve() == ROOT / 'ultralytics-main/ultralytics/nn/modules/csr_p3.py',
            'Wrong CSR import')
    info.update(csr_module=module.__file__, cudnn=torch.backends.cudnn.version())
    return info


def build(variant='cbr_lif_csr_p3_v1', nc=80, baseline=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant='cbr_lif_csr_p3_v1', zero=False):
    parent_pair.source_contract()
    require(variant in VARIANTS and type(model) is RTDETRDetectionModel, 'Unknown variant/model type')
    expected = deepcopy(YAML.load(MODEL_DIR / VARIANTS[variant][1]))
    row = expected['head'][17-len(expected['backbone'])]
    require(row == [5, 1, 'Conv', [256, 1, 1, 'None', 1, 1, False]], 'Parent lateral projection changed')
    row[2], row[3] = 'CSRConv', [256, 1, 1, 'None', 1, 1, False, 32, 7]
    require(all(model.yaml[k] == expected[k] for k in ('backbone', 'head', 'scales')), 'Unexpected topology change')
    lateral = model.model[17]
    require(type(lateral) is CSRConv and lateral.f == 5, 'CSR location/type changed')
    require(type(lateral.act) is torch.nn.Identity, 'Original act=False lost')
    require(lateral.conv.in_channels == 128 and lateral.conv.out_channels == 256 and
            lateral.conv.kernel_size == lateral.conv.stride == (1, 1), 'Lateral geometry changed')
    require(sum(isinstance(m, CurvedSamplingResidual) for m in model.modules()) == 1, 'Exactly one CSR required')
    csr = lateral.csr
    require(sum(p.numel() for p in csr.parameters()) == 17548, 'CSR parameter delta changed')
    require(tuple(csr.theta.shape) == (2, 32, 7), 'Theta contract changed')
    require(all(is_added(k) for k in model.state_dict() if '.csr.' in k), 'Unexpected CSR state location')
    require(model.model[18].f == [-2, -1] and model.model[26].f == [19, 22, 25], 'Concat/decoder wiring changed')
    head = model.model[-1]
    require(head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3 and
            head.decoder.eval_idx == 2, 'Decoder dimensions changed')
    if variant == 'cbr_lif_csr_p3_v1':
        require(type(model.model[20]) is LIFDown and type(head) is RTDETRDecoderCBR, 'Original pair missing')
        require(head.cbr.rho == head.cbr.normal_fraction == .10, 'Original CBR changed')
    else:
        require(type(model.model[20]) is Conv and not hasattr(head, 'cbr'), 'Single-module ablation contains pair')
    fused = not hasattr(lateral, 'bn')
    if head.nc == 1:
        require(sum(p.numel() for p in model.parameters()) == COUNTS[variant][int(fused)], 'nc=1 parameter count changed')
    if fused:
        require(lateral.forward.__func__ is CSRConv.forward_fuse, 'Fused forward lost CSR branch')
    if zero:
        require(all(torch.count_nonzero(p) == 0 for p in csr.offset_pw.parameters()), 'Offsets must start zero')
        require(torch.count_nonzero(csr.theta) == 0, 'Theta must start uniform')
        require(all(torch.count_nonzero(p.weight) > 0 for p in (csr.in_proj, csr.offset_dw, csr.out_proj)),
                'Input/depthwise/output projections must start nonzero')
        require(torch.count_nonzero(csr.offset_dw.bias) == 0, 'Depthwise bias must start zero')
    return dict(lateral=17, concat=18, repc3=19, downsample=20, decoder=26, fused=fused)


def controlled_models(source, variant='cbr_lif_csr_p3_v1'):
    # Reuse the audited successful pair builder; it verifies the exact source hash,
    # nc=80, epoch=-1, empty optimizer/EMA/scaler and all 533 C2 states.
    c2, pair, inherited = parent_pair.controlled_models(source)
    parent = pair if variant == 'cbr_lif_csr_p3_v1' else c2
    target = build(variant)
    fresh, common = target.state_dict(), parent.state_dict()
    parent_constructor = build(variant, baseline=True).state_dict()
    require(all(k in fresh and torch.equal(v, fresh[k]) for k, v in parent_constructor.items()),
            'CSR construction consumed public RNG')
    extra = set(fresh) - set(common)
    require(extra == {k for k in fresh if is_added(k)} and len(extra) == 7, 'Unexpected added state')
    require(extra <= dict(target.named_parameters()).keys(), 'Unexpected CSR buffer')
    other = build('csr_p3_v1' if variant == 'cbr_lif_csr_p3_v1' else 'cbr_lif_csr_p3_v1')
    require(all(torch.equal(fresh[k], other.state_dict()[k]) for k in extra), 'Variant CSR initializations differ')
    require(all(common[k].shape == fresh[k].shape for k in common), 'Public state shape mismatch')
    target.load_state_dict({**fresh, **common}, strict=True)
    require(all(torch.equal(v, target.state_dict()[k]) for k, v in common.items()), 'Common values changed')
    verify_model(target, variant, zero=True)
    report = dict(variant=variant, base_commit=BASE_COMMIT, c2_commit=C2_COMMIT,
                  source=str(Path(source).resolve()), source_sha256=SOURCE_SHA256, source_nc=80, target_nc=80,
                  COMMON=[dict(name=k, shape=list(v.shape), equal=True, sha256=tensor_hash(v)) for k,v in common.items()],
                  NEW_TRAINABLE=sorted(extra), NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
                  ALLOWED_CLASS_ADAPTATION=[], new_parameters=17548, public_constructor_equal=True,
                  variants_new_state_exact=True, inherited_pair_audit=inherited,
                  new_initial_hashes={k: tensor_hash(fresh[k]) for k in sorted(extra)})
    return parent, target, report


def native_rebuild(cfg, weights, data):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = dict(nc=data['nc'], channels=data.get('channels', 3))
    return RTDETRTrainer.get_model(trainer, cfg=deepcopy(cfg), weights=weights, verbose=False)


def build_training_model(cfg, weights, data, variant='cbr_lif_csr_p3_v1', zero=False):
    require(weights is not None, 'Controlled weights required')
    # Parent uses exactly the RNG about to be used by the real target, without
    # consuming that RNG. Nine adapted nc=1 classifier tensors are compared too.
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), weights, data)
    target = native_rebuild(cfg, weights, data)
    verify_model(target, variant, zero=zero)
    before, after = weights.state_dict(), target.state_dict()
    allowed = {'model.26.denoising_class_embed.weight', 'model.26.enc_score_head.weight', 'model.26.enc_score_head.bias'}
    allowed |= {f'model.26.dec_score_head.{i}.{part}' for i in range(3) for part in ('weight', 'bias')}
    require(set(before) == set(after), 'Native Trainer state inventory changed')
    adapted = {k for k in before if before[k].shape != after[k].shape}
    require(adapted == (allowed if weights.model[-1].nc != data['nc'] else set()), 'Unexpected class adaptation')
    require(all(torch.equal(v, after[k]) for k, v in before.items() if k not in adapted), 'Trainer lost initialized state')
    require(all(torch.equal(v, after[k]) for k, v in parent.state_dict().items()), 'nc=1 common initialization differs')
    report = dict(COMMON=sorted(parent.state_dict()), NEW_TRAINABLE=sorted(k for k in after if is_added(k)),
                  NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
                  ALLOWED_CLASS_ADAPTATION=[dict(name=k, source_shape=list(before[k].shape), target_shape=list(after[k].shape),
                      parent_equal=True, sha256=tensor_hash(after[k])) for k in sorted(adapted)],
                  native_get_model=True, common_nc1_exact=True, loaded_exact=len(before)-len(adapted),
                  parent_rng_isolated=True, csr_preserved=True)
    return target, report


def initialize(source, output, variant='cbr_lif_csr_p3_v1'):
    output = Path(output)
    require(not output.exists(), f'Existing initialization preserved: {output}')
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, 'model': str(MODEL_DIR / VARIANTS[variant][0]), 'task': 'detect'}
    target.task, target.pt_path = 'detect', str(output.resolve())
    checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
                      optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
                      date=datetime.now(timezone.utc).isoformat(), csr_p3_provenance=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save(checkpoint, stream)
    restored = RTDETR(str(output)).model
    require(set(target.state_dict()) == set(restored.state_dict()) and
            all(torch.equal(v, restored.state_dict()[k]) for k,v in target.state_dict().items()), 'Init reload differs')
    # Actually execute native adaptation immediately; the formal API repeats it
    # at its own seeded construction point and must retain these invariants.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        _, adaptation = build_training_model(target.yaml, restored, dict(nc=1, channels=3), variant)
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True,
                  status='PASSED', runtime=runtime(), trainer_nc1=adaptation)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=VARIANTS)
    for name in ('source', 'output', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.report, initialize(args.source, args.output, args.variant))
