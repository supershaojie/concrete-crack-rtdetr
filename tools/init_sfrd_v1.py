"""Strict CBR/LIF reference -> SFR/SFR-D initialization and native nc adaptation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch
import init_c19_lif_v1 as parent
from init_c19_lif_v1 import ROOT, MODEL_DIR, SOURCE_SHA256, require, sha256, write_json, runtime
from sfrd_v1_svd import SITES, factors, tensor_hash
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules.sfr_d import SFRStage, SpatialFactorConv, DirectionalCalibration, validate_source
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE_SHA = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
VARIANTS = {
    'cbr_lif_sfrd_v1': ('rtdetr-resnet18-lite-cbr-lif-sfrd-v1.yaml', 'pair', True, 18780549),
    'cbr_lif_sfr_control_v1': ('rtdetr-resnet18-lite-cbr-lif-sfr-control-v1.yaml', 'pair', False, 18773509),
    'sfrd_v1': ('rtdetr-resnet18-lite-sfrd-v1.yaml', 'C2', True, 18713556),
}
REMOVED = {f'model.{i}.blocks.1.branch2b.conv.weight' for i in SITES}
CLASS_KEYS = {'model.26.'+s for s in ('denoising_class_embed.weight', 'enc_score_head.weight', 'enc_score_head.bias')}
CLASS_KEYS |= {f'model.26.dec_score_head.{i}.{s}' for i in range(3) for s in ('weight', 'bias')}


def new_keys(variant):
    names = ['A_1x3.weight', 'B_3x1.weight']
    if VARIANTS[variant][2]:
        names += ['gate.dw_h.weight', 'gate.dw_v.weight', 'gate.dw_v.bias']
    return {f'model.{i}.blocks.1.branch2b.{n}' for i in SITES for n in names}


def build(variant, nc=80):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][0]), nc=nc, verbose=False)


def copy_bn_metadata(source, target):
    """State dicts omit these attributes; preserve them explicitly across reconstruction."""
    source_modules = dict(source.named_modules())
    report = {}
    for name, module in target.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            other = source_modules[name]
            row = {k: getattr(other, k) for k in ('eps', 'momentum', 'affine', 'track_running_stats')}
            require(module.affine == other.affine and module.track_running_stats == other.track_running_stats,
                    'BN state-layout mismatch: '+name)
            module.eps, module.momentum = other.eps, other.momentum
            report[name] = row
    return report


def verify_model(model, variant, zero=False):
    parent.source_contract()
    expected = YAML.load(MODEL_DIR / VARIANTS[variant][0])
    reference = YAML.load(MODEL_DIR / parent.CONFIGS[VARIANTS[variant][1]])
    require(type(model) is RTDETRDetectionModel and len(model.model) == 27, 'Native 27-layer topology required')
    require(all(model.yaml[k] == expected[k] for k in ('backbone', 'head', 'scales')), 'Unexpected model YAML')
    require(model.yaml['head'] == reference['head'], 'Neck/CBR/LIF moved')
    for idx, node in enumerate(model.yaml['backbone']):
        if idx not in SITES:
            require(node == reference['backbone'][idx], 'Non-SFR backbone node changed')
    rows = []
    for i, (c, r) in SITES.items():
        stage = model.model[i]; block = stage.blocks[1]; path = block.branch2b
        require(type(stage) is SFRStage and len(stage.blocks) == 2 and block.shortcut, 'Wrong SFR block')
        first = stage.blocks[0]
        require(first.branch2a.conv.stride == (2, 2) and not first.shortcut and
                isinstance(first.short.pool, torch.nn.AvgPool2d), 'Stage downsampling changed')
        require(block.branch2a.conv.in_channels == block.branch2a.conv.out_channels == c and
                block.branch2a.conv.stride == (1, 1), 'Complete first convolution changed')
        require(type(path) is SpatialFactorConv and tuple(path.A_1x3.weight.shape) == (r,c,1,3) and
                tuple(path.B_3x1.weight.shape) == (c,r,3,1), 'Wrong factors')
        require(hasattr(path, 'gate') == VARIANTS[variant][2], 'Control contains gate or gate missing')
        if zero and hasattr(path, 'gate'):
            require(all(torch.count_nonzero(p) == 0 for p in path.gate.parameters()), 'Initial gate is nonzero')
        rows.append(dict(site=f'model.{i}.blocks.1.branch2b', C=c, rank=r, gate_parameters=11*r if hasattr(path,'gate') else 0))
    require(sum(isinstance(m, SpatialFactorConv) for m in model.modules()) == 2, 'Extra SFR path')
    require(sum(p.numel() for m in model.modules() if isinstance(m, DirectionalCalibration) for p in m.parameters()) ==
            (7040 if VARIANTS[variant][2] else 0), 'Gate parameter count')
    require(sum(p.numel() for p in model.parameters()) == VARIANTS[variant][3] if model.model[-1].nc == 1 else True,
            'Unfused nc1 parameter count mismatch')
    return rows


def controlled_models(source, cache):
    baseline, pair, parent_report = parent.controlled_models(source)
    original = torch_load(source, map_location='cpu')['model']
    source_bn_metadata = copy_bn_metadata(original, baseline)
    copy_bn_metadata(original, pair)
    for i, (c, _) in SITES.items():
        validate_source(pair.model[i].blocks[1], c)
    factor_state, svd_report = factors(pair, cache)
    models, reports = {}, {}
    for variant, (_, kind, _, _) in VARIANTS.items():
        reference = pair if kind == 'pair' else baseline
        target = build(variant)
        source_state, fresh = reference.state_dict(), target.state_dict()
        common = set(source_state) - REMOVED
        require(set(source_state)-set(fresh) == REMOVED and set(fresh)-set(source_state) == new_keys(variant),
                'Unknown missing/unexpected initialization key')
        # Constructor RNG comparison uses uninitialized committed reference, not the pretrained source.
        public_fresh = parent.build(kind).state_dict()
        require(all(torch.equal(fresh[k], public_fresh[k]) for k in common), 'SFR constructor disturbed public RNG')
        state = {k: source_state[k] for k in common}
        state.update({k: factor_state[k] if k in factor_state else fresh[k] for k in new_keys(variant)})
        require(set(state) == set(fresh) and all(v.shape == fresh[k].shape for k,v in state.items()), 'Invalid strict target state')
        target.load_state_dict(state, strict=True)
        metadata = copy_bn_metadata(reference, target)
        actual = target.state_dict()
        require(all(torch.equal(v, actual[k]) for k,v in state.items()), 'Strict state load changed a value')
        sites = verify_model(target, variant, zero=True)
        target.sfrd_variant = variant
        reports[variant] = dict(status='PASSED', source_sha256=SOURCE_SHA256, source_nc=80, target_nc=80,
            reference_kind=kind, public_constructor_exact=True, strict_load=True,
            COMMON=[dict(key=k, shape=list(state[k].shape), sha256=tensor_hash(state[k]), equal=True) for k in sorted(common)],
            REMOVED=sorted(REMOVED), NEW_TRAINABLE=[dict(key=k, shape=list(state[k].shape), sha256=tensor_hash(state[k])) for k in sorted(new_keys(variant))],
            MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[], bn_metadata=metadata, sites=sites,
            factors_sha256=svd_report['file_sha256'], base_sha=BASE_SHA)
        models[variant] = target
    shared = set.intersection(*(set(m.state_dict()) for m in models.values()))
    require(all(torch.equal(models['sfrd_v1'].state_dict()[k], m.state_dict()[k]) for m in models.values() for k in shared),
            'Cross-variant common/factor states differ')
    return baseline, pair, models, dict(variants=reports, cross_variant_shared_states_exact=len(shared),
                                      parent_public_initialization=parent_report, source_bn_metadata=source_bn_metadata, svd=svd_report)


def build_training_model(cfg, weights, data, variant, initial=False):
    """Native constructor/class-head RNG plus explicit nine-key adaptation and strict load."""
    require(weights is not None and data['nc'] == 1 and data['channels'] == 3, 'Controlled nc1 RGB weights required')
    before = weights.state_dict()
    with torch.random.fork_rng(devices=[]):
        control = parent.native_rebuild(str(MODEL_DIR / parent.CONFIGS[VARIANTS[variant][1]]), None, 1, 3)
    trainer = RTDETRTrainer.__new__(RTDETRTrainer); trainer.data = data
    target = RTDETRTrainer.get_model(trainer, cfg=deepcopy(cfg), weights=None, verbose=False)
    fresh = target.state_dict()
    require(set(before) == set(fresh), 'Unknown Trainer missing/unexpected keys')
    changed = {k for k in before if before[k].shape != fresh[k].shape}
    require(changed == (CLASS_KEYS if weights.model[-1].nc == 80 else set()), 'Unexpected class adaptation')
    state = {k: fresh[k] if k in changed else before[k] for k in fresh}
    require(all(torch.equal(state[k], control.state_dict()[k]) for k in changed), 'Native class RNG changed')
    target.load_state_dict(state, strict=True)
    metadata = copy_bn_metadata(weights, target)
    require(all(torch.equal(v, target.state_dict()[k]) for k,v in state.items()), 'Trainer lost migrated state')
    target.sfrd_variant = variant
    verify_model(target, variant, zero=initial)
    report = dict(status='PASSED', native_constructor=True, strict_load=True, loaded_exact=len(state)-len(changed),
        MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[], ALLOWED_CLASS_ADAPTATION=sorted(changed),
        native_class_initial_values_exact=True, initial_gate_checked=initial, bn_metadata=metadata,
        state_sha256={k:tensor_hash(v) for k,v in state.items()})
    return target, report


def initialize(source, cache, output_dir, report_dir):
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    require(all(not (output_dir/(v+'_init.pt')).exists() for v in VARIANTS), 'Existing init preserved; choose a new output directory')
    _, _, models, report = controlled_models(source, cache)
    for variant, target in models.items():
        out = output_dir/(variant+'_init.pt')
        target.eval(); target.args = {**DEFAULT_CFG_DICT, 'model':str(MODEL_DIR/VARIANTS[variant][0]), 'task':'detect'}
        target.task, target.pt_path = 'detect', str(out.resolve())
        checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
            optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
            date=datetime.now(timezone.utc).isoformat(), sfrd_v1_provenance=report['variants'][variant])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('xb') as stream: torch.save(checkpoint, stream)
        restored = RTDETR(str(out)).model
        require(all(torch.equal(v, restored.state_dict()[k]) for k,v in target.state_dict().items()), 'Checkpoint reload mismatch')
        report['variants'][variant].update(output=str(out.resolve()), output_sha256=sha256(out), reload_exact=True)
    report.update(status='PASSED', source=str(Path(source).resolve()), source_sha256=SOURCE_SHA256, runtime=runtime())
    write_json(report_dir/'svd_audit.json', report.pop('svd'))
    write_json(report_dir/'initialization_audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'cache', 'output-dir', 'report-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    initialize(args.source, args.cache, args.output_dir, args.report_dir)


if __name__ == '__main__':
    main()
