"""Fixed nc80 ImageNet source -> original parent -> TRC; audited native nc1 rebuild."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import torch
from init_lif_down import ROOT, MODEL_DIR, SOURCE_SHA256, require, sha256, write_json, runtime as base_runtime
from init_c19_lif_v1 import controlled_models as original_models, native_rebuild, source_contract
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import AIFI_TRC, LIFDown, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML

BASE_COMMIT = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
DEFAULT_VARIANT = 'cbr_lif_trc_v1'
VARIANTS = {
    'cbr_lif_trc_v1': ('rtdetr-resnet18-lite-cbr-lif-trc-v1.yaml', 'rtdetr-resnet18-lite-cbr-lif-down.yaml',
                       'cbr_lif_trc_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
    'trc_v1': ('rtdetr-resnet18-lite-trc-v1.yaml', 'rtdetr-resnet18-lite.yaml',
               'trc_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
}
NEW_KEYS = {'model.9.trc.descriptor.weight', 'model.9.trc.coefficient.weight', 'model.9.trc.coefficient.bias'}
CLASS_KEYS = {'model.26.denoising_class_embed.weight', 'model.26.enc_score_head.weight', 'model.26.enc_score_head.bias'} | {
    f'model.26.dec_score_head.{i}.{p}' for i in range(3) for p in ('weight', 'bias')}


def is_added(key):
    return key in NEW_KEYS


def runtime():
    info = base_runtime()
    from ultralytics.nn.modules import trc_aifi
    require(Path(trc_aifi.__file__).resolve() == ROOT/'ultralytics-main/ultralytics/nn/modules/trc_aifi.py', 'Wrong TRC import')
    info.update(trc_module=trc_aifi.__file__, base_commit=BASE_COMMIT)
    return info


def build(variant=DEFAULT_VARIANT, nc=80, parent=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR/VARIANTS[variant][int(parent)]), nc=nc, verbose=False)


def verify_model(model, variant=DEFAULT_VARIANT, zero=False):
    source_contract()
    require(variant in VARIANTS, 'Unknown TRC variant')
    reference = YAML.load(MODEL_DIR/VARIANTS[variant][1])
    reference['head'][1][2] = 'AIFI_TRC'
    require(type(model) is RTDETRDetectionModel, 'Native RTDETRDetectionModel required')
    require(all(model.yaml[k] == reference[k] for k in ('backbone', 'head', 'scales')), 'Only parent layer9 may change')
    require(len(model.model) == 27 and type(model.model[9]) is AIFI_TRC, 'Layer9 AIFI_TRC required')
    require(model.model[-1].f == [19,22,25], 'Decoder input topology changed')
    require(model.model[9].ma.embed_dim == 256 and model.model[9].ma.num_heads == 8 and model.model[9].fc1.out_features == 1024,
            'AIFI dimensions changed')
    require(sum(p.numel() for p in model.model[9].trc.parameters()) == 10248, 'TRC parameter count changed')
    require({k for k in model.state_dict() if '.trc.' in k} == NEW_KEYS, 'Unexpected TRC state')
    combo = variant == DEFAULT_VARIANT
    require(sum(type(m) is LIFDown for m in model.modules()) == int(combo), 'Incorrect LIF inventory')
    require(sum(type(m) is RTDETRDecoderCBR for m in model.modules()) == int(combo), 'Incorrect CBR inventory')
    if combo:
        require(type(model.model[20]) is LIFDown and hasattr(model.model[20], 'bn'), 'Original LIF BN must survive fusion')
    if zero:
        require(all(torch.count_nonzero(p) == 0 for p in model.model[9].trc.coefficient.parameters()), 'Initial coefficient must be zero')
        require(torch.count_nonzero(model.model[9].trc.descriptor.weight) > 0, 'Descriptor must be nonzero')
    return dict(layer=9, channels=256, heads=8, head_dim=32, ffn=1024, tokens_at_640=400,
                cbr=combo, lif=combo, new_parameters=10248)


def controlled_models(source, variant=DEFAULT_VARIANT):
    # Reuse the original audited C2->CBR+LIF procedure, never trained best/last.
    baseline, combo, parent_report = original_models(source)
    parent = combo if variant == DEFAULT_VARIANT else baseline
    candidate = build(variant)
    constructor_parent = build(variant, parent=True).state_dict()
    fresh = candidate.state_dict()
    require(set(fresh)-set(constructor_parent) == NEW_KEYS and not set(constructor_parent)-set(fresh), 'Unexpected state delta')
    require(all(torch.equal(v, fresh[k]) for k,v in constructor_parent.items()), 'New construction consumed public RNG')
    state = parent.state_dict()
    require(set(state) == set(constructor_parent), 'Parent state inventory changed')
    candidate.load_state_dict({**fresh, **state}, strict=True)
    require(all(torch.equal(v,candidate.state_dict()[k]) for k,v in state.items()), 'Parent parameters or BN buffers changed')
    require(sum(p.numel() for p in candidate.parameters())-sum(p.numel() for p in parent.parameters()) == 10248, 'Wrong delta')
    verify_model(candidate,variant,zero=True)
    report = dict(variant=variant, base_commit=BASE_COMMIT, source=str(Path(source).resolve()), source_sha256=SOURCE_SHA256,
                  source_nc=80, target_nc=80, seed=42, new_parameters=10248, COMMON=sorted(state),
                  NEW_TRAINABLE=sorted(NEW_KEYS), NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
                  ALLOWED_CLASS_ADAPTATION=[], public_constructor_exact=True, parent_state_and_bn_exact=True,
                  trc_initial_sha256={k:hashlib.sha256(fresh[k].numpy().tobytes()).hexdigest() for k in sorted(NEW_KEYS)},
                  parent_initialization=dict(source_sha256=parent_report['source_sha256'], new_parameters=parent_report['new_parameters']),
                  formal_optimizer_steps=0)
    return parent, candidate, report


def build_training_model(cfg, weights, data, variant=DEFAULT_VARIANT):
    # Both calls enter with the same RNG; only the native candidate call advances it.
    # No zeroing here: the same audit is safe for learned checkpoints / resume.
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(MODEL_DIR/VARIANTS[variant][1]), weights, data['nc'], data['channels'])
    model = native_rebuild(cfg, weights, data['nc'], data['channels'])
    verify_model(model, variant)
    before, after = weights.state_dict(), model.state_dict()
    require(set(before) == set(after), 'Native rebuild missing/unexpected tensors')
    changed = {k for k,v in before.items() if v.shape != after[k].shape}
    require(changed == (CLASS_KEYS if weights.model[-1].nc != data['nc'] else set()), 'Unexpected nc shape adaptation')
    require(all(torch.equal(v,after[k]) for k,v in before.items() if k not in changed), 'Native load lost common/TRC state')
    require(all(torch.equal(v,after[k]) for k,v in parent.state_dict().items()), 'Parent native class adaptation differs')
    model.nc = data['nc']
    return model, dict(status='PASSED',native_get_model=True,loaded_exact=len(before)-len(changed),
        parent_common_and_bn_exact=True, new_state_preserved=True, MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
        ALLOWED_CLASS_ADAPTATION=[dict(name=k,source_shape=list(before[k].shape),target_shape=list(after[k].shape)) for k in sorted(changed)])


def initialize(source, output, variant=DEFAULT_VARIANT):
    output = Path(output)
    require(not output.exists(), f'Existing initialization protected: {output}')
    _,model,report = controlled_models(source,variant)
    model.eval(); model.args = {**DEFAULT_CFG_DICT,'model':str(MODEL_DIR/VARIANTS[variant][0]),'task':'detect'}
    model.task,model.pt_path = 'detect',str(output.resolve())
    ckpt = dict(epoch=-1,best_fitness=None,model=deepcopy(model).float(),ema=None,updates=None,optimizer=None,scaler=None,
                train_args=model.args,train_metrics=None,train_results=None,date=datetime.now(timezone.utc).isoformat(),
                trc_v1_provenance=report)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f: torch.save(ckpt,f)
    restored = RTDETR(str(output)).model
    require(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()), 'Initialization reload differs')
    report.update(status='PASSED',output=str(output.resolve()),output_sha256=sha256(output),reload_exact=True,runtime=runtime())
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','report'): p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--variant',choices=VARIANTS,default=DEFAULT_VARIANT)
    args=p.parse_args(); torch.set_num_threads(4)
    require(not args.report.exists(), 'Existing initialization report protected')
    write_json(args.report,initialize(args.source,args.output,args.variant))
