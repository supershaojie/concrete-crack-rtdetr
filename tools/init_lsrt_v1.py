"""Strict, reproducible LSRT initialization; no optimizer updates or trained source."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ultralytics-main'))
from init_c19_lif_v1 import (SOURCE_SHA256, MODEL_DIR, require, sha256, write_json,
                            controlled_models as controlled_pair, source_contract, native_rebuild)
from ultralytics import RTDETR
import ultralytics
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules import LIFDown, RTDETRDecoder, RTDETRDecoderCBR, Conv, LSRTConcat
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE_COMMIT = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
VARIANTS = {
    'cbr_lif_lsrt_v1': ('rtdetr-resnet18-lite-cbr-lif-lsrt-v1.yaml',
                        'rtdetr-resnet18-lite-cbr-lif-down.yaml',
                        'cbr_lif_lsrt_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
    'lsrt_v1': ('rtdetr-resnet18-lite-lsrt-v1.yaml', 'rtdetr-resnet18-lite.yaml',
                'lsrt_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
}
LSRT_KEYS = {'model.18.lsrt.' + name for name in
             ('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'out_proj.weight', 'rel_bias')}


def git(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + ROOT.as_posix(), *args],
                                   cwd=ROOT, text=True).strip()


def runtime():
    require(Path(ultralytics.__file__).resolve() == ROOT/'ultralytics-main/ultralytics/__init__.py', 'Wrong ultralytics import')
    return dict(python=platform.python_version(), executable=sys.executable, torch=str(torch.__version__),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, commit=git('rev-parse', 'HEAD'))


def fingerprint(variant, source, initialized):
    """Actual source bytes normalized only for checkout line endings, including untracked task code."""
    files = set()
    for folder in ('tools', 'ultralytics-main/ultralytics'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.suffix in {'.py', '.yaml', '.sh'} and '__pycache__' not in p.parts)
    files.add(ROOT/'docs/c19_lif_v1/c2_args.yaml')
    rows = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(files)}
    return dict(commit=git('rev-parse', 'HEAD'),
                source_tree_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                config_sha256=hashlib.sha256((MODEL_DIR/VARIANTS[variant][0]).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                source_sha256=sha256(source), initialization_sha256=sha256(initialized))


def is_added(key):
    return key in LSRT_KEYS


def build(variant='cbr_lif_lsrt_v1', nc=80, baseline=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR/VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant='cbr_lif_lsrt_v1', zero=False):
    require(variant in VARIANTS, 'Unknown LSRT variant')
    source_contract()
    parent = YAML.load(MODEL_DIR/VARIANTS[variant][1])
    expected = deepcopy(parent)
    expected['head'][10] = [[16, 17, 15], 1, 'LSRTConcat', [32, 8.0]]
    require(type(model) is RTDETRDetectionModel, 'Native RTDETR model required')
    require(all(model.yaml[k] == expected[k] for k in ('backbone', 'head', 'scales')), 'Unexpected graph change')
    require(type(model.model[18]) is LSRTConcat and model.model[18].f == [16, 17, 15], 'Wrong LSRT routing')
    require({15, 16, 17} <= set(model.save), 'Missing saved LSRT inputs')
    require(type(model.model[16]) is torch.nn.Upsample and model.model[16].mode == 'nearest' and model.model[16].scale_factor == 2, 'Original nearest lost')
    require(type(model.model[20]) is (LIFDown if variant == 'cbr_lif_lsrt_v1' else Conv), 'Wrong P3 downsample')
    head = model.model[26]
    require(type(head) is (RTDETRDecoderCBR if variant == 'cbr_lif_lsrt_v1' else RTDETRDecoder), 'Wrong decoder')
    require(head.f == [19, 22, 25] and head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3, 'Decoder changed')
    reference = YAML.load(MODEL_DIR/'rtdetr-resnet18-lite.yaml')
    require(model.yaml['backbone'] == reference['backbone'], 'R18 backbone changed')
    require(sum(p.numel() for n,p in model.named_parameters() if is_added(n)) == 32777, 'LSRT parameter count mismatch')
    require({n for n in model.state_dict() if '.lsrt.' in n} == LSRT_KEYS, 'Unexpected LSRT state')
    m = model.model[18].lsrt
    if zero:
        require(all(p.requires_grad for p in model.parameters()), 'Frozen training parameters are not allowed')
        require(torch.count_nonzero(m.out_proj.weight) == 0 and torch.count_nonzero(m.rel_bias) == 0, 'LSRT zero initialization lost')
        require(all(torch.count_nonzero(p.weight) > 0 for p in (m.q_proj, m.k_proj, m.v_proj)), 'Internal projections must be nonzero')
        if variant == 'cbr_lif_lsrt_v1':
            require(torch.count_nonzero(model.model[20].O_proj.weight) == 0, 'LIF initial output changed')
            require(all(torch.count_nonzero(v) == 0 for v in head.cbr.offset_out.parameters()), 'CBR initial output changed')
    return dict(layer=18, inputs=[16,17,15], output_channels=512, decoder=head.__class__.__name__,
                queries=head.num_queries, decoder_layers=len(head.decoder.layers),
                DN=head.num_denoising, label_noise_ratio=head.label_noise_ratio, box_noise_scale=head.box_noise_scale,
                parameters=sum(p.numel() for p in model.parameters()), added_parameters=32777)


def controlled_models(source, variant='cbr_lif_lsrt_v1'):
    # Parent helper audits the fixed source, original C2/LIF/CBR values and all 533 C2 states.
    c2, pair, parent_audit = controlled_pair(source)
    parent = pair if variant == 'cbr_lif_lsrt_v1' else c2
    target = build(variant)
    fresh_parent = build(variant, baseline=True)
    fresh, public = target.state_dict(), fresh_parent.state_dict()
    require(set(fresh)-set(public) == LSRT_KEYS and not set(public)-set(fresh), 'Only five LSRT tensors may be added')
    require(all(torch.equal(v, fresh[k]) for k,v in public.items()), 'LSRT construction consumed public RNG')
    target.load_state_dict({**fresh, **parent.state_dict()}, strict=True)
    require(all(torch.equal(v, target.state_dict()[k]) for k,v in parent.state_dict().items()), 'Parent parameter/buffer migration differs')
    other = build('lsrt_v1' if variant == 'cbr_lif_lsrt_v1' else 'cbr_lif_lsrt_v1')
    require(all(torch.equal(target.state_dict()[k], other.state_dict()[k]) for k in LSRT_KEYS), 'Two variants have different LSRT init')
    graph = verify_model(target, variant, zero=True)
    report = dict(status='PASSED', variant=variant, source=str(Path(source).resolve()), source_sha256=SOURCE_SHA256,
                  base_commit=BASE_COMMIT, source_nc=80, target_nc=80, formal_optimizer_steps=0,
                  COMMON=[dict(name=k, shape=list(v.shape), equal=True) for k,v in parent.state_dict().items()],
                  NEW_TRAINABLE=sorted(LSRT_KEYS), NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
                  ALLOWED_CLASS_ADAPTATION=[], public_constructor_exact=True, cross_variant_lsrt_exact=True,
                  graph=graph, original_parent_audit=parent_audit)
    return parent, target, report


def build_training_model(cfg, weights, data, variant='cbr_lif_lsrt_v1'):
    # Both true native get_model calls see identical RNG. Only target call advances caller RNG.
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(MODEL_DIR/VARIANTS[variant][1]), weights, data['nc'], data.get('channels',3))
    target = native_rebuild(cfg, weights, data['nc'], data.get('channels',3))
    verify_model(target, variant, zero=False)
    require(all(p.requires_grad for p in target.parameters()), 'Native training reconstruction froze parameters')
    before, after = weights.state_dict(), target.state_dict()
    allowed = {'model.26.'+n for n in ('denoising_class_embed.weight','enc_score_head.weight','enc_score_head.bias')}
    allowed |= {f'model.26.dec_score_head.{i}.{suffix}' for i in range(3) for suffix in ('weight','bias')}
    changed = {k for k in before if before[k].shape != after[k].shape}
    require(set(before) == set(after), 'Trainer missing/unexpected states')
    require(changed == (allowed if weights.model[-1].nc != data['nc'] else set()), 'Unexpected nc shape adaptation')
    require(all(torch.equal(v,after[k]) for k,v in before.items() if k not in changed), 'Trainer lost initialized/learned values')
    require(all(torch.equal(v,after[k]) for k,v in parent.state_dict().items()), 'Real nc1 parent/common tensors differ')
    report = dict(status='PASSED', native_get_model=True, public_states_exact=len(parent.state_dict()),
                  loaded_exact=len(before)-len(changed), MISSING=[],UNEXPECTED=[],SHAPE_MISMATCH=[],
                  ALLOWED_CLASS_ADAPTATION=[dict(name=k, source_shape=list(before[k].shape), target_shape=list(after[k].shape),
                                               rule='unchanged native constructor; equal to parent at same RNG') for k in sorted(changed)],
                  LSRT_preserved=True, formal_optimizer_steps=0)
    return target, report


def initialize(source, output, variant='cbr_lif_lsrt_v1'):
    output = Path(output)
    require(not output.exists(), 'Existing initialization preserved: '+str(output))
    _, model, report = controlled_models(source, variant)
    model.eval()
    model.args = {**DEFAULT_CFG_DICT, 'model': str(MODEL_DIR/VARIANTS[variant][0]), 'task':'detect'}
    model.task, model.pt_path = 'detect', str(output.resolve())
    ckpt = dict(epoch=-1, best_fitness=None, model=deepcopy(model).float(), ema=None, updates=None,
                optimizer=None, scaler=None, train_args=model.args, train_metrics=None, train_results=None,
                date=datetime.now(timezone.utc).isoformat(), lsrt_v1_provenance=report)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f: torch.save(ckpt,f)
    restored = RTDETR(str(output)).model
    require(all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items()), 'Checkpoint reload differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        _, report['trainer_nc1'] = build_training_model(restored.yaml,restored,dict(nc=1,channels=3),variant)
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True, runtime=runtime())
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, default='cbr_lif_lsrt_v1')
    for arg in ('source','output','report'): parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(4)
    write_json(args.report,initialize(args.source,args.output,args.variant))
