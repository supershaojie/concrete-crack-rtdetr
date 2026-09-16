"""Controlled RCS-Q initialization; native class adaptation and exact common-state audits."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ultralytics-main'))
import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

MODEL_DIR = ROOT / 'ultralytics-main/ultralytics/cfg/models/rt-detr'
SOURCE_SHA256 = 'fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e'
BASE_COMMIT = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
VARIANTS = {
    'cbr_lif_rcsq_v1': ('rtdetr-resnet18-lite-cbr-lif-rcsq.yaml', 'rtdetr-resnet18-lite-cbr-lif-down.yaml',
                        'cbr_lif_rcsq_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
    'rcsq_v1': ('rtdetr-resnet18-lite-rcsq.yaml', 'rtdetr-resnet18-lite.yaml',
                'rcsq_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
}
CONFIGS = {'baseline': VARIANTS['rcsq_v1'][1], 'combo': VARIANTS['cbr_lif_rcsq_v1'][1],
           **{key: value[0] for key, value in VARIANTS.items()}}
ORIGINAL = {'rcsq_v1': 'baseline', 'cbr_lif_rcsq_v1': 'combo'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + ROOT.as_posix(), *args], cwd=ROOT, text=True).strip()


def code_fingerprint():
    """LF-normalized content identity, portable across checkouts; HEAD is recorded separately."""
    candidates = list((ROOT / 'ultralytics-main/ultralytics').rglob('*.py'))
    candidates += list(MODEL_DIR.glob('*.yaml'))
    candidates += list((ROOT / 'tools').glob('*rcsq*.py')) + list((ROOT / 'tools').glob('*rcsq*.sh'))
    candidates += [ROOT / 'tools/c19_lif_v1_data.py', ROOT / 'tools/train_c19_lif_v1.py',
                   ROOT / 'tools/c19_lif_v1_results.py', ROOT / 'docs/c19_lif_v1/c2_args.yaml',
                   ROOT / 'docs/c19_lif_v1/c2_data.yaml', ROOT / 'docs/c19_lif_v1/summary.json',
                   ROOT / 'tools/check_lif_down.py', ROOT / 'tools/c19_lif_v1_probe.py',
                   ROOT / 'tools/c19_lif_v1_diagnostic.py', ROOT / 'tools/c19_lif_v1_cutoff.py',
                   ROOT / 'docs/rcsq_v1/cbr_lif_authoritative_args.yaml', ROOT / 'docs/rcsq_v1/c2_data.yaml']
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
             for p in sorted(set(candidates)) if p.is_file()}
    return dict(algorithm='sha256(path + LF-normalized content sha256), sorted JSON', files=files,
                sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


def runtime():
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / 'ultralytics-main'), 'Wrong ultralytics import')
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, commit=git('rev-parse', 'HEAD'),
                status=git('status', '--short'), code_fingerprint=code_fingerprint()['sha256'])


def build(kind='cbr_lif_rcsq_v1', nc=80):
    require(kind in CONFIGS, 'Unknown configuration ' + str(kind))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / CONFIGS[kind]), nc=nc, verbose=False)


def is_added(key):
    return '.rcsq.' in key


def class_allowlist(model):
    prefix = f'model.{len(model.model) - 1}.'
    return {prefix + 'denoising_class_embed.weight', prefix + 'enc_score_head.weight', prefix + 'enc_score_head.bias'} | {
        prefix + f'dec_score_head.{i}.{suffix}' for i in range(3) for suffix in ('weight', 'bias')}


def verify_model(model, variant='cbr_lif_rcsq_v1', zero=False):
    from ultralytics.nn.modules import LIFDown, RTDETRDecoderCBR
    from ultralytics.nn.modules.rcs_q import RegionCenteredSupportQuery, RTDETRDecoderRCSQ, RTDETRDecoderCBRRCSQ
    require(variant in VARIANTS and type(model) is RTDETRDetectionModel, 'Native RCS-Q detection model required')
    expected = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected['head'][-1][2] = 'RTDETRDecoderCBRRCSQ' if variant == 'cbr_lif_rcsq_v1' else 'RTDETRDecoderRCSQ'
    require(all(model.yaml[k] == expected[k] for k in ('backbone', 'head', 'scales')), 'Only final Decoder class may change')
    head = model.model[-1]
    require(type(head) is (RTDETRDecoderCBRRCSQ if variant == 'cbr_lif_rcsq_v1' else RTDETRDecoderRCSQ), 'Wrong Decoder')
    require(head.f == [19, 22, 25] and head.hidden_dim == 256 and head.num_queries == 300 and
            len(head.decoder.layers) == 3 and head.decoder.eval_idx == 2, 'Original Decoder geometry changed')
    require(sum(isinstance(m, RegionCenteredSupportQuery) for m in model.modules()) == 1, 'Exactly one shared RCS-Q')
    require(sum(p.numel() for p in head.rcsq.parameters()) == 58370, 'RCS-Q parameter count must be 58,370')
    require(not any(any(tag in type(m).__name__.lower() for tag in ('nbr', 'sfr', 'lsrt')) for m in model.modules()), 'Unrelated module')
    if variant == 'cbr_lif_rcsq_v1':
        require(isinstance(head, RTDETRDecoderCBR) and type(model.model[20]) is LIFDown and
                sum(isinstance(m, LIFDown) for m in model.modules()) == 1, 'Original CBR/LIF location changed')
        require(sum(p.numel() for p in head.cbr.parameters()) == 45889 and head.cbr.rho == head.cbr.normal_fraction == .1,
                'Original CBR contract changed')
        if zero:
            require(torch.count_nonzero(model.model[20].O_proj.weight) == 0, 'Original LIF zero output changed')
            require(all(torch.count_nonzero(v) == 0 for v in head.cbr.offset_out.parameters()), 'Original CBR zero output changed')
    else:
        require(not any(isinstance(m, (LIFDown, RTDETRDecoderCBR)) for m in model.modules()), 'Single-module variant contains CBR/LIF')
    if zero:
        m = head.rcsq
        require(torch.count_nonzero(m.out_proj.weight) == 0, 'RCS-Q output must start exactly zero')
        require(all(torch.count_nonzero(getattr(m, name).weight) > 0 for name in ('p3_proj', 'q_proj', 'k_proj', 'v_proj')), 'RCS-Q upstream initialization is zero')
        require(torch.equal(m.query_norm.weight, torch.ones_like(m.query_norm.weight)) and torch.count_nonzero(m.query_norm.bias) == 0,
                'Unexpected LayerNorm initialization')
        require(torch.count_nonzero(m.null_logit.weight) == 0 and
                torch.allclose(m.null_logit.bias, torch.full_like(m.null_logit.bias, __import__('math').log(25)), atol=0, rtol=0),
                'Unexpected null initialization')
    if head.nc == 1 and not model.is_fused():
        require(sum(p.numel() for p in model.parameters()) == (20208135 if variant == 'cbr_lif_rcsq_v1' else 20141142),
                'Unexpected total nc=1 parameter count')
    return dict(decoder=len(model.model) - 1, decoder_inputs=head.f, rcsq_parameters=58370)


def tensor_digest(tensor):
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def controlled_models(source):
    source = Path(source)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, 'Unified initialization source SHA256 mismatch')
    checkpoint = torch_load(source, map_location='cpu')
    require(checkpoint.get('epoch') == -1 and all(checkpoint.get(k) is None for k in
            ('ema', 'optimizer', 'scaler', 'updates', 'train_metrics', 'train_results', 'best_fitness')), 'Source contains trained state')
    original = deepcopy(checkpoint['model']).float()
    require(original.model[-1].nc == 80, 'Expected fixed nc=80 source')
    models = {kind: build(kind) for kind in CONFIGS}
    source_state, base_state = original.state_dict(), models['baseline'].state_dict()
    require(set(source_state) == set(base_state) and len(source_state) == 533, 'Source state inventory differs from original baseline')
    require(all(original.yaml[k] == models['baseline'].yaml[k] for k in ('backbone', 'head', 'scales')), 'Source architecture mismatch')
    require(all(source_state[k].shape == base_state[k].shape for k in source_state), 'Source shape mismatch')
    report = dict(source=str(source.resolve()), source_sha256=SOURCE_SHA256, source_nc=80, seed=42, base_sha=BASE_COMMIT,
                  formal_training_updates=0, variants={}, missing=[], unexpected=[], shape_mismatch=[])
    # Constructor audit precedes source migration: RNG isolation protects all public tensors.
    for kind, model in models.items():
        state = model.state_dict()
        require(all(torch.equal(value, state[key]) for key, value in base_state.items()), f'{kind} changed public constructor state')
        require(all(key in state for key in source_state), 'Missing common state')
    for model in models.values():
        state = model.state_dict()
        model.load_state_dict({**state, **source_state}, strict=True)
    for variant in VARIANTS:
        target, parent = models[variant], models[ORIGINAL[variant]]
        state, common = target.state_dict(), parent.state_dict()
        extra = set(state) - set(common)
        require(extra == {k for k in state if is_added(k)} and extra <= dict(target.named_parameters()).keys(), 'Unexpected new tensors/buffers')
        require(all(torch.equal(v, state[k]) for k, v in common.items()), f'{variant} common state mismatch')
        verify_model(target, variant, zero=True)
        report['variants'][variant] = dict(common_tensors=len(common), common_exact=True,
            common=[dict(name=k, shape=list(v.shape), sha256=tensor_digest(v)) for k, v in common.items()],
            new_trainable=sorted(extra), new_buffer=[], new_parameters=58370, missing=[], unexpected=[], shape_mismatch=[],
            ALLOWED_CLASS_ADAPTATION=[], parent=ORIGINAL[variant])
    a, b = [models[v].model[-1].rcsq.state_dict() for v in VARIANTS]
    require(set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), 'Both variants need identical RCS-Q initialization')
    report['shared_rcsq_exact'] = True
    return models, report


def native_rebuild(cfg, weights, nc=1, channels=3):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = dict(nc=nc, channels=channels)
    return RTDETRTrainer.get_model(trainer, cfg=deepcopy(cfg), weights=weights, verbose=False)


def build_training_model(cfg, weights, data, variant='cbr_lif_rcsq_v1', zero=True):
    """Audit the real native Trainer result, including the nine nc80->nc1 tensors."""
    require(weights is not None and data['nc'] == 1, 'Controlled weights and nc=1 required')
    before = weights.state_dict()
    channels = data.get('channels', 3)
    with torch.random.fork_rng(devices=[]):
        reference = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), weights, 1, channels)
    target = native_rebuild(cfg, weights, 1, channels)
    verify_model(target, variant, zero=zero)
    after = target.state_dict()
    require(set(before) == set(after), 'Native Trainer added/dropped state keys')
    changed = {k for k in before if before[k].shape != after[k].shape}
    expected = class_allowlist(target) if weights.model[-1].nc != 1 else set()
    require(changed == expected, f'Class shape changes outside precise allowlist: {changed ^ expected}')
    require(all(torch.equal(v, after[k]) for k, v in before.items() if k not in changed), 'Native Trainer failed exact initialized/checkpoint reload')
    require(all(torch.equal(v, after[k]) for k, v in reference.state_dict().items()), 'Native nc=1 original-model common state differs')
    report = dict(native_get_model=True, target_nc=1, common_exact=True, parent=ORIGINAL[variant],
                  loaded_exact=len(before) - len(changed), new_trainable=[k for k in after if is_added(k)],
                  missing=[], unexpected=[], shape_mismatch=[], formal_training_updates=0,
                  ALLOWED_CLASS_ADAPTATION=[dict(name=k, source_shape=list(before[k].shape), target_shape=list(after[k].shape),
                    rule='Native RTDETRTrainer constructor, same RNG as original reference') for k in sorted(changed)])
    return target, report


def initialized_path(init_dir, variant):
    return Path(init_dir) / (variant + '_controlled_init.pt')


def initialize(source, init_dir, variant='both'):
    require(variant == 'both' or variant in VARIANTS, 'Unknown variant')
    chosen = list(VARIANTS) if variant == 'both' else [variant]
    init_dir = Path(init_dir)
    require(all(not initialized_path(init_dir, name).exists() for name in chosen), 'Existing initialization preserved')
    models, audit = controlled_models(source)
    audit.update(code_fingerprint=code_fingerprint()['sha256'], runtime=runtime(), outputs={}, native_nc1={})
    for name in chosen:
        target = models[name]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            adapted, row = build_training_model(deepcopy(target.yaml), target, dict(nc=1, channels=3), name)
        audit['native_nc1'][name] = row
        del adapted
        output = initialized_path(init_dir, name)
        target.eval()
        target.args = {**DEFAULT_CFG_DICT, 'model': str(MODEL_DIR / CONFIGS[name]), 'task': 'detect'}
        target.task, target.pt_path = 'detect', str(output.resolve())
        provenance = dict(source_sha256=SOURCE_SHA256, variant=name, code_fingerprint=audit['code_fingerprint'],
                          formal_training_updates=0, state_sha256={k: tensor_digest(v) for k, v in target.state_dict().items()})
        checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
                          optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
                          date=datetime.now(timezone.utc).isoformat(), rcsq_v1_provenance=provenance)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            torch.save(checkpoint, stream)
        restored = RTDETR(str(output)).model
        require(all(torch.equal(v, restored.state_dict()[k]) for k, v in target.state_dict().items()), 'Checkpoint serialization differs')
        audit['outputs'][name] = dict(path=str(output.resolve()), sha256=sha256(output), reload_exact=True)
    audit['status'] = 'PASSED'
    write_json(init_dir / ('initialization.json' if variant == 'both' else variant + '_initialization.json'), audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--init-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=['both', *VARIANTS], default='both')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = initialize(args.source, args.init_dir, args.variant)
    print(json.dumps(dict(status=result['status'], outputs=result['outputs']), indent=2))
