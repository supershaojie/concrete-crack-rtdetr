"""Pinned CQS identities, strict initialization, recipe and read-only data inventory."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ultralytics-main'))
import torch
import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import LIFDown, RTDETRDecoderCBR, RTDETRDecoderCBRCQS
from ultralytics.nn.modules.cqs import CQS_CONFIG
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from init_c19_lif_v1 import (SOURCE_SHA256, MODULE_HASHES, source_contract, controlled_models,
                              native_rebuild, build as parent_build, require, sha256)

BASE_SHA = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
BRANCH = 'exp-rtdetr-r18-lite-cqs-v1'
REMOTE = 'https://github.com/supershaojie/concrete-crack-rtdetr.git'
MODEL_DIR = ROOT / 'ultralytics-main/ultralytics/cfg/models/rt-detr'
YAML_PATH = MODEL_DIR / 'rtdetr-resnet18-lite-cbr-lif-down-cqs.yaml'
PARENT_YAML = MODEL_DIR / 'rtdetr-resnet18-lite-cbr-lif-down.yaml'
MAIN = Path(os.environ.get('CQS_V1_MAIN', '/root/autodl-tmp/projects/Crack_RTDETR')).resolve()
OUT = ROOT / 'outputs/cqs_v1'
RUN_NAME = 'cqs_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug'
RUN = MAIN / 'runs/c_series' / RUN_NAME
INIT = ROOT / 'weights/cqs_v1_controlled_init.pt'
SESSION = 'cqs-v1-training'
SERVER_PYTHON = '/root/miniconda3/envs/rtdetr/bin/python'


def utc():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return dict(value=None, nonfinite=True, original=repr(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True, encoding='utf-8').strip()


def digest_json(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    files = set((ROOT / 'ultralytics-main/ultralytics').rglob('*.py'))
    files.update(ROOT.glob('tools/*cqs_v1*'))
    files.update(ROOT.glob('tools/*c19_lif_v1*.py'))
    files.update(ROOT.glob('tools/*lif_down*.py'))
    files.update((YAML_PATH, PARENT_YAML, ROOT/'docs/cqs_v1/research.yaml', ROOT/'docs/cqs_v1/parent_args.yaml',
                  ROOT/'docs/cqs_v1/parent_data_identity.json'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(files) if p.is_file()}


def runtime():
    require(Path(ultralytics.__file__).resolve() == ROOT/'ultralytics-main/ultralytics/__init__.py',
            'ultralytics import is outside the CQS worktree')
    source_contract()
    info = dict(python=platform.python_version(), executable=sys.executable, torch=str(torch.__version__),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, ultralytics_version=ultralytics.__version__,
                source_sha=git('rev-parse', 'HEAD'), base_sha=BASE_SHA,
                model_type='RTDETRDetectionModel', head_type='RTDETRDecoderCBRCQS',
                criterion_type='RTDETRDetectionLoss')
    if torch.cuda.is_available():
        info['gpu_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True)
        info['driver'] = result.stdout.strip()
    for name in ('numpy', 'scipy', 'cv2', 'torchvision'):
        module = __import__(name)
        info[name] = module.__version__
    return info


def verify_model(model, require_nc1=True):
    require(type(model) is RTDETRDetectionModel, 'Unexpected model type')
    expected = YAML.load(PARENT_YAML)
    expected['head'][-1][2] = 'RTDETRDecoderCBRCQS'
    require(all(model.yaml[k] == expected[k] for k in ('backbone', 'head', 'scales')), 'Only head class may change')
    head = model.model[26]
    require(type(head) is RTDETRDecoderCBRCQS and head.cqs_config == CQS_CONFIG, 'CQS head/config missing or disabled')
    require(type(model.model[20]) is LIFDown and hasattr(model.model[20], 'bn'), 'Original LIF BN protection missing')
    require(head.f == [19, 22, 25] and head.hidden_dim == 256 and head.num_queries == 300,
            'Decoder routing/dimensions changed')
    require(len(head.decoder.layers) == 3 and head.decoder.eval_idx == 2 and head.num_denoising == 100,
            'Decoder/DN structure changed')
    require(head.cbr.rho == head.cbr.normal_fraction == .10, 'Original CBR rho changed')
    if require_nc1:
        require(head.nc == 1, 'Actual CQS requires nc=1')
    # Generic is_fused() counts remaining backbone BNs and misclassifies this R18.
    # Node8 is an ordinary Conv whose BN is removed by the native fuse path.
    if hasattr(model.model[8], 'bn') and head.nc == 1:
        require(sum(v.numel() for v in model.parameters()) == 20149765 and len(model.state_dict()) == 552,
                'Unfused nc1 state/parameter count changed')
    return head


def build(nc=1):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(YAML_PATH), nc=nc, verbose=False)


def compare_states(reference, target):
    a, b = reference.state_dict(), target.state_dict()
    require(set(a) == set(b), f'State keys differ: missing={set(a)-set(b)}, extra={set(b)-set(a)}')
    pa, pb = dict(reference.named_parameters()), dict(target.named_parameters())
    require(set(pa) == set(pb), 'Parameter names differ')
    require(set(dict(reference.named_buffers())) == set(dict(target.named_buffers())), 'Buffer names differ')
    rows = []
    for k, value in a.items():
        require(value.shape == b[k].shape and torch.equal(value.detach().cpu(), b[k].detach().cpu()), f'State differs: {k}')
        rows.append(dict(name=k, kind='parameter' if k in pa else 'buffer', shape=list(value.shape),
                         equal=True, sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
    return dict(states=len(rows), parameters=sum(v.numel() for v in target.parameters()),
                added_parameters=0, missing=[], unexpected=[], shape_mismatch=[], values=rows)


def training_rebuild(cfg, weights, data, initial=True):
    require(weights is not None, 'Controlled CQS weights required')
    # Both calls start at the same global RNG. The reference audit consumes no global RNG.
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(PARENT_YAML), weights, data['nc'], data['channels'])
    model = native_rebuild(cfg, weights, data['nc'], data['channels'])
    verify_model(model)
    report = compare_states(parent, model)
    before, after = weights.state_dict(), model.state_dict()
    changed = {k for k in before if before[k].shape != after[k].shape}
    prefix = 'model.26.'
    allowed = {prefix+'denoising_class_embed.weight', prefix+'enc_score_head.weight', prefix+'enc_score_head.bias'}
    allowed |= {prefix+f'dec_score_head.{i}.{s}' for i in range(3) for s in ('weight','bias')}
    require(changed == (allowed if weights.model[-1].nc != data['nc'] else set()), 'Unexpected class adaptation')
    require(set(before) == set(after), 'Unexpected Trainer key filtering')
    require(all(torch.equal(v, after[k]) for k, v in before.items() if k not in changed), 'Native Trainer reload lost values')
    report.update(native_get_model=True, class_adaptation=sorted(changed), exact_loaded=len(before)-len(changed),
                  source_nc=weights.model[-1].nc, target_nc=data['nc'], initial_audit=initial)
    return model, report


def initialize(source, output=INIT):
    source, output = Path(source), Path(output)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, 'Missing/wrong public untrained source SHA256')
    ctor = compare_states(parent_build(nc=80), build(80))
    _, parent, provenance = controlled_models(source)
    model = build(80)
    model.load_state_dict(parent.state_dict(), strict=True)
    loaded = compare_states(parent, model)
    model.eval()
    model.args = {**DEFAULT_CFG_DICT, 'model': str(YAML_PATH), 'task': 'detect'}
    model.task, model.pt_path = 'detect', str(output.resolve())
    model.cqs_identity = dict(base_sha=BASE_SHA, config=CQS_CONFIG, public_source_sha256=SOURCE_SHA256)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), f'Existing initialization preserved: {output}')
    with output.open('xb') as stream:
        torch.save(dict(epoch=-1, best_fitness=None, model=deepcopy(model).float(), ema=None, updates=None,
                        optimizer=None, scaler=None, train_args=model.args, train_metrics=None,
                        train_results=None, date=utc(), cqs_provenance=provenance), stream)
    restored = RTDETR(str(output)).model
    compare_states(model, restored)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        _, trainer_audit = training_rebuild(str(YAML_PATH), restored, dict(nc=1, channels=3))
    return dict(status='PASS', source=source, source_sha256=SOURCE_SHA256, output=output,
                output_sha256=sha256(output), constructor=ctor, loaded=loaded, trainer=trainer_audit,
                module_hashes=source_contract(), parent_provenance=provenance, reload_exact=True)


def recipe(parent_args, init=INIT):
    actual = YAML.load(parent_args)
    expected = YAML.load(ROOT/'docs/cqs_v1/parent_args.yaml')
    require(set(actual) == set(expected), f'Parent args fields differ: {set(actual)^set(expected)}')
    # Only historical mother model workspace can differ; all 109 fields are audited.
    historical = {expected['model'], str(expected['model']).replace('-gatefix/', '/')}
    diffs = {k: [expected[k], actual[k]] for k in expected if type(expected[k]) is not type(actual[k]) or expected[k] != actual[k]}
    require(set(diffs) <= {'model'} and actual['model'] in historical, f'Authoritative parent args mismatch: {diffs}')
    target = dict(actual)
    target.update(model=str(Path(init).resolve()), name=RUN_NAME, save_dir=str(RUN))
    require(Path(actual['project']).resolve() == RUN.parent, 'Project relocation needs verified content identity')
    rows = [dict(field=k, parent=actual[k], cqs=target[k], changed=actual[k] != target[k]) for k in sorted(actual)]
    require({r['field'] for r in rows if r['changed']} == {'model', 'name', 'save_dir'}, 'Unexpected formal recipe changes')
    return target, dict(fields=len(actual), appendix_diff=diffs, fields_diff=rows)


def inventory(data_yaml, destination=None, check_counts=True, splits=('train','val','test')):
    """Read-only content identities. No dataset loader/cache writes and no model inference."""
    import numpy as np
    cfg = YAML.load(data_yaml)
    names = cfg.get('names')
    require(names in ({0:'crack'}, ['crack']), 'Expected one crack class')
    base = Path(cfg['path']).resolve()
    results = {}
    expected = dict(train=(6048,45573), val=(1728,12840), test=(864,6663))
    for split in splits:
        location = base / cfg[split]
        require(location.resolve() == (base/'images'/split).resolve() and location.is_dir(), 'Split identity/path changed')
        files = sorted(p for p in location.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
        require(files, f'Empty split: {split}')
        records, boxes = [], 0
        for image in files:
            label = base/'labels'/split/image.relative_to(location).with_suffix('.txt')
            require(label.is_file(), f'Missing label: {label}')
            values = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
            require(all(len(r) == 5 and r[0] == 0 and np.isfinite(r).all() and
                        all(0 <= v <= 1 for v in r[1:]) and r[3] > 0 and r[4] > 0 for r in values), f'Invalid label: {label}')
            boxes += len(values)
            records.append(dict(image=image.relative_to(base).as_posix(), image_sha256=sha256(image),
                                label=label.relative_to(base).as_posix(), label_sha256=sha256(label), instances=len(values)))
        require(boxes > 0, f'Missing GT: {split}')
        if check_counts:
            require((len(files), boxes) == expected[split], f'{split} counts differ: {(len(files), boxes)} != {expected[split]}')
        paths_hash=hashlib.sha256('\n'.join(r['image'] for r in records).encode()).hexdigest()
        labels_hash=hashlib.sha256('\n'.join(r['label']+':'+r['label_sha256'] for r in records).encode()).hexdigest()
        if check_counts:
            mother=read_json(ROOT/'docs/cqs_v1/parent_data_identity.json')[split]
            require(paths_hash==mother['split_paths_sha256'] and labels_hash==mother['label_inventory_sha256'],
                    f'{split} paths/labels differ from successful mother inventory')
        results[split] = dict(images=len(files), instances=boxes, content_sha256=digest_json(records),
                              split_paths_sha256=paths_hash,label_inventory_sha256=labels_hash)
        if destination:
            write_json(Path(destination)/f'{split}_manifest.json', records)
    return dict(data_yaml=str(Path(data_yaml).resolve()), data_sha256=sha256(data_yaml), root=str(base), splits=results)


def verify_delivery():
    record = read_json(OUT/'delivery.json')
    require(record['sha'] == git('rev-parse','HEAD') and record['branch'] == BRANCH and
            Path(record['worktree']).resolve() == ROOT, 'Run official sync FULL_SHA for this worktree first')
    require(git('remote','get-url','origin') == REMOTE and git('branch','--show-current') == BRANCH, 'Repository/branch identity changed')
    require(not git('status','--porcelain','--untracked-files=normal'), 'CQS source worktree is dirty; preserve and review changes')
    require((ROOT/'.git').is_file() and ROOT != MAIN, 'Use independent linked CQS worktree')
    return record


def fingerprint(prepared):
    info = runtime()
    environment = {k:v for k,v in info.items() if k not in ('source_sha','base_sha')}
    current = dict(source=source_hashes(), research=YAML.load(ROOT/'docs/cqs_v1/research.yaml'), environment=environment,
                   public_source=sha256(prepared['source']), initialization=sha256(INIT),
                   parent_args=sha256(prepared['parent_args']), train_args=YAML.load(OUT/'train_args.yaml'),
                   data=inventory(prepared['data']))
    current['amp_resources']={row['name']:sha256(row['path']) for row in read_json(OUT/'amp_resources.json')}
    current['device_environment']={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER')}
    return dict(sha256=digest_json(current), identity=current)
