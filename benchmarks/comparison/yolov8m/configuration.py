"""Strict candidate YAML and immutable per-run snapshots; no detector imports."""
from __future__ import annotations

import math
from pathlib import Path
import re
import sys
import uuid

from support import (HERE, ROOT, LOCK, atomic_bytes, adapter_hash, canonical, digest,
                     git, load_yaml, read_json, sha256, write_json)

DEFAULT_CONFIG = HERE / 'default_config.yaml'
RUN_ROOT = ROOT / 'outputs/yolov8m-configurable'
PILOT = Path('/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-coco-b19-pilot')
DEFAULT_SOURCE = PILOT / '.vendor/yolov8m-coco-b19-pilot/ultralytics-v8.3.20'
DEFAULT_WEIGHT = PILOT / '.runtime/yolov8m-coco-b19-pilot/assets/yolov8m.pt'
DEFAULT_DATA = Path('/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml')
DEFAULT_DATA_ROOT = Path('/root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det')

INT_RANGES = {'epochs': (1, 10000), 'patience': (0, 10000), 'workers': (0, 128),
              'close_mosaic': (0, 10000), 'nbs': (1, 65536)}
NUM_RANGES = {
    'lr0': (0, 1), 'lrf': (0, 1), 'momentum': (0, 1), 'weight_decay': (0, 1),
    'warmup_epochs': (0, 10000), 'warmup_momentum': (0, 1), 'warmup_bias_lr': (0, 1),
    'box': (0, 100), 'cls': (0, 100), 'dfl': (0, 100),
    'hsv_h': (0, 1), 'hsv_s': (0, 1), 'hsv_v': (0, 1),
    'degrees': (0, 180), 'translate': (0, 1), 'scale': (0, 1),
    'shear': (0, 180), 'perspective': (0, 1),
    'flipud': (0, 1), 'fliplr': (0, 1), 'mosaic': (0, 1),
    'mixup': (0, 1), 'cutmix': (0, 1),
}
FIXED = {'batch': 16, 'imgsz': 640, 'device': 0, 'seed': 42, 'deterministic': True,
         'amp': True, 'cache': False, 'rect': False, 'multi_scale': False,
         'bgr': 0.0, 'copy_paste': 0.0}


def safe_run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('run-id must be 1-80 ASCII letters/digits/underscore/hyphen, starting with a letter/digit')
    return value


def yaml_mapping(raw):
    import yaml
    class StrictLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError('Configuration keys must be strings')
            if key in result:
                raise ValueError('Duplicate configuration field: ' + key)
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    cfg = yaml.load(raw.decode('utf-8-sig'), Loader=StrictLoader)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a YAML mapping')
    return cfg


def resolve_config(user):
    defaults = load_yaml(DEFAULT_CONFIG)
    unknown = set(user) - set(defaults)
    if unknown:
        raise ValueError('Unknown or unsupported detection configuration fields: ' + ', '.join(sorted(unknown)))
    values = {**defaults, **user}
    for key, expected in FIXED.items():
        actual = values[key]
        valid_type = (type(actual) is bool if type(expected) is bool else
                      type(actual) is int if type(expected) is int else type(actual) in (int, float))
        if not valid_type or actual != expected:
            raise ValueError(f'{key} is fixed at {expected!r} for this implementation/protocol')
    for key, (low, high) in INT_RANGES.items():
        if type(values[key]) is not int or not low <= values[key] <= high:
            raise ValueError(f'{key} must be an integer in [{low}, {high}]')
    for key, (low, high) in NUM_RANGES.items():
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be a finite number in [{low}, {high}]')
    if values['lr0'] <= 0 or not 0 < values['momentum'] < 1:
        raise ValueError('lr0 must be positive; momentum (SGD momentum / Adam beta1) must be strictly between 0 and 1')
    if type(values['cos_lr']) is not bool:
        raise ValueError('cos_lr must be a boolean')
    if values['optimizer'] not in ('SGD', 'Adam', 'AdamW'):
        raise ValueError('optimizer must be SGD, Adam or AdamW; auto is unsupported')
    if values['optimizer'] != 'SGD' and values['warmup_momentum'] != defaults['warmup_momentum']:
        raise ValueError('warmup_momentum is only used by native SGD; keep its default for Adam/AdamW')
    if values['close_mosaic'] > values['epochs'] or values['warmup_epochs'] > values['epochs']:
        raise ValueError('close_mosaic and warmup_epochs must not exceed epochs')
    if not any(values[k] > 0 for k in ('box', 'cls', 'dfl')):
        raise ValueError('At least one native detection loss gain must be positive')
    # Non-tuning algorithm/protocol fields remain the committed pilot defaults.
    return {**load_yaml(HERE / 'recipe.yaml'), **values}


def freeze_config(run, raw, paths, run_id, require_clean=True):
    import yaml
    safe_run_id(run_id)
    resolved = resolve_config(yaml_mapping(raw))
    if require_clean and git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Commit implementation changes before preparing a formal run')
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    atomic_bytes(run / 'user_config.yaml', raw)
    atomic_bytes(run / 'resolved_config.yaml', yaml.safe_dump(
        resolved, sort_keys=False, allow_unicode=True).encode('utf-8'))
    write_json(run / 'runtime_paths.json', paths)
    write_json(run / 'run_id.json', {'run_id': run_id, 'run_uuid':uuid.uuid4().hex,
                                     'experiment': 'yolov8m-configurable'})
    from export import SETTINGS, POSTPROCESSING, evaluator_api
    write_json(run / 'config_identity.json', {
        'schema_version': 1, 'config_sha256': digest(canonical(resolved)),
        'user_config_sha256': sha256(run / 'user_config.yaml'),
        'resolved_file_sha256': sha256(run / 'resolved_config.yaml'),
        'runtime_paths_sha256': sha256(run / 'runtime_paths.json'),
        'run_id_sha256': sha256(run / 'run_id.json'),
        'model_code_sha': git('rev-parse', 'HEAD'), 'adapter_sha256': adapter_hash(),
        'upstream': read_json(LOCK), 'augmentation': read_json(HERE / 'augmentation.lock.json'),
        'initialization': read_json(HERE / 'initialization.json'),
        'public_inference': SETTINGS, 'postprocessing': POSTPROCESSING,
        'evaluation_config_sha256': evaluator_api().POLICY_SHA})
    return resolved


def frozen_config(run, check_code=True):
    run = Path(run)
    record = read_json(run / 'config_identity.json')
    for name, field in (('user_config.yaml', 'user_config_sha256'),
                        ('resolved_config.yaml', 'resolved_file_sha256'),
                        ('runtime_paths.json', 'runtime_paths_sha256'),
                        ('run_id.json', 'run_id_sha256')):
        if sha256(run / name) != record[field]:
            raise ValueError('Frozen run artifact changed: ' + name)
    resolved = load_yaml(run / 'resolved_config.yaml')
    if digest(canonical(resolved)) != record['config_sha256']:
        raise ValueError('Frozen resolved configuration identity differs')
    if check_code and (git('rev-parse', 'HEAD') != record['model_code_sha']
                       or adapter_hash() != record['adapter_sha256']
                       or git('status', '--porcelain', '--untracked-files=normal')):
        raise ValueError('Frozen run requires its original clean implementation HEAD/content')
    # Do not re-merge current defaults or consult the original candidate path.
    return resolved


def frozen_paths(run):
    frozen_config(run)
    paths = read_json(Path(run) / 'runtime_paths.json')
    if Path(paths['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Use the frozen Python interpreter for this run: ' + paths['python'])
    return paths
