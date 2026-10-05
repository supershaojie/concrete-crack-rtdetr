"""Strict candidate configuration and immutable per-run identity; no model imports."""
from __future__ import annotations
import math
import os
from pathlib import Path
import re
import shlex
import sys
import uuid
from support import (HERE, ROOT, SOURCE, ASSET, LOCK, atomic_bytes, adapter_hash,
                     canonical, digest, git, load_yaml, read_json, sha256, write_json)

DEFAULT_CONFIG = HERE / 'default_config.yaml'
RUN_ROOT = ROOT / 'outputs/yolo11m-configurable'
DEFAULT_SOURCE, DEFAULT_WEIGHT = SOURCE, ASSET
DEFAULT_DATA = Path('/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml')
DEFAULT_DATA_ROOT = Path('/root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det')
INT_RANGES = {'epochs': (1, 10000), 'patience': (0, 10000), 'batch': (1, 128),
              'workers': (0, 128), 'close_mosaic': (0, 10000), 'nbs': (1, 65536)}
NUM_RANGES = {
    'lr0': (0, 1), 'lrf': (0, 1), 'momentum': (0, 1), 'weight_decay': (0, 1),
    'warmup_epochs': (0, 10000), 'warmup_momentum': (0, 1), 'warmup_bias_lr': (0, 1),
    'box': (0, 100), 'cls': (0, 100), 'dfl': (0, 100),
    'hsv_h': (0, 1), 'hsv_s': (0, 1), 'hsv_v': (0, 1), 'degrees': (0, 180),
    'translate': (0, 1), 'scale': (0, 1), 'shear': (0, 180), 'perspective': (0, .001),
    'flipud': (0, 1), 'fliplr': (0, 1), 'mosaic': (0, 1), 'mixup': (0, 1), 'cutmix': (0, 1),
}
BOOL_FIELDS = ('cos_lr', 'amp', 'deterministic')
TUNABLE = set(INT_RANGES) | set(NUM_RANGES) | set(BOOL_FIELDS) | {'optimizer'}

def safe_run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('run-id must be 1-80 ASCII letters/digits/underscore/hyphen, starting with a letter/digit')
    return value

def yaml_mapping(raw):
    import yaml
    class StrictLoader(yaml.SafeLoader):
        pass
    StrictLoader.add_implicit_resolver('tag:yaml.org,2002:float',
        re.compile(r'^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+$'),list('-+0123456789.'))
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

def cli_overrides(items=()):
    import yaml
    result = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep or not key or key.strip() != key or not value.strip():
            raise ValueError('--set requires key=value (use explicit null for null)')
        if key in result:
            raise ValueError('Duplicate --set field: ' + key)
        scalar = yaml.safe_load(value)
        if isinstance(scalar,str) and re.fullmatch(r'[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+',value):
            scalar = float(value)
        result[key] = scalar
    return result

def resolve_config(user=None, overrides=None):
    defaults = {**load_yaml(HERE / 'recipe.yaml'), **load_yaml(DEFAULT_CONFIG)}
    user, overrides = user or {}, overrides or {}
    unknown = (set(user) | set(overrides)) - set(defaults)
    if unknown:
        raise ValueError('Unknown or unsupported configuration fields: ' + ', '.join(sorted(unknown)))
    # Invalid YAML cannot be hidden by a later valid --set.
    for values in ({**defaults, **user}, {**defaults, **user, **overrides}):
        for key, expected in defaults.items():
            if key in TUNABLE:
                continue
            actual = values[key]
            correct_type = (type(actual) is type(expected) or (type(expected) is float and type(actual) is int))
            if not correct_type or actual != expected:
                raise ValueError(f'{key} is fixed at {expected!r} for this implementation/protocol')
        for key, (low, high) in INT_RANGES.items():
            if type(values[key]) is not int or not low <= values[key] <= high:
                raise ValueError(f'{key} must be an integer in [{low}, {high}]; AutoBatch is forbidden')
        for key, (low, high) in NUM_RANGES.items():
            value = values[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{key} must be a finite number in [{low}, {high}]')
        for key in BOOL_FIELDS:
            if type(values[key]) is not bool:
                raise ValueError(key + ' must be a boolean')
        if not isinstance(values['optimizer'], str) or values['optimizer'] not in ('SGD', 'Adam', 'AdamW'):
            raise ValueError('optimizer must be SGD, Adam or AdamW; auto is forbidden')
        if values['lr0'] <= 0 or not 0 < values['momentum'] < 1:
            raise ValueError('lr0 must be positive; momentum / Adam beta1 must be strictly between 0 and 1')
    if values['optimizer'] != 'SGD' and values['warmup_momentum'] != defaults['warmup_momentum']:
        raise ValueError('warmup_momentum is inactive for native Adam/AdamW; keep the default')
    if values['close_mosaic'] > values['epochs'] or values['warmup_epochs'] > values['epochs']:
        raise ValueError('close_mosaic and warmup_epochs must not exceed epochs')
    if not any(values[k] > 0 for k in ('box', 'cls', 'dfl')):
        raise ValueError('At least one native detection loss gain must be positive')
    return values

def candidate(config=None, items=(), clone=None):
    if config is not None and clone is not None:
        raise ValueError('--config and --clone-config-from are mutually exclusive')
    raw = Path(config).read_bytes() if config else b'# No input YAML: committed defaults were used.\n{}\n'
    user = yaml_mapping(raw)
    origin = {'mode': 'yaml' if config else 'committed_defaults', 'path': str(Path(config).absolute()) if config else None}
    if clone is not None:
        clone = Path(clone) if Path(clone).is_absolute() else RUN_ROOT / safe_run_id(str(clone))
        user = frozen_config(clone, check_code=False)
        import yaml
        raw = yaml.safe_dump(user, sort_keys=False).encode('utf-8')
        origin = {'mode': 'clone_frozen_recipe_only', 'run': str(clone.absolute()),
                  'config_sha256': read_json(clone/'config_identity.json')['config_sha256'],
                  'training_state_inherited': False}
    overrides = cli_overrides(items)
    return resolve_config(user, overrides), raw, overrides, origin

def freeze_config(run, raw, paths, run_id, require_clean=True, overrides=None, origin=None, command=None, smoke=False):
    import yaml
    safe_run_id(run_id)
    user = yaml_mapping(raw)
    if smoke:
        if not run_id.startswith('SMOKE_ONLY_') or user.pop('imgsz',None) != 64:
            raise ValueError('Internal smoke requires SMOKE_ONLY_ ID and explicit imgsz=64')
    resolved = resolve_config(user, overrides)
    if smoke:
        if resolved['epochs'] > 3 or resolved['batch'] > 2 or resolved['workers'] > 2:
            raise ValueError('SMOKE_ONLY budget exceeds 3 epochs / batch2 / workers2')
        resolved['imgsz'] = 64
    if require_clean and git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Commit implementation changes before preparing a formal run')
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    atomic_bytes(run/'user_config.yaml', raw)
    atomic_bytes(run/'resolved_config.yaml', yaml.safe_dump(resolved, sort_keys=False).encode('utf-8'))
    write_json(run/'cli_overrides.json', overrides or {})
    write_json(run/'config_input.json', origin or {'mode': 'yaml_bytes'})
    write_json(run/'runtime_paths.json', paths)
    write_json(run/'run_id.json', {'run_id': run_id, 'run_uuid': uuid.uuid4().hex, 'experiment': 'yolo11m-configurable'})
    argv = [str(v) for v in (command or [sys.executable, *sys.argv])]
    write_json(run/'launch_command.json', {'argv': argv, 'bash': shlex.join(argv),
        'shell_invocation': os.environ.get('YOLO11M_LAUNCH_COMMAND'), 'cwd': str(Path.cwd())})
    from ultralytics.cfg import get_cfg
    from support import native_recipe
    native = native_recipe(resolved)
    native.update(model=paths.get('weights', resolved['model']), data=str(run/'data.yaml'),
                  project=str(run), name='train', exist_ok=False, resume=False)
    write_json(run/'native_args.json', vars(get_cfg(overrides=native)))
    from export import SETTINGS, POSTPROCESSING, evaluator_api
    files = ('user_config.yaml', 'resolved_config.yaml', 'cli_overrides.json', 'config_input.json',
             'runtime_paths.json', 'run_id.json', 'launch_command.json', 'native_args.json')
    write_json(run/'config_identity.json', {
        'schema_version': 2, 'config_sha256': digest(canonical(resolved)),
        'scope':'SMOKE_ONLY' if smoke else 'CONFIGURABLE_FORMAL',
        'worktree_dirty_at_freeze':bool(git('status','--porcelain','--untracked-files=normal')),
        'default_config_sha256': sha256(DEFAULT_CONFIG), 'recipe_sha256': sha256(HERE/'recipe.yaml'),
        'input_file_sha256': digest(raw) if origin and origin['mode'] == 'yaml' else None,
        'cli_overrides_sha256': digest(canonical(overrides or {})),
        'files': {name: sha256(run/name) for name in files},
        'model_code_sha': git('rev-parse', 'HEAD'), 'adapter_sha256': adapter_hash(),
        'upstream': read_json(LOCK), 'augmentation': read_json(HERE/'augmentation.lock.json'),
        'initialization': read_json(HERE/'initialization.json'),
        'public_inference': SETTINGS, 'postprocessing': POSTPROCESSING,
        'evaluation_config_sha256': evaluator_api().POLICY_SHA})
    return resolved

def frozen_config(run, check_code=True):
    run = Path(run)
    record = read_json(run/'config_identity.json')
    for name, expected in record['files'].items():
        if sha256(run/name) != expected:
            raise ValueError('Frozen run artifact changed: ' + name)
    resolved = load_yaml(run/'resolved_config.yaml')
    if digest(canonical(resolved)) != record['config_sha256']:
        raise ValueError('Frozen resolved configuration identity differs')
    if check_code and (git('rev-parse', 'HEAD') != record['model_code_sha']
                      or adapter_hash() != record['adapter_sha256']
                      or (record.get('scope') != 'SMOKE_ONLY' and git('status', '--porcelain', '--untracked-files=normal'))):
        raise ValueError('Frozen run requires its original clean implementation HEAD/content')
    return resolved

def interpreter_path(path):
    # abspath preserves Linux venv links; resolve() can silently select the mother Python.
    return os.path.normcase(os.path.abspath(os.fspath(path)))

def frozen_paths(run):
    frozen_config(run)
    paths = read_json(Path(run)/'runtime_paths.json')
    if interpreter_path(paths['python']) != interpreter_path(sys.executable):
        raise ValueError('Use the frozen venv interpreter for this run: ' + paths['python'])
    if interpreter_path(paths['sys_prefix']) != interpreter_path(sys.prefix):
        raise ValueError('Frozen sys.prefix differs from the current interpreter')
    return paths

def schema():
    defaults = resolve_config({})
    return {'schema_version': 2, 'precedence': ['committed_defaults', 'config_yaml_or_cloned_recipe', 'cli_set'],
            'integers': {k:list(v) for k,v in INT_RANGES.items()},
            'finite_numbers': {k:list(v) for k,v in NUM_RANGES.items()}, 'booleans': list(BOOL_FIELDS),
            'optimizer': ['SGD', 'Adam', 'AdamW'],
            'fixed': {k: v for k, v in defaults.items() if k not in TUNABLE},
            'inactive_detection_fields': ['erasing', 'auto_augment'],
            'cross_constraints': ['lr0>0', '0<momentum<1', 'close_mosaic<=epochs',
                'warmup_epochs<=epochs', 'any positive box/cls/dfl', 'Adam warmup_momentum must remain default']}
