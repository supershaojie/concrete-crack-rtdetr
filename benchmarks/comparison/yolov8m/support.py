"""Shared helpers for the isolated YOLOv8m entry points (no detector imports)."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMON = HERE.parent
sys.path.insert(0, str(COMMON))
from common import canonical, digest, load_yaml, sha256

SOURCE = Path(os.environ.get('YOLOV8M_SOURCE', ROOT / '.vendor/yolov8m-coco-b19-pilot/ultralytics-v8.3.20'))
ASSET = Path(os.environ.get('YOLOV8M_WEIGHT', ROOT / '.runtime/yolov8m-coco-b19-pilot/assets/yolov8m.pt'))
RECIPE = HERE / 'recipe.yaml'
LOCK = HERE / 'upstream.lock.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A short sibling name also works in deeply nested Windows worktrees.
    tmp = path.parent / ('.tmp_' + uuid.uuid4().hex)
    try:
        with tmp.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path, value):
    atomic_bytes(path, canonical(value))


@contextmanager
def local_lock(path):
    """Nonblocking OS lock, released on process exit; only this run/cache is locked."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if not path.stat().st_size:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('This YOLOv8m run/cache is already in use: ' + str(path)) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + Path(cwd).resolve().as_posix(),
                                    *args], cwd=cwd, text=True, encoding='utf-8').strip()


def adapter_hash():
    files = sorted(p for p in HERE.rglob('*') if p.suffix in {'.py', '.yaml', '.json', '.patch', '.txt'}
                   and 'runtime_configs' not in p.relative_to(HERE).parts)
    return digest(canonical({p.relative_to(HERE).as_posix(): digest(p.read_bytes().replace(b'\r\n', b'\n'))
                             for p in files}))


def source_hash(source):
    paths = git('ls-files', 'ultralytics', 'pyproject.toml', 'LICENSE', cwd=source).splitlines()
    paths = [p for p in paths if Path(p).suffix in {'.py', '.yaml', '.toml'} or p == 'LICENSE']
    return digest(canonical({p: digest((source / p).read_bytes().replace(b'\r\n', b'\n')) for p in paths}))


def checked_source(source=SOURCE):
    source = Path(source).resolve()
    lock = read_json(LOCK)
    if sha256(HERE / lock['patch']) != lock['patch_sha256']:
        raise ValueError('Committed compatibility patch checksum differs')
    if git('rev-parse', 'HEAD', cwd=source) != lock['commit']:
        raise ValueError('Official source commit differs from upstream.lock.json')
    if source_hash(source) != lock['patched_source_sha256']:
        raise ValueError('Official source/patch content differs; inspect modifications or run bootstrap')
    unknown = git('ls-files', '--others', '--exclude-standard', '--', 'ultralytics', cwd=source)
    if unknown:
        raise ValueError('Untracked files in official package: ' + unknown)
    return source, lock


def checked_weight(path=ASSET):
    path = Path(path).resolve()
    weight = read_json(LOCK)['weights']
    if path.stat().st_size != weight['bytes'] or sha256(path) != weight['sha256']:
        raise ValueError('COCO initialization weight checksum differs: ' + str(path))
    return path


def configure(runtime, source=SOURCE):
    """Call BEFORE importing Ultralytics, including in spawned data workers."""
    if os.environ.get('YOLOV8M_FROZEN_RUN'):
        from configuration import frozen_config
        frozen_config(Path(os.environ['YOLOV8M_FROZEN_RUN']))
    source, lock = checked_source(source)
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    for key, value in {'YOLO_CONFIG_DIR': runtime / 'ultralytics-settings',
                       'MPLCONFIGDIR': runtime / 'matplotlib', 'YOLO_AUTOINSTALL': 'false',
                       'YOLOV8M_SOURCE': source, 'YOLOV8M_RUNTIME': runtime}.items():
        os.environ[key] = str(value)
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    import ultralytics
    actual = Path(ultralytics.__file__).resolve()
    if actual != source / 'ultralytics/__init__.py' or ultralytics.__version__ != lock['version']:
        raise RuntimeError('Wrong ultralytics import: ' + str(actual))
    from ultralytics.utils import SETTINGS
    SETTINGS.update({key: str(runtime / key) for key in ('weights_dir', 'runs_dir', 'datasets_dir')})
    SETTINGS.update({key: False for key in ('sync', 'hub', 'wandb', 'comet', 'clearml', 'dvc',
                     'mlflow', 'neptune', 'raytune', 'tensorboard') if key in SETTINGS})
    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks = lambda instance: None
    result = {'ultralytics_file': str(actual), 'version': ultralytics.__version__,
              'commit': lock['commit'], 'patch_sha256': lock['patch_sha256'],
              'patched_source_sha256': lock['patched_source_sha256'], 'adapter_sha256': adapter_hash(),
              'settings_dir': os.environ['YOLO_CONFIG_DIR'], 'python': sys.executable}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def recipe():
    return load_yaml(RECIPE)

def native_recipe(config=None):
    """Old get_cfg accepts native fields; the dataset consumes CutMix separately."""
    return {k:v for k,v in (recipe() if config is None else config).items()
            if k not in ('initialization_type', 'cutmix')}

def initialization_type(config=None):
    value = read_json(HERE/'initialization.json')['initialization_type']
    cfg = recipe() if config is None else config
    if value != 'coco_detection_pretrained' or cfg['initialization_type'] != value or cfg['pretrained'] is not True:
        raise ValueError('Recipe and explicit initialization mode disagree')
    return value

def model_yaml(source=SOURCE):
    return Path(source)/'ultralytics/cfg/models/v8/yolov8.yaml'

def model_config(source=SOURCE):
    cfg = load_yaml(model_yaml(source))
    cfg['scale'] = 'm'
    return cfg

def initialization_record(source=SOURCE, config=None):
    weight = read_json(LOCK)['weights']
    return {'initialization_type':initialization_type(config), 'pretraining_source':'official COCO detection (80 classes)',
            'source_sha256':weight['sha256'], 'source_bytes':weight['bytes'], 'source_url':weight['url'],
            'model_yaml':str(model_yaml(source).resolve()), 'model_yaml_sha256':sha256(model_yaml(source)),
            'scale':'m', 'nc':1, 'seed':42,
            'native_constants':'BN, Detect bias priors and fixed DFL projection retain official definitions'}

def validate_checkpoint(ckpt, identity, resume=False, config=None):
    cfg = recipe() if config is None else config
    if ckpt.get('comparison_identity') != identity:
        raise ValueError('Checkpoint model/initialization/data/config/run identity differs')
    state = ckpt.get('comparison_training_state', {})
    required = {'model', 'optimizer', 'ema', 'scaler', 'scheduler', 'rng', 'augmentation_closed', 'optimizer_steps'}
    if not required <= state.keys() or ckpt.get('pilot_recipe') != cfg:
        raise ValueError('Pilot checkpoint is missing verified full training state/recipe')
    if identity.get('scope') in ('PILOT_FORMAL', 'CONFIGURABLE_FORMAL'):
        saved = ckpt.get('train_args', {})
        for key,value in native_recipe(cfg).items():
            if key in ('model','resume'):
                continue  # runtime paths/explicit resume replace these two values
            actual = saved.get(key)
            if (str(actual) != str(value) if key == 'device' else actual != value):
                raise ValueError('Checkpoint native training configuration differs: ' + key)
    if ckpt.get('comparison_initialization') is None or ckpt['comparison_initialization']['source_sha256'] != identity['source_sha256']:
        raise ValueError('Original COCO initialization provenance is missing or different')
    if resume and (state.get('optimizer') is None or not 0 <= ckpt['epoch'] < cfg['epochs'] - 1):
        raise ValueError('Checkpoint is complete or missing resume state')


def run_identity(manifest, source_identity, require_clean=True, run_id=None, config=None, environment=None, run_uuid=None):
    if require_clean and git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Commit adapter changes before formal training; worktree must be clean')
    cfg = recipe() if config is None else config
    init = initialization_record(Path(source_identity['ultralytics_file']).parents[1], cfg)
    if run_id is None:
        raise ValueError('Pilot requires a unique experiment/run UUID')
    if config is not None and not run_id.startswith('SMOKE_') and run_uuid is None:
        raise ValueError('A configurable formal run requires its frozen unique run UUID')
    return {'model': 'official_yolov8m_coco_b19_pilot_to_crack',
            **init, 'run_id':run_id, 'scope':'SMOKE_ONLY' if run_id.startswith('SMOKE_') else
            ('CONFIGURABLE_FORMAL' if config is not None else 'PILOT_FORMAL'),
            'model_code_sha': git('rev-parse', 'HEAD'),
            'dataset_identity_sha256': manifest['dataset_identity_sha256'],
            'recipe_sha256': digest(canonical(cfg)), 'adapter_sha256': adapter_hash(),
            'upstream_commit': source_identity['commit'], 'patch_sha256': source_identity['patch_sha256'],
            'initialization_sha256': read_json(LOCK)['weights']['sha256'],
            'b19_recipe_sha256':digest(canonical(cfg)),
            **({'config_sha256':digest(canonical(cfg))} if config is not None else {}),
            **({'run_uuid':run_uuid} if run_uuid is not None else {}),
            **({'environment_sha256':digest(canonical(environment))} if environment is not None else {}),
            'augmentation_source_sha256':digest(canonical(read_json(HERE/'augmentation.lock.json'))),
            'training_preprocessing':'mother square stretch + additive MotherHSV; B19 online parameters only'}


def runtime_environment():
    """Probe only; never install into or modify the reused environment."""
    from bootstrap import environment_probe
    result = environment_probe()
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    result['pip_freeze_sha256'] = digest('\n'.join(sorted(freeze.splitlines())).encode())
    result['python_executable_sha256'] = sha256(sys.executable)
    return result


def verify_environment(run):
    actual = runtime_environment()
    if actual != read_json(Path(run) / 'environment.json'):
        raise ValueError('Frozen training environment differs; use the original interpreter/packages')
    return actual


def status(run, step, state, **values):
    write_json(Path(run) / (step + '_status.json'), {'step': step, 'status': state, **values})
