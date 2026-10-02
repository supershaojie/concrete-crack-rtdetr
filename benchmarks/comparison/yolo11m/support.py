"""Shared helpers for the isolated YOLO11m entry points (no detector imports)."""
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

SOURCE = ROOT / '.vendor/yolo11m-scratch/ultralytics-v8.3.20'
RECIPE = HERE / 'recipe.yaml'
LOCK = HERE / 'upstream.lock.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
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
            raise RuntimeError('This YOLO11m run/cache is already in use: ' + str(path)) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + str(Path(cwd).resolve()),
                                    *args], cwd=cwd, text=True, encoding='utf-8').strip()


def adapter_hash():
    files = sorted(p for p in HERE.rglob('*') if p.suffix in {'.py', '.yaml', '.json', '.patch', '.txt'})
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


def checked_weight(path=None):
    raise ValueError('YOLO11m scratch forbids external pretrained weights')


def configure(runtime, source=SOURCE):
    """Call BEFORE importing Ultralytics, including in spawned data workers."""
    source, lock = checked_source(source)
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    for key, value in {'YOLO_CONFIG_DIR': runtime / 'ultralytics-settings',
                       'MPLCONFIGDIR': runtime / 'matplotlib', 'YOLO_AUTOINSTALL': 'false',
                       'YOLO11M_SOURCE': source, 'YOLO11M_RUNTIME': runtime}.items():
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

def initialization_type():
    value = read_json(HERE/'initialization.json')['initialization_type']
    if value != 'random':
        raise ValueError('This branch accepts random initialization only')
    if (value == 'random') != (recipe()['pretrained'] is False):
        raise ValueError('Recipe and explicit initialization mode disagree')
    return value

def model_yaml(source=SOURCE):
    return Path(source)/'ultralytics/cfg/models/11/yolo11.yaml'

def model_config(source=SOURCE):
    cfg = load_yaml(model_yaml(source))
    cfg['scale'] = 'm'
    cfg['nc'] = 1
    if cfg['scales']['m'] != [.5, 1., 512]:
        raise ValueError('Pinned YOLO11 M scaling differs')
    return cfg

def initialization_record(source=SOURCE):
    mode = initialization_type()
    return {'initialization_type':mode, 'pretraining_source':None if mode=='random' else 'COCO',
            'pretrained_tensors_loaded':0 if mode=='random' else None,
            'model_yaml':str(model_yaml(source).resolve()), 'model_yaml_sha256':sha256(model_yaml(source)),
            'scale':'m', 'nc':1, 'seed':42,
            'native_constants':'BN, Detect bias priors and fixed DFL projection retain official definitions'}

def validate_checkpoint(ckpt, identity, resume=False):
    if ckpt.get('comparison_identity') != identity or identity.get('initialization_type') != 'random':
        raise ValueError('Checkpoint model/initialization/data/config/run identity differs')
    if ckpt.get('checkpoint_schema') != 'yolo11m_scratch_fp32_resume_v1':
        raise ValueError('Checkpoint schema lacks verified YOLO11m resume state')
    saved = ckpt.get('train_args', {})
    if digest(canonical(saved)) != ckpt.get('comparison_expanded_args_sha256'):
        raise ValueError('Checkpoint expanded training arguments changed')
    if saved.get('pretrained') is not False:
        raise ValueError('Scratch checkpoint claims pretrained initialization')
    if identity.get('run_scope') == 'formal':
        for key, value in recipe().items():
            if key not in ('model', 'resume') and saved.get(key) != value:
                raise ValueError('Checkpoint frozen recipe mismatch: ' + key)
    if not resume:
        return
    required = ('optimizer', 'model', 'ema', 'updates', 'scaler', 'scheduler',
                'comparison_stopper', 'comparison_best_epoch', 'comparison_initialization_sha256')
    if any(ckpt.get(key) is None for key in required):
        raise ValueError('Checkpoint missing optimizer/model/EMA/scaler/scheduler/stopper/initialization state')
    epoch, epochs = ckpt.get('epoch', -1), saved.get('epochs', 0)
    best_epoch = ckpt['comparison_best_epoch']
    stop = ckpt['comparison_stopper']
    if (not 0 <= epoch < epochs - 1 or not 1 <= best_epoch <= epoch+1
            or stop.get('best_epoch') != best_epoch or stop.get('best_fitness') != ckpt.get('best_fitness')
            or stop.get('patience') != saved.get('patience')
            or epoch+1-best_epoch >= saved['patience']):
        raise ValueError('Checkpoint complete or best/patience state inconsistent')
    if not isinstance(ckpt['scaler'], dict) or (saved.get('amp') and 'scale' not in ckpt['scaler']):
        raise ValueError('Checkpoint missing active AMP scaler state')


def run_identity(manifest, source_identity, require_clean=True, run_id=None):
    if require_clean and git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Commit adapter changes before formal training; worktree must be clean')
    init = initialization_record(Path(source_identity['ultralytics_file']).parents[1])
    return {'model': 'official_yolo11m_random_to_crack',
            'run_scope': 'synthetic_smoke' if str(run_id).startswith('SMOKE_') else 'formal',
            **init, 'run_id':run_id, 'model_code_sha': git('rev-parse', 'HEAD'),
            'dataset_identity_sha256': manifest['dataset_identity_sha256'],
            'recipe_sha256': digest(canonical(recipe())), 'adapter_sha256': adapter_hash(),
            'upstream_commit': source_identity['commit'], 'patch_sha256': source_identity['patch_sha256'],
            'initialization_sha256': None}


def status(run, step, state, **values):
    write_json(Path(run) / (step + '_status.json'), {'step': step, 'status': state, **values})
