"""Shared helpers for the isolated YOLOv13l entry points (no detector imports)."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
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

SOURCE = ROOT / '.vendor/yolov13l-scratch/yolov13'
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
            raise RuntimeError('This YOLOv13l run/cache is already in use: ' + str(path)) from exc
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
    paths = git('ls-files', 'ultralytics', 'pyproject.toml', 'LICENSE', 'requirements.txt', cwd=source).splitlines()
    paths = [p for p in paths if Path(p).suffix in {'.py', '.yaml', '.toml'} or p in {'LICENSE', 'requirements.txt'}]
    return digest(canonical({p: digest((source / p).read_bytes().replace(b'\r\n', b'\n')) for p in paths}))


def checked_source(source=SOURCE):
    source = Path(source).resolve()
    lock = read_json(LOCK)
    if git('remote', 'get-url', 'origin', cwd=source) != lock['repository']:
        raise ValueError('Upstream remote differs from official iMoonLab repository')
    if sha256(HERE / lock['patch']) != lock['patch_sha256']:
        raise ValueError('Committed compatibility patch checksum differs')
    if git('rev-parse', 'HEAD', cwd=source) != lock['commit']:
        raise ValueError('Official source commit differs from upstream.lock.json')
    if source_hash(source) != lock['patched_source_sha256']:
        raise ValueError('Official source/patch content differs; inspect modifications or run bootstrap')
    unknown = git('ls-files', '--others', '--exclude-standard', '--', 'ultralytics', cwd=source)
    unknown='\n'.join(p for p in unknown.splitlines() if not ('/__pycache__/' in p and p.endswith('.pyc')))
    if unknown:
        raise ValueError('Untracked files in official package: ' + unknown)
    return source, lock


def configure(runtime, source=SOURCE):
    """Call BEFORE importing Ultralytics, including in spawned data workers."""
    sys.dont_write_bytecode = True
    source, lock = checked_source(source)
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    for key, value in {'YOLO_CONFIG_DIR': runtime / 'ultralytics-settings',
                       'MPLCONFIGDIR': runtime / 'matplotlib', 'YOLO_AUTOINSTALL': 'false',
                       'YOLOV13L_SOURCE': source, 'YOLOV13L_RUNTIME': runtime}.items():
        os.environ[key] = str(value)
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    os.environ['YOLOV13_ATTENTION_BACKEND'] = 'native'
    sys.path.insert(0, str(source))
    import ultralytics
    actual = Path(ultralytics.__file__).resolve()
    if actual != source / 'ultralytics/__init__.py' or ultralytics.__version__ != lock['version']:
        raise RuntimeError('Wrong ultralytics import: ' + str(actual))
    import torch
    from ultralytics.nn.modules import block
    if block.USE_FLASH_ATTN:
        raise RuntimeError('Frozen native backend unexpectedly changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from ultralytics.utils import SETTINGS
    SETTINGS.update({key: str(runtime / key) for key in ('weights_dir', 'runs_dir', 'datasets_dir')})
    SETTINGS.update({key: False for key in ('sync', 'hub', 'wandb', 'comet', 'clearml', 'dvc',
                     'mlflow', 'neptune', 'raytune', 'tensorboard') if key in SETTINGS})
    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks = lambda instance: None
    import importlib.metadata as metadata
    versions={n:metadata.version(n) for n in ('torch','torchvision','numpy','opencv-python','pandas','seaborn','ultralytics-thop','pillow','scipy','matplotlib','pyyaml','huggingface-hub','safetensors')}
    result = {'runtime_versions':versions, 'python_version':sys.version, 'ultralytics_file': str(actual), 'version': ultralytics.__version__,
              'commit': lock['commit'], 'patch_sha256': lock['patch_sha256'],
              'patched_source_sha256': lock['patched_source_sha256'], 'adapter_sha256': adapter_hash(),
              'settings_dir': os.environ['YOLO_CONFIG_DIR'], 'python': os.path.abspath(sys.executable),
              'attention_backend': 'native', 'attention_operation': lock['backend_operation'],
              'internal_flash_half_casts': False, 'tf32': False,
              'sys_prefix': sys.prefix, 'sys_base_prefix': sys.base_prefix}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def recipe():
    return load_yaml(RECIPE)

def initialization_type():
    if read_json(HERE/'initialization.json')['initialization_type'] != 'random' or recipe()['pretrained'] is not False:
        raise ValueError('Only random initialization with pretrained=False is allowed')
    return 'random'


def model_yaml(source=SOURCE):
    return Path(source)/'ultralytics/cfg/models/v13/yolov13.yaml'


def model_config(source=SOURCE):
    cfg = load_yaml(model_yaml(source))
    if set(cfg['scales']) != {'n','s','l','x'} or cfg['scales']['l'] != [1.0,1.0,512]:
        raise ValueError('Official n/s/l/x scale dictionary differs')
    cfg.update(scale='l', nc=1)
    return cfg


def initialization_record(source=SOURCE):
    initialization_type()
    return {'initialization_type':'random', 'pretraining_source':None, 'pretrained_tensors_loaded':0,
            'model_yaml':str(model_yaml(source).resolve()), 'model_yaml_sha256':sha256(model_yaml(source)),
            'scale':'l', 'nc':1, 'seed':42,
            'effective_model_config_sha256':digest(canonical(model_config(source))),
            'native_constants':'Official FullPAD gate=0; ABlock trunc_normal(std=.02); BN; Detect biases; fixed DFL arange(16); HyperACE Xavier prototypes; A2C2f gamma=.01'}


@contextmanager
def no_external_initialization():
    """Fail on actual loading calls during fresh construction, not just config flags."""
    import torch
    from unittest.mock import patch
    from ultralytics.nn.tasks import BaseModel
    from ultralytics.utils import downloads
    def forbidden(*args, **kwargs):
        raise RuntimeError('External pretrained state/download forbidden for scratch construction')
    with ExitStack() as stack:
        for obj, name in ((torch, 'load'), (torch.nn.Module, 'load_state_dict'),
                          (torch.hub, 'load_state_dict_from_url'), (BaseModel, 'load'),
                          (downloads, 'attempt_download_asset'), (downloads, 'safe_download')):
            stack.enter_context(patch.object(obj, name, side_effect=forbidden))
        yield


def validate_checkpoint(ckpt, identity, resume=False):
    if ckpt.get('comparison_identity') != identity or identity.get('initialization_type') != 'random':
        raise ValueError('Checkpoint model/initialization/data/config/run identity differs')
    if resume:
        required = ('optimizer', 'ema', 'updates', 'scaler', 'comparison_best_epoch',
                    'comparison_initialization_sha256', 'comparison_stopper', 'comparison_model_state',
                    'comparison_rng', 'comparison_scheduler', 'train_results')
        if any(ckpt.get(k) is None for k in required):
            raise ValueError('Checkpoint missing optimizer/EMA/scaler/best/initialization/RNG/scheduler resume state')
        epochs = ckpt['train_args']['epochs']
        if not 0 <= ckpt['epoch'] < epochs-1:
            raise ValueError('Checkpoint is complete or epoch invalid')
        best = ckpt['comparison_best_epoch']
        if not isinstance(best, int) or not 1 <= best <= ckpt['epoch']+1:
            raise ValueError('Invalid best epoch')
        stopper = ckpt['comparison_stopper']
        if stopper['best_epoch'] != best or stopper['best_fitness'] != ckpt['best_fitness']:
            raise ValueError('Checkpoint best/patience state differs')
        if ckpt['epoch']+1-best >= ckpt['train_args']['patience']:
            raise ValueError('Checkpoint already reached patience')


def run_identity(manifest, source_identity, require_clean=True, run_id=None):
    if require_clean and git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Commit adapter changes before formal training; worktree must be clean')
    if not run_id:
        raise ValueError('An explicit run UUID is required')
    init = initialization_record(Path(source_identity['ultralytics_file']).parents[1])
    return {'model':'official_yolov13l_random_to_crack', **init,
            'run_id':run_id, 'scope':'smoke' if run_id.startswith('SMOKE_') else 'formal',
            'model_code_sha':git('rev-parse','HEAD'), 'dataset_identity_sha256':manifest['dataset_identity_sha256'],
            'recipe_sha256':digest(canonical(recipe())), 'adapter_sha256':adapter_hash(),
            'upstream_commit':source_identity['commit'], 'patch_sha256':source_identity['patch_sha256'],
            'patched_source_sha256':source_identity['patched_source_sha256'],
            'attention_backend':source_identity['attention_backend'],
            'environment_sha256':digest(canonical({'versions':source_identity['runtime_versions'],'python_version':source_identity['python_version']})),
            'initialization_sha256':None}


def status(run, step, state, **values):
    write_json(Path(run)/(step+'_status.json'), {'step':step, 'status':state, **values})
