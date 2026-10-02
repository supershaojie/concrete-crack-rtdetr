"""Identity, atomic outputs and run-local locks. No detector imports."""
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

NAME = 'fasterrcnn-r50-fpn-scratch'
VENDOR = ROOT / '.vendor' / NAME
VISION = VENDOR / 'vision-v0.16.2'
AUGMENT = VENDOR / 'ultralytics-v8.3.20'
RUNTIME = ROOT / '.runtime' / NAME
ENV = ROOT / '.envs' / NAME
LOCK = HERE / 'upstream.lock.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.atomic-' + uuid.uuid4().hex[:12] + '.partial')
    with tmp.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    atomic_bytes(path, canonical(value))


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + Path(cwd).resolve().as_posix(),
        *args], cwd=cwd, text=True, encoding='utf-8').strip()


def text_hash(path):
    return digest(Path(path).read_bytes().replace(b'\r\n', b'\n'))


def code_identity():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.json')) + list(HERE.glob('*.yaml')) + list(HERE.glob('*.txt'))
    paths += [ROOT/'scripts/autodl_fasterrcnn_r50_fpn.sh', COMMON/'common.py', COMMON/'dataset.py',
              COMMON/'evaluation/evaluate.py', COMMON/'evaluation/native_metrics.py',
              ROOT/'docs/comparison/evidence/yolov8m_delivery_validation.json']
    return digest(canonical({p.relative_to(ROOT).as_posix(): text_hash(p) for p in sorted(paths)}))


def recipe():
    cfg = load_yaml(HERE / 'recipe.yaml')
    validate_recipe(cfg)
    return cfg


def validate_recipe(cfg):
    unsupported = {'model': 'fasterrcnn_resnet50_fpn', 'optimizer': 'SGD', 'lr_schedule': 'warmup_then_cosine',
        'gradient_accumulation_steps': 1, 'ema': False, 'gradient_clip': None, 'resume': False,
        'copy_paste': 0.0, 'cutmix': 0.0, 'erasing': 0.0, 'bgr': 0.0, 'auto_augment': None}
    for key, expected in unsupported.items():
        if cfg.get(key) != expected:
            raise ValueError('Unsupported/inactive recipe option must not be silently ignored: ' + key)
    if len({cfg[k] for k in ('input_height', 'input_width', 'model_transform_min_size', 'model_transform_max_size')}) != 1:
        raise ValueError('Data and model transforms must use exactly the same square size')
    if not cfg['amp_init_scale'] > 0 or cfg['amp_growth_interval'] < 1:
        raise ValueError('Invalid AMP scale configuration')


def checked_source(path, key):
    spec = read_json(LOCK)[key]
    if git('rev-parse', 'HEAD', cwd=path) != spec['commit']:
        raise ValueError('Wrong pinned source commit: ' + str(path))
    if git('status', '--porcelain', '--untracked-files=normal', cwd=path):
        raise ValueError('Source tree is modified; no automatic reset: ' + str(path))
    paths = git('ls-files', cwd=path).splitlines()
    files = {p: text_hash(path/p) for p in paths if p.endswith(('.py', '.yaml', '.toml')) or p == 'LICENSE'}
    if digest(canonical(files)) != spec['source_sha256']:
        raise ValueError('Pinned source content hash differs: ' + str(path))
    if key == 'augmentation' and text_hash(path/'LICENSE') != spec['license_sha256']:
        raise ValueError('Augmentation license hash differs')
    return {'commit': spec['commit'], 'tag': spec['tag'], 'source_sha256': digest(canonical(files)),
            'license_sha256': text_hash(path/'LICENSE'), 'text_hash_normalization': 'CRLF_to_LF'}


def configure_augmentation(runtime):
    """Own pinned data library, also called before imports in spawned workers."""
    runtime = Path(runtime).absolute()
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(runtime/'ultralytics-settings'),
                      MPLCONFIGDIR=str(runtime/'matplotlib'), YOLO_AUTOINSTALL='false',
                      FRCNN_DATA_RUNTIME=str(runtime))
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(AUGMENT))
    import ultralytics
    if Path(ultralytics.__file__).resolve() != AUGMENT/'ultralytics/__init__.py':
        raise ValueError('Refusing an installed or mother Ultralytics package')
    if ultralytics.__version__ != read_json(LOCK)['augmentation']['version']:
        raise ValueError('Wrong augmentation library version')
    from ultralytics.utils import SETTINGS
    SETTINGS.update({k: str(runtime/k) for k in ('weights_dir', 'runs_dir', 'datasets_dir')})
    SETTINGS.update({k: False for k in ('sync', 'hub', 'wandb', 'comet', 'clearml', 'dvc',
        'mlflow', 'neptune', 'raytune', 'tensorboard') if k in SETTINGS})


def environment(strict=True):
    import importlib.metadata as metadata
    import torch
    import torchvision
    import numpy
    from torchvision._internally_replaced_utils import _get_extension_path
    target = (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0]) == ('2.1.2', '0.16.2')
    if strict and not target:
        raise ValueError('Formal runs require torch 2.1.2 + torchvision 0.16.2 in the private environment')
    if strict and numpy.__version__ != '1.26.4':
        raise ValueError('Expected pinned NumPy 1.26.4')
    vision_source = checked_source(VISION, 'vision') if strict else None
    correspondence = {'status': 'NOT_TARGET_VERSION_LOCAL_COMPATIBILITY_ONLY'}
    if target:
        vision_source = checked_source(VISION, 'vision')
        installed = Path(torchvision.__file__).parent
        compared = {}
        for name in git('ls-files', 'torchvision', cwd=VISION).splitlines():
            if name.endswith('.py') and name != 'torchvision/version.py':
                actual = installed / Path(name).relative_to('torchvision')
                if not actual.is_file() or text_hash(actual) != text_hash(VISION/name):
                    raise ValueError('Installed TorchVision Python source differs from fixed tag: ' + name)
                compared[name] = text_hash(actual)
        correspondence = {'status': 'PACKAGE_PYTHON_SOURCE_EQUALS_PINNED_TAG',
                          'files': len(compared), 'sha256': digest(canonical(compared)),
                          'wheel_git_version': getattr(torchvision.version, 'git_version', None)}
    names = ('torch', 'torchvision', 'numpy', 'opencv-python', 'pillow', 'pyyaml', 'scipy', 'tqdm')
    return {'python': os.path.abspath(sys.executable), 'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
            'python_version': sys.version, 'target_version': target,
            'versions': {n: metadata.version(n) for n in names}, 'torch_file': torch.__file__,
            'torchvision_file': torchvision.__file__, 'torch_git_version': torch.version.git_version,
            'torchvision_extension_sha256': sha256(_get_extension_path('_C')),
            'torchvision_wheel_metadata': metadata.distribution('torchvision').read_text('WHEEL'),
            'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'vision_source': vision_source, 'package_source_correspondence': correspondence,
            'augmentation_source': checked_source(AUGMENT, 'augmentation')}


def run_identity(manifest, run_uuid, cfg, env, scope='FORMAL', require_clean=True):
    if require_clean and git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Formal training requires a committed clean worktree')
    return {'model': 'Faster R-CNN (ResNet-50-FPN)', 'initialization_type': 'random',
            'pretraining_source': None, 'pretrained_tensors_loaded': 0, 'run_uuid': run_uuid,
            'scope': scope, 'model_code_sha': git('rev-parse', 'HEAD'), 'code_sha256': code_identity(),
            'dataset_identity_sha256': manifest['dataset_identity_sha256'],
            'recipe_sha256': digest(canonical(cfg)), 'recipe': cfg,
            'torch': env['versions']['torch'], 'torchvision': env['versions']['torchvision'],
            'runtime_abi_sha256': digest(canonical({k: env[k] for k in ('versions', 'python_version',
                'torch_git_version', 'cuda_build', 'cudnn', 'torchvision_extension_sha256', 'package_source_correspondence')})),
            'vision_source': env['vision_source'], 'augmentation_source': env['augmentation_source'],
            'upstream_lock_sha256': text_hash(LOCK)}


@contextmanager
def local_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if not path.stat().st_size:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('This experiment run/cache is already in use: ' + str(path)) from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def status(run, stage, state, **values):
    write_json(Path(run)/(stage+'_status.json'), {'stage': stage, 'status': state, **values})
