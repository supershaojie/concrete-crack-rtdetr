"""Atomic artifacts, actual environment identity, and run-local locks."""
from __future__ import annotations
from contextlib import contextmanager
import importlib.metadata as metadata
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
NAME = 'fasterrcnn-r50-fpn-configurable'
RUNTIME = ROOT / '.runtime' / NAME
ENV = ROOT / '.envs' / NAME
LOCK = HERE / 'upstream.lock.json'
RUN_ROOT = ROOT / 'outputs' / NAME

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.atomic-' + uuid.uuid4().hex + '.partial')
    with tmp.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def write_json(path, value):
    atomic_bytes(path, canonical(value))

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-c', 'safe.directory='+Path(cwd).absolute().as_posix(),
        *args], cwd=cwd, text=True, encoding='utf-8').strip()

def text_hash(path):
    return digest(Path(path).read_bytes().replace(b'\r\n', b'\n'))

def code_identity():
    paths = [p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.txt','.md')]
    paths += [ROOT/'scripts/autodl_fasterrcnn_r50_fpn_configurable.sh', COMMON/'common.py',
              COMMON/'dataset.py', COMMON/'evaluation/evaluate.py', COMMON/'evaluation/native_metrics.py',
              ROOT/'docs/comparison/evidence/yolov8m_delivery_validation.json']
    return digest(canonical({p.relative_to(ROOT).as_posix(): text_hash(p) for p in sorted(paths)}))

def recipe():
    from configuration import resolve_config
    return resolve_config()

def validate_recipe(cfg):
    from configuration import validate_config
    validate_config(cfg)

def installed_version(name):
    try: return metadata.version(name)
    except metadata.PackageNotFoundError: return None

def environment(strict=True):
    import torch, torchvision, numpy, cv2
    from torchvision._internally_replaced_utils import _get_extension_path
    target = (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0]) == ('2.1.2','0.16.2')
    if strict and (not target or numpy.__version__ != '1.26.4'):
        raise ValueError('Formal runs require torch2.1.2 / torchvision0.16.2 / numpy1.26.4; bootstrap an isolated environment')
    installed = Path(torchvision.__file__).parent
    source_files = [installed/'models/detection/faster_rcnn.py', installed/'models/detection/backbone_utils.py',
                    installed/'models/detection/roi_heads.py', installed/'models/detection/rpn.py',
                    installed/'models/detection/transform.py']
    return {'python': os.path.abspath(sys.executable), 'prefix':sys.prefix, 'base_prefix':sys.base_prefix,
        'python_version':sys.version, 'target_version':target,
        'versions':{n:installed_version(n) for n in ('torch','torchvision','numpy','opencv-python','pillow','pyyaml','scipy','tqdm','matplotlib','pycocotools')},
        'torch_file':torch.__file__, 'torchvision_file':torchvision.__file__, 'opencv_file':cv2.__file__,
        'torch_git_version':torch.version.git_version, 'vision_source':read_json(LOCK)['vision'],
        'wheel_git_version':getattr(torchvision.version,'git_version',None),
        'torchvision_extension_path':_get_extension_path('_C'),
        'torchvision_extension_sha256':sha256(_get_extension_path('_C')),
        'torchvision_wheel_metadata':metadata.distribution('torchvision').read_text('WHEEL'),
        'installed_source_sha256':{str(p.relative_to(installed)):text_hash(p) for p in source_files},
        'torch_build':torch.__config__.show(), 'cuda_build':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
        'cuda_available':torch.cuda.is_available(),
        'devices':[{'index':i,'name':torch.cuda.get_device_name(i),'capability':list(torch.cuda.get_device_capability(i)),
                    'total_memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]}

def run_identity(manifest, run_uuid, cfg, env, initialization, scope='FORMAL', require_clean=True):
    if require_clean and git('status','--porcelain','--untracked-files=normal'):
        raise ValueError('Formal training requires a committed clean worktree')
    return {'model':'Faster R-CNN (ResNet-50-FPN)', 'initialization_type':cfg['initialization'],
        'initialization_identity':initialization, 'run_uuid':run_uuid, 'scope':scope,
        'model_code_sha':git('rev-parse','HEAD'), 'code_sha256':code_identity(),
        'dataset_identity_sha256':manifest['dataset_identity_sha256'], 'recipe_sha256':digest(canonical(cfg)),
        'recipe':cfg, 'torch':env['versions']['torch'], 'torchvision':env['versions']['torchvision'],
        'runtime_abi_sha256':digest(canonical(env)), 'vision_source':env['vision_source'],
        'upstream_lock_sha256':text_hash(LOCK)}

@contextmanager
def local_lock(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if not path.stat().st_size:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('This run/cache is already active: '+str(path)) from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def status(run, stage, state, **values):
    write_json(Path(run)/(stage+'_status.json'),{'stage':stage,'status':state,**values})
