"""Shared read-only source identities, atomic artifacts and per-run locks."""
from __future__ import annotations
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMON = HERE.parent
sys.path.append(str(COMMON))
from common import canonical, digest, load_yaml, sha256
SOURCE = ROOT / '.vendor/dfine-m-configurable/D-FINE'
ASSET = ROOT / '.runtime/dfine-m-configurable/assets/dfine_m_coco.pth'
LOCK = HERE / 'upstream.lock.json'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic_bytes(path, raw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / ('.tmp_' + uuid.uuid4().hex)
    try:
        with temp.open('xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()

def write_json(path, obj):
    atomic_bytes(path, canonical(obj))

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-c', 'safe.directory='+Path(cwd).resolve().as_posix(), *args],
        cwd=str(cwd), text=True, encoding='utf-8').strip()

def adapter_hash():
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p.suffix in {'.py','.yaml','.json','.txt'})
    paths += [ROOT/'scripts/autodl_dfine_m_configurable.sh']
    return digest(canonical({str(p.relative_to(ROOT).as_posix()):digest(p.read_bytes().replace(b'\r\n',b'\n'))
                             for p in paths if p.exists()}))

def checked_source(source=SOURCE):
    source=Path(source).resolve(); lock=read_json(LOCK)
    if git('rev-parse','HEAD',cwd=source)!=lock['commit']:
        raise ValueError('Official D-FINE source commit differs')
    for name,expected in lock['files_lf_sha256'].items():
        if digest((source/name).read_bytes().replace(b'\r\n',b'\n'))!=expected:
            raise ValueError('Locked official file changed: '+name)
    if git('status','--porcelain','--untracked-files=normal',cwd=source):
        raise ValueError('Official source checkout must be unchanged and clean')
    return source,lock

def checked_weight(path=ASSET):
    path=Path(path).resolve(); expected=read_json(LOCK)['weights']
    if any(x in path.name.lower() for x in ('obj365','obj2coco','objects365')):
        raise ValueError('Only the locked COCO-only M asset is allowed')
    if path.stat().st_size!=expected['bytes'] or sha256(path)!=expected['sha256']:
        raise ValueError('Locked COCO-only M weight checksum differs: '+str(path))
    return path

@contextmanager
def local_lock(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size==0: stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e: raise RuntimeError('This D-FINE run/cache is already active: '+str(path)) from e
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

def evaluator_api():
    name='_dfine_public_evaluate'
    if name not in sys.modules:
        sys.path.insert(0,str(COMMON/'evaluation'))
        spec=importlib.util.spec_from_file_location(name,COMMON/'evaluation/evaluate.py')
        mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    return sys.modules[name]

def environment():
    import importlib.metadata as metadata
    import platform
    import torch, torchvision, numpy, PIL, cv2, scipy, yaml
    from torchvision.ops import box_convert
    probe=box_convert(torch.tensor([[.5,.5,.2,.2]]), 'cxcywh','xyxy')
    if not torch.isfinite(probe).all(): raise RuntimeError('Torchvision box operator failed')
    packages={n:metadata.version(n) for n in ('torch','torchvision','numpy','Pillow','scipy','PyYAML','matplotlib')}
    packages['opencv']=cv2.__version__
    try: packages['faster-coco-eval']=metadata.version('faster-coco-eval')
    except metadata.PackageNotFoundError: packages['faster-coco-eval']=None
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    return {'python':os.path.abspath(sys.executable),'sys_prefix':os.path.abspath(sys.prefix),
        'python_version':platform.python_version(),'platform':platform.platform(),'packages':packages,
        'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'import_paths':{'torch':torch.__file__,'torchvision':torchvision.__file__},
        'pip_freeze_sha256':digest('\n'.join(sorted(freeze.splitlines())).encode())}

def status(run, stage, state, **kwargs):
    from datetime import datetime,timezone
    write_json(Path(run)/(stage+'_status.json'),{'stage':stage,'status':state,'pid':os.getpid(),
        'updated_utc':datetime.now(timezone.utc).isoformat(),**kwargs})
