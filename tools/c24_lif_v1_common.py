"""Experiment identity and small atomic evidence helpers; standard library only."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ.get('C24_LIF_V1_MAIN','/root/autodl-tmp/projects/Crack_RTDETR'))
EXPERIMENT = 'c24_lif_v1'
BRANCH = 'codex/rtdetr-c24-lif-v1'
NAME = 'c24_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug'
SESSION = 'c24_lif_v1-training'
MODEL = 'rtdetr-resnet18-lite-scca-lif-down.yaml'
BASE = '0e95bbade3558b0d2b77c5531483c60810391d88'
SOURCE_SHA256 = 'fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e'

def require(value, message):
    if not value: raise RuntimeError(message)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    tmp.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n').encode())
    os.replace(tmp,path)

def read_json(path,default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).is_file() else default

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT).decode().strip()

def paths():
    return dict(run=MAIN/'runs/c_series'/NAME,launch=ROOT/'outputs'/EXPERIMENT,
        init=ROOT/'weights/c24_lif_v1_controlled_init.pt',source=MAIN/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt',
        c2_args=MAIN/'runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml',
        data=MAIN/'configs/crack_autodl.yaml',download=MAIN/'downloads'/EXPERIMENT)

def fingerprint():
    files=['ultralytics-main/ultralytics/nn/'+p for p in ['tasks.py','modules/__init__.py','modules/scca_aifi.py',
        'modules/lif_down.py','modules/head.py','modules/transformer.py']]
    files+=['ultralytics-main/ultralytics/cfg/models/rt-detr/'+MODEL,'docs/c24_lif_v1/c2_args.yaml','docs/c24_lif_v1/c2_data.yaml']
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'tools').glob('*c24_lif_v1*')) if p.is_file()]
    # LF canonicalization is used for cross-platform code/config identity only, never module provenance.
    return {n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in files}
