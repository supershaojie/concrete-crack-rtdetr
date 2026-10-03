"""Utilities shared only by this adapter; never imports a detector."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
COMMON = HERE.parent
sys.path.append(str(COMMON))
from common import canonical, digest, load_yaml, sha256, write_json

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def git(*args, cwd=PROJECT):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + str(Path(cwd).resolve()),
                                    '-C', str(cwd), *args], text=True).strip()

def initialization_type():
    mode = read_json(HERE/'recipe.json')['initialization_type']
    if mode not in ('random', 'coco_detection_pretrained'):
        raise ValueError('Unknown initialization mode: ' + mode)
    return mode

def initialization_record(upstream):
    mode = initialization_type()
    yaml = Path(upstream)/'models/yolov5m.yaml'
    return {'initialization_type':mode, 'pretraining_source':None if mode=='random' else 'COCO',
            'pretrained_tensors_loaded':0 if mode=='random' else 475,
            'coco_source_sha256':None if mode=='random' else read_json(HERE/'upstream.lock.json')['weights']['sha256'],
            'b19_recipe_sha256':sha256(HERE/'hyp.yaml'),
            'b19_source_sha256':sha256(HERE/'b19_source.json'),
            'model_yaml':str(yaml), 'model_yaml_sha256':sha256(yaml),
            'scale':'m', 'nc':1, 'seed':42,
            'native_constants':'BN, Detect bias priors and anchors retain official initialization'}

def identity():
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p.suffix in {'.py', '.yaml', '.json', '.patch', '.txt'})
    return {'project_commit': git('rev-parse', 'HEAD'),
            'adapter_sha256': digest(canonical({p.relative_to(HERE).as_posix(): sha256(p) for p in paths})),
            'recipe_sha256': sha256(HERE/'recipe.json'), 'hyp_sha256': sha256(HERE/'hyp.yaml'),
            'patch_sha256': sha256(HERE/'upstream.patch')}

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(canonical(value))
    os.replace(tmp, path)

def environment():
    from importlib.metadata import version, distributions, PackageNotFoundError
    from importlib.util import find_spec
    names = ['torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python', 'PyYAML', 'scipy',
             'pandas', 'seaborn', 'matplotlib', 'tqdm', 'thop', 'tensorboard', 'GitPython']
    result = {'python': sys.version, 'executable': sys.executable, 'prefix':sys.prefix,
              'base_prefix':sys.base_prefix, 'executable_realpath':str(Path(sys.executable).resolve())}
    result['all_distributions'] = sorted([d.metadata['Name'],d.version] for d in distributions() if d.metadata['Name'])
    result['package_import_paths']={name:find_spec(name).origin if find_spec(name) else None
        for name in ('torch','torchvision','numpy','PIL','cv2','yaml','pandas','scipy','matplotlib','IPython')}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'NOT_INSTALLED'
    return result
