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
    return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()

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
    names = ['torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python', 'PyYAML', 'scipy',
             'pandas', 'seaborn', 'matplotlib', 'tqdm', 'thop', 'tensorboard', 'GitPython']
    result = {'python': sys.version, 'executable': sys.executable}
    result['all_distributions'] = sorted([d.metadata['Name'],d.version] for d in distributions() if d.metadata['Name'])
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'NOT_INSTALLED'
    return result
