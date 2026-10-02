"""Fetch pinned official assets and prepare a private environment; never train."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import uuid

from support import (ASSET, HERE, LOCK, ROOT, SOURCE, checked_source, checked_weight,
                     git, read_json, sha256, source_hash, write_json)


def environment_probe():
    import torch
    import torchvision
    import numpy as np
    import cv2
    import pandas, seaborn, thop, yaml, PIL, scipy, matplotlib, psutil, requests, tqdm, cpuinfo
    from packaging.version import Version
    if Version(np.__version__) != Version('1.26.4'):
        raise RuntimeError('This experiment freezes NumPy 1.26.4 (v8.3.20 itself allows >=1.23)')
    if Version(torch.__version__.split('+')[0]) < Version('2.1'):
        raise RuntimeError('This checkpoint-loading compatibility patch is verified with Torch >=2.1')
    torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.]]), torch.tensor([0.5]), 0.7)
    versions = {n: metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'opencv-python',
        'pandas', 'seaborn', 'ultralytics-thop', 'pillow', 'scipy', 'matplotlib', 'pyyaml')}
    return {'python': sys.executable, 'python_version': sys.version, 'versions': versions,
            'torch_cuda_build': torch.version.cuda, 'cpu_nms': 'passed'}


def call(args, **kwargs):
    print('EXEC ' + json.dumps([str(v) for v in args]), flush=True)
    subprocess.run([str(v) for v in args], check=True, **kwargs)


def bootstrap(args):
    lock = read_json(LOCK)
    patch = HERE / lock['patch']
    if sha256(patch) != lock['patch_sha256']:
        raise ValueError('Committed patch hash differs')
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        call(['git', 'clone', '--branch', lock['tag'], '--single-branch', '--depth', '1', lock['repository'], SOURCE])
    if git('rev-parse', 'HEAD', cwd=SOURCE) != lock['commit']:
        raise ValueError('Existing upstream tree is not the pinned commit; refusing to reset it')
    if source_hash(SOURCE) != lock['patched_source_sha256']:
        if git('status', '--porcelain', cwd=SOURCE):
            raise ValueError('Upstream has unexpected changes; refusing to overwrite')
        call(['git', '-C', SOURCE, 'apply', '--check', patch])
        call(['git', '-C', SOURCE, 'apply', patch])
    checked_source()
    if not ASSET.exists():
        ASSET.parent.mkdir(parents=True, exist_ok=True)
        temporary = ASSET.with_suffix('.download')
        if temporary.exists():
            raise FileExistsError('Incomplete prior download retained: ' + str(temporary))
        print('Download official frozen URL: ' + lock['weights']['url'], flush=True)
        urllib.request.urlretrieve(lock['weights']['url'], temporary)
        checked_weight(temporary)
        temporary.rename(ASSET)
    checked_weight()
    state = ROOT / '.runtime/yolov8m'
    probe = [args.base_python, HERE / 'bootstrap.py', '--probe']
    base = subprocess.run([str(x) for x in probe], capture_output=True, text=True)
    write_json(state / 'base_environment_probe.json', {'exit_code': base.returncode,
               'stdout': base.stdout, 'stderr': base.stderr, 'base_python': str(args.base_python)})
    if args.reuse_base:
        if base.returncode:
            raise RuntimeError('Base dependencies incompatible; use default private venv (no base package changes)')
        python = Path(args.base_python).resolve()
    else:
        env = ROOT / '.envs/yolov8m'
        python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not python.exists():
            call([args.base_python, '-m', 'venv', '--system-site-packages', env])
        # Probe exact overlay versions before installing. --no-deps cannot upgrade inherited Torch.
        code = ('import importlib.metadata as m,json; '
                'names={d.metadata["Name"].lower() for d in m.distributions()}; '
                'print(json.dumps({n:m.version(n) for n in names}))')
        installed = json.loads(subprocess.check_output([str(python), '-c', code], text=True))
        needed = []
        for line in (HERE / 'requirements-overlay.txt').read_text().splitlines():
            if line and not line.startswith('#'):
                name, version = line.split('==')
                if installed.get(name.lower()) != version:
                    needed.append(line)
        if needed:
            call([python, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps', *needed])
    result = subprocess.run([str(python), str(HERE / 'bootstrap.py'), '--probe'],
                            capture_output=True, text=True)
    write_json(state / 'environment_probe.json', {'exit_code': result.returncode,
               'stdout': result.stdout, 'stderr': result.stderr})
    if result.returncode:
        raise RuntimeError('Isolated dependency probe failed; inspect .runtime/yolov8m/environment_probe.json: ' + result.stderr)
    (state / 'python_path.txt').write_text(str(python.resolve()) + '\n', encoding='utf-8')
    check_output = state / ('check_' + uuid.uuid4().hex[:8] + '.json')
    call([python, HERE / 'run.py', 'check', '--output', check_output])
    write_json(state / 'bootstrap.json', {'status': 'ASSETS_AND_IMPORT_VERIFIED_NO_TRAINING',
        'upstream': lock, 'python': str(python.resolve()), 'environment': json.loads(result.stdout),
        'check_report': str(check_output)})
    print('BOOTSTRAP COMPLETE. Python: ' + str(python.resolve()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-python', type=Path, default=Path(sys.executable))
    p.add_argument('--reuse-base', action='store_true', help='Read-only reuse after successful dependency probe; never pip into it')
    p.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.probe:
        print(json.dumps(environment_probe()))
    else:
        bootstrap(args)


if __name__ == '__main__':
    main()
