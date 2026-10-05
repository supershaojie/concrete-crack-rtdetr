"""Fetch pinned official assets and prepare a private environment; never train."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import urllib.request
import uuid

# Probes/venv subprocesses may read another experiment's packages, never write its pyc caches.
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

from support import (ASSET, HERE, LOCK, ROOT, SOURCE, checked_source, checked_weight,
                     git, read_json, sha256, source_hash, write_json, initialization_type)


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
    return {'python': sys.executable, 'sys_prefix':sys.prefix, 'sys_base_prefix':sys.base_prefix,
            'python_version': sys.version, 'versions': versions,
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
        if args.reuse_source:
            checked_source(args.reuse_source)
            call(['git', '-c', 'safe.directory=' + args.reuse_source.absolute().as_posix(),
                  'clone', '--no-hardlinks', '--no-checkout', args.reuse_source, SOURCE])
            call(['git', '-c', 'safe.directory=' + SOURCE.absolute().as_posix(),
                  '-C', SOURCE, 'checkout', '--detach', lock['commit']])
        else:
            call(['git', 'clone', '--branch', lock['tag'], '--single-branch', '--depth', '1', lock['repository'], SOURCE])
    if git('rev-parse', 'HEAD', cwd=SOURCE) != lock['commit']:
        raise ValueError('Existing upstream tree is not the pinned commit; refusing to reset it')
    if source_hash(SOURCE) != lock['patched_source_sha256']:
        if git('status', '--porcelain', cwd=SOURCE):
            raise ValueError('Upstream has unexpected changes; refusing to overwrite')
        call(['git', '-C', SOURCE, 'apply', '--check', patch])
        call(['git', '-C', SOURCE, 'apply', patch])
    checked_source()
    initialization_type()
    if not ASSET.exists():
        ASSET.parent.mkdir(parents=True, exist_ok=True)
        if args.reuse_asset:
            original = checked_weight(args.reuse_asset)
            print('Reusing verified ORIGINAL COCO asset: ' + str(original), flush=True)
            with original.open('rb') as src, ASSET.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        else:
            temporary = ASSET.with_suffix('.download')
            if temporary.exists():
                raise FileExistsError('Incomplete prior download retained: ' + str(temporary))
            print('Download official frozen URL: ' + lock['weights']['url'], flush=True)
            progress = {'last':-1}
            def downloaded(block, size, total):
                received = min(block*size,total) if total > 0 else block*size
                if received-progress['last'] >= 5*1024*1024 or received == total:
                    print(f'COCO asset: {received}/{total} bytes',flush=True)
                    progress['last'] = received
            urllib.request.urlretrieve(lock['weights']['url'], temporary, reporthook=downloaded)
            checked_weight(temporary)
            temporary.rename(ASSET)
    checked_weight()
    state = ROOT / '.runtime/yolo11m-configurable'
    probe = [args.base_python, HERE / 'bootstrap.py', '--probe']
    base = subprocess.run([str(x) for x in probe], capture_output=True, text=True)
    write_json(state / 'base_environment_probe.json', {'exit_code': base.returncode,
               'stdout': base.stdout, 'stderr': base.stderr, 'base_python': str(args.base_python)})
    env = ROOT / '.envs/yolo11m-configurable'
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        call([args.base_python, '-m', 'venv', '--copies', '--system-site-packages', env])
    if python.is_symlink():
        raise ValueError('Pilot venv Python must be a copy, not a link into the base environment')
    actual = json.loads(subprocess.check_output([str(python), '-c',
        'import sys,json; print(json.dumps({"python":sys.executable,"prefix":sys.prefix,"base":sys.base_prefix}))'], text=True))
    if Path(actual['prefix']).resolve() != env.resolve() or Path(actual['prefix']).resolve() == Path(actual['base']).resolve():
        raise ValueError('Refusing pip outside the private pilot environment: ' + str(actual))
    if os.path.normcase(os.path.abspath(actual['python'])) != os.path.normcase(str(python.absolute())):
        raise ValueError('Venv sys.executable does not match the launched interpreter: ' + str(actual))
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
        raise RuntimeError('Isolated dependency probe failed; inspect .runtime/yolo11m-configurable/environment_probe.json: ' + result.stderr)
    (state / 'python_path.txt').write_text(str(python.absolute()) + '\n', encoding='utf-8')
    check_output = state / ('check_' + uuid.uuid4().hex[:8] + '.json')
    call([python, HERE / 'run.py', 'check', '--output', check_output])
    write_json(state / 'bootstrap.json', {'status': 'ASSETS_AND_IMPORT_VERIFIED_NO_TRAINING',
        'upstream': lock, 'python': str(python.absolute()), 'environment': json.loads(result.stdout),
        'check_report': str(check_output)})
    print('BOOTSTRAP COMPLETE. Python: ' + str(python.absolute()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-python', type=Path, default=Path(sys.executable))
    p.add_argument('--reuse-asset', type=Path, help='Read-only verified original official yolo11m.pt, copied into pilot assets')
    p.add_argument('--reuse-source', type=Path, help='Read-only verified pinned upstream; clone into this experiment')
    p.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.probe:
        print(json.dumps(environment_probe()))
    else:
        bootstrap(args)


if __name__ == '__main__':
    main()
