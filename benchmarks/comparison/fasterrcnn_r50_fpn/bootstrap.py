"""Private --copies venv; pinned official sources/wheels; never install into the base."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from support import (AUGMENT, ENV, HERE, LOCK, RUNTIME, VISION, atomic_bytes, checked_source,
                     environment, local_lock, read_json, write_json)


def call(argv, **kwargs):
    argv = [str(v) for v in argv]
    print('EXEC ' + json.dumps(argv), flush=True)
    return subprocess.run(argv, check=True, **kwargs)


def absolute_python(path):
    # Never Path.resolve(): Linux venv python may otherwise become the base interpreter.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def assert_interpreter(python, prefix):
    code = 'import sys,json,os;print(json.dumps(dict(executable=os.path.abspath(sys.executable),prefix=sys.prefix,base_prefix=sys.base_prefix)))'
    observed = json.loads(subprocess.check_output([str(python), '-c', code], text=True))
    if os.path.normcase(observed['executable']) != os.path.normcase(str(absolute_python(python))):
        raise ValueError('Subprocess interpreter path differs from recorded private Python')
    if os.path.normcase(os.path.abspath(observed['prefix'])) != os.path.normcase(str(absolute_python(prefix))):
        raise ValueError('Private sys.prefix mismatch')
    if observed['prefix'] == observed['base_prefix']:
        raise ValueError('Refusing a base environment interpreter')
    return observed


def bootstrap(base_python):
    base_python = absolute_python(base_python)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with local_lock(RUNTIME/'bootstrap.lock'):
        attempt = RUNTIME/'bootstrap_attempts'/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        base_version = json.loads(subprocess.check_output([str(base_python), '-c',
            'import sys,json;print(json.dumps(list(sys.version_info[:3])))'], text=True))
        if base_version[0] != 3 or not 9 <= base_version[1] <= 11:
            raise ValueError('Pinned wheels require Python 3.9-3.11; choose a compatible --base-python, '
                             'without changing the running rtdetr environment')
        lock = read_json(LOCK)
        for path, key in ((VISION, 'vision'), (AUGMENT, 'augmentation')):
            spec = lock[key]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                call(['git', '-c', 'core.autocrlf=false', 'clone', '--depth', '1', '--single-branch',
                      '--branch', spec['tag'], spec['repository'], path])
            checked_source(path, key)  # existing wrong/dirty/partial directories are never reset
        code = ('import torch,torchvision,json;print(json.dumps(dict(torch=torch.__version__,'
                'vision=torchvision.__version__,cuda=torch.version.cuda)))')
        base = subprocess.run([str(base_python), '-c', code], capture_output=True, text=True)
        write_json(attempt/'base_probe.json', {'exit_code': base.returncode, 'stdout': base.stdout,
                                             'stderr': base.stderr, 'base_python': str(base_python)})
        try:
            versions = json.loads(base.stdout) if base.returncode == 0 else {}
        except ValueError:
            versions = {}
        compatible = (str(versions.get('torch', '')).split('+')[0] == '2.1.2' and
                      str(versions.get('vision', '')).split('+')[0] == '0.16.2' and
                      versions.get('cuda') in ('11.8', '12.1'))
        python = absolute_python(ENV/('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
        if not ENV.exists():
            argv = [base_python, '-m', 'venv', '--copies']
            if compatible:
                argv += ['--system-site-packages']
            call([*argv, ENV])
            write_json(RUNTIME/'venv_origin.json', {'base': str(base_python), 'inherit_readonly': compatible,
                                                  'prefix': str(ENV), 'copies': True})
        origin = read_json(RUNTIME/'venv_origin.json')
        if origin['prefix'] != str(ENV):
            raise ValueError('Existing private environment identity mismatch')
        interpreter = assert_interpreter(python, ENV)
        if not origin['inherit_readonly']:
            current = subprocess.run([str(python), '-c', code], capture_output=True, text=True)
            observed = json.loads(current.stdout) if current.returncode == 0 else {}
            if observed.get('torch') != '2.1.2+cu118' or observed.get('vision') != '0.16.2+cu118':
                call([python, '-m', 'pip', 'install', '--disable-pip-version-check',
                    'torch==2.1.2', 'torchvision==0.16.2', '--index-url', lock['torch']['wheel_index'],
                    '--report', attempt/'torch_install.json'])
        # Explicit pins; pip writes only this venv. No Ultralytics install or weight downloads.
        call([python, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', HERE/'requirements.txt',
              '--report', attempt/'dependencies_install.json'])
        call([python, HERE/'bootstrap.py', '--probe', '--output', attempt/'environment.json'])
        (attempt/'pip_freeze.txt').write_text(subprocess.check_output(
            [str(python), '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
        # Executed in its own process before any formal model or RNG state exists.
        call([python, HERE/'run.py', 'check', '--output', attempt/'model_probe.json'])
        write_json(attempt/'bootstrap.json', {'status': 'PINNED_ENVIRONMENT_AND_ISOLATED_PROBE_PASSED',
            'python': str(python), 'interpreter': interpreter, 'environment': read_json(attempt/'environment.json'),
            'base_unchanged': True, 'formal_capacity_batch16_640': 'NOT_TESTED', 'weights_downloaded': False})
        atomic_bytes(RUNTIME/'python_path.txt', (str(python)+'\n').encode())
        atomic_bytes(RUNTIME/'latest_bootstrap.txt', (str(attempt)+'\n').encode())
        print('BOOTSTRAP PASSED: '+str(python), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-python', type=Path, default=Path(sys.executable))
    p.add_argument('--probe', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.probe:
        report = environment(strict=True)
        if a.output is None:
            p.error('--probe requires --output')
        write_json(a.output, report)
    else:
        bootstrap(a.base_python)


if __name__ == '__main__':
    main()
