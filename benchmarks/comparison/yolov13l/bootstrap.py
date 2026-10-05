"""Fetch one official commit and prepare only this experiment's environment; never train."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import shutil
import urllib.request
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
from support import (HERE, ROOT, LOCK, SOURCE, ASSET, checked_source, checked_weight,
                     initialization_type, git, read_json, sha256, source_hash, write_json, local_lock)


def executable_path(value):
    # Do not resolve(): Linux venv symlinks can resolve back into the base environment.
    if not os.path.dirname(str(value)) and shutil.which(str(value)):
        value = shutil.which(str(value))
    return Path(os.path.abspath(os.path.expanduser(str(value))))


def environment_probe():
    import torch, torchvision, numpy as np
    import cv2, pandas, seaborn, thop, yaml, PIL, scipy, matplotlib, psutil, requests, tqdm, cpuinfo
    import huggingface_hub, safetensors
    from packaging.version import Version
    if np.__version__ != '1.26.4':
        raise RuntimeError('Frozen NumPy must be 1.26.4')
    pair = (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0])
    if pair not in {('2.1.2','0.16.2'),('2.2.2','0.17.2'),('2.7.1','0.22.1')}:
        raise RuntimeError('Unverified Torch/TorchVision pair '+str(pair)+'. Use --fresh-torch in a NEW private env')
    torchvision.ops.nms(torch.tensor([[0.,0.,2.,2.]]),torch.tensor([.5]),.7)
    versions = {n:metadata.version(n) for n in ('torch','torchvision','numpy','opencv-python','pandas',
                'seaborn','ultralytics-thop','pillow','scipy','matplotlib','pyyaml','huggingface-hub','safetensors')}
    driver = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],
                            capture_output=True,text=True) if __import__('shutil').which('nvidia-smi') else None
    return {'python':str(executable_path(sys.executable)), 'sys_prefix':sys.prefix,'sys_base_prefix':sys.base_prefix,
            'python_version':sys.version, 'versions':versions,'torch_cuda_build':torch.version.cuda,
            'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            'driver':driver.stdout.strip() if driver else None,'cpu_nms':'passed',
            'backend':'native','flash_parity':'NOT_VERIFIED; native backend frozen; no Flash wheel installed by bootstrap'}


def verify_frozen_environment(environment=None):
    environment=environment or environment_probe()
    frozen=read_json(ROOT/'.runtime/yolov13l-configurable/environment_identity.json')
    from support import digest
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    actual={'python':str(executable_path(sys.executable)),'sys_prefix':environment['sys_prefix'],
            'python_version':environment['python_version'],'versions':environment['versions'],
            'torch_cuda_build':environment['torch_cuda_build'],
            'pip_freeze_sha256':digest(freeze.replace('\r\n','\n').encode()),'backend':'native'}
    if actual!=frozen:
        raise RuntimeError('Frozen private environment differs; inspect instead of changing an active run')
    return actual


def call(args, **kwargs):
    print('EXEC '+json.dumps([str(x) for x in args]),flush=True)
    subprocess.run([str(x) for x in args],check=True,**kwargs)


def prepare_source(reuse=None):
    lock=read_json(LOCK)
    patch=HERE/lock['patch']
    if sha256(patch)!=lock['patch_sha256']:
        raise ValueError('Committed patch hash differs')
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True,exist_ok=True)
        if reuse:
            checked_source(reuse)
            call(['git', '-c', 'safe.directory='+Path(reuse).absolute().as_posix(),
                  '-c','safe.directory='+(Path(reuse).absolute()/'.git').as_posix(),
                  'clone', '--no-hardlinks', '--no-checkout', Path(reuse).absolute().as_posix(), SOURCE])
            call(['git','-C',SOURCE,'remote','set-url','origin',lock['repository']])
            call(['git','-C',SOURCE,'config','core.autocrlf','false'])
            call(['git','-C',SOURCE,'checkout','--detach',lock['commit']])
        else:
            call(['git','init',SOURCE])
            call(['git','-C',SOURCE,'remote','add','origin',lock['repository']])
        call(['git','-C',SOURCE,'config','core.autocrlf','false'])
    if git('remote','get-url','origin',cwd=SOURCE)!=lock['repository']:
        raise ValueError('Existing vendor remote differs')
    head=subprocess.run(['git','-C',str(SOURCE),'rev-parse','--verify','HEAD'],capture_output=True,text=True)
    if head.returncode:
        if git('ls-files',cwd=SOURCE) or git('ls-files','--others','--exclude-standard',cwd=SOURCE):
            raise ValueError('Uninitialized vendor directory contains files; inspect it manually')
        call(['git','-C',SOURCE,'fetch','--depth','1','origin',lock['commit']])
        call(['git','-C',SOURCE,'checkout','--detach',lock['commit']])
    if git('rev-parse','HEAD',cwd=SOURCE)!=lock['commit']:
        raise ValueError('Vendor is not pinned commit; refusing reset')
    if source_hash(SOURCE)!=lock['patched_source_sha256']:
        if git('status','--porcelain',cwd=SOURCE):
            raise ValueError('Unexpected vendor changes; refusing overwrite')
        call(['git','-C',SOURCE,'apply','--check',patch])
        call(['git','-C',SOURCE,'apply',patch])
    checked_source()
    return lock


def prepare_asset(reuse=None):
    initialization_type()
    if ASSET.exists():
        return checked_weight()
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    temporary = ASSET.with_suffix('.download')
    if temporary.exists():
        raise FileExistsError('Incomplete prior asset download retained: '+str(temporary))
    lock = read_json(LOCK)['weights']
    if reuse:
        original = checked_weight(reuse)
        with original.open('rb') as src, temporary.open('xb') as dst:
            shutil.copyfileobj(src, dst)
    else:
        print('Download verified official COCO URL: '+lock['url'], flush=True)
        request = urllib.request.Request(lock['url'], headers={'User-Agent':'Crack-RTDETR-YOLOv13L'})
        with urllib.request.urlopen(request, timeout=60) as src, temporary.open('xb') as dst:
            size = 0
            while True:
                block = src.read(4*1024*1024)
                if not block:
                    break
                dst.write(block)
                size += len(block)
                print(f'COCO asset: {size}/{lock["bytes"]} bytes',flush=True)
    checked_weight(temporary)  # identity is checked before any torch deserialization
    temporary.rename(ASSET)
    return checked_weight()


def bootstrap(args):
    state=ROOT/'.runtime/yolov13l-configurable';state.mkdir(parents=True,exist_ok=True)
    with local_lock(state/'.bootstrap.lock'):
        lock=prepare_source(args.reuse_source)
        prepare_asset(args.reuse_asset)
        base=executable_path(args.base_python)
        probe=subprocess.run([str(base),str(HERE/'bootstrap.py'),'--probe'],capture_output=True,text=True)
        write_json(state/'base_environment_probe.json',{'exit_code':probe.returncode,'stdout':probe.stdout,'stderr':probe.stderr,'base_python':str(base)})
        frozen=state/'environment_identity.json'
        env=executable_path(args.env_dir) if args.env_dir else (
            Path(read_json(frozen)['sys_prefix']) if frozen.exists() else ROOT/'.envs/yolov13l-configurable')
        if not env.resolve().is_relative_to((ROOT/'.envs').resolve()):
            raise ValueError('Private environment must stay within this worktree .envs directory')
        python=executable_path(env/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))
        frozen=state/'environment_identity.json'
        if frozen.exists() and not python.is_file():
            raise RuntimeError('Frozen environment disappeared; inspect before replacing')
        if not python.is_file():
            options=[] if args.fresh_torch else ['--system-site-packages']
            call([base,'-m','venv','--copies',*options,env])
        # Verify the actual prefix before ANY pip mutation.
        raw=subprocess.check_output([str(python),'-c','import sys,json; print(json.dumps([sys.executable,sys.prefix,sys.base_prefix]))'],text=True)
        observed, prefix, base_prefix=json.loads(raw)
        if executable_path(prefix)!=executable_path(env) or executable_path(observed)!=python or prefix==base_prefix:
            raise RuntimeError('Private interpreter/prefix mismatch; base environment is protected')
        pycfg=(env/'pyvenv.cfg').read_text().lower()
        if args.fresh_torch and 'include-system-site-packages = true' in pycfg:
            raise RuntimeError('--fresh-torch requires a NEW environment without inherited packages; do not delete a used env')
        if not frozen.exists():
            if args.fresh_torch:
                call([python,'-m','pip','install','--disable-pip-version-check','torch==2.2.2','torchvision==0.17.2',
                      '--index-url','https://download.pytorch.org/whl/cu121'])
                call([python,'-m','pip','install','--disable-pip-version-check','-r',HERE/'requirements-runtime.txt'])
            code='import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"].lower():d.version for d in m.distributions()}))'
            # metadata.version respects venv precedence; duplicate inherited names must not overwrite it.
            installed=json.loads(subprocess.check_output([str(python),'-c',
                'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in {d.metadata["Name"].lower() for d in m.distributions()}}))'],text=True))
            needed=[]
            for line in (HERE/'requirements-overlay.txt').read_text().splitlines():
                if line and not line.startswith('#'):
                    name,version=line.split('==')
                    if installed.get(name.lower())!=version: needed.append(line)
            if needed: call([python,'-m','pip','install','--disable-pip-version-check','--no-deps',*needed])
        result=subprocess.run([str(python),str(HERE/'bootstrap.py'),'--probe'],capture_output=True,text=True)
        write_json(state/'environment_probe.json',{'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:
            raise RuntimeError('Private probe failed; no base packages changed. Inspect environment_probe.json. '+result.stderr)
        environment=json.loads(result.stdout)
        freeze=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
        from support import digest
        identity={'python':str(python),'sys_prefix':environment['sys_prefix'],'python_version':environment['python_version'],
                  'versions':environment['versions'],'torch_cuda_build':environment['torch_cuda_build'],
                  'pip_freeze_sha256':digest(freeze.replace('\r\n','\n').encode()),'backend':'native'}
        if frozen.exists() and read_json(frozen)!=identity:
            raise RuntimeError('Frozen environment changed; no automatic reinstall or backend switch')
        check=state/('check_'+uuid.uuid4().hex[:8]+'.json')
        call([python,HERE/'run.py','check','--output',check])
        write_json(frozen,identity)
        (state/'pip_freeze.txt').write_text(freeze,encoding='utf-8')
        (state/'python_path.txt').write_text(str(python)+'\n',encoding='utf-8')
        write_json(state/'bootstrap.json',{'status':'PINNED_SOURCE_ENV_MODEL_VERIFIED_NO_TRAINING',
                   'upstream':lock,'python':str(python),'environment':environment,'check_report':str(check)})
        print('BOOTSTRAP COMPLETE. Python: '+str(python),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-python',type=Path,default=Path(sys.executable))
    p.add_argument('--env-dir',type=Path,help='Optional new environment within this worktree .envs; preserves failed environments')
    p.add_argument('--fresh-torch',action='store_true',help='NEW copies venv, torch2.2.2/vision0.17.2 cu121; never installs into base')
    p.add_argument('--reuse-source',type=Path,help='Read-only verified pinned iMoonLab tree; clone into this worktree')
    p.add_argument('--reuse-asset',type=Path,help='Read-only original official yolov13l.pt; copy after SHA256 verification')
    p.add_argument('--probe',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.probe: print(json.dumps(environment_probe()))
    else: bootstrap(args)


if __name__=='__main__': main()
