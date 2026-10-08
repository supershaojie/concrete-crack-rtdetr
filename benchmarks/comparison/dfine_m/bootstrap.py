"""Bounded observable source/asset acquisition and read-only environment reuse."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from support import (HERE,LOCK,ROOT,checked_source,checked_weight,environment,read_json,write_json)

def bounded(command,cwd=None,timeout=180):
    for attempt in range(1,4):
        print('Bootstrap attempt '+str(attempt)+': '+' '.join(map(str,command)),flush=True)
        try:
            subprocess.run(command,cwd=cwd,check=True,timeout=timeout)
            return
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
            if attempt==3: raise
            time.sleep(3)

def acquire_source(path):
    path=Path(path).absolute(); lock=read_json(LOCK)
    if path.exists():
        checked_source(path); print('Reuse locked official source: '+str(path),flush=True); return
    path.parent.mkdir(parents=True,exist_ok=True)
    bounded(['git','clone','--no-checkout',lock['repository'],str(path)])
    bounded(['git','-c','core.autocrlf=false','checkout','--detach',lock['commit']],cwd=str(path))
    checked_source(path)

def acquire_weight(path):
    path=Path(path).absolute(); rec=read_json(LOCK)['weights']
    if path.is_symlink(): raise ValueError('Weight cache symlink is not allowed')
    if path.exists(): checked_weight(path); print('Reuse checked COCO weight: '+str(path),flush=True); return
    if path.name!=rec['filename']: raise ValueError('New weight download must retain the locked filename')
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix('.partial')
    for attempt in range(1,4):
        print(f'COCO download attempt {attempt}/3: {rec["url"]}',flush=True)
        try:
            started=time.monotonic(); last=0
            with urllib.request.urlopen(rec['url'],timeout=45) as response,temporary.open('wb') as f:
                size=0
                while True:
                    block=response.read(1024*1024)
                    if not block: break
                    f.write(block); size+=len(block)
                    if size>rec['bytes'] or time.monotonic()-started>600: raise RuntimeError('Bounded download exceeded size/time')
                    if size-last>=5*1024*1024 or size==rec['bytes']:
                        print(f'COCO download {size}/{rec["bytes"]} bytes ({size/rec["bytes"]:.1%})',flush=True); last=size
            checked_weight(temporary)
            temporary.rename(path); return
        except Exception:
            if attempt==3: raise
            time.sleep(3)

def core_compatible():
    try:
        if sys.version_info<(3,9): return False
        import numpy,torch,torchvision
        if tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))<(2,0): return False
        if int(numpy.__version__.split('.')[0])>=2 and torch.__version__.startswith('2.1.'): return False
        inp=torch.randn(1,1,4,4,requires_grad=True); grid=torch.zeros(1,2,2,2)
        torch.nn.functional.grid_sample(inp,grid,align_corners=False).sum().backward()
        from torchvision.ops import box_convert
        box_convert(torch.tensor([[.5,.5,.2,.2]]),'cxcywh','xyxy')
        return True
    except (ImportError,RuntimeError,AttributeError): return False

def missing_extras():
    import importlib
    result=[]
    for module,requirement in [('scipy','scipy==1.11.4'),('PIL','Pillow==10.4.0'),
        ('cv2','opencv-python==4.10.0.84'),('yaml','PyYAML==6.0.2'),('matplotlib','matplotlib==3.8.4')]:
        try: importlib.import_module(module)
        except (ImportError,RuntimeError,AttributeError): result.append(requirement)
    return result

def compatible():
    return core_compatible() and not missing_extras()


def bootstrap(cfg,new_environment=False):
    from configuration import paths
    locations=paths(cfg); acquire_source(locations['source']); acquire_weight(locations['weights'])
    cache=ROOT/'.runtime/dfine-m-configurable'; cache.mkdir(parents=True,exist_ok=True)
    if compatible() and not new_environment:
        interpreter=os.path.abspath(sys.executable); mode='readonly_reuse'
    else:
        overlay=core_compatible() and not new_environment
        if not overlay and not (3,9)<=sys.version_info[:2]<=(3,11):
            raise ValueError('Fresh Torch2.1.2 environment needs Python3.9-3.11 base (reference3.10.14)')
        env=ROOT/('.envs/dfine-m-configurable-overlay' if overlay else '.envs/dfine-m-configurable')
        interpreter=str(env/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))
        if env.exists():
            probe=[interpreter,'-c',"import sys; sys.path.insert(0,sys.argv[1]); from bootstrap import compatible; assert compatible()",str(HERE)]
            bounded(probe,timeout=90); mode='readonly_existing_isolated_environment'
        else:
            command=[sys.executable,'-m','venv','--copies']
            if overlay: command.append('--system-site-packages')
            bounded([*command,str(env)],timeout=120)
            install=[interpreter,'-m','pip','install','--disable-pip-version-check','--timeout','45','--retries','2']
            if overlay:
                import torch,torchvision,numpy
                constraints=cache/'overlay-core-constraints.txt'
                constraints.write_text('torch=='+torch.__version__+'\ntorchvision=='+torchvision.__version__+
                    '\nnumpy=='+numpy.__version__+'\n',encoding='utf-8')
                # Missing small extras only; constrain shared core versions, never pip into the base environment.
                bounded([*install,'-c',str(constraints),*missing_extras()],timeout=900)
                mode='isolated_small_overlay_preserve_core'
            else:
                bounded([*install,'-r',str(HERE/'requirements-lock.txt')],timeout=1800)
                mode='fresh_isolated_venv'
            bounded([interpreter,'-c',"import sys; sys.path.insert(0,sys.argv[1]); from bootstrap import compatible; assert compatible()",str(HERE)],timeout=90)
    result={'python':interpreter,'mode':mode,'source':locations['source'],'weights':locations['weights'],
        'academically_accelerated_entry':os.environ.get('DFINE_ACCELERATION_STATUS','not_observed'),
        'proxy_env_keys_present':[k for k in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY') if os.environ.get(k)],
        'caller_environment':environment() if compatible() else {'python':sys.executable,'compatible':False}}
    write_json(cache/'bootstrap.json',result)
    (cache/'python.path').write_text(interpreter+'\n',encoding='utf-8')
    print('Bootstrap ready: '+interpreter+' ('+mode+')',flush=True)
    return result
