"""Read-only compatible reuse, private overlay, or pinned --copies environment."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid
from support import ENV,HERE,LOCK,RUNTIME,atomic_bytes,local_lock,read_json,sha256,text_hash,write_json

def call(argv,timeout=1800):
    print('EXEC '+json.dumps([str(v) for v in argv]),flush=True)
    for attempt in range(1,4):
        try: return subprocess.run([str(v) for v in argv],check=True,timeout=timeout)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
            if attempt==3: raise
            print(f'Retry {attempt}/2 after failed stage',flush=True); time.sleep(2*attempt)

def version_probe(python):
    code='import torch,torchvision,numpy,sys,json,os;print(json.dumps(dict(torch=torch.__version__,vision=torchvision.__version__,numpy=numpy.__version__,cuda=torch.version.cuda,prefix=sys.prefix,python=os.path.abspath(sys.executable))))'
    result=subprocess.run([str(python),'-c',code],capture_output=True,text=True,timeout=90)
    try: value=json.loads(result.stdout) if result.returncode==0 else {}
    except ValueError: value={}
    return value,{'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}

def source_probe():
    import torchvision
    from torchvision._internally_replaced_utils import _get_extension_path
    lock=read_json(LOCK)['vision']; cache=RUNTIME/'vision-source'; cache.mkdir(exist_ok=True)
    files=('torchvision/models/detection/faster_rcnn.py','torchvision/models/detection/backbone_utils.py',
        'torchvision/ops/feature_pyramid_network.py','torchvision/models/detection/rpn.py','torchvision/models/detection/roi_heads.py','torchvision/models/detection/transform.py',
        'references/detection/train.py','references/detection/coco_eval.py','LICENSE')
    inventory={}
    for name in files:
        target=cache/name; target.parent.mkdir(parents=True,exist_ok=True)
        url='https://raw.githubusercontent.com/pytorch/vision/'+lock['commit']+'/'+name
        if not target.is_file():
            for attempt in range(1,4):
                try:
                    print('Official source '+name+f' [{attempt}/3]',flush=True)
                    with urllib.request.urlopen(url,timeout=45) as response: raw=response.read()
                    atomic_bytes(target,raw); break
                except Exception:
                    if attempt==3: raise
                    time.sleep(2*attempt)
        if name.startswith('torchvision/'):
            installed=Path(torchvision.__file__).parent/Path(name).relative_to('torchvision')
            if text_hash(installed)!=text_hash(target): raise ValueError('Wheel source differs from locked tag: '+name)
        inventory[name]={'url':url,'sha256':sha256(target)}
    write_json(RUNTIME/'source_correspondence.json',{'status':'WHEEL_MODEL_PYTHON_SOURCE_EQUALS_LOCKED_COMMIT',
        'commit':lock['commit'],'tag':lock['tag'],'files':inventory,'extension':_get_extension_path('_C'),
        'extension_sha256':sha256(_get_extension_path('_C')),'compiled_ops_origin':'installed torchvision wheel; tested on selected device during per-run capacity probe'})

def bootstrap(base_python,cuda_wheel='cu118'):
    python=Path(os.path.abspath(str(base_python))); RUNTIME.mkdir(parents=True,exist_ok=True)
    with local_lock(RUNTIME/'bootstrap.lock'):
        attempt=RUNTIME/'bootstrap_attempts'/uuid.uuid4().hex; attempt.mkdir(parents=True)
        if Path('/etc/network_turbo').is_file():
            speed=subprocess.run(['bash','-lc','source /etc/network_turbo'],capture_output=True,text=True,timeout=30)
            write_json(attempt/'acceleration.json',{'available':True,'exit_code':speed.returncode,'stdout':speed.stdout,'stderr':speed.stderr})
            # Exported proxy variables must be inherited by subsequent stages, not lost in a subshell.
            env_result=subprocess.run(['bash','-lc','source /etc/network_turbo >/dev/null 2>&1; env -0'],capture_output=True,timeout=30)
            if env_result.returncode==0:
                for item in env_result.stdout.decode().split('\0'):
                    key,sep,val=item.partition('=')
                    if sep and key.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'): os.environ[key]=val
        else: write_json(attempt/'acceleration.json',{'available':False,'status':'no /etc/network_turbo entry'})
        versions,report=version_probe(python); write_json(attempt/'base_probe.json',report)
        target=versions.get('torch','').split('+')[0]=='2.1.2' and versions.get('vision','').split('+')[0]=='0.16.2' and versions.get('cuda') in ('11.8','12.1')
        if target:
            check=subprocess.run([str(python),str(HERE/'bootstrap.py'),'--check-existing'],capture_output=True,text=True,timeout=180)
            write_json(attempt/'existing_probe.json',{'exit_code':check.returncode,'stdout':check.stdout,'stderr':check.stderr})
            if check.returncode==0:
                print('Reuse compatible environment read-only: '+str(python),flush=True)
                selected=python; mode='read_only_existing'
            else: selected=None
        else: selected=None
        if selected is None:
            selected=Path(os.path.abspath(str(ENV/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))))
            if not ENV.exists():
                info=json.loads(subprocess.check_output([str(python),'-c','import sys,json; print(json.dumps(list(sys.version_info[:2])))'],text=True))
                if info[0]!=3 or not 9<=info[1]<=11: raise ValueError('Choose a Python3.9-3.11 base; no existing environment will be changed')
                argv=[python,'-m','venv','--copies']
                if target: argv.append('--system-site-packages')
                call([*argv,ENV],timeout=180)
                write_json(RUNTIME/'venv_origin.json',{'base':str(python),'prefix':str(ENV),'copies':True,'inherited_read_only':target,'wheel':cuda_wheel})
            origin=read_json(RUNTIME/'venv_origin.json')
            if origin['prefix']!=str(ENV) or origin['wheel']!=cuda_wheel: raise ValueError('Existing private environment identity differs')
            observed,_=version_probe(selected)
            if not (observed.get('torch','').split('+')[0]=='2.1.2' and observed.get('vision','').split('+')[0]=='0.16.2'):
                call([selected,'-m','pip','install','--disable-pip-version-check','--timeout','45','--retries','2',
                    'torch==2.1.2','torchvision==0.16.2','--index-url','https://download.pytorch.org/whl/'+cuda_wheel,'--report',attempt/'torch_install.json'])
            call([selected,'-m','pip','install','--disable-pip-version-check','--timeout','45','--retries','2',
                '-r',HERE/'requirements.txt','--report',attempt/'dependency_install.json'])
            mode='isolated_overlay' if origin['inherited_read_only'] else 'new_pinned_environment'
        call([selected,HERE/'bootstrap.py','--verify-source'],timeout=600)
        call([selected,HERE/'bootstrap.py','--check-existing'],timeout=180)
        info=json.loads(subprocess.check_output([str(selected),'-c','import os,sys,json;print(json.dumps(dict(python=os.path.abspath(sys.executable),prefix=sys.prefix,base_prefix=sys.base_prefix)))'],text=True))
        if os.path.normcase(info['python'])!=os.path.normcase(str(selected)): raise ValueError('Interpreter path differs; no symlink resolution permitted')
        write_json(attempt/'bootstrap.json',{'status':'PASSED','mode':mode,'python':str(selected),'interpreter':info,
            'source':read_json(RUNTIME/'source_correspondence.json'),'capacity':'per-run actual configuration probe still required','formal_server_verified':False})
        atomic_bytes(RUNTIME/'python_path.txt',(str(selected)+'\n').encode())
        atomic_bytes(RUNTIME/'latest_bootstrap.txt',(str(attempt)+'\n').encode())
        print('BOOTSTRAP PASSED '+str(selected),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-python',type=Path,default=Path(sys.executable))
    parser.add_argument('--cuda-wheel',choices=('cu118','cu121'),default='cu118')
    parser.add_argument('--verify-source',action='store_true'); parser.add_argument('--check-existing',action='store_true')
    args=parser.parse_args()
    if args.verify_source: return source_probe()
    if args.check_existing:
        from support import environment
        import torch,torchvision
        env=environment(strict=True)
        if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
        b=torch.tensor([[0.,0.,10.,10.]],device='cuda'); s=torch.ones(1,device='cuda')
        torchvision.ops.nms(b,s,.7)
        x=torch.ones(1,2,16,16,device='cuda',requires_grad=True)
        torchvision.ops.roi_align(x,[b],7).sum().backward()
        if x.grad is None: raise ValueError('ROIAlign backward missing')
        if any(env['versions'][n] is None for n in ('pycocotools','matplotlib','tqdm','scipy')): raise ValueError('Missing required dependencies')
        print(json.dumps(env)); return
    bootstrap(args.base_python,args.cuda_wheel)
if __name__=='__main__': main()
