"""Pinned official Flash assets/build in this worktree's private environment only."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from support import ROOT, HERE, read_json, sha256, write_json

RELEASE_API = 'https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.7.3'
SOURCE_COMMIT = '89c5a7dd4e6a8644575bd0c04a286f48c42763ec'


def compatibility():
    import torch
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3,10):
        raise RuntimeError('This Flash environment requires Linux x86_64 CPython3.10; no automatic Python change')
    if torch.__version__.split('+')[0] != '2.2.2' or torch.version.cuda != '12.1':
        raise RuntimeError('Flash environment requires torch2.2.2 cu121; no automatic Torch upgrade')
    return {'python_tag':'cp310','torch_tag':'2.2','cuda_tag':'cu12',
            'abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),'platform':'linux_x86_64'}


def select_wheel(release, target):
    if release.get('tag_name') != 'v2.7.3':
        raise ValueError('Wrong Flash release tag')
    abi = 'TRUE' if target['abi'] else 'FALSE'
    pattern = (r'^flash_attn-2\.7\.3\+'+re.escape(target['cuda_tag'])+
               'torch'+re.escape(target['torch_tag'])+'cxx11abi'+abi+'-'+
               target['python_tag']+'-'+target['python_tag']+'-'+target['platform']+r'\.whl$')
    matches = [a for a in release['assets'] if re.fullmatch(pattern,a['name'])]
    if len(matches)>1:
        raise ValueError('Ambiguous official wheel metadata')
    return matches[0] if matches else None


def download(url, destination):
    request = urllib.request.Request(url,headers={'User-Agent':'Crack-RTDETR-YOLOv13L-Flash'})
    with urllib.request.urlopen(request,timeout=60) as src, Path(destination).open('xb') as dst:
        shutil.copyfileobj(src,dst,4*1024*1024)


def run_git(*args, cwd=None):
    for attempt in range(3):
        result = subprocess.run(['git','-c','http.version=HTTP/1.1',*map(str,args)],cwd=cwd)
        if result.returncode == 0:
            return
    raise subprocess.CalledProcessError(result.returncode,result.args)


def install():
    # Only called by the private Python, after toolchain/Torch/runtime installation.
    env_root = (ROOT/'.envs').resolve()
    if not Path(sys.prefix).resolve().is_relative_to(env_root) or sys.prefix == sys.base_prefix:
        raise RuntimeError('Flash installation must use this worktree private interpreter')
    target = compatibility()
    state = ROOT/'.runtime/yolov13l-configurable/flash-install'
    state.mkdir(parents=True,exist_ok=True)
    receipt = state/'receipt.json'
    if receipt.exists():
        from backend import flash_identity, flash_operator_probe
        saved = read_json(receipt)
        if saved['flash_identity'] != flash_identity() or saved['target'] != target:
            raise RuntimeError('Existing Flash receipt differs; refuse reinstall')
        flash_operator_probe()
        return saved
    request = urllib.request.Request(RELEASE_API,headers={'User-Agent':'Crack-RTDETR-YOLOv13L-Flash'})
    with urllib.request.urlopen(request,timeout=60) as response:
        release = json.load(response)
    write_json(state/'official_release_metadata.json',release)
    asset = select_wheel(release,target)
    if asset:
        url = asset['browser_download_url']
        if not url.startswith('https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/'):
            raise ValueError('Wheel URL is outside the official fixed release')
        wheel = state/asset['name']
        if not wheel.exists():
            download(url,wheel)
        actual_sha = sha256(wheel)
        if wheel.stat().st_size != asset['size'] or (asset.get('digest') and asset['digest'] != 'sha256:'+actual_sha):
            raise ValueError('Downloaded Flash wheel size/digest differs')
        provenance = {'mode':'official_release_wheel','url':url,'asset_id':asset['id'],
                      'bytes':wheel.stat().st_size,'sha256':actual_sha,
                      'release_digest':asset.get('digest'),
                      'trust_note':'Actual downloaded SHA256; this historical release supplies no asset digest when release_digest is null'}
    else:
        nvcc, compiler = shutil.which('nvcc'), shutil.which('g++')
        if not nvcc or not compiler:
            raise RuntimeError('No matching wheel; pinned source build requires nvcc and g++; no fallback/upgrade')
        tools = {tool:subprocess.check_output([exe,'--version'],text=True) for tool,exe in [('nvcc',nvcc),('g++',compiler)]}
        if 'release 12.' not in tools['nvcc']:
            raise RuntimeError('Pinned cu121 source build requires CUDA12 nvcc')
        source = state/'source'
        if source.exists():
            raise FileExistsError('Prior source-build directory retained; inspect failure instead of overwriting')
        run_git('clone','--no-checkout','https://github.com/Dao-AILab/flash-attention.git',source)
        run_git('-C',source,'checkout','--detach',SOURCE_COMMIT)
        run_git('-C',source,'submodule','update','--init','--recursive','--depth','1')
        dist = state/'built-wheels';dist.mkdir()
        env = {**os.environ,'MAX_JOBS':'2','NVCC_THREADS':'2','FLASH_ATTENTION_FORCE_BUILD':'TRUE'}
        subprocess.run([sys.executable,'-m','pip','wheel','--no-build-isolation','--no-deps','--wheel-dir',str(dist),str(source)],check=True,env=env)
        wheels = list(dist.glob('flash_attn-2.7.3*.whl'))
        if len(wheels)!=1:
            raise RuntimeError('Pinned source build did not produce exactly one 2.7.3 wheel')
        wheel = wheels[0]
        provenance = {'mode':'pinned_source_build','repository':'https://github.com/Dao-AILab/flash-attention.git',
                      'commit':SOURCE_COMMIT,'toolchain':tools,'MAX_JOBS':2,'NVCC_THREADS':2,
                      'sha256':sha256(wheel),'bytes':wheel.stat().st_size}
    write_json(state/'asset_before_install.json',provenance)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps',str(wheel)],check=True)
    from backend import flash_identity, flash_operator_probe
    saved = {'version':'2.7.3','target':target,'asset':provenance,
             'flash_identity':flash_identity(),'real_operator_probe':flash_operator_probe()}
    write_json(receipt,saved)
    return saved


if __name__=='__main__':
    print(json.dumps(install()))
