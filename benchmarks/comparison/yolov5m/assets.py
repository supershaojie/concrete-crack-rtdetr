"""Prepare isolated pinned YOLOv5 and hash-verified original COCO assets."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import shutil
import urllib.request
from support import HERE, git, read_json, sha256, write_json, initialization_type

def verify(root, mode=None):
    mode = mode or initialization_type()
    if mode=='coco':mode='coco_detection_pretrained'
    if mode not in ('random', 'coco_detection_pretrained'):
        raise ValueError('Unknown initialization mode')
    root = Path(root).resolve()
    lock = read_json(HERE/'upstream.lock.json')
    upstream = root/'upstream'
    if git('rev-parse', 'HEAD', cwd=upstream) != lock['commit']:
        raise ValueError('Upstream HEAD differs from lock')
    diff = subprocess.check_output(['git', '-c', 'safe.directory='+str(upstream),
                                   '-C', str(upstream), 'diff', '--binary', '--no-ext-diff', 'HEAD'])
    # Normalize CRLF emitted by Windows Git. Patch remains text and is replayed by git apply.
    expected = (HERE/'upstream.patch').read_bytes().replace(b'\r\n', b'\n')
    if diff.replace(b'\r\n', b'\n') != expected:
        raise ValueError('Upstream modifications differ from the reviewed patch')
    if git('ls-files', '--others', '--exclude-standard', cwd=upstream):
        raise ValueError('Untracked upstream files; refusing ambiguous runtime')
    weights = root/'yolov5m.pt' if mode=='coco_detection_pretrained' else None
    if weights and (weights.stat().st_size != lock['weights']['bytes'] or sha256(weights) != lock['weights']['sha256']):
        raise ValueError('Official pretrained checkpoint hash/size mismatch')
    if not (upstream/'models/yolov5m.yaml').is_file():
        raise FileNotFoundError('Pinned official M architecture missing')
    return {'upstream': str(upstream), 'weights': str(weights) if weights else None, 'lock': lock,
            'initialization_type':mode,
            'patch_sha256': sha256(HERE/'upstream.patch')}

def prepare(root, mode=None, weights_from=None):
    mode = mode or initialization_type()
    if mode=='coco':mode='coco_detection_pretrained'
    if mode not in ('random', 'coco_detection_pretrained'):
        raise ValueError('Unknown initialization mode')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = read_json(HERE/'upstream.lock.json')
    upstream = root/'upstream'
    if not upstream.exists():
        subprocess.run(['git', 'clone', '--progress', '--depth', '1', '--branch', lock['tag'], lock['repository'], str(upstream)], check=True)
    if git('rev-parse', 'HEAD', cwd=upstream) != lock['commit']:
        raise ValueError('Tag/checkout differs from pinned full commit')
    if not git('status', '--porcelain', cwd=upstream):
        subprocess.run(['git', '-C', str(upstream), 'apply', '--check', str(HERE/'upstream.patch')], check=True)
        subprocess.run(['git', '-C', str(upstream), 'apply', str(HERE/'upstream.patch')], check=True)
    weights = root/'yolov5m.pt'
    print('Source/patch ready: '+str(upstream),flush=True)
    if mode=='coco_detection_pretrained' and not weights.exists() and weights_from:
        source=Path(weights_from)
        if source.stat().st_size != lock['weights']['bytes'] or sha256(source) != lock['weights']['sha256']:
            raise ValueError('--weights-from must be the exact original official COCO asset')
        shutil.copyfile(source,weights)
        print('Reused hash-verified official COCO bytes: '+str(source),flush=True)
    if mode=='coco_detection_pretrained' and not weights.exists():
        partial = root/'yolov5m.pt.partial'
        if partial.exists():
            raise FileExistsError('Inspect/remove incomplete download explicitly: ' + str(partial))
        with urllib.request.urlopen(lock['weights']['url'], timeout=120) as src, partial.open('xb') as dst:
            while True:
                block = src.read(1024*1024)
                if not block:
                    break
                dst.write(block)
                print(f'COCO download: {dst.tell()}/{lock["weights"]["bytes"]} bytes',flush=True)
        if partial.stat().st_size != lock['weights']['bytes'] or sha256(partial) != lock['weights']['sha256']:
            raise ValueError('Downloaded bytes do not match lock; partial retained')
        partial.rename(weights)
    result = verify(root, mode)
    write_json(root/'verified.json', result)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--initialization', choices=['random','coco','coco_detection_pretrained'], default=initialization_type())
    p.add_argument('--weights-from',type=Path,help='optional existing ORIGINAL official yolov5m.pt; exact hash/size required')
    a = p.parse_args()
    print(verify(a.assets,a.initialization) if a.verify_only else prepare(a.assets,a.initialization,a.weights_from))
