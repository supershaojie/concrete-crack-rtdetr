"""Fetch only pinned YOLOv5 v7.0 and its official M checkpoint; verify every reuse."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import urllib.request
from support import HERE, git, read_json, sha256, write_json

def verify(root):
    root = Path(root).resolve()
    lock = read_json(HERE/'upstream.lock.json')
    upstream = root/'upstream'
    if git('rev-parse', 'HEAD', cwd=upstream) != lock['commit']:
        raise ValueError('Upstream HEAD differs from lock')
    diff = subprocess.check_output(['git', '-C', str(upstream), 'diff', '--binary', '--no-ext-diff', 'HEAD'])
    # Normalize CRLF emitted by Windows Git. Patch remains text and is replayed by git apply.
    expected = (HERE/'upstream.patch').read_bytes().replace(b'\r\n', b'\n')
    if diff.replace(b'\r\n', b'\n') != expected:
        raise ValueError('Upstream modifications differ from the reviewed patch')
    if git('ls-files', '--others', '--exclude-standard', cwd=upstream):
        raise ValueError('Untracked upstream files; refusing ambiguous runtime')
    weights = root/'yolov5m.pt'
    if weights.stat().st_size != lock['weights']['bytes'] or sha256(weights) != lock['weights']['sha256']:
        raise ValueError('Official pretrained checkpoint hash/size mismatch')
    return {'upstream': str(upstream), 'weights': str(weights), 'lock': lock,
            'patch_sha256': sha256(HERE/'upstream.patch')}

def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = read_json(HERE/'upstream.lock.json')
    upstream = root/'upstream'
    if not upstream.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', lock['tag'], lock['repository'], str(upstream)], check=True)
    if git('rev-parse', 'HEAD', cwd=upstream) != lock['commit']:
        raise ValueError('Tag/checkout differs from pinned full commit')
    if not git('status', '--porcelain', cwd=upstream):
        subprocess.run(['git', '-C', str(upstream), 'apply', '--check', str(HERE/'upstream.patch')], check=True)
        subprocess.run(['git', '-C', str(upstream), 'apply', str(HERE/'upstream.patch')], check=True)
    weights = root/'yolov5m.pt'
    if not weights.exists():
        partial = root/'yolov5m.pt.partial'
        if partial.exists():
            raise FileExistsError('Inspect/remove incomplete download explicitly: ' + str(partial))
        with urllib.request.urlopen(lock['weights']['url'], timeout=120) as src, partial.open('xb') as dst:
            while True:
                block = src.read(1024*1024)
                if not block:
                    break
                dst.write(block)
        if partial.stat().st_size != lock['weights']['bytes'] or sha256(partial) != lock['weights']['sha256']:
            raise ValueError('Downloaded bytes do not match lock; partial retained')
        partial.rename(weights)
    result = verify(root)
    write_json(root/'verified.json', result)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    print(verify(a.assets) if a.verify_only else prepare(a.assets))
