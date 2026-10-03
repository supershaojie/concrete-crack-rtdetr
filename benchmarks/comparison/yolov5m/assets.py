"""One fresh patched source runtime, reusing pinned raw Git objects and original COCO bytes."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
from support import HERE, git, read_json, sha256, write_json

DEFAULT_CACHE = Path('/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot/outputs/yolov5m-coco-b19-pilot/assets')

def official_weight(path):
    path = Path(path).resolve()
    lock = read_json(HERE/'upstream.lock.json')['weights']
    if not path.is_file() or path.stat().st_size != lock['bytes'] or sha256(path) != lock['sha256']:
        raise ValueError('Official yolov5m.pt hash/size mismatch: ' + str(path))
    return path

def verify(root, mode=None):
    if mode not in (None, 'coco', 'coco_detection_pretrained'):
        raise ValueError('Native fine-tuning always starts from official COCO weights')
    root = Path(root).resolve()
    lock = read_json(HERE/'upstream.lock.json')
    upstream = root/'upstream'
    if git('rev-parse', 'HEAD', cwd=upstream) != lock['commit']:
        raise ValueError('Upstream HEAD differs from lock')
    diff = subprocess.check_output(['git', '-c', 'safe.directory='+upstream.as_posix(),
                                   '-C', str(upstream), 'diff', '--binary', '--no-ext-diff', 'HEAD'])
    expected = (HERE/'upstream.patch').read_bytes().replace(b'\r\n', b'\n')
    if diff.replace(b'\r\n', b'\n') != expected:
        raise ValueError('Runtime source modifications differ from the reviewed native patch')
    if git('ls-files', '--others', '--exclude-standard', cwd=upstream):
        raise ValueError('Untracked upstream source files; refusing ambiguous runtime')
    record = read_json(root/'weights-source.json')
    weights = official_weight(record['path'])
    if record['sha256'] != lock['weights']['sha256']:
        raise ValueError('Weight provenance record differs')
    return {'upstream': str(upstream), 'weights': str(weights), 'lock': lock,
            'initialization_type': 'coco_detection_pretrained',
            'patch_sha256': sha256(HERE/'upstream.patch')}

def prepare(root, source_assets=DEFAULT_CACHE):
    root, source_assets = Path(root).resolve(), Path(source_assets).resolve()
    if root.exists():
        return verify(root)  # no silent patch repair or cache replacement
    if root == source_assets:
        raise ValueError('New runtime must not overwrite an old patched asset directory')
    lock = read_json(HERE/'upstream.lock.json')
    source = source_assets/'upstream'
    if git('rev-parse', 'HEAD', cwd=source) != lock['commit']:
        raise ValueError('Cached raw source Git commit differs from v7.0 lock')
    weights = official_weight(source_assets/'yolov5m.pt')
    root.mkdir(parents=True, exist_ok=False)
    upstream = root/'upstream'
    # Local clone reads committed objects, ignoring the old worktree's preprocessing patch.
    # Independent objects survive a later cleanup of the old source checkout.
    subprocess.run(['git', '-c', 'safe.directory='+source.as_posix(), '-c', 'safe.directory='+(source/'.git').as_posix(), 'clone', '--no-hardlinks',
                    '--no-checkout', '--', str(source), str(upstream)], check=True)
    subprocess.run(['git', '-C', str(upstream), 'config', 'core.autocrlf', 'false'], check=True)
    subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', lock['commit']], check=True)
    subprocess.run(['git', '-C', str(upstream), 'apply', '--check', str(HERE/'upstream.patch')], check=True)
    subprocess.run(['git', '-C', str(upstream), 'apply', str(HERE/'upstream.patch')], check=True)
    write_json(root/'weights-source.json', {'path': str(weights), 'sha256': sha256(weights),
                                          'bytes': weights.stat().st_size, 'reused_without_copy': True})
    result = verify(root)
    write_json(root/'verified.json', result)
    print('Reused official weight bytes; fresh native source runtime: '+str(upstream), flush=True)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True, help='new native runtime directory')
    p.add_argument('--asset-cache', type=Path, default=DEFAULT_CACHE, help='existing pilot asset directory, read-only')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    print(verify(a.assets) if a.verify_only else prepare(a.assets, a.asset_cache))
