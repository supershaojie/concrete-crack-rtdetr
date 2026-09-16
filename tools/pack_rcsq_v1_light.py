"""Bounded RCS-Q evidence/source archive; train/val/test separated, no weights/data or per-image predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

from init_rcsq_v1 import ROOT, code_fingerprint, git, require, sha256, write_json

MAX_BYTES = 16_000_000


def package(output, preflight=None, init_dir=None, plan_dir=None, train=None, val=None, test=None):
    output = Path(output).resolve()
    require(not output.exists() and not Path(str(output) + '.sha256').exists(), 'Existing archive preserved')
    entries, omitted = {}, []

    def add(path, name):
        if path.is_symlink():
            omitted.append(dict(path=name, reason='symlink'))
        elif path.stat().st_size > 2_000_000:
            omitted.append(dict(path=name, reason='per-file light cap', bytes=path.stat().st_size, sha256=sha256(path)))
        else:
            entries[name] = path.read_bytes()

    for label, folder in [('preflight', preflight), ('initialization', init_dir), ('plan', plan_dir),
                          ('train', train), ('val', val), ('test', test)]:
        if folder is None:
            continue
        folder = Path(folder).resolve(strict=True)
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path == output:
                continue
            rel = path.relative_to(folder)
            if any(part in ('weights', '__pycache__', 'dataset', 'datasets') for part in rel.parts):
                continue
            suffix = path.suffix.lower()
            curve = suffix == '.png' and (path.name == 'results.png' or 'curve' in path.name or path.name.startswith('confusion_matrix'))
            if suffix in ('.json', '.yaml', '.csv') or curve:
                add(path, label + '/' + rel.as_posix())
            elif suffix in ('.log', '.txt') and (path.name.endswith('.log') or path.name in ('pip_freeze.txt', 'process_exit_code.txt')):
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 32768))
                    entries[label + '/' + rel.as_posix() + '.tail.txt'] = stream.read(32768)
    for name in code_fingerprint()['files']:
        # Package all code actually used by the gate, including inherited numerical helpers.
        add(ROOT / name, 'source/' + name)
    for path in sorted((ROOT / 'docs/rcsq_v1').glob('*')):
        if path.is_file() and path.suffix in ('.md', '.json', '.yaml'):
            add(path, 'source/' + path.relative_to(ROOT).as_posix())
    evidence = dict(created=datetime.now(timezone.utc).isoformat(), commit=git('rev-parse', 'HEAD'),
                    git_status=git('status', '--short'), code_fingerprint=code_fingerprint()['sha256'], omitted=omitted,
                    excluded='all weights, datasets, reference ZIPs, archives, per-image predictions and full log files',
                    scope='existing evidence only; archive creation does not train, evaluate or imply checks passed')
    entries['PACKAGE_INFO.json'] = (json.dumps(evidence, ensure_ascii=False, indent=2) + '\n').encode()
    manifest = [dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in sorted(entries.items())]
    entries['MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    require(sum(len(data) for data in entries.values()) <= 64_000_000, 'Uncompressed light archive exceeds cap')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, data in sorted(entries.items()):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    payload = buffer.getvalue()
    require(len(payload) <= MAX_BYTES, 'Compressed light archive exceeds 16 MB cap')
    # Read every member back before exposing a completed artifact.
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        require(len(archive.getnames()) == len(entries), 'Archive member inventory differs')
        for row in manifest:
            require(hashlib.sha256(archive.extractfile(row['path']).read()).hexdigest() == row['sha256'], 'Archive integrity check failed')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(payload)
    digest = sha256(output)
    with Path(str(output) + '.sha256').open('x', encoding='utf-8') as stream:
        stream.write(digest + '  ' + output.name + '\n')
    write_json(Path(str(output) + '.manifest.json'), manifest)
    return dict(path=str(output), sha256=digest, bytes=len(payload), members=len(entries), omitted=omitted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('preflight', 'init-dir', 'plan-dir', 'train', 'val', 'test'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.preflight, args.init_dir, args.plan_dir, args.train, args.val, args.test), indent=2))
