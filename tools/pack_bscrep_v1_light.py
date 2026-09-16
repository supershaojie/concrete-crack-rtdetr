"""Bounded light package with per-file SHA256; excludes data, weights and predictions."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from init_bscrep_v1 import ROOT, VARIANTS, require, sha256
from bscrep_v1_common import DEFAULT_VARIANT, paths, git


def package(output, variant=DEFAULT_VARIANT, main=None, extra=()):
    output = Path(output).resolve()
    require(not output.exists(), 'Existing package protected')
    checksum = output.with_suffix(output.suffix + '.sha256')
    require(not checksum.exists(), 'Existing package checksum protected')
    p = paths(variant, main)
    sources = [('training', p['run']), ('launch', p['launch']), ('docs', ROOT/'docs/bscrep_v1')]
    sources += [(f'evidence_{i}', Path(folder)) for i, folder in enumerate(extra)]
    allow = {'.json', '.yaml', '.yml', '.csv', '.md', '.txt'}
    figure_names = {'results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png',
                    'BoxPR_curve.png', 'BoxF1_curve.png', 'BoxP_curve.png', 'BoxR_curve.png'}
    selected, excluded = {}, []
    for label, folder in sources:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(folder)
            accepted = (path.suffix.lower() in allow or path.name in figure_names)
            accepted &= not any(part in {'weights', 'datasets', '__pycache__'} for part in rel.parts)
            accepted &= 'prediction' not in path.name.lower() and path.stat().st_size <= 8*1024*1024
            if accepted:
                selected[f'{label}/{rel.as_posix()}'] = path
            else:
                excluded.append(f'{label}/{rel.as_posix()}')
    # Reproducible source for the new implementation and every reused runtime tool/module.
    tracked = git('ls-files', '--cached', '--others', '--exclude-standard', 'tools', 'ultralytics-main/ultralytics', 'docs/bscrep_v1', '.gitattributes').splitlines()
    for rel in tracked:
        path = ROOT/rel
        if path.suffix in {'.py', '.yaml', '.yml', '.sh'} or rel == '.gitattributes':
            selected['source/'+rel] = path
    total = sum(path.stat().st_size for path in selected.values())
    require(total <= 64*1024*1024, 'Light package exceeds 64 MiB uncompressed; inspect rather than silently drop evidence')
    manifest = dict(variant=variant, commit=git('rev-parse', 'HEAD'), working_tree_status=git('status', '--porcelain'), formal_training='read separate launch state',
                    policy='no dataset/weights/reference archive/large predictions; max file 8 MiB, total 64 MiB',
                    files={name: dict(bytes=path.stat().st_size, sha256=sha256(path)) for name, path in selected.items()},
                    excluded=excluded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, path in selected.items():
            archive.add(path, arcname=name, recursive=False)
        raw = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
        item = tarfile.TarInfo('manifest.json'); item.size = len(raw)
        archive.addfile(item, io.BytesIO(raw))
    # Verify actual package bytes against the manifest before returning.
    with tarfile.open(output) as archive:
        for name, row in manifest['files'].items():
            require(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == row['sha256'], 'Package integrity failed')
    with checksum.open('x', encoding='utf-8') as stream:
        stream.write(sha256(output)+'  '+output.name+'\n')
    return dict(path=str(output), sha256=sha256(output), files=len(selected), uncompressed_bytes=total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument('--main', type=Path)
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.variant, args.main, args.evidence), indent=2))
