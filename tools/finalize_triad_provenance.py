"""Fingerprint this change's text artifacts with canonical LF bytes before committing."""
from __future__ import annotations
from pathlib import Path
from triad_compat import ROOT,BASE_COMMIT,git,sha256,write_json


def finalize():
    names=set(git('diff','--name-only',BASE_COMMIT).splitlines())
    names.update(n for n in git('ls-files','--others','--exclude-standard','-z').split('\0') if n)
    inventory='docs/triad_compat/FILES.md';manifest='docs/triad_compat/source_files_sha256.json'
    names.update((inventory,manifest))
    for name in sorted(names-{manifest,inventory}):
        path=ROOT/name
        if path.is_file() and path.suffix in {'.py','.sh','.yaml','.json','.md'}:
            data=path.read_bytes();normalized=data.replace(b'\r\n',b'\n')
            if data!=normalized:path.write_bytes(normalized)
    text='# 修改与新增文件\n\n仅列本兼容分支相对 C2 的文件；不包含用户结果包或权重。\n\n'
    for name in sorted(names):text+=f'- `{name}`\n'
    (ROOT/inventory).write_bytes(text.encode('utf-8'))
    files={name:dict(sha256=sha256(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in sorted(names-{manifest})}
    write_json(ROOT/manifest,dict(c2_commit=BASE_COMMIT,branch='codex/rtdetr-triad-compat',
        format='SHA256 of canonical LF bytes, no BOM; manifest excludes itself because Git commit authenticates its own bytes',
        final_commit='Resolve this artifact through its containing Git commit; see final delivery FULL SHA',files=files))
    manifest_path=ROOT/manifest
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b'\r\n',b'\n'))
    print('Fingerprinted',len(files),'files; no checkpoints/packages included')


if __name__=='__main__':finalize()
