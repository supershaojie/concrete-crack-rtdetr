"""LIGHT size, content, JSON completeness, and old-evidence preservation tests."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from pack_c19_lif_v1_light import package,MAX_BYTES
from c19_lif_v1_diagnostic import atomic_json


def run(folder,source):
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAILED',tests={})
    try:
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        original={str(p):digest(p) for p in source.rglob('*') if p.is_file() and p.stat().st_size<2_000_000}
        result=package(source,folder/'actual_LIGHT.tar.gz')
        assert result['bytes']<=MAX_BYTES
        with tarfile.open(result['path']) as t:
            names=t.getnames()
            assert not any(n.endswith(('.pt','.npz','.png','.jpg','.gz')) for n in names)
            for m in t.getmembers():
                if m.name.endswith('.json'):json.loads(t.extractfile(m).read())
            assert any('fuse_diagnostic.json' in n for n in names)
        report['tests']['real_LIGHT']=result
        assert original=={str(p):digest(p) for p in source.rglob('*') if p.is_file() and p.stat().st_size<2_000_000}
        report['tests']['source_preserved']=True
        fixture=folder/'large_fixture';fixture.mkdir()
        with (fixture/'weights.pt').open('wb') as f:f.seek(100_000_000);f.write(b'x')
        atomic_json(fixture/'checks.json',dict(status='FAILED',long_list=['record']*100000))
        (fixture/'preflight.log').write_bytes(b'bounded log line\n'*100000)
        small=package(fixture,folder/'bounded_LIGHT.tar.gz')
        with tarfile.open(small['path']) as t:
            assert 'weights.pt' not in t.getnames()
            summary=json.loads(t.extractfile('checks.json.summary.json').read())
            assert summary['summary']['long_list']['count']==100000
            assert len(t.extractfile('preflight.log.tail.txt').read())<=32768
        report['tests']['large_weights_excluded_structured_JSON_and_tail']=True
        try:package(fixture,folder/'over_cap.tar.gz',limit=1024)
        except ValueError:assert not (folder/'over_cap.tar.gz').exists()
        else:raise AssertionError('Hard cap ignored')
        report['tests']['hard_cap_no_partial_output']=True
        try:package(fixture,folder/'invalid_limit.tar.gz',limit=MAX_BYTES+1)
        except ValueError:pass
        else:raise AssertionError('Raised cap accepted')
        report['tests']['cannot_raise_8MB_cap']=True
        before=digest(folder/'actual_LIGHT.tar.gz')
        try:package(fixture,folder/'actual_LIGHT.tar.gz')
        except FileExistsError:assert digest(folder/'actual_LIGHT.tar.gz')==before
        else:raise AssertionError('Old archive overwritten')
        report['tests']['old_archive_preserved']=True
        report['status']='PASSED'
    finally:atomic_json(folder/'tests.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();run(a.output,a.source)
