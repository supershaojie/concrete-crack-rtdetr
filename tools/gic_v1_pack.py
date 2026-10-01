"""Offline evidence archive only. No torch, data decoding, inference, or downloads."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import tarfile

from gic_v1_common import ROOT,OUT,RUN,COUNTS,read_json,write_json,require,sha256,digest,now,source_identity


def integrity():
    missing=[]
    plan=read_json(OUT/'prepare.json',{})
    complete=read_json(OUT/'training_completed.json',{})
    training=read_json(OUT/'training.json',{})
    preflight=read_json(OUT/'preflight.json',{})
    identity=complete.get('binding')
    for name in ('prepare.json','initialization.json','nc1_loading.json','structure.json','data_snapshot.json',
                 'recipe_audit.json','train_args.yaml','mother_args.yaml','data_config.yaml','source_snapshot.tar.gz',
                 'source_from_mother.patch','training_setup.json','trainer_loading.json','training_completed.json','gic_diagnostics.jsonl','metrics.html'):
        if not (OUT/name).is_file():missing.append(name)
    if not identity or complete.get('status')!='COMPLETE' or training.get('status')!='COMPLETE':missing.append('successful completed training')
    if preflight.get('status')!='PASS' or preflight.get('binding')!=identity or preflight.get('exit_code')!=0:missing.append('same-identity successful preflight')
    for p,h in preflight.get('artifacts',{}).items():
        if not Path(p).is_file() or sha256(p)!=h:missing.append('preflight artifact '+p)
    if plan:
        if plan.get('development') or plan.get('recipe_audit',{}).get('actual_status')!='CHECKED':missing.append('formal mother recipe audit')
        if plan['code']['functional_sha256']!=source_identity(clean=False)['functional_sha256']:missing.append('unchanged functional code')
        for name,h in plan.get('artifacts',{}).items():
            if not (OUT/name).is_file() or sha256(OUT/name)!=h:missing.append('prepared artifact '+name)
        if identity and plan['code']['functional_sha256']!=identity['identity']['functional_sha256']:missing.append('prepared/training binding')
    for name in ('weights/best.pt','args.yaml','results.csv','results.png'):
        if not (RUN/name).is_file():missing.append('training/'+name)
    best_hash=complete.get('weights',{}).get('best',{}).get('sha256')
    if not best_hash or not (RUN/'weights/best.pt').is_file() or sha256(RUN/'weights/best.pt')!=best_hash:missing.append('fixed selected best SHA256')
    for kind in ('training','finish'):
        r=read_json(OUT/(kind+'.json'),{})
        e=read_json(Path(r['folder'])/'exit.json',{}) if r.get('folder') else {}
        if r.get('status')!='COMPLETE' or e.get('python')!=0 or e.get('tee')!=0 or e.get('binding')!=identity:missing.append(kind+' successful Python/tee exit evidence')
    exports={}
    for split in ('val','test'):
        r=read_json(OUT/f'evaluation_{split}.json',{})
        if r.get('status')!='COMPLETE' or r.get('exit_code')!=0 or not r.get('folder'):
            missing.append(split+' independent evaluation');continue
        if r.get('split')!=split or (r.get('images'),r.get('ground_truths'))!=COUNTS[split]:missing.append(split+' split coverage')
        if r.get('best_sha256')!=best_hash or r.get('identity',{}).get('binding')!=identity:missing.append(split+' best/identity')
        if r!=read_json(Path(r['folder'])/'metrics.json'):missing.append(split+' original metrics')
        if not {'queries_gt.jsonl.gz','raw_stats.npz','curves.npz'}<=set(r.get('artifacts',{})):missing.append(split+' all-query/GT and metric exports')
        for name,info in r.get('artifacts',{}).items():
            p=Path(r['folder'])/name
            if not p.is_file() or sha256(p)!=info['sha256']:missing.append(split+' artifact '+name)
        exports[split]=r.get('artifacts',{}).get('queries_gt.jsonl.gz',{}).get('sha256')
    for name in ('threshold_analysis.json','gic_analysis.json'):
        r=read_json(OUT/name,{})
        if r.get('status')!='COMPLETE' or r.get('best_sha256')!=best_hash or r.get('source_sha256')!=exports:missing.append(name+' matching complete offline analysis')
    return sorted(set(missing))


def verify_archive(path):
    with tarfile.open(path,'r:gz') as archive:
        manifest=json.load(archive.extractfile('MANIFEST.json'))
        names=archive.getnames()
        require(len(names)==len(set(names)) and set(names)=={r['path'] for r in manifest}|{'MANIFEST.json'},'Archive inventory mismatch')
        for row in manifest:
            p=PurePosixPath(row['path']);require(not p.is_absolute() and '..' not in p.parts,'Unsafe archive member')
            member=archive.getmember(row['path']);require(member.isfile() and member.size==row['bytes'],'Archive type/size mismatch')
            h=hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            require(h.hexdigest()==row['sha256'],'Archive content mismatch: '+row['path'])
    return manifest


def package():
    OUT.mkdir(parents=True,exist_ok=True)
    missing=integrity();status='PARTIAL' if missing else 'COMPLETE'
    complete=read_json(OUT/'training_completed.json',{})
    # A verified complete same-run package is immutable/reusable, including after a new finish dispatch.
    old=read_json(OUT/'package.json',{})
    if status=='COMPLETE' and old.get('status')=='COMPLETE' and old.get('binding')==complete.get('binding'):
        path=Path(old['path'])
        if path.is_file() and sha256(path)==old.get('sha256'):
            verify_archive(path);return dict(old,reused=True)
    files=[]
    for p in OUT.rglob('*'):
        if not p.is_file() or p.suffix in ('.pt','.pth','.tmp') or p.name.startswith('GIC_v1_'):continue
        if p.name in ('package.json','run.lock','lock_owner.json','pack.log','pack_exit.json'):continue
        files.append((p,'evidence/'+p.relative_to(OUT).as_posix()))
    for p in RUN.rglob('*'):
        if p.is_file() and (p.suffix in ('.yaml','.json','.csv','.png','.log') or p==RUN/'weights/best.pt'):
            files.append((p,'training/'+p.relative_to(RUN).as_posix()))
    for p in (ROOT/'docs/gic_v1').rglob('*'):
        if p.is_file():files.append((p,'docs/'+p.relative_to(ROOT/'docs/gic_v1').as_posix()))
    files.append((ROOT/'server_commands_gic_v1.md','server_commands_gic_v1.md'))
    inventory=[dict(path=rel,bytes=p.stat().st_size,sha256=sha256(p)) for p,rel in files if p.is_file()]
    signature=digest(dict(status=status,missing=missing,files=inventory))
    if old.get('payload_sha256')==signature and Path(old['path']).is_file() and sha256(old['path'])==old['sha256']:
        verify_archive(old['path']);return dict(old,reused=True)
    destination=OUT/f'GIC_v1_{status}_{now()}.tar.gz'
    package_status=dict(status=status,missing=missing,time=now(),binding=complete.get('binding'),
                        best_included=any(rel=='training/weights/best.pt' for _,rel in files))
    with tarfile.open(destination,'w:gz') as archive:
        for p,rel in files:
            if p.is_file():archive.add(p,arcname=rel,recursive=False)
        content=(json.dumps(package_status,ensure_ascii=False,indent=2)+'\n').encode()
        entry=tarfile.TarInfo('PACKAGE_STATUS.json');entry.size=len(content);archive.addfile(entry,io.BytesIO(content))
        inventory.append(dict(path=entry.name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        content=(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n').encode()
        entry=tarfile.TarInfo('MANIFEST.json');entry.size=len(content);archive.addfile(entry,io.BytesIO(content))
    verify_archive(destination)
    result=dict(status=status,path=str(destination),sha256=sha256(destination),payload_sha256=signature,
                binding=complete.get('binding'),missing=missing,archive_readback='PASS',files=len(inventory))
    destination.with_suffix(destination.suffix+'.sha256').write_text(result['sha256']+'  '+destination.name+'\n',encoding='utf-8')
    write_json(OUT/'package.json',result)
    return result
