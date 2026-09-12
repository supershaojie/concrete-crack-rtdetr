"""Standard-library evidence packaging: no torch, checkpoint loading, GPU or inference."""
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import hashlib
import io
import json
import os
import tarfile
import uuid
from c24_lif_v1_common import *

LIMIT=8_000_000
PRIORITY=('first_failure','first_error','name','status','error','key','mode','device','shape','shape_a','shape_b',
    'finite','atol','rtol','max_abs','max_rel','max_coordinate','over_tolerance','candidate_ids_a','candidate_ids_b',
    'natural_relation','operator_status','boundary','replay','parent_control','stages','cases','continuous','result')

def compact(value,depth=0):
    if isinstance(value,dict):
        keys=[k for k in PRIORITY if k in value]+[k for k in value if k not in PRIORITY]
        keep=keys[:80]
        out={k:compact(value[k],depth+1) for k in keep}
        if len(keep)<len(keys):out['_omitted_keys']=dict(count=len(keys)-80,sample=keys[80:90])
        return out
    if isinstance(value,list):
        if len(value)>1000:return dict(original_items=len(value),first=[compact(v,depth+1) for v in value[:12]],last=[compact(v,depth+1) for v in value[-4:]],summary=True)
        return [compact(v,depth+1) for v in value]
    if isinstance(value,str) and len(value)>16000:return dict(characters=len(value),head=value[:1000],tail=value[-4000:],summary=True)
    return value

def encode(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def json_evidence(path):
    # Our stages are small by construction. Foreign giant records are never uploaded.
    if path.stat().st_size>64_000_000:
        return encode(dict(status='RETAINED_SERVER',path=str(path),bytes=path.stat().st_size,sha256=sha256(path),
            reason='Exceeds parser memory budget; inspect the separately included first-failure stage JSON.',
            extraction_command=f"python -c 'import json; p={str(path)!r}; r=json.load(open(p)); print(r.get(\"first_failure\",r.get(\"first_error\")))'"))
    try:
        value=json.loads(path.read_text(encoding='utf-8'));return encode(compact(value))
    except (ValueError,UnicodeError) as error:
        return encode(dict(status='INVALID_JSON',path=str(path),bytes=path.stat().st_size,error=str(error)))

def verify_archive(path):
    with tarfile.open(path,'r:gz') as t:
        names=t.getnames();require(len(names)==len(set(names)),'Duplicate archive member')
        manifest=json.load(t.extractfile('MANIFEST.json'))
        require(set(names)==set(manifest)|{'MANIFEST.json'},'Manifest mismatch')
        for name,row in manifest.items():
            pure=PurePosixPath(name);require(not pure.is_absolute() and '..' not in pure.parts,'Unsafe archive path')
            m=t.getmember(name);require(m.isfile() and m.size==row['bytes'],'Member size mismatch')
            require(hashlib.sha256(t.extractfile(name).read()).hexdigest()==row['sha256'],'Member hash mismatch')
    return manifest

def archive(entries,destination,limit=None):
    destination=Path(destination).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    require(not any(Path(str(destination)+s).exists() for s in ['', '.sha256','.inventory.json','.verification.json']),'Existing package preserved')
    manifest={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in entries.items()}
    payload={**entries,'MANIFEST.json':encode(manifest)}
    tmp=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('xb') as f,tarfile.open(fileobj=f,mode='w|gz') as t:
        for name,data in sorted(payload.items()):
            member=tarfile.TarInfo(name);member.size=len(data);t.addfile(member,io.BytesIO(data))
    require(limit is None or tmp.stat().st_size<=limit,f'Archive exceeds hard limit; not published: {tmp}')
    verify_archive(tmp);os.link(tmp,destination);tmp.unlink()
    digest=sha256(destination)
    with Path(str(destination)+'.sha256').open('x') as f:f.write(digest+'  '+destination.name+'\n')
    write_json(str(destination)+'.inventory.json',manifest)
    write_json(str(destination)+'.verification.json',dict(status='PASSED',bytes=destination.stat().st_size,sha256=digest,members=len(manifest)))
    print(f'PACKAGE {destination}\nBYTES {destination.stat().st_size}\nSHA256 {digest}',flush=True)
    return destination

def pack_light(launch=None,destination=None):
    p=paths();launch=Path(launch or p['launch']).resolve()
    destination=destination or p['download']/('c24_lif_v1_LIGHT_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.tar.gz')
    entries={};reductions=[]
    for f in sorted(launch.rglob('*')) if launch.exists() else []:
        if not f.is_file() or f.is_symlink() or any(part.startswith(('init_','history_','update_','fuse_reload_','preflight_previous_','previous_launch_')) for part in f.relative_to(launch).parts):continue
        # Only current experiment metadata and stage reports. No prediction collections.
        if f.suffix=='.json':
            name='metadata/'+f.relative_to(launch).as_posix()
            if f.stat().st_size<=250_000:
                try:json.loads(f.read_text(encoding='utf-8'));entries[name]=f.read_bytes()
                except (ValueError,UnicodeError):entries[name]=json_evidence(f)
            else:
                entries[name]=json_evidence(f);reductions.append(dict(path=str(f),action='structured summary',original_bytes=f.stat().st_size))
        elif f.suffix in ('.log','.txt','.yaml','.sh') and f.name not in ('predictions_gt.jsonl',):
            cap=160_000 if f.suffix=='.log' else 40_000
            with f.open('rb') as stream:
                start=max(0,f.stat().st_size-cap);stream.seek(start);content=stream.read(cap)
            entries['excerpts/'+f.relative_to(launch).as_posix()+'.txt']=f'Original {f}; bytes {start}:{start+len(content)}\n'.encode()+content
            if start:reductions.append(dict(path=str(f),action='tail only',omitted_bytes=start))
    for f in sorted((ROOT/'docs/c24_lif_v1').glob('*')):
        if f.is_file() and f.suffix in ('.json','.yaml','.md'):
            entries['source_docs/'+f.name]=json_evidence(f) if f.suffix=='.json' else f.read_bytes()[:40000]
    for name in ['ultralytics-main/ultralytics/nn/modules/scca_aifi.py','ultralytics-main/ultralytics/nn/modules/lif_down.py',
                 'tools/c24_lif_v1_numerics.py','tools/check_c24_lif_v1.py','tools/train_c24_lif_v1.py']:
        f=ROOT/name
        if f.is_file():entries['source/'+name]=f.read_bytes()
    # Deterministic reduction before compression; incompressible input also fits.
    for prefix in ('excerpts/','source/','source_docs/'):
        for name in list(entries):
            if sum(map(len,entries.values()))<=6_500_000:break
            if name.startswith(prefix):reductions.append(dict(path=name,action='omitted for byte budget'));entries.pop(name)
    if sum(map(len,entries.values()))>6_500_000:
        for name,data in sorted(list(entries.items()),key=lambda kv:len(kv[1]),reverse=True):
            if sum(map(len,entries.values()))<=6_500_000:break
            if name.startswith('metadata/'):
                value=json.loads(data);entries[name]=encode(compact(value));reductions.append(dict(path=name,action='additional structured summary'))
    entries['PACKAGE.json']=encode(dict(experiment=EXPERIMENT,commit=git('rev-parse','HEAD'),created=datetime.now(timezone.utc).isoformat(),
        kind='diagnostic_light',hard_limit=LIMIT,model_deserialization=False,preflight_rerun=False,reductions=reductions,
        omitted_by_policy=['weights','activations','fixtures','predictions','images','datasets','nested_source_archives']))
    require(sum(map(len,entries.values()))<=7_500_000,'Critical evidence alone exceeds budget; package refused, server files retained')
    return archive(entries,destination,LIMIT)

def pack_complete():
    p=paths();launch=p['launch'];run=p['run']
    from train_c24_lif_v1 import run_state,verify_delivery
    verify_delivery();require(run_state()=='SUCCESS','Successful completed training required')
    reports={s:read_json(launch/('evaluation_'+s)/'metrics.json') for s in ('val','test')}
    require(all(r and r.get('status')=='completed' for r in reports.values()),'Missing completed val/test; packaging never runs evaluation')
    best=sha256(run/'weights/best.pt');last=sha256(run/'weights/last.pt')
    require(all(r['checkpoint_sha256']==best and r['policy']=='corrected_sorted_conf_mask_v1' and r['data_sha256']==sha256(p['data']) for r in reports.values()),'Evaluation identity mismatch')
    require((reports['val']['images'],reports['val']['ground_truth'])==(1728,12840) and
        (reports['test']['images'],reports['test']['ground_truth'])==(864,6663),'Split coverage mismatch')
    entries={};missing=[]
    for s,folder in [('training',run),('val',launch/'evaluation_val'),('test',launch/'evaluation_test')]:
        for f in folder.rglob('*'):
            if not f.is_file() or f.is_symlink():continue
            if f.name in ['metrics.json','args.yaml','results.csv','results.png'] or f.suffix=='.png' and any(v in f.name for v in ['PR_curve','F1_curve','P_curve','R_curve','confusion_matrix']):
                entries[s+'/'+f.relative_to(folder).as_posix()]=f.read_bytes()
        for curve in ['PR_curve','F1_curve','P_curve','R_curve','confusion_matrix']:
            if not any(k.startswith(s+'/') and curve in k for k in entries):missing.append(s+'/'+curve)
    require(not missing,'Missing curves: '+str(missing))
    for name in ['initialization.json','nc1_loading.json','training_setup.json','actual_train_args.yaml','train_args.yaml','parameter_diff.json',
        'authoritative_c2_args.yaml','data_config.yaml','environment.json','pip_freeze.txt','plan.json','state.json','exit_code.json','process_exit_code.txt']:
        f=launch/name;require(f.is_file(),'Missing evidence: '+name);entries['launch/'+name]=f.read_bytes()
    for f in (launch/'preflight').glob('*.json'):entries['preflight/'+f.name]=json_evidence(f)
    for name in ['console.log','preflight.log']:
        f=launch/name
        with f.open('rb') as stream:stream.seek(max(0,f.stat().st_size-200000));entries['log_tails/'+name+'.txt']=stream.read()
    source=subprocess.check_output(['git','archive','--format=tar.gz','HEAD','ultralytics-main/ultralytics','tools','docs/c24_lif_v1','.gitattributes'],cwd=ROOT)
    entries['source_at_commit.tar.gz']=source
    predictions={s:dict(status='EXPORTED_SERVER_ONLY' if (launch/('evaluation_'+s)/'predictions_gt.jsonl.gz').is_file() else 'NOT_RUN',
        sha256=r.get('predictions_gt_sha256')) for s,r in reports.items()}
    entries['PACKAGE.json']=encode(dict(kind='complete_analysis_evidence',commit=git('rev-parse','HEAD'),best_sha256=best,last_sha256=last,
        weights='retained server; excluded by default',predictions=predictions,missing=[],precision_recall='Each model own F1 operating point'))
    return archive(entries,p['download']/('c24_lif_v1_complete_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.tar.gz'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack-light','pack-complete']);p.add_argument('--launch',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode=='pack-light':pack_light(a.launch,a.output)
    else:require(a.launch is None and a.output is None,'Complete pack uses fixed experiment paths');pack_complete()
