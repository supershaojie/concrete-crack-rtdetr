#!/usr/bin/env python3
"""Inspect or gracefully stop only the two named legacy COCO jobs on this server.

Reads /proc and tmux. Writes a NEW audit directory; never edits old logs/checkpoints.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import pickletools
from pathlib import Path
import signal
import shutil
import subprocess
import time
import zipfile

BASE=Path('/root/autodl-tmp/projects')
MODELS=('yolov5m','yolov8m')


def processes():
    rows={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat=(entry/'stat').read_text().rsplit(')',1)[1].split()
            rows[int(entry.name)]={'pid':int(entry.name),'ppid':int(stat[1]),'start_ticks':stat[19],
                'argv':(entry/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0'),
                'cwd':str((entry/'cwd').resolve())}
        except (OSError,ValueError,IndexError):
            pass
    return rows


def under(path,root):
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def training_process(row,model):
    """Match exact entry script, action and contained output, never just 'python'."""
    root=BASE/('Crack_RTDETR-bench-'+model)
    argv=row['argv']
    for i,arg in enumerate(argv[:-1]):
        path=Path(arg)
        if not path.is_absolute():
            path=Path(row['cwd'])/path
        if path.resolve() not in {root/'benchmarks/comparison'/model/'run.py',
                                  root/'benchmarks/comparison'/model/'worker.py'}:
            continue
        if argv[i+1] not in ('train','pipeline') or '--run' not in argv:
            continue
        idx=argv.index('--run')+1
        if idx>=len(argv):
            continue
        run=Path(argv[idx])
        if not run.is_absolute():
            run=Path(row['cwd'])/run
        if under(run,root/'outputs'/model):
            return str(run.resolve())
    return None


def pane_inventory(model):
    command=['tmux','list-panes','-s','-t','=comparison-'+model,'-F',
             '#{pane_id}\t#{pane_pid}\t#{pane_current_path}\t#{pane_current_command}\t#{pane_start_command}']
    result=subprocess.run(command,capture_output=True,text=True)
    return {'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}


def snapshot(model):
    rows=processes()
    matches=[]
    for row in rows.values():
        run=training_process(row,model)
        if run:
            ancestry=[]
            parent=row['ppid']
            while parent in rows and parent not in ancestry:
                ancestry.append(parent)
                parent=rows[parent]['ppid']
            matches.append({**row,'run':run,'ancestor_pids':ancestry})
    # Inspect native train scripts in this worktree as well. Unknown ones block automatic stop/start.
    root=BASE/('Crack_RTDETR-bench-'+model)
    unrecognized=[r for r in rows.values() if under(r['cwd'],root)
                  and Path(r['argv'][0]).name.startswith('python')
                  and any(Path(x).name=='train.py' for x in r['argv'])
                  and r['pid'] not in {m['pid'] for m in matches}]
    return {'model':model,'session':pane_inventory(model),'training_processes':matches,
            'unrecognized_native_train_processes':unrecognized}


def checkpoint_metadata(path):
    # Read the serialized epoch scalar without unpickling executable model objects.
    with zipfile.ZipFile(path) as archive:
        failed=archive.testzip()
        if failed:
            raise ValueError('Incomplete checkpoint ZIP member: '+failed)
        names=[n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(names)!=1:
            raise ValueError('Ambiguous checkpoint pickle')
        expect_epoch=False
        for op,arg,_ in pickletools.genops(archive.read(names[0])):
            if expect_epoch and op.name in ('BININT','BININT1','BININT2','INT','LONG1','LONG4'):
                return {'checkpoint_epoch_zero_based':arg,'completed_epochs':arg+1,'zip_crc':'passed'}
            if expect_epoch and op.name not in ('BINPUT','LONG_BINPUT','MEMOIZE'):
                raise ValueError('Unexpected epoch serialization')
            if op.name in ('BINUNICODE','SHORT_BINUNICODE','UNICODE') and arg=='epoch':
                expect_epoch=True
    raise ValueError('Checkpoint epoch unavailable')


def preserve_checkpoints(run,model,destination):
    root=Path(run)
    weights=root/('native/weights' if model=='yolov5m' else 'train/weights')
    destination.mkdir(parents=True,exist_ok=False)
    receipts={}
    for name in ('last.pt','best.pt'):
        path=weights/name
        if not path.exists():
            continue  # No completed epoch yet; nothing to fabricate.
        last_error=None
        for attempt in range(3):
            target=destination/(str(attempt)+'_'+name)
            try:
                before=path.stat()
                shutil.copyfile(path,target)
                meta=checkpoint_metadata(target)
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise ValueError('Checkpoint changed while copying')
                receipts[name]={'preserved_copy':str(target),**meta}
                break
            except (OSError,ValueError,zipfile.BadZipFile) as exc:
                last_error=str(exc)
                time.sleep(1)
        else:
            raise RuntimeError('No verified stable '+name+'; no stop signal sent: '+str(last_error))
    (destination/'receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    return receipts


def archive_metadata(run,model):
    """Keep native status/CSV and inspect checkpoint epochs without unpickling model objects."""
    root=Path(run)
    paths=(['status.json','native/epoch_state.json','native/training_complete.json','native/results.csv']
           if model=='yolov5m' else ['train_status.json','training_progress.json','train/results.csv','identity.json'])
    evidence={}
    for name in paths:
        path=root/name
        if path.is_file():
            raw=path.read_bytes()
            value={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            if path.suffix=='.json':
                value['record']=json.loads(raw)
            elif path.suffix=='.csv':
                rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
                value['csv_rows']=len(rows)
                value['last_row']=rows[-1] if rows else None
                value['note']='CSV can be newer than the last complete checkpoint'
            evidence[name]=value
    weights=root/('native/weights' if model=='yolov5m' else 'train/weights')
    for path in weights.glob('*.pt'):
        stat=path.stat()
        value={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        try:
            value.update(checkpoint_metadata(path))
        except (OSError,ValueError,zipfile.BadZipFile) as exc:
            value['integrity_error']=str(exc)
            value['note']='Use the verified preserved pre-stop copy; incomplete current checkpoint is retained as evidence'
        evidence[str(path.relative_to(root))]=value
    return {'run':str(root),'initialization_type':'coco','pretraining_source':'COCO',
            'reported_status':'see native evidence; interrupted is never relabeled as 200 epochs',
            'evidence':evidence}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','stop','assert-stopped'])
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--audit-root',type=Path,required=True)
    a=p.parse_args()
    if os.name!='posix' or not Path('/proc').is_dir():
        p.error('Run this helper on the Linux server only')
    before=snapshot(a.model)
    print(json.dumps(before,ensure_ascii=False,indent=2),flush=True)
    out=a.audit_root/(a.model+'_'+a.action+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    (out/'before.json').write_text(json.dumps(before,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.action=='inspect':
        return
    if before['unrecognized_native_train_processes']:
        raise SystemExit('Unrecognized native trainer in the named worktree; inspect before stopping/starting')
    matches=before['training_processes']
    if a.action=='assert-stopped':
        if matches:
            raise SystemExit('Legacy training is still running; scratch launch must wait')
        print('No legacy '+a.model+' training process remains.')
        return
    # If both supervisor and worker match, signal the supervisor only.
    ids={r['pid'] for r in matches}
    controllers=[r for r in matches if not ids.intersection(r['ancestor_pids'])]
    for index,run in enumerate(sorted({r['run'] for r in matches})):
        print('Preserving verified complete checkpoints before signalling: '+run,flush=True)
        preserve_checkpoints(run,a.model,out/('checkpoints_'+str(index)))
    for row in controllers:
        current=processes().get(row['pid'])
        if current is None:
            continue
        if current['start_ticks']!=row['start_ticks'] or current['argv']!=row['argv'] or training_process(current,a.model)!=row['run']:
            raise SystemExit('Process identity changed; no signal sent')
        print('SIGINT to verified PID '+str(row['pid'])+': '+repr(row['argv']),flush=True)
        os.kill(row['pid'],signal.SIGINT)
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        after=snapshot(a.model)
        if not after['training_processes']:
            break
        time.sleep(1)
    after=snapshot(a.model)
    (out/'after.json').write_text(json.dumps(after,ensure_ascii=False,indent=2),encoding='utf-8')
    receipts=[archive_metadata(run,a.model) for run in sorted({r['run'] for r in matches})]
    (out/'preserved_runs.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
    if after['training_processes'] or after['unrecognized_native_train_processes']:
        raise SystemExit('Job has not exited; inspect audit. No SIGKILL or global process cleanup was attempted.')
    print('Verified legacy training stopped; original logs, last/best and actual epoch records retained: '+str(out))


if __name__=='__main__':
    main()
