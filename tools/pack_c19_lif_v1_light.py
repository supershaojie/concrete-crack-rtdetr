"""Upload-safe diagnostic archive: JSON, bounded log tails, source; hard 8 MB cap."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

MAX_BYTES=8_000_000
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def compact(value,depth=0):
    """Structured omission markers; never truncate text and pretend it is JSON."""
    if depth>5:return dict(omitted='deep detail',type=type(value).__name__,count=len(value) if isinstance(value,(dict,list)) else None)
    if isinstance(value,dict):
        if 'max_abs_error' in value:return {k:value[k] for k in ('key','status','stage','max_abs_error','max_rel_error','allclose_failed_count','index','a','b') if k in value}
        return {k:compact(v,depth+1) for k,v in value.items() if k not in ('gradients','graph','dataset','samples','mapping','trainer_rebuild','parameters')}
    if isinstance(value,list):
        if len(value)>32:return dict(omitted='long list in total-report summary',count=len(value))
        return [compact(v,depth+1) for v in value]
    if isinstance(value,str) and len(value)>4096:return dict(omitted='long string',characters=len(value),prefix=value[:1024])
    return value


def package(preflight,output,launch=None,limit=MAX_BYTES):
    if not 0<limit<=MAX_BYTES:raise ValueError('Archive limit must be at most 8,000,000 bytes')
    preflight=Path(preflight).resolve(strict=True);output=Path(output)
    if output.exists():raise FileExistsError('Existing LIGHT archive preserved: '+str(output))
    entries={};omitted=[]
    def json_entry(name,value):entries[name]=(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')
    # Only named evidence, not arbitrary datasets or user files. No .pt/.npz/images.
    patterns=['fuse_diagnostic.json','checks.json','precision_smoke.json','preflight_only.json','initialization.json']
    for name in patterns:
        for path in sorted(preflight.rglob(name)):
            if path.is_symlink():omitted.append(dict(path=str(path),reason='symlink'));continue
            relative=path.relative_to(preflight).as_posix()
            if path.stat().st_size>16_000_000:
                json_entry(relative+'.summary.json',dict(path=relative,bytes=path.stat().st_size,sha256=digest(path),omitted='oversized JSON; no partial JSON emitted'));continue
            data=json.loads(path.read_text(encoding='utf-8'))
            # Per-mode diagnostics keep all candidate logits, A/B IDs/margins and
            # replay comparisons. Only redundant aggregate reports are summarized.
            if name in ('checks.json','initialization.json','preflight_only.json','precision_smoke.json'):
                json_entry(relative+'.summary.json',dict(original_sha256=digest(path),original_bytes=path.stat().st_size,summary=compact(data)))
            else:json_entry(relative,data)
    folders=[preflight]+([Path(launch).resolve(strict=True)] if launch else [])
    for directory in folders:
        for name in ('preflight.log','console.log','launch_state.json','preflight_only.json'):
            path=directory/name
            if not path.is_file() or path.is_symlink():continue
            relative=('launch/' if directory!=preflight else '')+name
            if name.endswith('.json'):json_entry(relative,json.loads(path.read_text(encoding='utf-8')))
            else:
                with path.open('rb') as f:f.seek(max(0,path.stat().st_size-32768));tail=f.read(32768)
                entries[relative+'.tail.txt']=tail
    for name in ('c19_lif_v1_diagnostic.py','c19_lif_v1_cutoff.py','c19_lif_v1_probe.py','check_c19_lif_v1.py',
                 'c19_lif_v1_precision.py','train_c19_lif_v1.py','preflight_c19_lif_v1.py','pack_c19_lif_v1_light.py'):
        entries['tools/'+name]=(ROOT/'tools'/name).read_bytes()
    info=dict(schema='c19_lif_light_v2',limit_bytes=limit,created=datetime.now(timezone.utc).isoformat(),
        preflight=str(preflight),excluded='all weights, .pt, full records, images, datasets, old archives',
        total_report='structured summary only; per-mode candidate/replay diagnostics retained',omitted=omitted,
        files={name:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in entries.items()})
    json_entry('LIGHT_PACKAGE_INFO.json',info)
    class CappedBuffer(io.BytesIO):
        def write(self,data):
            if self.tell()+len(data)>limit:raise ValueError('LIGHT archive exceeds hard size limit; no archive delivered')
            return super().write(data)
    buffer=CappedBuffer()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in entries.items():
            item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
    data=buffer.getvalue()
    if len(data)>limit:raise ValueError('LIGHT cap exceeded')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:f.write(data)
    result=dict(path=str(output.resolve()),bytes=len(data),limit_bytes=limit,sha256=hashlib.sha256(data).hexdigest(),files=len(entries))
    output.with_suffix(output.suffix+'.sha256').write_text(result['sha256']+'  '+output.name+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--launch',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();package(a.preflight,a.output,a.launch)
