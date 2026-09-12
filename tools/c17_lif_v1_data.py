"""Read-only split inventories; no dataset edits or inference."""
from pathlib import Path
import hashlib
import json
from init_c17_lif_v1 import require,sha256


def manifest(root):
    root=Path(root).resolve();report=dict(root=str(root),splits={})
    for split in ('train','val','test'):
        images=sorted(p for p in (root/'images'/split).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
        require(images,'Empty split: '+split)
        names=[];labels=[];instances=0
        for image in images:
            rel=image.relative_to(root).as_posix();names.append(rel)
            label=root/'labels'/split/image.relative_to(root/'images'/split).with_suffix('.txt')
            require(label.is_file(),'Missing label: '+str(label))
            data=label.read_bytes();rows=[r.split() for r in data.decode().splitlines() if r.strip()]
            require(all(len(r)==5 and r[0]=='0' for r in rows),'Invalid label format/class')
            instances+=len(rows);labels.append([label.relative_to(root).as_posix(),hashlib.sha256(data).hexdigest()])
        require(len(names)==len(set(names)),'Duplicate image path')
        report['splits'][split]=dict(images=len(names),instances=instances,
            split_paths_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
            label_contents_sha256=hashlib.sha256(json.dumps(labels,separators=(',',':')).encode()).hexdigest())
    return report


def verify_manifest(actual, expected):
    require(actual['splits']==expected['splits'],'Dataset split inventory/labels differ from audited local C2 data')
    require((actual['splits']['val']['images'],actual['splits']['val']['instances'])==(1728,12840),'Val counts changed')
    require((actual['splits']['test']['images'],actual['splits']['test']['instances'])==(864,6663),'Test counts changed')
