"""Read-only split fingerprints and bounded training-image loading."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from init_c19_lif_v1 import require,sha256

def dataset_inventory(dataset):
    """Read paths/labels only; no image decoding or full-split model evaluation."""
    result={}
    for split in ('train','val','test'):
        files=sorted(p for p in (dataset/'images'/split).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
        require(files,'Missing split '+split)
        paths=[];labels=[];count=0
        for image in files:
            label=dataset/'labels'/split/image.relative_to(dataset/'images'/split).with_suffix('.txt')
            raw=label.read_bytes();rows=[l.split() for l in raw.decode().splitlines() if l.strip()]
            require(all(len(r)==5 and int(r[0])==0 and np.isfinite(np.asarray(r,dtype=float)).all() for r in rows),'Invalid label')
            paths.append(image.relative_to(dataset).as_posix());labels.append(label.relative_to(dataset).as_posix()+':'+sha256(label));count+=len(rows)
        result[split]=dict(images=len(files),boxes=count,split_paths_sha256=hashlib.sha256('\n'.join(paths).encode()).hexdigest(),
                           label_inventory_sha256=hashlib.sha256('\n'.join(labels).encode()).hexdigest())
    return result


def real_batch(dataset,size=160,count=2):
    import cv2
    files=sorted((dataset/'images/train').glob('*.jpg'))[:count]
    require(len(files)==count,'Insufficient bounded real train samples')
    tensors=[];boxes=[];classes=[];indices=[];records=[]
    for i,p in enumerate(files):
        image=cv2.imread(str(p));require(image is not None,'Unreadable image')
        label=dataset/'labels/train'/p.with_suffix('.txt').name
        rows=[list(map(float,line.split())) for line in label.read_text().splitlines() if line.strip()]
        tensors.append(torch.from_numpy(cv2.resize(image,(size,size))[:,:,::-1].copy()).permute(2,0,1).float()/255)
        boxes.extend(r[1:] for r in rows);classes.extend([r[0]] for r in rows);indices.extend([i]*len(rows))
        records.append(dict(image=p.name,image_sha256=sha256(p),label_sha256=sha256(label),instances=len(rows)))
    return dict(img=torch.stack(tensors),bboxes=torch.tensor(boxes).reshape(-1,4),cls=torch.tensor(classes).reshape(-1,1),
                batch_idx=torch.tensor(indices)),records
