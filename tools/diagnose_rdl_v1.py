"""固定 seed=42 的母版 train/val RDL 现象诊断；默认各128张，禁止test和长训。"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from init_c19_lif_v1 import ROOT, require, sha256, write_json, verify_model
from ultralytics import RTDETR
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.rdl import decompose
from ultralytics.utils.metrics import bbox_iou
from rdl_v1 import TRAINED_SHA, OUT, runtime


def load_batch(files,dataset,split,size=640):
    import cv2
    tensors=[];boxes=[];classes=[];indices=[];records=[]
    for i,path in enumerate(files):
        label=dataset/'labels'/split/path.relative_to(dataset/'images'/split).with_suffix('.txt')
        image=cv2.imread(str(path));require(image is not None,f'Unreadable image: {path}')
        rows=[list(map(float,line.split())) for line in label.read_text().splitlines() if line.strip()]
        require(all(len(row)==5 and row[0]==0 for row in rows),'GT schema differs')
        tensors.append(torch.from_numpy(cv2.resize(image,(size,size),interpolation=cv2.INTER_LINEAR)[:,:,::-1].copy()).permute(2,0,1).float()/255)
        boxes.extend(row[1:] for row in rows);classes.extend(int(row[0]) for row in rows);indices.extend([i]*len(rows))
        records.append(dict(image=path.relative_to(dataset).as_posix(),image_sha256=sha256(path),label_sha256=sha256(label),
                            original_hw=list(image.shape[:2]),instances=len(rows)))
    batch=dict(img=torch.stack(tensors),bboxes=torch.tensor(boxes).reshape(-1,4),cls=torch.tensor(classes,dtype=torch.long),
               batch_idx=torch.tensor(indices,dtype=torch.long),gt_groups=[r['instances'] for r in records])
    return batch,records


def summary(values):
    a=np.asarray(values,dtype=np.float64)
    if not a.size:return dict(count=0)
    require(np.isfinite(a).all(),'Nonfinite diagnostic statistic')
    return dict(count=int(a.size),mean=float(a.mean()),min=float(a.min()),p10=float(np.quantile(a,.1)),
                median=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),max=float(a.max()))


@torch.no_grad()
def diagnose(weights,dataset,device='cuda:0',batch_size=2,limit=128):
    weights,dataset=Path(weights).resolve(),Path(dataset).resolve()
    require(1<=limit<=128 and 1<=batch_size<=16,'Diagnosis is bounded to <=128 images/split and batch<=16')
    require(sha256(weights)==TRAINED_SHA,'Trained mother checkpoint provenance/SHA differs')
    report=dict(status='FAIL',runtime=runtime(),checkpoint=str(weights),checkpoint_sha256=TRAINED_SHA,seed=42,
                preprocessing='eval FP32; OpenCV INTER_LINEAR stretch 640x640; BGR->RGB /255; no random augmentation',
                batch=batch_size,limit_per_split=limit,device=device,scope='mechanism only; no AP evaluation; train/val only')
    samples={};pair_rows=[]
    try:
        model=RTDETR(str(weights)).model.float().to(device).eval();verify_model(model)
        criterion=RTDETRDetectionLoss(nc=1,use_vfl=True)
        for split in ('train','val'):
            files=sorted(p for p in (dataset/'images'/split).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
            require(files,'Missing split '+split)
            files=random.Random(42).sample(files,min(limit,len(files)));samples[split]=[]
            for start in range(0,len(files),batch_size):
                batch,records=load_batch(files[start:start+batch_size],dataset,split)
                samples[split].extend(records)
                batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                pred,details=model.predict(batch['img'],return_cbr_details=True)
                b,s,eb,es,dn=pred[1];require(dn is None,'Eval diagnostics must have no DN')
                indices=criterion.matcher(b[-1],s[-1],batch['bboxes'],batch['cls'],batch['gt_groups'])
                idx,gt_idx=criterion._get_index(indices)
                before,after,v=details['before'][idx],details['after'][idx],details['tanh_offsets'][idx]
                gt=batch['bboxes'][gt_idx]
                out,inside,ev=decompose(before,gt,v,batch['img'].shape[-2:])
                iou0=bbox_iou(before,gt,xywh=True).flatten().cpu().tolist()
                iou1=bbox_iou(after,gt,xywh=True).flatten().cpu().tolist()
                evidence={k:value.cpu().tolist() for k,value in ev.items()}
                short=(gt[:,2:]*640).min(-1).values.cpu().tolist()
                source_idx,query_idx=idx[0].tolist(),idx[1].tolist()
                for j,(im,q) in enumerate(zip(source_idx,query_idx)):
                    row=dict(split=split,image=records[im]['image'],query=q,short_edge_gt_px=short[j],
                             iou_before=iou0[j],iou_after=iou1[j])
                    row.update({k:value[j] for k,value in evidence.items()});pair_rows.append(row)
                print(f'diagnose {split}: {min(start+batch_size,len(files))}/{len(files)}',flush=True)
        report['splits']={}
        for split in ('train','val'):
            rows=[r for r in pair_rows if r['split']==split]
            def aggregate(selected):
                edges=max(4*len(selected),1)
                result=dict(M=len(selected),outside_side_fraction=sum(abs(e)>a for r in selected for e,a in zip(r['ebar'],r['a']))/edges,
                            saturated_target_fraction=sum(abs(v)==1 for r in selected for v in r['vstar'])/edges,
                            L_out=sum(sum(r['out_sides']) for r in selected)/edges,L_in=sum(sum(r['in_sides']) for r in selected)/edges,
                            a_protected=sum(sum(r['a_protected']) for r in selected),e_protected=sum(sum(r['e_protected']) for r in selected))
                for key in ('a','S','abs_v','eta','ebar'):result[key]=summary([abs(v) if key=='ebar' else v for r in selected for v in r[key]])
                for key in ('iou_before','iou_after'):result[key]=summary([r[key] for r in selected])
                return result
            totals=aggregate(rows);totals['images']=len(samples[split]);totals['L_RDL']=totals['L_out']+totals['L_in']
            totals['short_edge_gt_pixels']={name:aggregate([r for r in rows if low<=r['short_edge_gt_px']<high])
                 for name,low,high in (('lt8',0,8),('8to16',8,16),('16to32',16,32),('ge32',32,float('inf')))}
            report['splits'][split]=totals
        require(sha256(weights)==TRAINED_SHA,'Checkpoint changed during diagnosis')
        report.update(status='PASS',interpretation='Descriptive only; cannot establish AP gain or global novelty. Full training and independent val/test required.')
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        folder=OUT/'diagnosis';folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'samples.json',samples);write_json(folder/'pairs.json',pair_rows)
        report['samples_sha256']=sha256(folder/'samples.json');report['pairs_sha256']=sha256(folder/'pairs.json')
        write_json(folder/'summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--batch',type=int,default=2);parser.add_argument('--limit',type=int,default=128)
    args=parser.parse_args();torch.set_num_threads(4);diagnose(args.weights,args.dataset,args.device,args.batch,args.limit)
