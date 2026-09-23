"""已训练母版 train/val 固定抽样诊断；不计算 AP、不读取 test、不启动训练。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import cv2
import torch
from init_c19_lif_v1 import require, sha256, verify_model, write_json, runtime
from ultralytics import RTDETR
from ultralytics.models.utils.ror import matched_loss
from ultralytics.models.utils.loss import RTDETRDetectionLoss

TRAINED_SHA = "24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa"


def load_images(files, dataset, device, size=640):
    images, boxes, classes, groups, records = [], [], [], [], []
    for path in files:
        image = cv2.imread(str(path)); require(image is not None, "Unreadable image: " + str(path))
        relative = path.relative_to(dataset / "images")
        label = (dataset / "labels" / relative).with_suffix('.txt')
        rows = [[float(v) for v in line.split()] for line in label.read_text().splitlines() if line.strip()]
        require(all(len(row)==5 and row[0]==0 for row in rows), "Invalid crack label")
        # Native RTDETRDataset.load_image: stretch, no letterbox, cv2.INTER_LINEAR.
        images.append(torch.from_numpy(cv2.resize(image,(size,size),interpolation=cv2.INTER_LINEAR)[:,:,::-1].copy()).permute(2,0,1))
        boxes.extend(row[1:] for row in rows);classes.extend(row[0] for row in rows);groups.append(len(rows))
        records.append(dict(image=path.relative_to(dataset).as_posix(),image_sha256=sha256(path),
                            label_sha256=sha256(label),ground_truth=len(rows)))
    return torch.stack(images).to(device).float()/255, dict(
        bboxes=torch.tensor(boxes,device=device,dtype=torch.float32).reshape(-1,4),
        cls=torch.tensor(classes,device=device,dtype=torch.long),gt_groups=groups), records


def diagnose(weights, dataset, output, device='0', count=128, batch=4):
    output, weights, dataset = Path(output), Path(weights), Path(dataset)
    require(1<=count<=128 and batch>=1,"Bounded diagnosis requires 1..128 images/split")
    if not weights.is_file():
        report=dict(status='PENDING_ASSET',expected_sha256=TRAINED_SHA,path=str(weights),reason='Trained parent checkpoint missing')
        write_json(output/'diagnosis.json',report);return report
    require(sha256(weights)==TRAINED_SHA,"Diagnosis requires verified trained parent; filename is insufficient")
    require(not (output/'diagnosis.json').exists(),"Preserve previous diagnostic report; choose new output")
    dev=torch.device('cuda:'+device if device.isdigit() else device)
    model=RTDETR(str(weights)).model.float().to(dev).eval();verify_model(model)
    matcher=RTDETRDetectionLoss(nc=1,use_vfl=True).matcher
    report=dict(status='RUNNING',runtime=runtime(),weights=str(weights.resolve()),weights_sha256=TRAINED_SHA,
                seed=42,max_images_per_split=count,batch=batch,device=str(dev),precision='FP32',mode='eval, no augmentation',
                preprocessing='native RT-DETR stretch 640x640, OpenCV INTER_LINEAR, BGR->RGB, /255',splits={})
    try:
        for split in ('train','val'):
            candidates=sorted(p for p in (dataset/'images'/split).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
            require(candidates,"Missing "+split)
            selected=sorted(random.Random(42).sample(candidates,min(count,len(candidates))))
            records, batches, rows=[],[],[]
            for start in range(0,len(selected),batch):
                images,targets,manifest=load_images(selected[start:start+batch],dataset,dev)
                with torch.no_grad():
                    result,details=model.predict(images,return_cbr_details=True)
                    boxes,scores,_,_,meta=result[1]
                    require(meta is None,"Eval diagnosis unexpectedly produced DN")
                    indices=matcher(boxes[-1],scores[-1],targets['bboxes'],targets['cls'],targets['gt_groups'])
                    raw,stats=matched_loss(details['before'],details['after'],scores[-1],targets,indices,detailed=True)
                for record,row in zip(manifest,stats): record.update(row)
                rows.extend(stats);records.extend(manifest)
                batches.append(dict(images=len(images),raw_loss=float(raw),full_weight_loss=float(raw)*.1))
            eligible=sum(r['eligible_pairs'] for r in rows);violating=sum(r['violating_pairs'] for r in rows)
            comparable=sum(r['comparable_pairs'] for r in rows)
            summary=dict(images=len(rows),positive_count_distribution=dict(Counter(r['positives'] for r in rows)),
                         images_with_two_same_class_positives=sum(r['comparable_pairs']>0 for r in rows),
                         comparable_unordered_pairs=comparable,eligible_pairs=eligible,violating_pairs=violating,
                         reversal_image_fraction=sum(r['eligible_pairs']>0 for r in rows)/len(rows),
                         violating_fraction_of_eligible=violating/eligible if eligible else 0.,
                         nonzero_batch_fraction=sum(r['raw_loss']>0 for r in batches)/len(batches),
                         batch_raw_mean=sum(r['raw_loss'] for r in batches)/len(batches),
                         batches=batches)
            report['splits'][split]=summary
            write_json(output/(split+'_samples.json'),records)
            print(split,{k:v for k,v in summary.items() if k!='batches'},flush=True)
        support=any(s['nonzero_batch_fraction']>0 for s in report['splits'].values())
        report['status']='OBSERVED_SUPPORT' if support else 'NO_OBSERVED_SUPPORT'
        report['interpretation']='Fixed sample mechanism evidence only; does not establish AP benefit or behavior throughout training.'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally: write_json(output/'diagnosis.json',report)
    return report
