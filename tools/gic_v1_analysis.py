"""Offline GIC sensitivity and detection error summaries from saved query/GT exports."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from gic_v1_common import OUT,require,sha256,read_json,write_json
import numpy as np
import torch
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.models.rtdetr.gic_loss import geometry_interval,positive_delta,aggregate
from ultralytics.utils.metrics import bbox_iou,box_iou
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.engine.validator import BaseValidator


def canonical(row):
    if 'predictions' in row:
        require(row['box_format']=='xyxy' and row['coordinate_space']=='original_image_pixels','Unknown mother export coordinates')
        h,w=row['original_size_hw'];scale=torch.tensor([640/w,640/h,640/w,640/h])
        boxes=torch.tensor([p['bbox'] for p in row['predictions']],dtype=torch.float32).reshape(-1,4)*scale
        scores=torch.tensor([p['score'] for p in row['predictions']],dtype=torch.float32)
        gt=torch.tensor([g['bbox'] for g in row['ground_truth']],dtype=torch.float32).reshape(-1,4)*scale
        return row['image'],boxes,scores,gt,(640,640)
    require(row['box_space']=='input_image_pixels_xyxy','Unknown GIC export coordinates')
    return row['image_id'],torch.tensor(row['boxes'],dtype=torch.float32).reshape(-1,4),torch.tensor(row['scores'],dtype=torch.float32),torch.tensor(row['gt_boxes'],dtype=torch.float32).reshape(-1,4),tuple(row['input_size_hw'])


def detection_pairs(gt,boxes,iou_threshold=.5):
    """Mother BaseValidator.match_predictions greedy matching, retaining GT IDs.

    No Hungarian assignment here. This is checked against the native TP mask.
    """
    iou=box_iou(gt,boxes).cpu().numpy()
    pairs=np.array(np.nonzero(iou>=iou_threshold)).T
    if len(pairs)>1:
        pairs=pairs[iou[pairs[:,0],pairs[:,1]].argsort()[::-1]]
        pairs=pairs[np.unique(pairs[:,1],return_index=True)[1]]
        pairs=pairs[np.unique(pairs[:,0],return_index=True)[1]]
    native=BaseValidator.match_predictions(SimpleNamespace(iouv=torch.tensor([iou_threshold])),torch.zeros(len(boxes)),torch.zeros(len(gt)),torch.from_numpy(iou))[:,0]
    mask=torch.zeros(len(boxes),dtype=torch.bool)
    if len(pairs):mask[pairs[:,1]]=True
    require(torch.equal(mask,native.cpu()),'Offline detection pairing differs from mother TP matcher')
    return pairs,iou


def analyze(path,limit=None,threshold=None):
    matcher=HungarianMatcher(cost_gain={'class':2,'bbox':5,'giou':2})
    gathered={k:[] for k in ('z','q','lo','hi','t','delta','gt')}
    images=[];saturated=0;input_hw=None
    scale_stats={k:dict(gt=0,tp=0,fn=0,matched_score_sum=0.) for k in ('lt4','4to16','16to32','ge32')}
    errors=dict(tp=0,fp=0,fn=0,duplicate_or_unassigned_iou_ge50=0,localization_iou10to50=0,background_iou_lt10=0)
    with gzip.open(path,'rt',encoding='utf-8') as stream,torch.no_grad():
        for line in stream:
            if limit is not None and len(images)>=limit:break
            image,boxes,scores,gt,hw=canonical(json.loads(line));images.append(image)
            require(len(scores)==300 and bool(torch.isfinite(scores).all()),'Incomplete/nonfinite all-query export')
            require(bool(((scores>=0)&(scores<=1)).all()),'Invalid probability')
            require(input_hw is None or input_hw==hw,'Mixed input dimensions need separate geometry summaries')
            input_hw=hw;h,w=hw
            normalizer=boxes.new_tensor([w,h,w,h]);b=xyxy2xywh(boxes)/normalizer;g=xyxy2xywh(gt)/normalizer
            # Inverse sigmoid is used only for offline Hungarian mechanism diagnostics.
            # Endpoints keep exact 0/1 matcher probabilities and are excluded from delta reconstruction.
            logits=torch.logit(scores)
            indices=matcher(b[None],logits[None,:,None],g,torch.zeros(len(g),dtype=torch.long),[len(g)])[0]
            src,dst=indices
            finite=torch.isfinite(logits[src]);saturated+=int((~finite).sum());src,dst=src[finite],dst[finite]
            if len(src):
                q=bbox_iou(b[src],g[dst],xywh=True).squeeze(-1)
                lo,hi=geometry_interval(b[src],g[dst],q,hw)
                d,t=positive_delta(logits[src],q,lo,hi,.5)
                for k,value in zip(gathered,(logits[src],q,lo,hi,t,d,g[dst])):gathered[k].append(value)
            if threshold is not None:
                order=scores.argsort(descending=True);keep=order[scores[order]>threshold]
                pairs,ious=detection_pairs(gt,boxes[keep])
                matched_gt=set(pairs[:,0].tolist());matched_pred=set(pairs[:,1].tolist())
                short=(gt[:,2:]-gt[:,:2]).min(-1).values
                gt_to_pred={int(a):int(b) for a,b in pairs}
                for i,side in enumerate(short.tolist()):
                    key='lt4' if side<4 else '4to16' if side<16 else '16to32' if side<32 else 'ge32'
                    r=scale_stats[key];r['gt']+=1;r['tp']+=int(i in matched_gt);r['fn']+=int(i not in matched_gt)
                    if i in gt_to_pred:r['matched_score_sum']+=float(scores[keep[gt_to_pred[i]]])
                errors['tp']+=len(pairs);errors['fp']+=len(keep)-len(pairs);errors['fn']+=len(gt)-len(pairs)
                for i in range(len(keep)):
                    if i in matched_pred:continue
                    overlap=float(ious[:,i].max()) if len(gt) else 0.
                    key='duplicate_or_unassigned_iou_ge50' if overlap>=.5 else 'localization_iou10to50' if overlap>=.1 else 'background_iou_lt10'
                    errors[key]+=1
    merged={k:torch.cat(v) for k,v in gathered.items()} if gathered['z'] else None
    diagnostic=aggregate(merged['z'],merged['q'],merged['lo'],merged['hi'],merged['t'],merged['delta'],merged['gt'],input_hw,.5) if merged else {'matched':0}
    for r in scale_stats.values():
        r['recall_iou50']=r['tp']/r['gt'] if r['gt'] else None
        r['matched_score_mean']=r.pop('matched_score_sum')/r['tp'] if r['tp'] else None
    return dict(images=len(images),image_ids=images,source_sha256=sha256(path),diagnostic=diagnostic,
                saturated_logits_excluded=saturated,threshold=threshold,scale_recall=scale_stats if threshold is not None else None,
                errors=errors if threshold is not None else None,
                scope='offline final-box Hungarian mechanism diagnostic; TP/FP/FN separately use native detection greedy matching',
                limitation='q is reconstructed from exported geometry; logits are inverse-sigmoid reconstructed; not the original training batch targets or AP benefit evidence')


def analyze_exports():
    records={s:read_json(OUT/f'evaluation_{s}.json',{}) for s in ('val','test')}
    require(all(r.get('status')=='COMPLETE' for r in records.values()),'Need successful val/test exports')
    sources={s:r['artifacts']['queries_gt.jsonl.gz']['sha256'] for s,r in records.items()}
    threshold=records['val']['native_max_f1_threshold']
    prior=read_json(OUT/'gic_analysis.json',{})
    best=records['val']['best_sha256']
    if prior.get('status')=='COMPLETE' and prior.get('source_sha256')==sources and prior.get('best_sha256')==best and prior.get('threshold')==threshold:return prior
    result=dict(status='COMPLETE',best_sha256=best,source_sha256=sources,threshold=threshold,
                scale_space='640x640 model input pixels',threshold_selection='val only; frozen for test',splits={})
    for s,r in records.items():
        path=Path(r['folder'])/'queries_gt.jsonl.gz';require(sha256(path)==sources[s],'Export changed')
        result['splits'][s]=analyze(path,threshold=threshold)
    write_json(OUT/'gic_analysis.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mother-val-export',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=64);args=parser.parse_args()
    require(0<args.limit<=64,'Mother diagnostic is bounded to 64 exported val images')
    torch.set_num_threads(4)
    result=analyze(args.mother_val_export,limit=args.limit)
    result.update(status='PASS',scope='fixed first <=64 mother val exports; no inference, no test, no eta/shift tuning')
    write_json(args.output,result);print(json.dumps({k:v for k,v in result.items() if k!='image_ids'},indent=2))
