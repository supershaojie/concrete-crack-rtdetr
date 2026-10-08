"""FP32 native postprocessor -> one letterbox inverse -> unchanged public evaluator."""
from __future__ import annotations
import gzip
import math
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from data import EvalDataset,eval_collate,inverse_boxes
from support import canonical,digest,evaluator_api,read_json,sha256,status,write_json
POSTPROCESSING=('Official DFINEPostProcessor sigmoid once + flattened topk300 over 300 queries/class0; '
    'postprocessor sizes are input640 WH, single round-derived letterbox inverse; public category1; '
    'stable score ordering; conf>0.001/max300; no NMS, clipping, TTA, deploy or fusion; FP32/autocast disabled')


def precision():
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')

def infer_rows(model,post,gt,root,cfg,stage):
    precision(); model.float().eval(); post.float().eval()
    if post.num_top_queries<300 or post.num_classes!=1 or post.remap_mscoco_category: raise ValueError('Wrong final postprocessor')
    dataset=EvalDataset(gt,root)
    loader=torch.utils.data.DataLoader(dataset,batch_size=cfg['eval_batch_size'],num_workers=cfg['eval_workers'],
        shuffle=False,collate_fn=eval_collate,persistent_workers=False)
    start=time.monotonic(); seen=0
    with torch.no_grad():
        for batch_index,(images,records,metas) in enumerate(loader,1):
            images=images.to(cfg['device'],dtype=torch.float32)
            with torch.autocast(device_type=torch.device(cfg['device']).type,enabled=False):
                output=model(images)
                if output['pred_logits'].shape[1:]!=(300,1): raise ValueError('Wrong final logits/query shape')
                # Passing original sizes here would incorrectly stretch padding back to the image.
                native=post(output,torch.full((len(images),2),640.,device=images.device))
            for pred,im,meta in zip(native,records,metas):
                if pred['labels'].numel() and (pred['labels']!=0).any(): raise ValueError('Native postprocessor category must be0')
                boxes=inverse_boxes(pred['boxes'],meta); scores=pred['scores'].float()
                if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all(): raise ValueError('Nonfinite native predictions')
                order=torch.argsort(scores,descending=True,stable=True)
                order=order[scores[order]>cfg['eval_conf']][:cfg['eval_max_det']]
                rows=[]
                for index in order.tolist():
                    box=boxes[index].cpu().tolist(); score=float(scores[index])
                    if not 0<=score<=1 or box[2]<=box[0] or box[3]<=box[1]: raise ValueError('Invalid final original-pixel box/score')
                    rows.append({'category_id':1,'bbox':box,'score':score})
                yield {'split':gt['info']['split'],'image_id':im['id'],'image':im['file_name'],'width':im['width'],
                    'height':im['height'],'box_format':'xyxy','coordinate_space':'original_image_pixels',
                    'letterbox':meta,'predictions':rows}
                seen+=1
            elapsed=time.monotonic()-start
            print(f'{stage} batch {batch_index}/{len(loader)} images {seen}/{len(dataset)} elapsed={elapsed:.1f}s speed={seen/max(elapsed,1e-9):.2f} images/s',flush=True)

def evaluate_rows(gt,rows,identity,curve_path=None):
    api=evaluator_api(); captured={}; original=api.ap_per_class
    def capture(*args,**kwargs):
        values=original(*args,**kwargs)
        captured.update({k:np.asarray(v).tolist() for k,v in zip(
            ('p_curve','r_curve','f1_curve','confidence','pr_precision'),values[7:12])})
        return values
    api.ap_per_class=capture
    try: result=api.evaluate(gt,rows,identity)
    finally: api.ap_per_class=original
    if curve_path is not None:
        write_json(curve_path,{'policy_sha256':api.POLICY_SHA,'units':'raw 0-1','iou_for_confidence_curves':0.5,
            'definition':'Unchanged common native_metrics.ap_per_class return values; P/R are common maximum-F1 point',**captured})
    return result

def training_val(model,post,manifest,cfg,identity,epoch):
    gt=read_json(Path(identity['run_path'])/'data/gt/val.json')
    public={**identity,'checkpoint_sha256':digest(canonical({'in_memory_selected':identity['run_uuid'],'epoch':epoch})),
        'evaluation_config_sha256':evaluator_api().POLICY_SHA,'postprocessing':POSTPROCESSING,
        'selected_source':'ema' if cfg['ema'] else 'model'}
    return evaluate_rows(gt,infer_rows(model,post,gt,manifest['data_root'],cfg,'VAL epoch '+str(epoch+1)),public)

def final_identity(run,split,cfg,manifest):
    run=Path(run); identity=read_json(run/'identity.json'); finished=read_json(run/'train_status.json')
    if finished.get('status')!='completed' or finished.get('completed_epochs')!=cfg['epochs'] or finished.get('exit_code')!=0:
        raise ValueError('Final val/test requires verified complete epoch-limit training')
    best=run/'train/weights/best.pth'
    if sha256(best)!=finished['best_sha256']: raise ValueError('Selected best changed')
    return {**identity,'checkpoint_sha256':sha256(best),'selected_source':'ema' if cfg['ema'] else 'model',
        'selected_key':'selected_state_dict','gt_sha256':sha256(run/f'data/gt/{split}.json'),
        'evaluation_config_sha256':evaluator_api().POLICY_SHA,'postprocessing':POSTPROCESSING,
        'split':split,'inference':{'imgsz':640,'batch_size':cfg['eval_batch_size'],'workers':cfg['eval_workers'],
            'fp32':True,'autocast':False,'tf32':False,'conf':cfg['eval_conf'],'max_det':cfg['eval_max_det'],'nms':False}}

def validate_cache(run,split,expected):
    run=Path(run); path=run/f'predictions/{split}.jsonl.gz'; receipt=run/f'predictions/{split}_complete.json'
    if not path.exists() and not receipt.exists(): return None
    if not path.exists() or not receipt.exists(): raise ValueError('Incomplete final prediction cache; artifacts retained')
    done=read_json(receipt); old,rows=evaluator_api().read_public(path)
    if old!=expected or done['sha256']!=sha256(path) or done['identity_sha256']!=digest(canonical(expected)):
        raise ValueError('Final prediction cache identity/content differs')
    gt=read_json(run/f'data/gt/{split}.json')
    # Coverage/order validated without GPU; metrics validation also verifies every box/category/dimension.
    if [r['image_id'] for r in rows]!=sorted(im['id'] for im in gt['images']): raise ValueError('Final prediction cache image coverage differs')
    return done

def export_split(run,split,cfg,manifest):
    from model import build,selected_checkpoint
    from configuration import paths
    run=Path(run); expected=final_identity(run,split,cfg,manifest)
    cached=validate_cache(run,split,expected)
    if cached: print('CPU reusable prediction cache: '+split,flush=True); return cached
    status(run,'export_'+split,'running'); start=time.monotonic()
    selected=selected_checkpoint(run/'train/weights/best.pth',read_json(run/'identity.json'))
    model,_,post,_=build(cfg,paths(cfg)['source'],cfg['device'])
    model.load_state_dict(selected['selected_state_dict'],strict=True); del selected
    destination=run/f'predictions/{split}.jsonl.gz'; destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    gt=read_json(run/f'data/gt/{split}.json'); seen=[]
    with gzip.open(temporary,'wt',encoding='utf-8',newline='\n') as f:
        f.write(canonical({'type':'metadata','schema_version':1,'identity':expected}).decode())
        for row in infer_rows(model,post,gt,manifest['data_root'],cfg,split.upper()):
            f.write(canonical(row).decode()); seen.append(row['image_id'])
    if seen!=sorted(im['id'] for im in gt['images']): raise ValueError('Export incomplete')
    temporary.rename(destination)
    done={'status':'completed','images':len(seen),'batches':math.ceil(len(seen)/cfg['eval_batch_size']),
        'instances':len(gt['annotations']),'elapsed_seconds':time.monotonic()-start,'sha256':sha256(destination),
        'identity_sha256':digest(canonical(expected)),'checkpoint_sha256':expected['checkpoint_sha256'],
        'selected_source':expected['selected_source'],'selected_key':'selected_state_dict','FP32':True,'NMS':False}
    write_json(run/f'predictions/{split}_complete.json',done); status(run,'export_'+split,'completed',**{k:v for k,v in done.items() if k!='status'})
    return done

def evaluate_split(run,split,cfg,manifest):
    run=Path(run); expected=final_identity(run,split,cfg,manifest); done=validate_cache(run,split,expected)
    if done is None: raise ValueError('Export full split before CPU evaluation')
    output=run/f'metrics/{split}.json'; complete=run/f'metrics/{split}_complete.json'
    if output.exists():
        cached=read_json(output)
        if (not complete.exists() or read_json(complete)['sha256']!=sha256(output)
            or cached['identity']!=expected or cached['predictions_sha256']!=done['sha256']): raise ValueError('CPU metric cache differs')
        return cached
    status(run,'cpu_'+split,'running'); start=time.monotonic()
    _,rows=evaluator_api().read_public(run/f'predictions/{split}.jsonl.gz')
    result=evaluate_rows(read_json(run/f'data/gt/{split}.json'),rows,expected,run/f'curves/{split}_raw.json')
    result.update(gt_sha256=expected['gt_sha256'],predictions_sha256=done['sha256'],units='raw fractions [0,1]',
        elapsed_seconds=time.monotonic()-start)
    write_json(output,result); write_json(complete,{'sha256':sha256(output),'identity_sha256':digest(canonical(expected))})
    from plotting import plot_curves
    plot_curves(run,split)
    status(run,'cpu_'+split,'completed',elapsed_seconds=result['elapsed_seconds'])
    return result

def show_summary(run):
    run=Path(run); summary={}
    print('\nSPLIT  images instances batches seconds  P(%)  R(%)  AP50(%) AP75(%) mAP50-95(%)',flush=True)
    for split in ('val','test'):
        metric=read_json(run/f'metrics/{split}.json'); done=read_json(run/f'predictions/{split}_complete.json')
        values=[metric[k] for k in ('precision','recall','AP50','AP75','mAP50_95')]
        print(f'{split.upper():5} {metric["images"]:6} {metric["ground_truth"]:9} {done["batches"]:7} {done["elapsed_seconds"]:7.1f} '+
            ' '.join(f'{x*100:9.4f}' if x is not None else 'undefined' for x in values),flush=True)
        summary[split]={'raw_0_1':dict(zip(('P','R','AP50','AP75','mAP50_95'),values)),
            'display_percent':dict(zip(('P','R','AP50','AP75','mAP50_95'),[x*100 if x is not None else None for x in values])),
            'inference':done,'metrics_sha256':sha256(run/f'metrics/{split}.json')}
    write_json(run/'summary.json',{'policy':evaluator_api().POLICY,'same_selected_checkpoint':
        summary['val']['inference']['checkpoint_sha256']==summary['test']['inference']['checkpoint_sha256'],**summary})
    return summary
