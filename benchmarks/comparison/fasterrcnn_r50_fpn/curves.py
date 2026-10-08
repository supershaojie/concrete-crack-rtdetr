"""Real training rows and curves returned by the existing public metric code."""
from __future__ import annotations
import csv
import io
from pathlib import Path
import numpy as np
from support import atomic_bytes, read_json, write_json

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def training_curves(run,history):
    if not history: return
    run=Path(run); folder=run/'train'; folder.mkdir(exist_ok=True)
    losses=('loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg')
    metrics=('P','R','AP50','AP75','mAP50-95')
    columns=['epoch',*losses,'total_loss','lr_first','lr_last','optimizer_steps','amp_scale','amp_enabled',*['val_'+k for k in metrics],'best_epoch']
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
    for row in history:
        writer.writerow({**{k:row[k] for k in ('epoch','total_loss','lr_first','lr_last','optimizer_steps','amp_scale','amp_enabled','best_epoch')},
            **row['losses'], **{'val_'+k:row['val_raw_0_1'][k] for k in metrics}})
    atomic_bytes(folder/'results.csv',stream.getvalue().encode())
    plt=plotting(); fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    x=[r['epoch'] for r in history]
    for ax,key in zip(axes.flat,(*losses,'total_loss','val')):
        if key=='val':
            for metric in metrics: ax.plot(x,[r['val_raw_0_1'][metric] for r in history],label=metric)
            ax.set_ylim(0,1); ax.legend(fontsize=8); ax.set_ylabel('Fraction')
        else:
            ax.plot(x,[r['losses'][key] if key in losses else r[key] for r in history]); ax.set_ylabel('Native loss')
        ax.set_title(key); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    axes.flat[4].twinx().plot(x,[r['lr_last'] for r in history],color='orange',label='LR last')
    buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=150); plt.close(fig)
    atomic_bytes(folder/'results.png',buf.getvalue())

def public_curves(run,split,gt_path,pred_path,result):
    import torch
    from unittest.mock import patch
    from export import evaluator_api
    from native_metrics import ap_per_class, smooth
    api=evaluator_api(); truths=read_json(gt_path); collected=[]; scores=[]
    original=api.Matcher
    class Capture(original):
        def match_predictions(self,*args,**kwargs):
            value=super().match_predictions(*args,**kwargs); collected.append(value.numpy()); return value
    _,records=api.read_public(pred_path)
    def capture_rows():
        for row in records:
            confidence=np.asarray([p['score'] for p in row['predictions']],dtype=np.float32)
            confidence=confidence[np.argsort(-confidence,kind='stable')]
            scores.append(confidence[confidence>api.POLICY['conf']][:api.POLICY['max_det']])
            yield row
    with patch.object(api,'Matcher',Capture): verified=api.evaluate(truths,capture_rows(),result['identity'])
    for k in ('precision','recall','AP50','AP75','mAP50_95'):
        if verified[k]!=result[k]: raise ValueError('Curve replay differs from public metrics')
    values=ap_per_class(np.concatenate(collected),np.concatenate(scores),np.ones(result['metric_predictions']),np.ones(result['ground_truth']),plot=False)
    p,r,f1,x,pr=values[7],values[8],values[9],values[10],values[11]
    point=int(smooth(f1.mean(0),.1).argmax())
    folder=Path(run)/'curves'/split; folder.mkdir(parents=True,exist_ok=True)
    raw={'identity':result['identity'],'policy':api.POLICY,'definition':{
        'PR':'precision envelope interpolated by native ap_per_class at IoU0.50 on recall grid',
        'confidence':'native interpolated P/R/F1 at IoU0.50; curves stored unsmoothed',
        'working_point':'argmax(native smooth(mean F1,0.1)); raw values preserved',
        'nms':'none added by curve replay','confusion_matrix':'not generated'},
        'x_0_1':x.tolist(),'P':p[0].tolist(),'R':r[0].tolist(),'F1':f1[0].tolist(),'PR_precision':pr[0].tolist(),
        'working_point':{'index':point,'confidence':float(x[point]),'P':float(p[0,point]),'R':float(r[0,point]),'F1':float(f1[0,point])}}
    write_json(folder/'curve_values.json',raw)
    plt=plotting()
    for name,y,label in [('PR',pr[0],'Recall'),('P',p[0],'Confidence'),('R',r[0],'Confidence'),('F1',f1[0],'Confidence')]:
        fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
        ax.plot(x,y,color='#165d9c'); ax.set(xlim=(0,1),ylim=(0,1),xlabel=label,ylabel='Precision' if name=='PR' else name,
            title=f'{split.upper()} {name} (public, IoU 0.50)'); ax.grid(alpha=.2)
        buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=160); plt.close(fig)
        atomic_bytes(folder/(name+'_curve.png'),buf.getvalue())

def native_auxiliary(run,split,gt_path,pred_path):
    """Official detection reference COCOeval is auxiliary; never replaces public selection/table."""
    from export import evaluator_api
    folder=Path(run)/'native_auxiliary'; folder.mkdir(exist_ok=True)
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        write_json(folder/(split+'.json'),{'status':'UNAVAILABLE_LOCAL_OPTIONAL_DEPENDENCY','main_metrics':'public metrics unchanged'})
        return
    import contextlib
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        coco=COCO(str(gt_path)); _,rows=evaluator_api().read_public(pred_path); detections=[]
        for row in rows:
            for item in row['predictions']:
                x1,y1,x2,y2=item['bbox']
                detections.append({'image_id':row['image_id'],'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'score':item['score']})
        if detections: dt=coco.loadRes(detections)
        else:
            dt=COCO(); dt.dataset={'images':coco.dataset['images'],'categories':coco.dataset['categories'],'annotations':[]}; dt.createIndex()
        ev=COCOeval(coco,dt,'bbox'); ev.params.imgIds=sorted(coco.getImgIds()); ev.evaluate(); ev.accumulate(); ev.summarize()
    atomic_bytes(folder/(split+'.log'),log.getvalue().encode())
    write_json(folder/(split+'.json'),{'status':'COMPLETED_AUXILIARY_COCOEVAL','reference':'torchvision v0.16.2 references/detection/coco_eval.py',
        'stats':ev.stats.tolist(),'maxDets':ev.params.maxDets,'units':'fraction','primary_policy':'corrected_sorted_conf_mask_v1',
        'selection_used':False,'score_nms':'already native model output; no extra NMS'})
