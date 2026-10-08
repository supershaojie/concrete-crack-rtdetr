"""Synthetic-only short GPU lifecycle. Never touches formal data or test tuning."""
from pathlib import Path
import time
import uuid
import numpy as np
import cv2
import torch
import yaml
from configuration import resolve_config,freeze
from data import prepare_inputs
from lifecycle import identity,preflight_process,train
from export import export_split,evaluate_split,show_summary
from model import load_for_visualization
from packaging_run import package
from support import ROOT,read_json,write_json

def synthetic_data(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    for split,count in [('train',4),('val',2),('test',2)]:
        for i in range(count):
            (root/f'images/{split}').mkdir(parents=True,exist_ok=True)
            (root/f'labels/{split}').mkdir(parents=True,exist_ok=True)
            h,w=[(100,321),(79,101),(96,128),(81,143)][i%4]
            image=np.full((h,w,3),155,np.uint8); cv2.line(image,(w//3,h//4),(2*w//3,3*h//4),(25,25,25),2)
            cv2.imwrite(str(root/f'images/{split}/{i}.png'),image)
            label='' if split=='train' and i==3 else '0 0.5 0.5 0.5 0.5\n'
            (root/f'labels/{split}/{i}.txt').write_bytes(label.encode())
    config={'path':str(root.absolute()),'train':'images/train','val':'images/val','test':'images/test','names':{0:'crack'},'nc':1}
    target=root/'data.yaml'; target.write_bytes(yaml.safe_dump(config).encode()); return target

def run_smoke(base=None):
    base=Path(base or ROOT/'outputs'/('dfine-smoke-'+time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6])).absolute()
    if base.exists(): raise ValueError('Smoke base must be a new isolated directory')
    base.mkdir(parents=True)
    data_yaml=synthetic_data(base/'synthetic_data')
    cfg=resolve_config({'epochs':2,'batch_size':1,'eval_batch_size':1,'train_workers':0,'eval_workers':0,
        'warmup_epochs':0,'close_mosaic':1,'preflight_steps':1,'print_freq':1,
        'data_yaml':str(data_yaml),'enforce_counts':False,'device':'cuda:0' if torch.cuda.is_available() else 'cpu'})
    run=base/'SMOKE_ONLY_lifecycle'; raw=yaml.safe_dump(cfg).encode(); freeze(run,run.name,cfg,raw,{},smoke=True)
    manifest=prepare_inputs(cfg,run); write_json(run/'identity.json',identity(run,cfg,manifest))
    preflight_process(run); train(run,epoch_budget=1)
    last=torch.load(run/'train/weights/last.pth',map_location='cpu',weights_only=False)
    first_updates=last['scheduler']['updates']; first_ema=last['ema']['updates']; del last
    train(run,resume=run/'train/weights/last.pth')
    for split in ('val','test'):
        export_split(run,split,cfg,manifest); evaluate_split(run,split,cfg,manifest)
    before=read_json(run/'predictions/val_complete.json'); export_split(run,'val',cfg,manifest)
    assert before==read_json(run/'predictions/val_complete.json'); show_summary(run)
    model,info=load_for_visualization(run/'resolved_config.yaml',run/'train/weights/best.pth','cpu')
    assert next(model.parameters()).requires_grad
    selected=torch.load(run/'train/weights/best.pth',map_location='cpu',weights_only=False)
    assert all(torch.equal(v.cpu(),selected['selected_state_dict'][k]) for k,v in model.state_dict().items())
    model.backbone(torch.zeros(1,3,640,640))[0].sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.backbone.parameters())
    del model,selected
    closed=read_json(run/'augmentation/epoch_002_counts.json'); assert all(v['triggered']==0 for v in closed.values())
    final=torch.load(run/'train/weights/last.pth',map_location='cpu',weights_only=False)
    assert first_updates==first_ema and final['ema']['updates']==final['scheduler']['updates']==8
    assert final['augmentation_closed']
    assert all(abs(v-cfg['lrf']*target)<1e-12 for v,target in zip(
        [g['lr'] for g in final['optimizer']['param_groups']],final['scheduler']['targets']))
    receipt={'scope':'SYNTHETIC_ONLY_NOT_FORMAL_METRICS','run':str(run),'device':cfg['device'],'batch_size':1,'imgsz':640,
        'epochs':2,'real_resume':True,'optimizer_updates':8,'ema_updates':8,'closed_mixers':closed,
        'cached_export_reused':True,'selected_state_visualization_equal':True,'gradient_backbone_passed':True,
        'formal_batch16_capacity':'NOT_TESTED_BY_THIS_SMOKE','formal_dataset_or_test_used':False}
    write_json(run/'smoke_validation.json',receipt)
    result=package(run); receipt['package']=result; write_json(base/'validation.json',receipt)
    print('SMOKE PASSED: '+str(base/'validation.json'),flush=True); return receipt
